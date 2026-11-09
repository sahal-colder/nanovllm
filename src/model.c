#include "nv.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

void nv_fatal(const char *message) {
    fprintf(stderr, "nanovllm: %s\n", message);
    exit(EXIT_FAILURE);
}

void *nv_alloc(size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) nv_fatal("allocation size overflow");
    void *p = calloc(count ? count : 1, size);
    if (!p) nv_fatal("out of memory");
    return p;
}

static float *tensor(NvModel *m, size_t *offset, size_t a, size_t b, size_t c) {
    size_t available = (m->mapping_size - *offset) / sizeof(float);
    if (a > available / b / c) nv_fatal("truncated checkpoint or invalid dimensions");
    float *p = (float *)((char *)m->mapping + *offset);
    *offset += a * b * c * sizeof(float);
    return p;
}

void nv_model_load(NvModel *m, const char *path) {
    *m = (NvModel){0};

    const uint32_t endian = 1;
    if (*(const unsigned char *)&endian != 1 || sizeof(float) != 4 || sizeof(int) != 4)
        nv_fatal("requires little-endian 32-bit int/float host");
    int fd = open(path, O_RDONLY);
    if (fd < 0) nv_fatal("cannot open model; run python3 tools/download.py");
    struct stat st;

    if (fstat(fd, &st) || st.st_size < 28 || (uintmax_t)st.st_size > SIZE_MAX)
        nv_fatal("invalid checkpoint size");
    m->mapping_size = (size_t)st.st_size;

    m->mapping = mmap(NULL, m->mapping_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->mapping == MAP_FAILED) nv_fatal("cannot map checkpoint");

    memcpy(&m->config, m->mapping, 28);
    NvConfig *p = &m->config;

    if (p->dim <= 0 || p->hidden_dim <= 0 || p->n_layers <= 0 || p->n_heads <= 0 ||
        p->n_kv_heads <= 0 || p->seq_len <= 0 || p->vocab_size == INT_MIN ||
        abs(p->vocab_size) < 259 || p->dim % p->n_heads ||
        (p->dim / p->n_heads) % 2 || p->n_heads % p->n_kv_heads)
        nv_fatal("invalid model config (expected llama2.c legacy FP32 format)");

    int shared = p->vocab_size > 0;
    p->vocab_size = abs(p->vocab_size);

    size_t d = (size_t)p->dim, h = (size_t)p->hidden_dim, l = (size_t)p->n_layers;
    size_t kv = (d / (size_t)p->n_heads) * (size_t)p->n_kv_heads;
    size_t offset = 28;
    NvWeights *w = &m->weights;

    w->embedding = tensor(m, &offset, (size_t)p->vocab_size, d, 1);
    w->att_norm = tensor(m, &offset, l, d, 1);
    w->wq = tensor(m, &offset, l, d, d);
    w->wk = tensor(m, &offset, l, kv, d);
    w->wv = tensor(m, &offset, l, kv, d);
    w->wo = tensor(m, &offset, l, d, d);
    w->ffn_norm = tensor(m, &offset, l, d, 1);
    w->w1 = tensor(m, &offset, l, h, d);
    w->w2 = tensor(m, &offset, l, d, h);
    w->w3 = tensor(m, &offset, l, h, d);
    w->final_norm = tensor(m, &offset, d, 1, 1);

    (void)tensor(m, &offset, (size_t)p->seq_len, d / (size_t)p->n_heads, 1);

    w->classifier = shared ? w->embedding : tensor(m, &offset, (size_t)p->vocab_size, d, 1);

    m->x = nv_alloc(d, sizeof(float));
    m->xb = nv_alloc(d, sizeof(float));
    m->xb2 = nv_alloc(d, sizeof(float));
    m->q = nv_alloc(d, sizeof(float));
    m->hb = nv_alloc(h, sizeof(float));
    m->hb2 = nv_alloc(h, sizeof(float));
    m->att = nv_alloc((size_t)p->n_heads * (size_t)p->seq_len, sizeof(float));
    m->logits = nv_alloc((size_t)p->vocab_size, sizeof(float));
}

void nv_model_free(NvModel *m) {
    munmap(m->mapping, m->mapping_size);
    free(m->x); free(m->xb); free(m->xb2); free(m->q);
    free(m->hb); free(m->hb2); free(m->att); free(m->logits);

    *m = (NvModel){0};
}

static void rmsnorm(float *out, const float *x, const float *weight, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / (float)n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = weight[i] * (scale * x[i]);
}

void nv_softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void matmul(float *out, const float *x, const float *w, int in, int rows) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < in; j++) sum += w[(size_t)i * (size_t)in + (size_t)j] * x[j];
        out[i] = sum;
    }
}

float *nv_forward(NvModel *m, NvCache *cache, const int *table, int token, int pos) {
    NvConfig *p = &m->config;
    NvWeights *w = &m->weights;
    int d = p->dim, h = p->hidden_dim, hs = d / p->n_heads;
    int kv = hs * p->n_kv_heads, group = p->n_heads / p->n_kv_heads;

    size_t dd = (size_t)d * (size_t)d, kd = (size_t)kv * (size_t)d;
    size_t hd = (size_t)h * (size_t)d;

    memcpy(m->x, w->embedding + (size_t)token * (size_t)d, (size_t)d * sizeof(float));
    for (int layer = 0; layer < p->n_layers; layer++) {
        size_t l = (size_t)layer;

        rmsnorm(m->xb, m->x, w->att_norm + l * (size_t)d, d);

        float *k = nv_cache_key(cache, table[pos / cache->block_size], layer, pos % cache->block_size);
        float *v = nv_cache_value(cache, table[pos / cache->block_size], layer, pos % cache->block_size);
        matmul(m->q, m->xb, w->wq + l * dd, d, d);
        matmul(k, m->xb, w->wk + l * kd, d, kv);
        matmul(v, m->xb, w->wv + l * kd, d, kv);

        for (int i = 0; i < d; i += 2) {
            float angle = (float)pos / powf(10000.0f, (float)(i % hs) / (float)hs);
            float cs = cosf(angle), sn = sinf(angle), q0 = m->q[i], q1 = m->q[i + 1];
            m->q[i] = q0 * cs - q1 * sn;
            m->q[i + 1] = q0 * sn + q1 * cs;
            if (i < kv) {
                float k0 = k[i], k1 = k[i + 1];
                k[i] = k0 * cs - k1 * sn;
                k[i + 1] = k0 * sn + k1 * cs;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int head = 0; head < p->n_heads; head++) {
            float *q = m->q + head * hs;

            float *att = m->att + (size_t)head * (size_t)p->seq_len;

            for (int t = 0; t <= pos; t++) {
                const float *kt = nv_cache_key(cache, table[t / cache->block_size], layer,
                                               t % cache->block_size) + (head / group) * hs;
                float score = 0;
                for (int j = 0; j < hs; j++) score += q[j] * kt[j];

                att[t] = score / sqrtf((float)hs);
            }
            nv_softmax(att, pos + 1);

            float *out = m->xb + head * hs;
            memset(out, 0, (size_t)hs * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                const float *vt = nv_cache_value(cache, table[t / cache->block_size], layer,
                                                 t % cache->block_size) + (head / group) * hs;
                for (int j = 0; j < hs; j++) out[j] += att[t] * vt[j];
            }
        }

        matmul(m->xb2, m->xb, w->wo + l * dd, d, d);
        for (int i = 0; i < d; i++) m->x[i] += m->xb2[i];

        rmsnorm(m->xb, m->x, w->ffn_norm + l * (size_t)d, d);
        matmul(m->hb, m->xb, w->w1 + l * hd, d, h);
        matmul(m->hb2, m->xb, w->w3 + l * hd, d, h);

        for (int i = 0; i < h; i++) m->hb[i] = (m->hb[i] * (1.0f / (1.0f + expf(-m->hb[i])))) * m->hb2[i];

        matmul(m->xb, m->hb, w->w2 + l * hd, h, d);
        for (int i = 0; i < d; i++) m->x[i] += m->xb[i];
    }

    rmsnorm(m->x, m->x, w->final_norm, d);

    matmul(m->logits, m->x, w->classifier, d, p->vocab_size);

    return m->logits;
}
