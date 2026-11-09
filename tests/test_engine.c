#include "nv.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static NvSampling greedy = {0, 1, 123, 1};

static const int prompt[] = {1, 259, 260, 261, 262, 260, 261, 262, 260};

static void invariant(const NvEngine *e) {
    for (int b = 0; b < e->cache.num_blocks; b++) {
        int refs = 0;
        for (int i = 0; i < e->num_requests; i++) {
            const NvSequence *s = e->requests[i];
            for (int j = 0; j < s->num_blocks; j++) refs += s->block_table[j] == b;
        }
        assert(refs == e->cache.blocks[b].refs);
    }
}

static void drain(NvEngine *e) {
    for (int step = 0; step < 10000; step++) {
        int used = nv_engine_step(e);
        assert(used <= e->options.token_budget);
        invariant(e);
        if (!used) return;
    }
    assert(!"scheduler did not make progress");
}

static void oracle(const char *root, const char *name) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s.bin", root, name);
    NvModel m;
    nv_model_load(&m, path);
    NvCache cache;
    nv_cache_init(&cache, &m.config, (NvOptions){32, 3, 1, 1, 1});
    snprintf(path, sizeof(path), "%s/%s.oracle", root, name);
    FILE *file = fopen(path, "rb");
    assert(file);
    int count;
    assert(fread(&count, 4, 1, file) == 1);
    int *tokens = nv_alloc((size_t)count, sizeof(int));
    assert(fread(tokens, 4, (size_t)count, file) == (size_t)count);

    int table[] = {7, 2, 19, 4, 11};
    float worst = 0;
    for (int t = 0; t < count; t++) {
        float *logits = nv_forward(&m, &cache, table, tokens[t], t);
        for (int i = 0; i < m.config.vocab_size; i++) {
            float expected;
            assert(fread(&expected, 4, 1, file) == 1);
            float error = fabsf(logits[i] - expected);
            if (error > worst) worst = error;

            assert(isfinite(logits[i]) && error < 2e-5f);
        }
    }
    printf("PASS %s dense oracle, max logit error %.9g\n", name, (double)worst);
    fclose(file); free(tokens); nv_cache_free(&cache); nv_model_free(&m);
}

static void batching(NvModel *m, NvSampling sampling) {
    NvEngine serial, batch;
    nv_engine_init(&serial, m, (NvOptions){16, 4, 1, 8, 0});
    nv_engine_init(&batch, m, (NvOptions){16, 4, 3, 24, 1});
    for (int i = 0; i < 3; i++) {
        int tokens[9];
        memcpy(tokens, prompt, sizeof(tokens));
        tokens[8] += i;

        assert(nv_engine_add(&serial, tokens, 9, 8, sampling) == i);
        assert(nv_engine_add(&batch, tokens, 9, 8, sampling) == i);
    }
    drain(&serial); drain(&batch);
    for (int i = 0; i < 3; i++) {
        assert(serial.requests[i]->num_tokens == 17);
        assert(batch.requests[i]->num_tokens == 17);

        assert(memcmp(serial.requests[i]->tokens, batch.requests[i]->tokens, 17 * sizeof(int)) == 0);
    }
    assert(batch.cache.hits > 0);
    assert(batch.forward_tokens < serial.forward_tokens);
    nv_engine_free(&serial); nv_engine_free(&batch);
    puts("PASS serial/batch/cache token equivalence");
}

static void pressure(NvModel *m, NvSampling sampling, int cache) {
    NvEngine serial, tight;
    nv_engine_init(&serial, m, (NvOptions){16, 4, 1, 8, 0});
    nv_engine_init(&tight, m, (NvOptions){4, 4, 3, 9, cache});
    for (int i = 0; i < 3; i++) {
        assert(nv_engine_add(&serial, prompt, 9, 8, sampling) == i);
        assert(nv_engine_add(&tight, prompt, 9, 8, sampling) == i);
    }
    drain(&serial); drain(&tight);
    assert(tight.preemptions > 0);
    for (int i = 0; i < 3; i++)
        assert(memcmp(serial.requests[i]->tokens, tight.requests[i]->tokens, 17 * sizeof(int)) == 0);
    nv_engine_free(&serial); nv_engine_free(&tight);
    puts("PASS KV pressure/preemption/replay token equivalence");
}

static void lifecycle(NvModel *m) {
    NvEngine e;
    nv_engine_init(&e, m, (NvOptions){8, 4, 2, 3, 1});
    assert(nv_engine_add(&e, prompt, 9, 8, greedy) == 0);
    assert(nv_engine_step(&e) > 0);
    assert(nv_engine_add(&e, prompt, 9, 8, greedy) == 1);
    assert(nv_engine_step(&e) > 0);
    assert(nv_engine_cancel(&e, 0));
    assert(!nv_engine_cancel(&e, 0));
    assert(!nv_engine_cancel(&e, 99));
    drain(&e);
    assert(e.requests[0]->status == NV_CANCELLED);
    assert(e.requests[1]->status == NV_DONE);

    uint64_t hits = e.cache.hits;
    assert(nv_engine_add(&e, prompt, 9, 8, greedy) == 2);
    drain(&e);
    assert(e.cache.hits > hits);

    assert(memcmp(e.requests[1]->tokens, e.requests[2]->tokens, 17 * sizeof(int)) == 0);

    assert(nv_engine_add(&e, prompt, 9, 60, greedy) == -1);
    assert(nv_engine_add(&e, prompt, 9, 40, greedy) == -1);
    int bad[] = {-1};
    assert(nv_engine_add(&e, bad, 1, 1, greedy) == -1);
    bad[0] = m->config.vocab_size;
    assert(nv_engine_add(&e, bad, 1, 1, greedy) == -1);
    assert(nv_engine_add(&e, prompt, 9, 0, greedy) == -1);
    nv_engine_free(&e);
    puts("PASS live admission, cancellation, idle prefix reuse, input bounds");
}

static void page_sizes(NvModel *m) {
    NvEngine serial;
    nv_engine_init(&serial, m, (NvOptions){64, 1, 1, 32, 0});
    assert(nv_engine_add(&serial, prompt, 9, 10, greedy) == 0);
    drain(&serial);
    int sizes[] = {1, 2, 3, 4, 7, 16, 64};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        NvEngine e;
        nv_engine_init(&e, m, (NvOptions){64, sizes[i], 3, 1, 1});
        for (int j = 0; j < 3; j++) assert(nv_engine_add(&e, prompt, 9, 10, greedy) == j);
        drain(&e);

        for (int j = 0; j < 3; j++)
            assert(memcmp(serial.requests[0]->tokens, e.requests[j]->tokens, 19 * sizeof(int)) == 0);
        nv_engine_free(&e);
    }

    NvEngine full;
    nv_engine_init(&full, m, (NvOptions){16, 4, 1, 32, 1});
    assert(nv_engine_add(&full, prompt, 9, 56, greedy) == 0);
    drain(&full);
    assert(full.requests[0]->num_tokens == 65);
    nv_engine_free(&full); nv_engine_free(&serial);
    puts("PASS varied page sizes, one-token budgets, full context boundary");
}

static void tokenizer(const char *root) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/tokenizer.bin", root);
    NvTokenizer t;
    nv_tokenizer_load(&t, path, 263);
    int n;
    int *tokens = nv_encode(&t, "ab", &n);
    assert(n == 3 && tokens[0] == 1 && tokens[1] == 259 && tokens[2] == 262);
    free(tokens);
    tokens = nv_encode(&t, "é", &n);
    assert(n == 4 && tokens[2] == 198 && tokens[3] == 172);
    free(tokens);
    tokens = nv_encode(&t, "", &n);
    assert(n == 1 && tokens[0] == 1);
    free(tokens);
    assert(strcmp(nv_decode(&t, 1, 259), "") == 0);
    assert((unsigned char)nv_decode(&t, 260, 198)[0] == 195);
    nv_tokenizer_free(&t);
    puts("PASS BPE merge, UTF-8 byte fallback, BOS/dummy prefix");
}

static void sampler(void) {
    float logits[] = {1, 10, -3};
    NvCandidate scratch[3];
    uint64_t rng = 1;
    assert(nv_sample(logits, 3, greedy, &rng, scratch) == 1);
    NvSampling o = {1, 0.01f, 1, 1};
    for (int i = 0; i < 50; i++) {
        float values[] = {1, 10, -3};
        assert(nv_sample(values, 3, o, &rng, scratch) == 1);
    }
    puts("PASS greedy and top-p nucleus sampling");
}

/* 检查相同后缀、不同前文的哈希不同，以及前一块重新分配后的缓存复用。 */
static void hash_cache(NvModel *m) {
    NvCache cache;
    nv_cache_init(&cache, &m->config, (NvOptions){4, 2, 1, 8, 1});
    /* 每块两个 token。A/B 首块不同，第二块同为 [260,261]，最后一个 token 留待计算。 */
    int a_tokens[] = {1, 259, 260, 261, 262};
    int b_tokens[] = {1, 260, 260, 261, 262};
    int a_table[2], b_table[2], query_table[2], replacement_table[1];
    NvSequence a = {.tokens = a_tokens, .num_tokens = 5, .block_table = a_table};
    NvSequence b = {.tokens = b_tokens, .num_tokens = 5, .block_table = b_table};
    NvSequence *sources[] = {&a, &b};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            sources[i]->block_table[j] = nv_cache_allocate(&cache);
            sources[i]->num_blocks++;
            sources[i]->computed += 2;
            nv_cache_publish(&cache, sources[i]);
        }
    }
    nv_cache_release(&cache, &a);
    nv_cache_release(&cache, &b);
    NvSequence query = {.tokens = b_tokens, .num_tokens = 5, .block_table = query_table};
    nv_cache_reuse(&cache, &query);
    if (query.computed != 4 || query_table[0] != b_table[0] || query_table[1] != b_table[1])
        nv_fatal("prefix hash did not distinguish different histories");
    nv_cache_release(&cache, &query);
    /* 重新分配 A 的首块，再发布相同前缀。旧第二块的前缀哈希仍然有效。 */
    int recycled = nv_cache_allocate(&cache);
    if (recycled != a_table[0]) nv_fatal("expected oldest idle prefix block to be recycled");
    replacement_table[0] = recycled;
    NvSequence replacement = {.tokens = a_tokens, .num_tokens = 5, .computed = 2,
        .block_table = replacement_table, .num_blocks = 1};
    nv_cache_publish(&cache, &replacement);
    nv_cache_release(&cache, &replacement);
    query.tokens = a_tokens;
    nv_cache_reuse(&cache, &query);
    if (query.computed != 4 || query_table[1] != a_table[1])
        nv_fatal("prefix hash changed after recycling and rebuilding an identical prefix");
    nv_cache_release(&cache, &query);
    nv_cache_free(&cache);
    puts("PASS prefix hash separation and reuse after block recycling");
}

int main(int argc, char **argv) {
    assert(argc == 2);
    oracle(argv[1], "gqa"); oracle(argv[1], "mha"); tokenizer(argv[1]); sampler();
    char path[4096];
    snprintf(path, sizeof(path), "%s/gqa.bin", argv[1]);
    NvModel m;
    nv_model_load(&m, path);
    batching(&m, greedy);

    batching(&m, (NvSampling){0.8f, 0.9f, 567, 1});
    pressure(&m, greedy, 0);
    pressure(&m, greedy, 1);
    pressure(&m, (NvSampling){0.8f, 0.9f, 567, 1}, 0);
    pressure(&m, (NvSampling){0.8f, 0.9f, 567, 1}, 1);
    lifecycle(&m); page_sizes(&m); hash_cache(&m);
    nv_model_free(&m);
    puts("All C tests passed");
    return 0;
}
