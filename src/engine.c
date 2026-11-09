#include "nv.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void nv_engine_init(NvEngine *e, NvModel *m, NvOptions o) {
    if (o.num_blocks <= 0 || o.block_size <= 0 || o.block_size > m->config.seq_len ||
        o.max_sequences <= 0 || o.token_budget <= 0) nv_fatal("invalid engine options");
    *e = (NvEngine){.model = m, .options = o};
    nv_cache_init(&e->cache, &m->config, o);
    e->candidates = nv_alloc((size_t)m->config.vocab_size, sizeof(NvCandidate));
}

void nv_engine_free(NvEngine *e) {
    for (int i = 0; i < e->num_requests; i++) {
        free(e->requests[i]->tokens);
        free(e->requests[i]->block_table);
        free(e->requests[i]);
    }
    free(e->requests); free(e->candidates);
    nv_cache_free(&e->cache);
    *e = (NvEngine){0};
}

int nv_engine_add(NvEngine *e, const int *tokens, int n, int max_new, NvSampling o) {
    int context = e->model->config.seq_len;
    if (!tokens || n < 1 || n > context || max_new < 1 || max_new > context - n + 1 ||
        !isfinite(o.temperature) || o.temperature < 0 || !isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1) return -1;

    size_t forwards = (size_t)n + (size_t)max_new - 1;

    if ((forwards - 1) / (size_t)e->options.block_size + 1 > (size_t)e->options.num_blocks) return -1;
    for (int i = 0; i < n; i++) if (tokens[i] < 0 || tokens[i] >= e->model->config.vocab_size) return -1;
    NvSequence *s = nv_alloc(1, sizeof(*s));

    *s = (NvSequence){.id = e->num_requests, .prompt_len = n, .num_tokens = n,
        .max_new_tokens = max_new, .sampling = o, .rng = o.seed ? o.seed : 1};

    s->tokens = nv_alloc((size_t)n + (size_t)max_new, sizeof(int));
    memcpy(s->tokens, tokens, (size_t)n * sizeof(int));

    s->block_table = nv_alloc(((size_t)context - 1) / (size_t)e->options.block_size + 1, sizeof(int));
    NvSequence **requests = realloc(e->requests, ((size_t)e->num_requests + 1) * sizeof(*requests));
    if (!requests) nv_fatal("out of memory adding request");
    e->requests = requests;
    e->requests[e->num_requests++] = s;
    return s->id;
}

int nv_engine_cancel(NvEngine *e, int id) {
    if (id < 0 || id >= e->num_requests) return 0;
    NvSequence *s = e->requests[id];
    if (s->status == NV_DONE || s->status == NV_CANCELLED) return 0;
    nv_cache_release(&e->cache, s);
    s->status = NV_CANCELLED;
    return 1;
}

static int allocate(NvEngine *e, NvSequence *s, int *active) {
    int block = nv_cache_allocate(&e->cache);
    if (block >= 0) return block;

    for (int i = e->num_requests - 1; i > s->id; i--) {
        NvSequence *victim = e->requests[i];
        if (victim->status != NV_ACTIVE) continue;
        nv_cache_release(&e->cache, victim);
        victim->status = NV_WAITING;
        (*active)--;
        e->preemptions++;
        block = nv_cache_allocate(&e->cache);
        if (block >= 0) return block;
    }
    return -1;
}

int nv_engine_step(NvEngine *e) {
    int active = 0, used = 0, scheduled = 0;

    for (int i = 0; i < e->num_requests; i++) if (e->requests[i]->status == NV_ACTIVE) active++;
    for (int i = 0; i < e->num_requests && used < e->options.token_budget; i++) {
        NvSequence *s = e->requests[i];
        if (s->status == NV_DONE || s->status == NV_CANCELLED) continue;
        if (s->status == NV_WAITING) {
            if (active >= e->options.max_sequences) continue;

            nv_cache_reuse(&e->cache, s);
            s->status = NV_ACTIVE;
            active++;
        }

        if (scheduled >= e->options.max_sequences) break;
        scheduled++;

        int quota = e->options.token_budget / e->options.max_sequences;
        if (quota < 1) quota = 1;
        for (int j = 0; j < quota && used < e->options.token_budget; j++) {
            if (s->computed / e->options.block_size == s->num_blocks) {
                int block = allocate(e, s, &active);
                if (block < 0) break;
                s->block_table[s->num_blocks++] = block;
            }

            float *logits = nv_forward(e->model, &e->cache, s->block_table,
                                      s->tokens[s->computed], s->computed);
            s->computed++;
            used++;
            e->forward_tokens++;

            nv_cache_publish(&e->cache, s);

            if (s->computed < s->num_tokens) continue;

            int token = nv_sample(logits, e->model->config.vocab_size, s->sampling,
                                  &s->rng, e->candidates);
            s->tokens[s->num_tokens++] = token;
            e->generated_tokens++;

            if (s->num_tokens - s->prompt_len == s->max_new_tokens ||
                (!s->sampling.ignore_eos && (token == 1 || token == 2))) {
                s->status = NV_DONE;
                nv_cache_release(&e->cache, s);
                active--;
            }

            break;
        }
    }
    e->steps++;

    if (!used) {
        for (int i = 0; i < e->num_requests; i++)
            assert(e->requests[i]->status == NV_DONE || e->requests[i]->status == NV_CANCELLED);
    }
    return used;
}
