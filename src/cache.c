#include "nv.h"
#include <assert.h>
#include <stdlib.h>

void nv_cache_init(NvCache *c, const NvConfig *p, NvOptions o) {
    *c = (NvCache){.num_blocks = o.num_blocks, .block_size = o.block_size,
        .layers = p->n_layers, .kv_dim = p->dim / p->n_heads * p->n_kv_heads,
        .enabled = o.prefix_cache};
    c->blocks = nv_alloc((size_t)c->num_blocks, sizeof(NvBlock));

    size_t count = (size_t)c->num_blocks;
    int factors[] = {c->block_size, c->layers, c->kv_dim};
    for (size_t i = 0; i < sizeof(factors) / sizeof(factors[0]); i++) {
        if (count > SIZE_MAX / (size_t)factors[i]) nv_fatal("KV cache size overflow");
        count *= (size_t)factors[i];
    }

    c->keys = nv_alloc(count, sizeof(float));
    c->values = nv_alloc(count, sizeof(float));
}

void nv_cache_free(NvCache *c) {
    free(c->blocks); free(c->keys); free(c->values);
    *c = (NvCache){0};
}

float *nv_cache_key(NvCache *c, int block, int layer, int offset) {
    return c->keys + (((size_t)block * (size_t)c->layers + (size_t)layer) *
           (size_t)c->block_size + (size_t)offset) * (size_t)c->kv_dim;
}

float *nv_cache_value(NvCache *c, int block, int layer, int offset) {
    return c->values + (((size_t)block * (size_t)c->layers + (size_t)layer) *
           (size_t)c->block_size + (size_t)offset) * (size_t)c->kv_dim;
}

int nv_cache_allocate(NvCache *c) {
    int best = -1;
    for (int i = 0; i < c->num_blocks; i++) {
        if (c->blocks[i].refs) continue;
        if (!c->blocks[i].prefix_len) { best = i; break; }

        if (best < 0 || c->blocks[i].age < c->blocks[best].age) best = i;
    }
    if (best < 0) return -1;
    c->blocks[best].refs = 1;
    c->blocks[best].prefix_len = 0;
    return best;
}

void nv_cache_release(NvCache *c, NvSequence *s) {
    for (int i = 0; i < s->num_blocks; i++) {
        NvBlock *b = &c->blocks[s->block_table[i]];
        assert(b->refs > 0);
        if (--b->refs == 0) b->age = ++c->clock;
    }
    s->computed = 0;
    s->num_blocks = 0;
}

/* FNV-1a 哈希链：hash(前驱哈希, 本块 token)。首块使用固定初值。
 * 同样的后缀 token 配上不同的前驱哈希，通常会得到不同的结果。
 * 复用仅比较前缀长度和哈希，接受哈希碰撞的可能性。
 */
static uint64_t hash_block(uint64_t hash, const int *tokens, int count) {
    for (int i = 0; i < count; i++) {
        /* 明确按低字节到高字节处理一个 token，避免依赖内存中的字节顺序。 */
        uint32_t token = (uint32_t)tokens[i];
        for (int byte = 0; byte < 4; byte++) {
            hash ^= (token >> (byte * 8)) & 255;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

void nv_cache_reuse(NvCache *c, NvSequence *s) {
    if (!c->enabled) return;

    /* 从请求开头逐块计算同一条哈希链，与发布缓存时的计算方式一致。 */
    uint64_t hash = UINT64_C(14695981039346656037);
    /* 至少留下最后一个输入 token 重新 forward，因为 KV 缓存中没有保存 logits。 */
    int limit = (s->num_tokens - 1) / c->block_size;
    for (int logical = 0; logical < limit; logical++) {
        int end = (logical + 1) * c->block_size;
        hash = hash_block(hash, s->tokens + logical * c->block_size, c->block_size);
        int found = -1;
        /* 长度和哈希相同即视为命中，线性扫描物理块。 */
        for (int i = 0; i < c->num_blocks; i++) {
            if (c->blocks[i].prefix_len == end && c->blocks[i].hash == hash) {
                found = i;
                break;
            }
        }
        /* 只能复用连续前缀。一块未命中，后面的 token 就交给模型继续计算。 */
        if (found < 0) break;
        /* 加入请求块表并增加引用；computed 前进后，引擎会跳过这些已有 K/V 的位置。 */
        c->blocks[found].refs++;
        s->block_table[s->num_blocks++] = found;
        s->computed += c->block_size;
        c->hits += (uint64_t)c->block_size;
    }
}

void nv_cache_publish(NvCache *c, const NvSequence *s) {
    /* 当前 token 的所有层已计算完成，且本块刚好写满时，才允许其他请求复用。 */
    if (!c->enabled || s->computed % c->block_size) return;
    int logical = s->computed / c->block_size - 1;
    NvBlock *b = &c->blocks[s->block_table[logical]];
    assert(b->refs == 1 && b->prefix_len == 0);
    /* 用当前请求的块表找到前一块，读取其已发布的哈希。
     * 只在计算新哈希时读取，无需保存前驱物理块编号。
     */
    uint64_t hash = logical > 0 ? c->blocks[s->block_table[logical - 1]].hash
                                : UINT64_C(14695981039346656037);
    b->hash = hash_block(hash, s->tokens + logical * c->block_size, c->block_size);
    /* 最后标记为已发布。例如块大小为 4，第二块的 prefix_len 是 8。 */
    b->prefix_len = s->computed;
}
