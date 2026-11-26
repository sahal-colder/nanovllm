#ifndef NV_ATTENTION_H
#define NV_ATTENTION_H
#include "miniinfer/operators.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Host shape + device-only pointers. A per-layer physical page contains
 * [2, page_tokens, kv_heads, head_dim], K then V. The pointer table permits
 * independently allocated physical pages. block_table is [batch,max_blocks],
 * lengths is [batch], q/out are contiguous [batch,heads,head_dim].
 * Inputs need only element alignment; head_dim 1..256, heads%kv_heads==0,
 * capacity max_blocks*page_tokens <= 65536.
 * The caller owns all buffers, clears error before a step, and keeps addresses
 * stable until its stream completes. Metadata changes are stream ordered.
 * Device error bits: 1 invalid length; 2 invalid physical page. Empty rows=0.
 * NaN/Inf inputs and overlapping outputs are outside this API's contract. */
typedef struct NvAttention {
    int batch, heads, kv_heads, head_dim, page_tokens, max_blocks, num_pages;
    void *const *d_pages;
    const int32_t *d_block_table, *d_lengths;
    int32_t *d_error;
} NvAttention;
size_t nv_attention_workspace_bytes(const NvAttention *a);
cudaError_t nv_paged_attention(mi_dtype dtype, const void *q, void *out,
    NvAttention a, float *workspace, size_t workspace_bytes, int optimized,
    cudaStream_t stream);
/* Adjacent-pair RoPE, matching nano-vLLM/llama2.c legacy checkpoints.
 * qkv = [batch,(heads+2*kv_heads)*head_dim]. len-1 is current token position.
 * No allocation/host synchronization. Current write pages must be exclusive. */
cudaError_t nv_rope_store(mi_dtype dtype, const void *qkv, void *q,
    NvAttention a, float theta, cudaStream_t stream);
cudaError_t nv_add(mi_dtype dtype, const void *a, const void *b, void *out,
    size_t n, cudaStream_t stream);
cudaError_t nv_swiglu(mi_dtype dtype, const void *gate_up, void *out,
    int batch, int hidden, cudaStream_t stream);
cudaError_t nv_embedding(mi_dtype dtype, const void *embedding, const int32_t *tokens,
    void *out, int batch, int hidden, int vocab, int32_t *error, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
