#ifndef NV_CUDA_H
#define NV_CUDA_H
#include "nv.h"
#include "nv_attention.h"
/* Plain C ownership records. h_* are host data; d_* are CUDA addresses.
 * No function table, opaque ctx, CUDA allocation or host wait in nv_cuda_step.
 * One stream owns all execution. Free only after Graph executions finish.
 * CPU scheduler owns block refs/prefixes; CUDA owns physical tensor storage.
 * Current-token pages must be exclusive, as guaranteed by nv_cache_publish. */
typedef struct NvCudaLayer {
    void *d_att_norm,*d_qkv,*d_o,*d_ffn_norm,*d_gate_up,*d_down;
} NvCudaLayer;
typedef struct NvCuda {
    NvConfig config;
    mi_dtype dtype;
    int batch,num_pages,page_tokens,max_blocks,use_graph;
    NvCudaLayer *h_layers;
    NvAttention *h_attention;
    void **h_page_allocations;
    void ***h_page_tables; /* per-layer host arrays of independently allocated GPU pages */
    void *d_embedding,*d_final_norm,*d_classifier;
    void *d_x,*d_norm,*d_qkv,*d_q,*d_att,*d_proj,*d_gate_up,*d_ffn,*d_logits;
    float *d_work;
    size_t work_bytes;
    int32_t *d_tokens,*d_lengths,*d_table,*d_error;
    int32_t *h_tokens,*h_lengths,*h_table,*h_error;
    void *h_output;
    float *h_logits;
    cudaStream_t stream;
    cudaEvent_t upload_ready;
    cudaGraph_t graph;
    cudaGraphExec_t replay;
} NvCuda;
/* Initialization/frees/metadata upload are outside Graph capture. Init follows
 * nano-vLLM's fail-fast allocation policy; normal APIs return CUDA errors. */
void nv_cuda_init(NvCuda *g,const NvModel *model,int pages,int page_tokens,int batch,mi_dtype dtype,int graph);
void nv_cuda_free(NvCuda *g);
cudaError_t nv_cuda_upload(NvCuda *g,const int32_t *tokens,const int32_t *lengths,const int32_t *table);
cudaError_t nv_cuda_step(NvCuda *g);
cudaError_t nv_cuda_capture(NvCuda *g);
cudaError_t nv_cuda_download(NvCuda *g);
float *nv_cuda_forward(NvCuda *g,const int *table,int token,int position);
#endif
