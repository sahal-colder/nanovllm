/* C11 host execution of nano-vLLM's existing Llama/legacy checkpoint math.
 * Original FP32 weight mapping, tokenizer, sampler and scheduler are reused.
 * CUDA kernels and capture-safe buffer management are new. */
#include "nv_cuda.h"
#include "numeric.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#define CU(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e));nv_fatal("CUDA initialization/execution failed");}}while(0)
#define TRY(x) do{cudaError_t e=(x);if(e!=cudaSuccess)return e;}while(0)
static void *device(size_t n){void*p;CU(cudaMalloc(&p,n));return p;}
static void *tensor(mi_dtype d,const float*x,size_t n){
    size_t bytes=n*nv_bytes(d);void*p=device(bytes);
    if(d==MI_F32)CU(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));
    else{void*h=nv_alloc(bytes,1);for(size_t i=0;i<n;i++)nv_pack(h,i,d,x[i]);CU(cudaMemcpy(p,h,bytes,cudaMemcpyHostToDevice));free(h);}
    return p;
}
static void *join(mi_dtype d,const float*a,size_t na,const float*b,size_t nb,const float*c,size_t nc){
    float*h=nv_alloc(na+nb+nc,sizeof(float));memcpy(h,a,na*4);memcpy(h+na,b,nb*4);if(nc)memcpy(h+na+nb,c,nc*4);
    void*p=tensor(d,h,na+nb+nc);free(h);return p;
}
void nv_cuda_init(NvCuda*g,const NvModel*m,int pages,int t,int batch,mi_dtype dtype,int graph){
    *g=(NvCuda){0};g->config=m->config;g->dtype=dtype;g->batch=batch;g->num_pages=pages;g->page_tokens=t;g->use_graph=graph;
    int d=m->config.dim,f=m->config.hidden_dim,layers=m->config.n_layers,heads=m->config.n_heads;
    if(dtype<MI_F32||dtype>MI_BF16||batch<1||batch>4096||pages<1||t<1||t>1024||d>16384||f>INT_MAX/2||d/heads>256||m->config.seq_len>65536)nv_fatal("unsupported CUDA model shape");
    g->max_blocks=(m->config.seq_len-1)/t+1;
    int kv=(d/heads)*m->config.n_kv_heads,qkv=d+2*kv;size_t bytes=nv_bytes(dtype),dd=(size_t)d*d,fd=(size_t)f*d,kd=(size_t)kv*d;
    CU(cudaStreamCreateWithFlags(&g->stream,cudaStreamNonBlocking));CU(cudaEventCreateWithFlags(&g->upload_ready,cudaEventDisableTiming));
    g->h_layers=nv_alloc(layers,sizeof(*g->h_layers));g->h_attention=nv_alloc(layers,sizeof(*g->h_attention));
    g->h_page_allocations=nv_alloc(pages,sizeof(void*));g->h_page_tables=nv_alloc(layers,sizeof(void**));
    size_t layer_page=(size_t)2*t*kv*bytes;
    for(int p=0;p<pages;p++){g->h_page_allocations[p]=device(layer_page*layers);CU(cudaMemsetAsync(g->h_page_allocations[p],0,layer_page*layers,g->stream));}
    g->d_tokens=device((size_t)batch*4);g->d_lengths=device((size_t)batch*4);g->d_table=device((size_t)batch*g->max_blocks*4);g->d_error=device(4);
    CU(cudaMallocHost((void**)&g->h_tokens,(size_t)batch*4));CU(cudaMallocHost((void**)&g->h_lengths,(size_t)batch*4));
    CU(cudaMallocHost((void**)&g->h_table,(size_t)batch*g->max_blocks*4));CU(cudaMallocHost((void**)&g->h_error,4));
    CU(cudaMallocHost(&g->h_output,(size_t)batch*m->config.vocab_size*bytes));g->h_logits=nv_alloc((size_t)batch*m->config.vocab_size,4);
    for(int l=0;l<layers;l++){
        const NvWeights*w=&m->weights;NvCudaLayer*lw=&g->h_layers[l];
        lw->d_att_norm=tensor(dtype,w->att_norm+(size_t)l*d,d);lw->d_ffn_norm=tensor(dtype,w->ffn_norm+(size_t)l*d,d);
        lw->d_qkv=join(dtype,w->wq+l*dd,dd,w->wk+l*kd,kd,w->wv+l*kd,kd);
        lw->d_o=tensor(dtype,w->wo+l*dd,dd);lw->d_gate_up=join(dtype,w->w1+l*fd,fd,w->w3+l*fd,fd,NULL,0);
        lw->d_down=tensor(dtype,w->w2+l*fd,fd);
        g->h_page_tables[l]=nv_alloc(pages,sizeof(void*));
        for(int p=0;p<pages;p++)g->h_page_tables[l][p]=(char*)g->h_page_allocations[p]+l*layer_page;
        void **dp=device((size_t)pages*sizeof(void*));CU(cudaMemcpyAsync(dp,g->h_page_tables[l],(size_t)pages*sizeof(void*),cudaMemcpyHostToDevice,g->stream));
        g->h_attention[l]=(NvAttention){batch,heads,m->config.n_kv_heads,d/heads,t,g->max_blocks,pages,(void*const*)dp,g->d_table,g->d_lengths,g->d_error};
    }
    g->d_embedding=tensor(dtype,m->weights.embedding,(size_t)m->config.vocab_size*d);
    g->d_classifier=tensor(dtype,m->weights.classifier,(size_t)m->config.vocab_size*d);g->d_final_norm=tensor(dtype,m->weights.final_norm,d);
    g->d_x=device((size_t)batch*d*bytes);g->d_norm=device((size_t)batch*d*bytes);g->d_qkv=device((size_t)batch*qkv*bytes);
    g->d_q=device((size_t)batch*d*bytes);g->d_att=device((size_t)batch*d*bytes);g->d_proj=device((size_t)batch*d*bytes);
    g->d_gate_up=device((size_t)batch*2*f*bytes);g->d_ffn=device((size_t)batch*f*bytes);g->d_logits=device((size_t)batch*m->config.vocab_size*bytes);
    g->work_bytes=nv_attention_workspace_bytes(&g->h_attention[0]);g->d_work=device(g->work_bytes);
    CU(cudaEventRecord(g->upload_ready,g->stream));CU(cudaStreamSynchronize(g->stream));
}
cudaError_t nv_cuda_upload(NvCuda*g,const int32_t*tokens,const int32_t*lengths,const int32_t*table){
    if(!g||!tokens||!lengths||!table)return cudaErrorInvalidValue;
    enum cudaStreamCaptureStatus status;TRY(cudaStreamIsCapturing(g->stream,&status));if(status!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    TRY(cudaEventSynchronize(g->upload_ready)); /* pinned staging cannot be overwritten until consumed */
    for(int b=0;b<g->batch;b++){
        if(tokens[b]<0||tokens[b]>=g->config.vocab_size||lengths[b]<1||lengths[b]>g->config.seq_len)return cudaErrorInvalidValue;
        for(int p=0;p<(lengths[b]-1)/g->page_tokens+1;p++)if(table[(size_t)b*g->max_blocks+p]<0||table[(size_t)b*g->max_blocks+p]>=g->num_pages)return cudaErrorInvalidValue;
        /* Simultaneous writes must not alias, even if read-only prefix pages do. */
        for(int other=0;other<b;other++)if(table[(size_t)b*g->max_blocks+(lengths[b]-1)/g->page_tokens]==table[(size_t)other*g->max_blocks+(lengths[other]-1)/g->page_tokens])return cudaErrorInvalidValue;
    }
    memcpy(g->h_tokens,tokens,(size_t)g->batch*4);memcpy(g->h_lengths,lengths,(size_t)g->batch*4);
    if(g->h_table!=table)memcpy(g->h_table,table,(size_t)g->batch*g->max_blocks*4);
    TRY(cudaMemcpyAsync(g->d_tokens,g->h_tokens,(size_t)g->batch*4,cudaMemcpyHostToDevice,g->stream));
    TRY(cudaMemcpyAsync(g->d_lengths,g->h_lengths,(size_t)g->batch*4,cudaMemcpyHostToDevice,g->stream));
    TRY(cudaMemcpyAsync(g->d_table,g->h_table,(size_t)g->batch*g->max_blocks*4,cudaMemcpyHostToDevice,g->stream));
    return cudaEventRecord(g->upload_ready,g->stream);
}
cudaError_t nv_cuda_step(NvCuda*g){
    int d=g->config.dim,f=g->config.hidden_dim,b=g->batch,kv=d/g->config.n_heads*g->config.n_kv_heads,qkv=d+2*kv;
    mi_dtype t=g->dtype;cudaStream_t s=g->stream;
    TRY(cudaMemsetAsync(g->d_error,0,4,s));
    TRY(nv_embedding(t,g->d_embedding,g->d_tokens,g->d_x,b,d,g->config.vocab_size,g->d_error,s));
    for(int l=0;l<g->config.n_layers;l++){
        NvCudaLayer*w=&g->h_layers[l];NvAttention a=g->h_attention[l];
        TRY(mi_rmsnorm_optimized(t,g->d_x,w->d_att_norm,g->d_norm,b,d,d,d,1.e-5f,s));
        TRY(mi_sgemv_optimized(t,w->d_qkv,g->d_norm,g->d_qkv,qkv,d,b,d,d,qkv,s));
        TRY(nv_rope_store(t,g->d_qkv,g->d_q,a,10000.f,s));
        TRY(nv_paged_attention(t,g->d_q,g->d_att,a,g->d_work,g->work_bytes,1,s));
        TRY(mi_sgemv_optimized(t,w->d_o,g->d_att,g->d_proj,d,d,b,d,d,d,s));
        TRY(nv_add(t,g->d_x,g->d_proj,g->d_x,(size_t)b*d,s));
        TRY(mi_rmsnorm_optimized(t,g->d_x,w->d_ffn_norm,g->d_norm,b,d,d,d,1.e-5f,s));
        TRY(mi_sgemv_optimized(t,w->d_gate_up,g->d_norm,g->d_gate_up,2*f,d,b,d,d,2*f,s));
        TRY(nv_swiglu(t,g->d_gate_up,g->d_ffn,b,f,s));
        TRY(mi_sgemv_optimized(t,w->d_down,g->d_ffn,g->d_proj,d,f,b,f,f,d,s));
        TRY(nv_add(t,g->d_x,g->d_proj,g->d_x,(size_t)b*d,s));
    }
    TRY(mi_rmsnorm_optimized(t,g->d_x,g->d_final_norm,g->d_norm,b,d,d,d,1.e-5f,s));
    return mi_sgemv_optimized(t,g->d_classifier,g->d_norm,g->d_logits,g->config.vocab_size,d,b,d,d,g->config.vocab_size,s);
}
cudaError_t nv_cuda_capture(NvCuda*g){
    if(g->replay)return cudaSuccess;
    TRY(nv_cuda_step(g));TRY(cudaStreamSynchronize(g->stream)); /* lazy module load before capture */
    TRY(cudaStreamBeginCapture(g->stream,cudaStreamCaptureModeThreadLocal));
    cudaError_t e=nv_cuda_step(g);
    cudaError_t end=cudaStreamEndCapture(g->stream,&g->graph);
    if(e!=cudaSuccess)return e;if(end!=cudaSuccess)return end;
    return cudaGraphInstantiate(&g->replay,g->graph,0);
}
cudaError_t nv_cuda_download(NvCuda*g){
    size_t count=(size_t)g->batch*g->config.vocab_size;
    TRY(cudaMemcpyAsync(g->h_output,g->d_logits,count*nv_bytes(g->dtype),cudaMemcpyDeviceToHost,g->stream));
    TRY(cudaMemcpyAsync(g->h_error,g->d_error,4,cudaMemcpyDeviceToHost,g->stream));TRY(cudaStreamSynchronize(g->stream));
    if(*g->h_error)return cudaErrorInvalidValue;
    for(size_t i=0;i<count;i++)g->h_logits[i]=nv_unpack(g->h_output,i,g->dtype);
    return cudaSuccess;
}
float *nv_cuda_forward(NvCuda*g,const int*table,int token,int pos){
    if(g->batch!=1||pos<0||pos>=g->config.seq_len)nv_fatal("invalid CUDA forward position/batch");
    CU(cudaEventSynchronize(g->upload_ready));
    for(int p=0;p<g->max_blocks;p++)g->h_table[p]=p<=pos/g->page_tokens?table[p]:-1;
    int32_t length=pos+1;CU(nv_cuda_upload(g,&token,&length,g->h_table));
    if(g->use_graph){CU(nv_cuda_capture(g));CU(cudaGraphLaunch(g->replay,g->stream));}else CU(nv_cuda_step(g));
    CU(nv_cuda_download(g));return g->h_logits;
}
void nv_cuda_free(NvCuda*g){
    if(!g||!g->stream)return;CU(cudaStreamSynchronize(g->stream));
    if(g->replay)CU(cudaGraphExecDestroy(g->replay));if(g->graph)CU(cudaGraphDestroy(g->graph));
    for(int l=0;l<g->config.n_layers;l++){
        NvCudaLayer*w=&g->h_layers[l];CU(cudaFree(w->d_att_norm));CU(cudaFree(w->d_qkv));CU(cudaFree(w->d_o));CU(cudaFree(w->d_ffn_norm));CU(cudaFree(w->d_gate_up));CU(cudaFree(w->d_down));
        CU(cudaFree((void*)g->h_attention[l].d_pages));free(g->h_page_tables[l]);
    }
    for(int p=0;p<g->num_pages;p++)CU(cudaFree(g->h_page_allocations[p]));
    free(g->h_page_allocations);free(g->h_page_tables);free(g->h_layers);free(g->h_attention);
    void*all[]={g->d_embedding,g->d_final_norm,g->d_classifier,g->d_x,g->d_norm,g->d_qkv,g->d_q,g->d_att,g->d_proj,g->d_gate_up,g->d_ffn,g->d_logits,g->d_work,g->d_tokens,g->d_lengths,g->d_table,g->d_error};
    for(size_t i=0;i<sizeof(all)/sizeof(*all);i++)CU(cudaFree(all[i]));
    CU(cudaFreeHost(g->h_tokens));CU(cudaFreeHost(g->h_lengths));CU(cudaFreeHost(g->h_table));CU(cudaFreeHost(g->h_error));CU(cudaFreeHost(g->h_output));free(g->h_logits);
    CU(cudaEventDestroy(g->upload_ready));CU(cudaStreamDestroy(g->stream));*g=(NvCuda){0};
}
