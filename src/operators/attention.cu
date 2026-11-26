#include "nv_attention.h"
#include "vector.cuh"
#include "attention_shape.cuh"
#include <limits.h>
#include <math.h>

extern "C" size_t nv_attention_workspace_bytes(const NvAttention *a) {
    if(!a || !nv_shape_ok(*a))return 0;
    size_t partitions=((size_t)a->max_blocks*a->page_tokens+63)/64;
    size_t stride=(a->head_dim+5)&~(size_t)3;
    size_t count=(size_t)a->batch*a->heads*partitions*stride;
    return count<=SIZE_MAX/sizeof(float)?count*sizeof(float):0;
}
template<typename T,bool V> __device__ float4 channels(const T*p,int d,int group,int lane){
    if(V){int j=group*128+lane*4;return j<d?mi_load_guarded(p+j,d-j):make_float4(0,0,0,0);}
    int j=group*128+lane;
    return make_float4(j<d?mi_read(p[j]):0,j+32<d?mi_read(p[j+32]):0,
        j+64<d?mi_read(p[j+64]):0,j+96<d?mi_read(p[j+96]):0);
}
template<typename T,bool V> __device__ void put_channels(T*p,int d,int group,int lane,float4 v){
    if(V){int j=group*128+lane*4;if(j<d)mi_store_guarded(p+j,d-j,v);return;}
    int j=group*128+lane;
    if(j<d)p[j]=mi_cast<T>(v.x);if(j+32<d)p[j+32]=mi_cast<T>(v.y);
    if(j+64<d)p[j+64]=mi_cast<T>(v.z);if(j+96<d)p[j+96]=mi_cast<T>(v.w);
}
/* Four independent warps/CTA. Each warp owns one (sequence,query-head,split).
 * Q and running PV stay in registers. No shared memory or CTA barriers.
 * Split chunks bound the serial online-softmax recurrence to 64 tokens.
 * Page table/pointer lookup occurs once per logical page, not per channel.
 * Masked tail entries are never loaded; GQA heads share the same KV group. */
template<typename T,bool V,bool PARTIAL,int GROUPS> __global__ void nv_attention_partition_kernel(
    const T*q,T*out,NvAttention a,float*work,int partitions,int span){
    size_t job=(size_t)blockIdx.x*4+(threadIdx.x>>5);
    size_t jobs=(size_t)a.batch*a.heads*partitions;
    if(job>=jobs)return;
    int lane=threadIdx.x&31,part=job%partitions;
    size_t bh=job/partitions;int head=bh%a.heads,b=bh/a.heads,d=a.head_dim;
    int length=a.d_lengths[b],capacity=a.max_blocks*a.page_tokens;
    if(length<0||length>capacity){if(!lane)atomicOr(a.d_error,1);length=0;}
    int begin=part*span,end=min(length,begin+span),kv=head/(a.heads/a.kv_heads);
    float4 query[GROUPS],acc[GROUPS];
    #pragma unroll
    for(int g=0;g<GROUPS;g++){query[g]=channels<T,V>(q+bh*d,d,g,lane);acc[g]=make_float4(0,0,0,0);}
    float maximum=-INFINITY,denom=0.f,scale=rsqrtf((float)d);
    for(int t=begin;t<end;){
        int logical=t/a.page_tokens,stop=min(end,(logical+1)*a.page_tokens);
        int physical=a.d_block_table[(size_t)b*a.max_blocks+logical];
        if(physical<0||physical>=a.num_pages){if(!lane)atomicOr(a.d_error,2);t=stop;continue;}
        const T*page=(const T*)a.d_pages[physical];
        if(!page){if(!lane)atomicOr(a.d_error,2);t=stop;continue;}
        for(;t<stop;t++){
            size_t offset=((size_t)(t%a.page_tokens)*a.kv_heads+kv)*d;
            const T*k=page+offset,*v=page+(size_t)a.page_tokens*a.kv_heads*d+offset;
            float dot=0.f;
            #pragma unroll
            for(int g=0;g<GROUPS;g++){
                float4 x=query[g],y=channels<T,V>(k,d,g,lane);
                dot=fmaf(x.x,y.x,dot);dot=fmaf(x.y,y.y,dot);dot=fmaf(x.z,y.z,dot);dot=fmaf(x.w,y.w,dot);
            }
            dot=mi_warp_sum(dot);float score=__shfl_sync(0xffffffffu,dot,0)*scale;
            float next=fmaxf(maximum,score),alpha=__expf(maximum-next),beta=__expf(score-next);
            denom=denom*alpha+beta;maximum=next;
            #pragma unroll
            for(int g=0;g<GROUPS;g++){
                float4 x=acc[g],y=channels<T,V>(v,d,g,lane);
                x.x=fmaf(beta,y.x,x.x*alpha);x.y=fmaf(beta,y.y,x.y*alpha);
                x.z=fmaf(beta,y.z,x.z*alpha);x.w=fmaf(beta,y.w,x.w*alpha);acc[g]=x;
            }
        }
    }
    if(PARTIAL){
        float*p=work+job*((d+5)&~3);
        #pragma unroll
        for(int g=0;g<GROUPS;g++)put_channels<float,V>(p,d,g,lane,acc[g]);
        if(!lane){p[d]=maximum;p[d+1]=denom;}
    }else{
        float inv=denom>0?1.f/denom:0.f;
        #pragma unroll
        for(int g=0;g<GROUPS;g++){float4 v=acc[g];v.x*=inv;v.y*=inv;v.z*=inv;v.w*=inv;put_channels<T,V>(out+bh*d,d,g,lane,v);}
    }
}
/* Merge unnormalized split outputs with their (max,sumexp), preserving the
 * exact online-softmax identity. Never materializes scores or probabilities. */
template<typename T> __global__ void nv_attention_merge_kernel(T*out,const float*work,NvAttention a,int partitions){
    size_t bh=blockIdx.x;int t=threadIdx.x,d=a.head_dim,stride=(d+5)&~3;
    extern __shared__ float weights[];
    float maximum=-INFINITY;
    for(int p=t;p<partitions;p+=256){const float*row=work+(bh*partitions+p)*stride;if(row[d+1]>0)maximum=fmaxf(maximum,row[d]);}
    maximum=mi_block_max(maximum);
    float sum=0.f;
    for(int p=t;p<partitions;p+=256){
        const float*row=work+(bh*partitions+p)*stride;
        float weight=row[d+1]>0?__expf(row[d]-maximum):0.f;weights[p]=weight;sum=fmaf(weight,row[d+1],sum);
    }
    sum=mi_block_sum(sum); // also publishes every shared weight before PV merge
    if(t<d){
        float value=0.f;
        for(int p=0;p<partitions;p++)value=fmaf(weights[p],work[(bh*partitions+p)*stride+t],value);
        out[bh*d+t]=mi_cast<T>(sum>0?value/sum:0.f);
    }
}
template<typename T,int G> static void launch_groups(const void*q,void*out,NvAttention a,float*w,int opt,cudaStream_t s){
    int capacity=a.max_blocks*a.page_tokens,parts=(capacity-1)/64+1;
    size_t bh=(size_t)a.batch*a.heads;
    if(!opt)nv_attention_partition_kernel<T,false,false,G><<<(bh+3)/4,128,0,s>>>((const T*)q,(T*)out,a,w,1,capacity);
    else if(opt==2||parts==1)nv_attention_partition_kernel<T,true,false,G><<<(bh+3)/4,128,0,s>>>((const T*)q,(T*)out,a,w,1,capacity);
    else {
        nv_attention_partition_kernel<T,true,true,G><<<(bh*parts+3)/4,128,0,s>>>((const T*)q,(T*)out,a,w,parts,64);
        nv_attention_merge_kernel<T><<<bh,256,(size_t)parts*sizeof(float),s>>>((T*)out,w,a,parts);
    }
}
template<typename T> static void launch(const void*q,void*out,NvAttention a,float*w,int opt,cudaStream_t s){
    if(a.head_dim<=128)launch_groups<T,1>(q,out,a,w,opt,s);else launch_groups<T,2>(q,out,a,w,opt,s);
}
extern "C" cudaError_t nv_paged_attention(mi_dtype dtype,const void*q,void*out,NvAttention a,float*w,size_t bytes,int opt,cudaStream_t s){
    if(!nv_shape_ok(a)||!q||!out||!a.d_pages||!a.d_block_table||!a.d_lengths||!a.d_error)return cudaErrorInvalidValue;
    size_t alignment=dtype==MI_F32?4:2;
    if((uintptr_t)q%alignment||(uintptr_t)out%alignment)return cudaErrorInvalidValue;
    if(opt<0||opt>2)return cudaErrorInvalidValue;
    if(opt==1&&a.max_blocks*a.page_tokens>64&&(!w||(uintptr_t)w%4||bytes<nv_attention_workspace_bytes(&a)))return cudaErrorInvalidValue;
    switch(dtype){
    case MI_F32:launch<float>(q,out,a,w,opt,s);break;
    case MI_F16:launch<__half>(q,out,a,w,opt,s);break;
    case MI_BF16:launch<__nv_bfloat16>(q,out,a,w,opt,s);break;
    default:return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
