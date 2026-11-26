#include "nv_attention.h"
#include "vector.cuh"
#include "attention_shape.cuh"
#include <math.h>
template<typename T> __global__ void nv_rope_store_kernel(const T*qkv,T*q,NvAttention a,float theta){
    int pairs=(a.heads+a.kv_heads)*(a.head_dim/2);
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=(size_t)a.batch*pairs)return;
    int b=idx/pairs,j=idx%pairs,head=j/(a.head_dim/2),pair=j%(a.head_dim/2),d=a.head_dim;
    int len=a.d_lengths[b],cap=a.max_blocks*a.page_tokens;
    if(len<1||len>cap){atomicOr(a.d_error,1);return;}
    int pos=len-1;size_t qkv_dim=(a.heads+2*a.kv_heads)*d;
    const T*src=qkv+(size_t)b*qkv_dim+(size_t)head*d+2*pair;
    float angle=(float)pos/powf(theta,(float)(2*pair)/d),sn,cs;sincosf(angle,&sn,&cs);
    float x=mi_read(src[0]),y=mi_read(src[1]);
    T v0=mi_cast<T>(x*cs-y*sn),v1=mi_cast<T>(x*sn+y*cs);
    if(head<a.heads){T*dst=q+((size_t)b*a.heads+head)*d+2*pair;dst[0]=v0;dst[1]=v1;}
    else {
        int physical=a.d_block_table[(size_t)b*a.max_blocks+pos/a.page_tokens];
        if(physical<0||physical>=a.num_pages||!a.d_pages[physical]){atomicOr(a.d_error,2);return;}
        T*page=(T*)a.d_pages[physical];int kv=head-a.heads;
        size_t off=((size_t)(pos%a.page_tokens)*a.kv_heads+kv)*d+2*pair;
        page[off]=v0;page[off+1]=v1;
        const T*v=src+(size_t)a.kv_heads*d;
        page[(size_t)a.page_tokens*a.kv_heads*d+off]=v[0];
        page[(size_t)a.page_tokens*a.kv_heads*d+off+1]=v[1];
    }
}
template<typename T> __global__ void nv_add_kernel(const T*a,const T*b,T*out,size_t n){
    size_t i=((size_t)blockIdx.x*blockDim.x+threadIdx.x)*4;if(i>=n)return;
    int left=(int)min((size_t)4,n-i);float4 x=mi_load_guarded(a+i,left),y=mi_load_guarded(b+i,left);
    mi_store_guarded(out+i,left,make_float4(x.x+y.x,x.y+y.y,x.z+y.z,x.w+y.w));
}
template<typename T> __global__ void nv_swiglu_kernel(const T*gu,T*out,int batch,int n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)batch*n)return;
    int b=i/n,j=i%n;float g=mi_read(gu[(size_t)b*2*n+j]),u=mi_read(gu[(size_t)b*2*n+n+j]);
    out[i]=mi_cast<T>((g/(1.f+expf(-g)))*u);
}
template<typename T> __global__ void nv_embedding_kernel(const T*w,const int32_t*tokens,T*out,int batch,int d,int vocab,int32_t*err){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)batch*d)return;
    int b=i/d,j=i%d,token=tokens[b];
    if(token<0||token>=vocab){atomicOr(err,4);out[i]=mi_cast<T>(0);return;}
    out[i]=w[(size_t)token*d+j];
}
#define NV_TYPES(CALL) switch(dtype){case MI_F32:{CALL(float);break;}case MI_F16:{CALL(__half);break;}case MI_BF16:{CALL(__nv_bfloat16);break;}default:return cudaErrorInvalidValue;}
extern "C" cudaError_t nv_rope_store(mi_dtype dtype,const void*in,void*q,NvAttention a,float theta,cudaStream_t s){
    if(!nv_shape_ok(a)||!in||!q||!a.d_pages||!a.d_lengths||!a.d_block_table||!a.d_error||a.head_dim%2||!isfinite(theta)||theta<=0)return cudaErrorInvalidValue;
    size_t align=dtype==MI_F32?4:2;if((uintptr_t)in%align||(uintptr_t)q%align)return cudaErrorInvalidValue;
    size_t count=(size_t)a.batch*(a.heads+a.kv_heads)*(a.head_dim/2);
    #define RUN(T) nv_rope_store_kernel<<<(count+255)/256,256,0,s>>>((const T*)in,(T*)q,a,theta)
    NV_TYPES(RUN)
    #undef RUN
    return cudaGetLastError();
}
extern "C" cudaError_t nv_add(mi_dtype dtype,const void*a,const void*b,void*out,size_t n,cudaStream_t s){
    if(!a||!b||!out||!n)return cudaErrorInvalidValue;
    #define RUN(T) nv_add_kernel<<<(n+1023)/1024,256,0,s>>>((const T*)a,(const T*)b,(T*)out,n)
    NV_TYPES(RUN)
    #undef RUN
    return cudaGetLastError();
}
extern "C" cudaError_t nv_swiglu(mi_dtype dtype,const void*in,void*out,int b,int n,cudaStream_t s){
    if(!in||!out||b<1||n<1)return cudaErrorInvalidValue;
    #define RUN(T) nv_swiglu_kernel<<<((size_t)b*n+255)/256,256,0,s>>>((const T*)in,(T*)out,b,n)
    NV_TYPES(RUN)
    #undef RUN
    return cudaGetLastError();
}
extern "C" cudaError_t nv_embedding(mi_dtype dtype,const void*w,const int32_t*t,void*out,int b,int d,int vocab,int32_t*err,cudaStream_t s){
    if(!w||!t||!out||!err||b<1||d<1||vocab<1)return cudaErrorInvalidValue;
    #define RUN(T) nv_embedding_kernel<<<((size_t)b*d+255)/256,256,0,s>>>((const T*)w,t,(T*)out,b,d,vocab,err)
    NV_TYPES(RUN)
    #undef RUN
    return cudaGetLastError();
}
