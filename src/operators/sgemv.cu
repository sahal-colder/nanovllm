#include "miniinfer/operators.h"
#include "vector.cuh"

/* Scalar, one CTA per row: deliberately simple reference GPU implementation. */
template<typename T> __global__ void mi_gemv_scalar_kernel(const T *w,const T *x,T *y,
    int m,int k,int lda,int xs,int ys) {
    const int row=blockIdx.x,b=blockIdx.y,t=threadIdx.x;
    float sum=0.f;
    for(size_t j=t;j<(size_t)k;j+=256)sum=fmaf(mi_read(w[(size_t)row*lda+j]),mi_read(x[(size_t)b*xs+j]),sum);
    __shared__ float tmp[256];
    tmp[t]=sum; __syncthreads();
    for(int d=128;d;d>>=1){if(t<d)tmp[t]+=tmp[t+d];__syncthreads();}
    if(!t)y[(size_t)b*ys+row]=mi_cast<T>(tmp[0]);
}

/* Four independent rows/CTA, one warp/row. No shared memory or CTA barrier.
 * Each warp covers adjacent 4-element groups, including scalar guarded tails.
 * W and X alignment is checked independently: offset views stay valid. */
template<typename T> __global__ void mi_gemv_vec4_kernel(const T *w,const T *x,T *y,
    int m,int k,int lda,int xs,int ys) {
    const int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),b=blockIdx.y;
    if(row>=m)return; // whole warp takes the same branch
    float sum=0.f;
    for(size_t j=lane*4;j<(size_t)k;j+=128){
        const float4 a=mi_load_guarded(w+(size_t)row*lda+j,k-j);
        const float4 v=mi_load_guarded(x+(size_t)b*xs+j,k-j);
        sum=fmaf(a.x,v.x,sum);sum=fmaf(a.y,v.y,sum);
        sum=fmaf(a.z,v.z,sum);sum=fmaf(a.w,v.w,sum);
    }
    sum=mi_warp_sum(sum);
    if(!lane)y[(size_t)b*ys+row]=mi_cast<T>(sum);
}
template<typename T> static void launch(bool opt,const void *w,const void *x,void *y,
    int m,int k,int b,int lda,int xs,int ys,cudaStream_t s) {
    if(opt)mi_gemv_vec4_kernel<<<dim3((m-1)/4+1,b),128,0,s>>>((const T*)w,(const T*)x,(T*)y,m,k,lda,xs,ys);
    else mi_gemv_scalar_kernel<<<dim3(m,b),256,0,s>>>((const T*)w,(const T*)x,(T*)y,m,k,lda,xs,ys);
}
static cudaError_t dispatch(bool opt,mi_dtype dt,const void *w,const void *x,void *y,
    int m,int k,int b,int lda,int xs,int ys,cudaStream_t s) {
    if(!w||!x||!y||m<=0||k<=0||b<=0||b>65535||lda<k||xs<k||ys<m)return cudaErrorInvalidValue;
    const size_t align=dt==MI_F32?4:2;
    if((uintptr_t)w%align||(uintptr_t)x%align||(uintptr_t)y%align)return cudaErrorInvalidValue;
    switch(dt){
    case MI_F32:launch<float>(opt,w,x,y,m,k,b,lda,xs,ys,s);break;
    case MI_F16:launch<__half>(opt,w,x,y,m,k,b,lda,xs,ys,s);break;
    case MI_BF16:launch<__nv_bfloat16>(opt,w,x,y,m,k,b,lda,xs,ys,s);break;
    default:return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
extern "C" cudaError_t mi_sgemv_baseline(mi_dtype d,const void*w,const void*x,void*y,
    int m,int k,int b,int lda,int xs,int ys,cudaStream_t s){return dispatch(false,d,w,x,y,m,k,b,lda,xs,ys,s);}
extern "C" cudaError_t mi_sgemv_optimized(mi_dtype d,const void*w,const void*x,void*y,
    int m,int k,int b,int lda,int xs,int ys,cudaStream_t s){return dispatch(true,d,w,x,y,m,k,b,lda,xs,ys,s);}
