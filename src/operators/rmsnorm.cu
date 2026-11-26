#include "miniinfer/operators.h"
#include "vector.cuh"
#include <math.h>

template<typename T> __global__ void mi_rms_scalar_kernel(const T*x,const T*g,T*y,int n,int xs,int ys,float eps){
    const T *row=x+(size_t)blockIdx.x*xs;
    T *out=y+(size_t)blockIdx.x*ys;
    float sum=0.f,scale=1.f;
    for(int i=threadIdx.x;i<n;i+=256){float v=mi_read(row[i]);sum=fmaf(v,v,sum);}
    sum=mi_block_sum(sum);
    if(!isfinite(sum)){
        float mx=0.f;
        for(int i=threadIdx.x;i<n;i+=256)mx=fmaxf(mx,fabsf(mi_read(row[i])));
        scale=mi_block_max(mx);sum=0.f;
        for(int i=threadIdx.x;i<n;i+=256){float v=mi_read(row[i])/scale;sum=fmaf(v,v,sum);}
        sum=mi_block_sum(sum);
    }
    float inv=rsqrtf(sum/n+(eps/scale)/scale);
    for(int i=threadIdx.x;i<n;i+=256)out[i]=mi_cast<T>((mi_read(row[i])/scale)*inv*mi_read(g[i]));
}

/* Input held in registers across the reduction; no second global input read.
 * PACKS specialization bounds register use for the actual hidden-size range. */
template<typename T,int PACKS> __global__ void mi_rms_vec4_kernel(const T*x,const T*g,T*y,int n,int xs,int ys,float eps){
    const T *row=x+(size_t)blockIdx.x*xs;
    T *out=y+(size_t)blockIdx.x*ys;
    float4 values[PACKS];float sum=0.f,scale=1.f;
    #pragma unroll
    for(int j=0;j<PACKS;j++){
        int i=(threadIdx.x+j*256)*4;
        float4 v=i<n?mi_load_guarded(row+i,n-i):make_float4(0,0,0,0);
        values[j]=v;sum=fmaf(v.x,v.x,sum);sum=fmaf(v.y,v.y,sum);
        sum=fmaf(v.z,v.z,sum);sum=fmaf(v.w,v.w,sum);
    }
    sum=mi_block_sum(sum);
    if(!isfinite(sum)){
        float mx=0.f;
        #pragma unroll
        for(int j=0;j<PACKS;j++){float4 v=values[j];mx=fmaxf(mx,fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w))));}
        scale=mi_block_max(mx);sum=0.f;
        #pragma unroll
        for(int j=0;j<PACKS;j++){
            float4 v=values[j];v.x/=scale;v.y/=scale;v.z/=scale;v.w/=scale;values[j]=v;
            sum=fmaf(v.x,v.x,sum);sum=fmaf(v.y,v.y,sum);sum=fmaf(v.z,v.z,sum);sum=fmaf(v.w,v.w,sum);
        }
        sum=mi_block_sum(sum);
    }
    float inv=rsqrtf(sum/n+(eps/scale)/scale);
    #pragma unroll
    for(int j=0;j<PACKS;j++){
        int i=(threadIdx.x+j*256)*4;
        if(i<n){
            float4 v=values[j],w=mi_load_guarded(g+i,n-i);
            v.x=v.x*inv*w.x;v.y=v.y*inv*w.y;v.z=v.z*inv*w.z;v.w=v.w*inv*w.w;
            mi_store_guarded(out+i,n-i,v);
        }
    }
}
template<typename T> static void launch(bool opt,const void*x,const void*g,void*y,int r,int n,int xs,int ys,float eps,cudaStream_t s){
    if(!opt){mi_rms_scalar_kernel<<<r,256,0,s>>>((const T*)x,(const T*)g,(T*)y,n,xs,ys,eps);return;}
    #define MI_RMS_LAUNCH(P) mi_rms_vec4_kernel<T,P><<<r,256,0,s>>>((const T*)x,(const T*)g,(T*)y,n,xs,ys,eps)
    if(n<=1024){MI_RMS_LAUNCH(1);}else if(n<=2048){MI_RMS_LAUNCH(2);}
    else if(n<=4096){MI_RMS_LAUNCH(4);}else if(n<=8192){MI_RMS_LAUNCH(8);}
    else {MI_RMS_LAUNCH(16);}
    #undef MI_RMS_LAUNCH
}
static cudaError_t dispatch(bool opt,mi_dtype d,const void*x,const void*g,void*y,int r,int n,int xs,int ys,float eps,cudaStream_t s){
    if(!x||!g||!y||r<=0||n<=0||n>16384||xs<n||ys<n||!isfinite(eps)||eps<=0.f)return cudaErrorInvalidValue;
    if(x==y && xs!=ys)return cudaErrorInvalidValue;
    size_t a=d==MI_F32?4:2;
    if((uintptr_t)x%a||(uintptr_t)g%a||(uintptr_t)y%a)return cudaErrorInvalidValue;
    switch(d){
    case MI_F32:launch<float>(opt,x,g,y,r,n,xs,ys,eps,s);break;
    case MI_F16:launch<__half>(opt,x,g,y,r,n,xs,ys,eps,s);break;
    case MI_BF16:launch<__nv_bfloat16>(opt,x,g,y,r,n,xs,ys,eps,s);break;
    default:return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
extern "C" cudaError_t mi_rmsnorm_baseline(mi_dtype d,const void*x,const void*g,void*y,int r,int n,int xs,int ys,float e,cudaStream_t s){return dispatch(false,d,x,g,y,r,n,xs,ys,e,s);}
extern "C" cudaError_t mi_rmsnorm_optimized(mi_dtype d,const void*x,const void*g,void*y,int r,int n,int xs,int ys,float e,cudaStream_t s){return dispatch(true,d,x,g,y,r,n,xs,ys,e,s);}
