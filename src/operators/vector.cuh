#ifndef MI_VECTOR_CUH
#define MI_VECTOR_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<typename T> __device__ float mi_read(T x);
template<> __device__ inline float mi_read(float x) { return x; }
template<> __device__ inline float mi_read(__half x) { return __half2float(x); }
template<> __device__ inline float mi_read(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ T mi_cast(float x);
template<> __device__ inline float mi_cast(float x) { return x; }
template<> __device__ inline __half mi_cast(float x) { return __float2half_rn(x); }
template<> __device__ inline __nv_bfloat16 mi_cast(float x) { return __float2bfloat16_rn(x); }

template<typename T> __device__ inline float4 mi_load4(const T *p) {
    const ushort4 u = *reinterpret_cast<const ushort4 *>(p);
    return make_float4(mi_read(*reinterpret_cast<const T *>(&u.x)),
        mi_read(*reinterpret_cast<const T *>(&u.y)),
        mi_read(*reinterpret_cast<const T *>(&u.z)),
        mi_read(*reinterpret_cast<const T *>(&u.w)));
}
template<> __device__ inline float4 mi_load4(const float *p) {
    return *reinterpret_cast<const float4 *>(p);
}
template<typename T> __device__ inline float4 mi_load_guarded(const T *p, int left) {
    if (left >= 4 && (reinterpret_cast<uintptr_t>(p) % (4 * sizeof(T))) == 0)
        return mi_load4(p);
    return make_float4(left > 0 ? mi_read(p[0]) : 0.f,
        left > 1 ? mi_read(p[1]) : 0.f, left > 2 ? mi_read(p[2]) : 0.f,
        left > 3 ? mi_read(p[3]) : 0.f);
}
template<typename T> __device__ inline void mi_store4(T *p, float4 v) {
    const T a=mi_cast<T>(v.x), b=mi_cast<T>(v.y), c=mi_cast<T>(v.z), d=mi_cast<T>(v.w);
    *reinterpret_cast<ushort4 *>(p) = make_ushort4(
        *reinterpret_cast<const unsigned short *>(&a), *reinterpret_cast<const unsigned short *>(&b),
        *reinterpret_cast<const unsigned short *>(&c), *reinterpret_cast<const unsigned short *>(&d));
}
template<> __device__ inline void mi_store4(float *p, float4 v) {
    *reinterpret_cast<float4 *>(p) = v;
}
template<typename T> __device__ inline void mi_store_guarded(T *p, int left, float4 v) {
    if (left >= 4 && (reinterpret_cast<uintptr_t>(p) % (4 * sizeof(T))) == 0) {
        mi_store4(p,v); return;
    }
    if(left>0)p[0]=mi_cast<T>(v.x);
    if(left>1)p[1]=mi_cast<T>(v.y);
    if(left>2)p[2]=mi_cast<T>(v.z);
    if(left>3)p[3]=mi_cast<T>(v.w);
}
__device__ inline float mi_warp_sum(float v) {
    #pragma unroll
    for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffffu,v,d);
    return v;
}
__device__ inline float mi_block_sum(float v) {
    __shared__ float sums[8];
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    v=mi_warp_sum(v);
    if(!lane)sums[warp]=v;
    __syncthreads();
    v=threadIdx.x<8?sums[lane]:0.f;
    if(!warp)v=mi_warp_sum(v);
    if(!threadIdx.x)sums[0]=v;
    __syncthreads();
    return sums[0];
}
__device__ inline float mi_block_max(float v) {
    __shared__ float maxima[8];
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    #pragma unroll
    for(int d=16;d;d>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffffu,v,d));
    if(!lane)maxima[warp]=v;
    __syncthreads();
    v=threadIdx.x<8?maxima[lane]:0.f;
    if(!warp) {
        #pragma unroll
        for(int d=16;d;d>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffffu,v,d));
    }
    if(!threadIdx.x)maxima[0]=v;
    __syncthreads();
    return maxima[0];
}
#endif
