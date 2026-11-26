#include "cache.h"
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void nv_bench_l2_flush_kernel(uint4 *p, size_t n) {
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
        uint4 v=p[i];v.x+=1;v.y+=3;v.z+=5;v.w+=7;p[i]=v;
    }
}
extern "C" cudaError_t nv_bench_cache_init(void **buffer,size_t *bytes){
    int device=0;cudaDeviceProp prop;
    cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
    e=cudaGetDeviceProperties(&prop,device);if(e!=cudaSuccess)return e;
    *bytes=128ull*1024*1024;
    if(*bytes<(size_t)prop.l2CacheSize*4)*bytes=(size_t)prop.l2CacheSize*4;
    fprintf(stderr,"GPU=%s SM_count=%d L2_bytes=%d eviction_bytes=%zu memory_bus_bits=%d\n",prop.name,prop.multiProcessorCount,prop.l2CacheSize,*bytes,prop.memoryBusWidth);
    e=cudaMalloc(buffer,*bytes);if(e!=cudaSuccess)return e;
    return cudaMemset(*buffer,0,*bytes);
}
extern "C" cudaError_t nv_bench_cache_flush(void *buffer,size_t bytes,cudaStream_t stream){
    nv_bench_l2_flush_kernel<<<1024,256,0,stream>>>((uint4*)buffer,bytes/sizeof(uint4));
    return cudaGetLastError();
}
