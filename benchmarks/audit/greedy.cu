#include "greedy.h"
#include "../../src/operators/vector.cuh"
#include <math.h>
#include <limits.h>
template<typename T> __global__ void nv_bench_greedy_kernel(const T*x,int n,int32_t*out){
    __shared__ float values[256];__shared__ int ids[256];
    int t=threadIdx.x,id=INT_MAX;float best=-INFINITY;
    for(int j=t;j<n;j+=256){float v=mi_read(x[j]);if(v>best||(v==best&&j<id)){best=v;id=j;}}
    values[t]=best;ids[t]=id;__syncthreads();
    for(int stride=128;stride;stride>>=1){
        if(t<stride){float v=values[t+stride];int j=ids[t+stride];
            if(v>values[t]||(v==values[t]&&j<ids[t])){values[t]=v;ids[t]=j;}}
        __syncthreads();
    }
    if(!t)*out=ids[0];
}
extern "C" cudaError_t nv_bench_greedy(mi_dtype dt,const void*x,int n,int32_t*out,cudaStream_t s){
    if(!x||!out||n<1)return cudaErrorInvalidValue;
    switch(dt){
    case MI_F32:nv_bench_greedy_kernel<<<1,256,0,s>>>((const float*)x,n,out);break;
    case MI_F16:nv_bench_greedy_kernel<<<1,256,0,s>>>((const __half*)x,n,out);break;
    case MI_BF16:nv_bench_greedy_kernel<<<1,256,0,s>>>((const __nv_bfloat16*)x,n,out);break;
    default:return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
