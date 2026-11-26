#ifndef NV_BENCH_GREEDY_H
#define NV_BENCH_GREEDY_H
#include "miniinfer/operators.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Benchmark prototype: finite logits, batch=1, ties choose smallest ID. */
cudaError_t nv_bench_greedy(mi_dtype dtype,const void*logits,int vocab,int32_t*out,cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
