#ifndef NV_BENCH_CACHE_H
#define NV_BENCH_CACHE_H
#include <cuda_runtime_api.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Benchmark-only eviction. Allocate before capture, touch every cache line
 * before EACH timed invocation, in the same stream and before start event. */
cudaError_t nv_bench_cache_init(void **buffer, size_t *bytes);
cudaError_t nv_bench_cache_flush(void *buffer, size_t bytes, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
