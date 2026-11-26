#ifndef MINIINFER_OPERATORS_H
#define MINIINFER_OPERATORS_H
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum mi_dtype { MI_F32 = 0, MI_F16 = 1, MI_BF16 = 2 } mi_dtype;

/* Device pointers, element strides, row-major W[M,K], X[B,K], Y[B,M].
 * Y = X W^T; no bias/alpha/beta. Positive dimensions, batch <= 65535.
 * Pointers need only their element's natural alignment; no 16-byte assumption.
 * Y must not alias W/X. Enqueue only: synchronize the supplied stream for errors.
 * F16/BF16 are IEEE 16-bit storage, with FP32 accumulation and rounded output.
 * These functions allocate no memory, own no state, and accept stream 0. */
cudaError_t mi_sgemv_baseline(mi_dtype dtype, const void *w, const void *x, void *y,
    int m, int k, int batch, int lda, int x_stride, int y_stride, cudaStream_t stream);
cudaError_t mi_sgemv_optimized(mi_dtype dtype, const void *w, const void *x, void *y,
    int m, int k, int batch, int lda, int x_stride, int y_stride, cudaStream_t stream);

/* y[r,i] = x[r,i] * rsqrt(mean(x[r,:]^2) + epsilon) * weight[i].
 * Finite inputs/weights; epsilon must be finite and positive. Hidden 1..16384.
 * Supports exact x==y in-place, but no other overlap. FP32 reduction; scaled
 * fallback avoids overflow of the square sum for large finite inputs.
 * Optimized entry always launches the guarded vector kernel (no dispatcher
 * fallback). Aligned groups load/store 4 elements; misaligned groups and tails
 * use safe element accesses in that same kernel. */
cudaError_t mi_rmsnorm_baseline(mi_dtype dtype, const void *x, const void *weight,
    void *y, int rows, int hidden, int x_stride, int y_stride, float epsilon,
    cudaStream_t stream);
cudaError_t mi_rmsnorm_optimized(mi_dtype dtype, const void *x, const void *weight,
    void *y, int rows, int hidden, int x_stride, int y_stride, float epsilon,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
