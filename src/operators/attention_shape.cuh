#ifndef NV_ATTENTION_SHAPE_CUH
#define NV_ATTENTION_SHAPE_CUH
#include "nv_attention.h"
static inline bool nv_shape_ok(const NvAttention&a){
    return a.batch>0&&a.batch<=4096&&a.heads>0&&a.heads<=1024&&a.kv_heads>0&&
        a.heads%a.kv_heads==0&&a.head_dim>0&&a.head_dim<=256&&a.page_tokens>0&&
        a.page_tokens<=1024&&a.max_blocks>0&&a.max_blocks<=65536/a.page_tokens&&a.num_pages>0;
}
#endif
