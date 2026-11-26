#ifndef NV_NUMERIC_H
#define NV_NUMERIC_H
#include "miniinfer/operators.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
static inline size_t nv_bytes(mi_dtype d){return d==MI_F32?4:2;}
static inline uint32_t nv_bits(float v){uint32_t u;memcpy(&u,&v,4);return u;}
static inline float nv_float(uint32_t u){float v;memcpy(&v,&u,4);return v;}
static inline uint16_t nv_half(float v){
    uint32_t u=nv_bits(v),s=(u>>16)&0x8000u,m=u&0x7fffffu;int e=(int)((u>>23)&255)-112;
    if(((u>>23)&255)==255)return (uint16_t)(s|0x7c00u|(m?0x200u:0));
    if(e>=31)return (uint16_t)(s|0x7c00u);
    if(e<=0){if(e< -10)return (uint16_t)s;m|=0x800000u;unsigned shift=14-e,q=m>>shift,r=m&((1u<<shift)-1),h=1u<<(shift-1);return (uint16_t)(s+q+(r>h||(r==h&&(q&1))));}
    uint32_t q=((uint32_t)e<<10)|(m>>13),r=m&8191u;return (uint16_t)(s+q+(r>4096u||(r==4096u&&(q&1))));
}
static inline void nv_pack(void*p,size_t i,mi_dtype d,float v){
    if(d==MI_F32)((float*)p)[i]=v;
    else if(d==MI_F16)((uint16_t*)p)[i]=nv_half(v);
    else{uint32_t u=nv_bits(v);((uint16_t*)p)[i]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
}
static inline float nv_unpack(const void*p,size_t i,mi_dtype d){
    if(d==MI_F32)return ((const float*)p)[i];uint16_t u=((const uint16_t*)p)[i];
    if(d==MI_BF16)return nv_float((uint32_t)u<<16);
    int e=(u>>10)&31,m=u&1023;float v=!e?ldexpf((float)m,-24):e==31?(m?NAN:INFINITY):ldexpf(1.f+m/1024.f,e-15);
    return u&0x8000u?-v:v;
}
static inline float nv_round(float v,mi_dtype d){uint32_t tmp=0;nv_pack(&tmp,0,d,v);return nv_unpack(&tmp,0,d);}
#endif
