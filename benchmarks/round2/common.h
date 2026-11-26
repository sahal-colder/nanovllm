#ifndef NV_BENCH_COMMON_H
#define NV_BENCH_COMMON_H
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "nv_cuda.h"
#include "../../src/numeric.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_));exit(1);}}while(0)
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
static inline float rnd(void){static uint32_t state=0x82a6371u;state^=state<<13;state^=state>>17;state^=state<<5;return (float)(state>>8)*(2.f/16777216.f)-1.f;}
static inline const char *dtype_name(mi_dtype d){return d==MI_F32?"f32":d==MI_F16?"f16":"bf16";}
static inline void *dev(size_t n){void*p;CHECK(cudaMalloc(&p,n));return p;}
static inline int cmp_double(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static inline double percentile(double*x,size_t n,double p){qsort(x,n,sizeof(double),cmp_double);return x[(size_t)ceil(n*p)-1];}
static inline double now_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
typedef struct Error {double abs,rel;size_t n;} Error;
static inline void compare(Error*e,double got,double ref,double atol,double rtol,const char*label){
    double a=fabs(got-ref),r=a/fmax(fabs(ref),1.e-12);
    if(!isfinite(got)||a>atol+rtol*fabs(ref)){fprintf(stderr,"FAIL %s got=%.10g ref=%.10g abs=%.6g rel=%.6g\n",label,got,ref,a,r);exit(1);}
    e->abs=fmax(e->abs,a);e->rel=fmax(e->rel,r);e->n++;
}
int attention_tests(void);
void attention_benchmark(int profile,int quick,int selected);
void attention_audit(mi_dtype dt,int length,int batch,int cold,int samples,int method,int profile);
int decode_tests(const char*root);
void decode_audit(const char*path,mi_dtype dt,int samples);
void decode_benchmark(const char*path,int trace,int selected);
#endif
