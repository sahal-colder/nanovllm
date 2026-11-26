/* C11 host only. Fixed-seed random inputs, every output checked, no timings
 * include allocation, copies or validation. CUDA graph replay amortizes CPU
 * launch gaps uniformly for baseline, optimized and cuBLAS. */
#include "miniinfer/operators.h"
#include <cublas_v2.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e_));exit(1);} } while(0)
#define BLAS(call) do { cublasStatus_t b_=(call); if(b_!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s:%d %s: cuBLAS %d\n",__FILE__,__LINE__,#call,(int)b_);exit(1);} } while(0)

static uint32_t rng=0x739ab21u;
static float random_float(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return (float)(rng>>8)*(2.f/16777216.f)-1.f;}
static size_t element_bytes(mi_dtype d){return d==MI_F32?4:2;}
static const char *dtype_name(mi_dtype d){return d==MI_F32?"f32":d==MI_F16?"f16":"bf16";}
static uint32_t bits(float v){uint32_t u;memcpy(&u,&v,4);return u;}
static float from_bits(uint32_t u){float v;memcpy(&v,&u,4);return v;}
static uint16_t to_half(float v){
    uint32_t u=bits(v),sign=(u>>16)&0x8000u,m=u&0x7fffffu;
    int e=(int)((u>>23)&255)-112;
    if(((u>>23)&255)==255)return (uint16_t)(sign|0x7c00u|(m?0x200u:0));
    if(e>=31)return (uint16_t)(sign|0x7c00u);
    if(e<=0){
        if(e < -10)return (uint16_t)sign;
        m|=0x800000u;unsigned shift=(unsigned)(14-e),q=m>>shift,rem=m&((1u<<shift)-1),half=1u<<(shift-1);
        return (uint16_t)(sign+q+(rem>half||(rem==half&&(q&1))));
    }
    uint32_t q=((uint32_t)e<<10)|(m>>13),rem=m&8191u;
    return (uint16_t)(sign+q+(rem>4096u||(rem==4096u&&(q&1))));
}
static float from_half(uint16_t u){
    int e=(u>>10)&31,m=u&1023;float v;
    if(!e)v=ldexpf((float)m,-24);
    else if(e==31)v=m?NAN:INFINITY;
    else v=ldexpf(1.f+m/1024.f,e-15);
    return u&0x8000u?-v:v;
}
static void store(void *p,size_t i,mi_dtype d,float v){
    if(d==MI_F32)((float*)p)[i]=v;
    else if(d==MI_F16)((uint16_t*)p)[i]=to_half(v);
    else {uint32_t u=bits(v);((uint16_t*)p)[i]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
}
static float load(const void*p,size_t i,mi_dtype d){
    if(d==MI_F32)return ((const float*)p)[i];
    uint16_t u=((const uint16_t*)p)[i];
    return d==MI_F16?from_half(u):from_bits((uint32_t)u<<16);
}
static float quantize(float v,mi_dtype d){uint32_t tmp=0;store(&tmp,0,d,v);return load(&tmp,0,d);}
static void *allocate(size_t n){void*p=malloc(n);if(!p){fprintf(stderr,"host allocation failed\n");exit(1);}return p;}

/* Plain C data for one invocation, no callbacks or opaque context. */
typedef struct op_case {
    int rms,m,k,b,lda,xs,ys,ow,ox,oy;
    mi_dtype dtype;
    size_t nw,nx,ny,bytes;
    void *hw,*hx,*hy,*dw_raw,*dx_raw,*dy_raw,*dw,*dx,*dy;
    float *reference;
    cudaStream_t stream;
    cublasHandle_t blas;
} op_case;
typedef struct errors {double abs,rel;size_t count;} errors;
static int last_method;

static void invoke(const op_case*c,int method){
    last_method=method;
    if(c->rms){
        if(method==0)CUDA(mi_rmsnorm_baseline(c->dtype,c->dx,c->dw,c->dy,c->m,c->k,c->xs,c->ys,1.e-5f,c->stream));
        else CUDA(mi_rmsnorm_optimized(c->dtype,c->dx,c->dw,c->dy,c->m,c->k,c->xs,c->ys,1.e-5f,c->stream));
    }else if(method==0)CUDA(mi_sgemv_baseline(c->dtype,c->dw,c->dx,c->dy,c->m,c->k,c->b,c->lda,c->xs,c->ys,c->stream));
    else if(method==1)CUDA(mi_sgemv_optimized(c->dtype,c->dw,c->dx,c->dy,c->m,c->k,c->b,c->lda,c->xs,c->ys,c->stream));
    else {
        const float alpha=1.f,beta=0.f;
        if(c->dtype==MI_F32&&c->b==1)
            BLAS(cublasSgemv(c->blas,CUBLAS_OP_T,c->k,c->m,&alpha,c->dw,c->lda,c->dx,1,&beta,c->dy,1));
        else {
            cudaDataType_t t=c->dtype==MI_F32?CUDA_R_32F:c->dtype==MI_F16?CUDA_R_16F:CUDA_R_16BF;
            BLAS(cublasGemmEx(c->blas,CUBLAS_OP_T,CUBLAS_OP_N,c->m,c->b,c->k,&alpha,
                c->dw,t,c->lda,c->dx,t,c->xs,&beta,c->dy,t,c->ys,
                CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
        }
    }
}
static void create_case(op_case*c,int rms,mi_dtype d,int m,int k,int b,int offset,int pad,cudaStream_t s,cublasHandle_t h){
    memset(c,0,sizeof(*c));c->rms=rms;c->dtype=d;c->m=m;c->k=k;c->b=b;c->lda=k+pad;
    c->xs=k+pad;c->ys=(rms?k:m)+pad;c->ow=offset;c->ox=offset?2:0;c->oy=offset?3:0;
    c->bytes=element_bytes(d);c->stream=s;c->blas=h;
    c->nw=(rms?(size_t)k:(size_t)m*c->lda)+c->ow+8;
    c->nx=(size_t)(rms?m:b)*c->xs+c->ox+8;
    c->ny=(size_t)(rms?m:b)*c->ys+c->oy+8;
    c->hw=allocate(c->nw*c->bytes);c->hx=allocate(c->nx*c->bytes);c->hy=allocate(c->ny*c->bytes);
    c->reference=allocate(c->ny*sizeof(float));
    for(size_t i=0;i<c->nw;i++)store(c->hw,i,d,-123.f);
    for(size_t i=0;i<c->nx;i++)store(c->hx,i,d,-123.f);
    for(size_t i=0;i<c->ny;i++){store(c->hy,i,d,-123.f);c->reference[i]=-123.f;}
    for(int r=0;r<(rms?1:m);r++)for(int j=0;j<k;j++)store(c->hw,c->ow+(size_t)r*c->lda+j,d,random_float());
    for(int r=0;r<(rms?m:b);r++)for(int j=0;j<k;j++)store(c->hx,c->ox+(size_t)r*c->xs+j,d,random_float());
    CUDA(cudaMalloc(&c->dw_raw,c->nw*c->bytes));CUDA(cudaMalloc(&c->dx_raw,c->nx*c->bytes));CUDA(cudaMalloc(&c->dy_raw,c->ny*c->bytes));
    c->dw=(char*)c->dw_raw+c->ow*c->bytes;c->dx=(char*)c->dx_raw+c->ox*c->bytes;c->dy=(char*)c->dy_raw+c->oy*c->bytes;
    CUDA(cudaMemcpyAsync(c->dw_raw,c->hw,c->nw*c->bytes,cudaMemcpyHostToDevice,s));
    CUDA(cudaMemcpyAsync(c->dx_raw,c->hx,c->nx*c->bytes,cudaMemcpyHostToDevice,s));
    CUDA(cudaMemcpyAsync(c->dy_raw,c->hy,c->ny*c->bytes,cudaMemcpyHostToDevice,s));
    CUDA(cudaStreamSynchronize(s));
}
static void destroy_case(op_case*c){
    CUDA(cudaFree(c->dw_raw));CUDA(cudaFree(c->dx_raw));CUDA(cudaFree(c->dy_raw));
    free(c->hw);free(c->hx);free(c->hy);free(c->reference);
}
static void cpu_reference(op_case*c){
    if(c->rms){
        for(int r=0;r<c->m;r++){
            double sum=0.;
            for(int j=0;j<c->k;j++){double x=load(c->hx,c->ox+(size_t)r*c->xs+j,c->dtype);sum+=x*x;}
            double inv=1./sqrt(sum/c->k+1.e-5);
            for(int j=0;j<c->k;j++){
                double v=load(c->hx,c->ox+(size_t)r*c->xs+j,c->dtype)*inv*load(c->hw,c->ow+j,c->dtype);
                c->reference[c->oy+(size_t)r*c->ys+j]=quantize((float)v,c->dtype);
            }
        }
    }else for(int b=0;b<c->b;b++)for(int r=0;r<c->m;r++){
        double sum=0.;
        for(int j=0;j<c->k;j++)sum+=(double)load(c->hw,c->ow+(size_t)r*c->lda+j,c->dtype)*load(c->hx,c->ox+(size_t)b*c->xs+j,c->dtype);
        c->reference[c->oy+(size_t)b*c->ys+r]=quantize((float)sum,c->dtype);
    }
}
static void blas_reference(op_case*c){
    invoke(c,2);CUDA(cudaStreamSynchronize(c->stream));
    CUDA(cudaMemcpy(c->hy,c->dy_raw,c->ny*c->bytes,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<c->ny;i++)c->reference[i]=load(c->hy,i,c->dtype);
}
static errors check_output(op_case*c){
    CUDA(cudaMemcpyAsync(c->hy,c->dy_raw,c->ny*c->bytes,cudaMemcpyDeviceToHost,c->stream));
    CUDA(cudaStreamSynchronize(c->stream));errors e={0.,0.,0};
    const double atol=c->dtype==MI_F32?1.e-4:c->dtype==MI_F16?1.e-3:8.e-3;
    const double rtol=c->dtype==MI_F32?2.e-4:c->dtype==MI_F16?2.e-3:1.6e-2;
    for(size_t i=0;i<c->ny;i++){
        double actual=load(c->hy,i,c->dtype),expected=c->reference[i];
        int data=i>=(size_t)c->oy&&i<(size_t)c->oy+(size_t)(c->rms?c->m:c->b)*c->ys&&
            (i-c->oy)%(size_t)c->ys<(size_t)(c->rms?c->k:c->m);
        double ae=fabs(actual-expected),re=ae/fmax(fabs(expected),1.e-12);
        if(!isfinite(actual)||(!data&&ae!=0.)||(data&&ae>atol+rtol*fabs(expected))){
            fprintf(stderr,"FAIL %s %s method=%d m=%d k=%d b=%d offset=%d index=%zu actual=%.9g expected=%.9g abs=%.4g rel=%.4g\n",
                c->rms?"rms":"gemv",dtype_name(c->dtype),last_method,c->m,c->k,c->b,c->ow,i,actual,expected,ae,re);exit(1);
        }
        if(data){if(ae>e.abs)e.abs=ae;if(re>e.rel)e.rel=re;e.count++;}
    }
    return e;
}
static int compare_double(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void measure(const op_case*c,int method,int samples,int inner,double *p50,double*p95){
    cudaGraph_t graph;cudaGraphExec_t replay;
    cudaEvent_t *start=allocate((size_t)inner*sizeof(*start)),*end=allocate((size_t)inner*sizeof(*end));
    for(int i=0;i<inner;i++){CUDA(cudaEventCreate(&start[i]));CUDA(cudaEventCreate(&end[i]));}
    for(int i=0;i<10;i++)invoke(c,method);
    CUDA(cudaStreamSynchronize(c->stream));
    CUDA(cudaStreamBeginCapture(c->stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<inner;i++){
        CUDA(cudaEventRecordWithFlags(start[i],c->stream,cudaEventRecordExternal));
        invoke(c,method);
        CUDA(cudaEventRecordWithFlags(end[i],c->stream,cudaEventRecordExternal));
    }
    CUDA(cudaStreamEndCapture(c->stream,&graph));
    CUDA(cudaGraphInstantiate(&replay,graph,0));
    size_t count=(size_t)samples*inner;
    double *times=allocate(count*sizeof(double));
    for(int i=0;i<samples;i++){
        CUDA(cudaGraphLaunch(replay,c->stream));CUDA(cudaStreamSynchronize(c->stream));
        for(int j=0;j<inner;j++){
            float ms;CUDA(cudaEventElapsedTime(&ms,start[j],end[j]));times[(size_t)i*inner+j]=ms*1000.;
        }
    }
    qsort(times,count,sizeof(double),compare_double);
    *p50=times[(size_t)ceil(.50*count)-1];*p95=times[(size_t)ceil(.95*count)-1];
    free(times);
    for(int i=0;i<inner;i++){CUDA(cudaEventDestroy(start[i]));CUDA(cudaEventDestroy(end[i]));}
    free(start);free(end);
    CUDA(cudaGraphExecDestroy(replay));CUDA(cudaGraphDestroy(graph));
}
static void benchmark_case(op_case*c,int samples,int inner){
    if(c->rms)cpu_reference(c);else blas_reference(c);
    double median[3]={0},tail[3]={0};errors err[3];int methods=c->rms?2:3;
    /* Rotate order by shape to reduce systematic baseline/optimized clock bias. */
    for(int i=0;i<methods;i++){
        int method=(i+c->m+c->b)%methods;
        invoke(c,method);CUDA(cudaStreamSynchronize(c->stream));err[method]=check_output(c);
        measure(c,method,samples,inner,&median[method],&tail[method]);
        (void)check_output(c);
    }
    /* Identical ideal unique tensor bytes for all implementations. This is a
     * useful-bandwidth model, never a measurement of DRAM traffic. */
    double bytes=c->bytes*(c->rms?(2.*c->m*c->k+c->k):((double)c->m*c->k+(double)c->b*(c->k+c->m)));
    for(int method=0;method<methods;method++){
        printf("%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.0f,%.6f,",
            c->rms?"rmsnorm":"sgemv",dtype_name(c->dtype),method==0?"baseline":method==1?"optimized":(c->dtype==MI_F32&&c->b==1?"cublasSgemv":"cublasGemmEx"),
            c->m,c->k,c->rms?c->m:c->b,c->ow,c->lda,c->xs,c->ys,c->rms?"cpu_f64":"cublas",
            median[method],tail[method],bytes,bytes/(median[method]*1.e-6)/1073741824.);
        if(!c->rms)printf("%.6f",median[2]/median[method]);
        printf(",%.9g,%.9g,%zu,%d,%d,0x739ab21\n",err[method].abs,err[method].rel,err[method].count,samples,inner);
    }
    fflush(stdout);
}
static void inplace_test(op_case*c){
    void *old=c->dy,*old_raw=c->dy_raw;int old_y=c->ys,old_oy=c->oy;size_t old_n=c->ny;
    float *old_ref=c->reference;void *old_host=c->hy;
    c->reference=allocate(c->nx*sizeof(float));c->hy=allocate(c->nx*c->bytes);
    for(size_t i=0;i<c->nx;i++)c->reference[i]=load(c->hx,i,c->dtype);
    for(int r=0;r<c->m;r++)for(int j=0;j<c->k;j++)c->reference[c->ox+(size_t)r*c->xs+j]=old_ref[old_oy+(size_t)r*old_y+j];
    c->dy=c->dx;c->dy_raw=c->dx_raw;c->ys=c->xs;c->oy=c->ox;c->ny=c->nx;
    for(int method=0;method<2;method++){
        CUDA(cudaMemcpyAsync(c->dx_raw,c->hx,c->nx*c->bytes,cudaMemcpyHostToDevice,c->stream));
        invoke(c,method);(void)check_output(c);
    }
    free(c->reference);free(c->hy);c->reference=old_ref;c->hy=old_host;
    c->dy=old;c->dy_raw=old_raw;c->ys=old_y;c->oy=old_oy;c->ny=old_n;
}
static void validation(cudaStream_t s,cublasHandle_t h){
    const int shapes[][2]={{1,1},{3,3},{7,17},{31,129},{65,513},{129,4099}};
    const int hidden[]={1,3,31,127,511,512,513,1024,1025,2048,2049,4096,4099,8192,8193,11008,11011,16384};
    size_t cases=0,checked=0;double maxabs[3]={0},maxrel[3]={0};
    for(int d=0;d<3;d++)for(int offset=0;offset<4;offset++){
        for(size_t sh=0;sh<sizeof(shapes)/sizeof(shapes[0]);sh++){
            op_case c;create_case(&c,0,(mi_dtype)d,shapes[sh][0],shapes[sh][1],offset?3:1,offset,offset?3:0,s,h);
            cpu_reference(&c);
            for(int method=0;method<3;method++){invoke(&c,method);errors e=check_output(&c);checked+=e.count;maxabs[d]=fmax(maxabs[d],e.abs);maxrel[d]=fmax(maxrel[d],e.rel);}
            destroy_case(&c);cases++;
        }
        for(size_t j=0;j<sizeof(hidden)/sizeof(hidden[0]);j++){
            op_case c;create_case(&c,1,(mi_dtype)d,offset?17:1,hidden[j],1,offset,offset?3:0,s,h);
            cpu_reference(&c);
            for(int method=0;method<2;method++){invoke(&c,method);errors e=check_output(&c);checked+=e.count;maxabs[d]=fmax(maxabs[d],e.abs);maxrel[d]=fmax(maxrel[d],e.rel);}
            inplace_test(&c);destroy_case(&c);cases++;
        }
    }
    /* Large, tiny and zero values exercise the stable RMS branch and epsilon.
     * Inputs remain random where the selected scale is nonzero. */
    for(int d=0;d<3;d++)for(int scenario=0;scenario<3;scenario++){
        op_case c;create_case(&c,1,(mi_dtype)d,3,4099,1,1,3,s,h);
        float scale=scenario==0?(d==MI_F16?1.e4f:1.e20f):scenario==1?1.e-20f:0.f;
        for(int r=0;r<c.m;r++)for(int j=0;j<c.k;j++)store(c.hx,c.ox+(size_t)r*c.xs+j,c.dtype,random_float()*scale);
        CUDA(cudaMemcpyAsync(c.dx_raw,c.hx,c.nx*c.bytes,cudaMemcpyHostToDevice,s));cpu_reference(&c);
        for(int method=0;method<2;method++){invoke(&c,method);errors e=check_output(&c);checked+=e.count;maxabs[d]=fmax(maxabs[d],e.abs);maxrel[d]=fmax(maxrel[d],e.rel);}
        inplace_test(&c);destroy_case(&c);cases++;
    }
    float *dummy;CUDA(cudaMalloc((void**)&dummy,64));
    if(mi_sgemv_optimized(MI_F32,dummy,dummy,dummy,1,3,1,2,3,1,s)!=cudaErrorInvalidValue||
       mi_rmsnorm_optimized(MI_F32,dummy,dummy,dummy,1,1,1,1,0.f,s)!=cudaErrorInvalidValue||
       mi_rmsnorm_optimized(MI_F32,dummy,dummy,dummy,1,16385,16385,16385,1.e-5f,s)!=cudaErrorInvalidValue){fprintf(stderr,"invalid argument validation failed\n");exit(1);}
    CUDA(cudaFree(dummy));
    fprintf(stderr,"PASS %zu random/boundary cases, %zu element comparisons plus in-place/canaries; seed=0x739ab21\n",cases,checked);
    for(int d=0;d<3;d++)fprintf(stderr,"%s max_abs_err=%.9g max_rel_err=%.9g (mixed abs/rel tolerance)\n",dtype_name((mi_dtype)d),maxabs[d],maxrel[d]);
}
static int positive(const char*s){char*end;long v=strtol(s,&end,10);if(*end||v<=0||v>10000){fprintf(stderr,"invalid positive integer: %s\n",s);exit(2);}return (int)v;}
int main(int argc,char**argv){
    int test=0,quick=0,samples=80,inner=0,selected=-1,profile=0;const char *op="all";
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--test"))test=1;
        else if(!strcmp(argv[i],"--quick"))quick=1;
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=positive(argv[++i]);
        else if(!strcmp(argv[i],"--inner")&&i+1<argc)inner=positive(argv[++i]);
        else if(!strcmp(argv[i],"--op")&&i+1<argc)op=argv[++i];
        else if(!strcmp(argv[i],"--profile")&&i+1<argc){profile=1;op=argv[++i];samples=1;inner=1;}
        else if(!strcmp(argv[i],"--dtype")&&i+1<argc){const char*p=argv[++i];selected=!strcmp(p,"f32")?0:!strcmp(p,"f16")?1:!strcmp(p,"bf16")?2:-2;}
        else {fprintf(stderr,"usage: %s [--test|--quick|--profile gemv|rms] [--op all|gemv|rms] [--dtype f32|f16|bf16] [--samples N] [--inner N]\n",argv[0]);return 2;}
    }
    if(selected==-2||(strcmp(op,"all")&&strcmp(op,"gemv")&&strcmp(op,"rms")))return 2;
    int count=0;cudaError_t available=cudaGetDeviceCount(&count);
    if(available!=cudaSuccess||!count){fprintf(stderr,"SKIP: no CUDA device: %s\n",cudaGetErrorString(available));return 77;}
    struct cudaDeviceProp device;CUDA(cudaGetDeviceProperties(&device,0));
    cublasHandle_t h;BLAS(cublasCreate(&h));int version;BLAS(cublasGetVersion(h,&version));
    BLAS(cublasSetMathMode(h,(cublasMath_t)(CUBLAS_PEDANTIC_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    cudaStream_t s;CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));BLAS(cublasSetStream(h,s));
    fprintf(stderr,"GPU=%s VRAM_bytes=%zu SM=%d.%d cuBLAS=%d math=FP32_PEDANTIC graph_event_timing=1\n",device.name,device.totalGlobalMem,device.major,device.minor,version);
    if(test)validation(s,h);
    else {
        puts("operator,dtype,implementation,m,k,batch,offset,lda,x_stride,y_stride,reference,p50_us,p95_us,ideal_bytes,effective_GiB_s,bandwidth_ratio_to_cublas,max_abs_err,max_rel_err,validated_elements,samples,inner,seed");
        const int shapes[][2]={{2048,2048},{4096,4096},{8192,4096},{11008,4096}};
        const int batches[]={1,2,8,32},hidden[]={512,1024,2048,4096,8192,11008},rows[]={1,8,128,512};
        for(int d=0;d<3;d++){
            if(selected>=0&&d!=selected)continue;
            if(strcmp(op,"rms"))for(int sh=0;sh<(quick||profile?1:4);sh++)for(int bi=0;bi<(quick||profile?1:4);bi++){
                op_case c;create_case(&c,0,(mi_dtype)d,profile?8192:shapes[sh][0],profile?4096:shapes[sh][1],batches[bi],0,0,s,h);
                benchmark_case(&c,samples,inner?inner:5);destroy_case(&c);
            }
            if(strcmp(op,"gemv"))for(int hi=0;hi<(quick||profile?1:6);hi++)for(int ri=0;ri<(quick||profile?1:4);ri++){
                op_case c;create_case(&c,1,(mi_dtype)d,quick||profile?128:rows[ri],quick||profile?4096:hidden[hi],1,0,0,s,h);
                benchmark_case(&c,samples,inner?inner:64);destroy_case(&c);
            }
        }
    }
    BLAS(cublasDestroy(h));CUDA(cudaStreamDestroy(s));return 0;
}
