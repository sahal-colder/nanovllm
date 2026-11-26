#include "common.h"
#include "../audit/cache.h"
#include <cuda_profiler_api.h>
typedef struct AttentionCase {
    NvAttention a;mi_dtype dtype;cudaStream_t stream;
    int32_t *table,*lengths;void *h_pages,*h_q,*h_y,*d_arena,*d_qraw,*d_yraw;
    void **h_pointers;void *d_q,*d_y;float *work_raw,*work;float *h_work;
    float *reference;size_t stride,page_elements,arena_elements,n,work_bytes,prefix,work_prefix;
} AttentionCase;
static size_t kv_offset(const AttentionCase*c,int b,int t,int kv,int d,int value){
    int physical=c->table[(size_t)b*c->a.max_blocks+t/c->a.page_tokens];
    return (size_t)physical*c->stride+c->prefix+(size_t)value*c->page_elements/2+
        ((size_t)(t%c->a.page_tokens)*c->a.kv_heads+kv)*c->a.head_dim+d;
}
static void create(AttentionCase*c,mi_dtype dt,int batch,int heads,int kv,int d,int length,int offset){
    memset(c,0,sizeof(*c));c->dtype=dt;CHECK(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));
    int blocks=(length>0?(length-1)/16+1:1),pages=batch*blocks;
    c->a=(NvAttention){batch,heads,kv,d,16,blocks,pages,NULL,NULL,NULL,NULL};
    c->page_elements=(size_t)2*16*kv*d;c->prefix=offset?3:4;
    c->stride=offset?c->page_elements+11:(c->page_elements+15)&~(size_t)3;c->arena_elements=(size_t)pages*c->stride;
    c->n=(size_t)batch*heads*d;size_t bytes=nv_bytes(dt);
    c->table=nv_alloc((size_t)batch*blocks,4);c->lengths=nv_alloc(batch,4);
    for(int i=0;i<pages;i++)c->table[i]=i;
    for(int i=pages-1;i>0;i--){int j=(int)((rnd()+1.f)*.4999f*(i+1));int tmp=c->table[i];c->table[i]=c->table[j];c->table[j]=tmp;}
    for(int b=0;b<batch;b++)c->lengths[b]=length>3*b?length-3*b:0;
    c->h_pages=nv_alloc(c->arena_elements,bytes);c->h_q=nv_alloc(c->n+11,bytes);c->h_y=nv_alloc(c->n+11,bytes);c->reference=nv_alloc(c->n,4);
    for(size_t i=0;i<c->arena_elements;i++)nv_pack(c->h_pages,i,dt,-123);
    for(int p=0;p<pages;p++)for(size_t j=0;j<c->page_elements;j++)nv_pack(c->h_pages,(size_t)p*c->stride+c->prefix+j,dt,rnd());
    /* seq=33 is an extreme-logit fixture: naive exp(score) would overflow. */
    for(size_t i=0;i<c->n+11;i++){nv_pack(c->h_q,i,dt,rnd()*(length==33?100.f:1.f));nv_pack(c->h_y,i,dt,-123);}
    /* Poison unread tail slots, so a masked load followed by 0*NaN also fails. */
    for(int b=0;b<batch;b++)if(c->lengths[b]%16){
        int end=(c->lengths[b]/16+1)*16;
        for(int t=c->lengths[b];t<end;t++)for(int h=0;h<kv;h++)for(int j=0;j<d;j++)for(int v=0;v<2;v++)
            nv_pack(c->h_pages,kv_offset(c,b,t,h,j,v),dt,NAN);
    }
    c->d_arena=dev(c->arena_elements*bytes);c->h_pointers=nv_alloc(pages,sizeof(void*));
    for(int p=0;p<pages;p++)c->h_pointers[p]=(char*)c->d_arena+((size_t)p*c->stride+c->prefix)*bytes;
    c->a.d_pages=dev((size_t)pages*sizeof(void*));c->a.d_block_table=dev((size_t)pages*4);c->a.d_lengths=dev((size_t)batch*4);c->a.d_error=dev(4);
    c->d_qraw=dev((c->n+11)*bytes);c->d_yraw=dev((c->n+11)*bytes);
    /* Offset=0 tests aligned q/output; offset=1 tests element-aligned views. */
    c->d_q=(char*)c->d_qraw+offset*bytes;c->d_y=(char*)c->d_yraw+offset*bytes;
    CHECK(cudaMemcpy(c->d_arena,c->h_pages,c->arena_elements*bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)c->a.d_pages,c->h_pointers,(size_t)pages*sizeof(void*),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)c->a.d_block_table,c->table,(size_t)pages*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)c->a.d_lengths,c->lengths,(size_t)batch*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(c->d_qraw,c->h_q,(c->n+11)*bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(c->d_yraw,c->h_y,(c->n+11)*bytes,cudaMemcpyHostToDevice));
    c->work_bytes=nv_attention_workspace_bytes(&c->a);c->work_raw=dev(c->work_bytes+32);c->work_prefix=offset?3:4;c->work=c->work_raw+c->work_prefix;c->h_work=nv_alloc(c->work_bytes+32,1);
    for(size_t i=0;i<(c->work_bytes+32)/4;i++)c->h_work[i]=-123.f;
    CHECK(cudaMemcpy(c->work_raw,c->h_work,c->work_bytes+32,cudaMemcpyHostToDevice));CHECK(cudaMemset(c->a.d_error,0,4));
    for(int b=0;b<batch;b++)for(int h=0;h<heads;h++){
        int len=c->lengths[b],kh=h/(heads/kv);double *scores=nv_alloc(len?len:1,sizeof(double)),max=-INFINITY,sum=0;
        for(int t=0;t<len;t++){
            double dot=0;for(int j=0;j<d;j++)dot+=(double)nv_unpack(c->h_q,offset+((size_t)b*heads+h)*d+j,dt)*nv_unpack(c->h_pages,kv_offset(c,b,t,kh,j,0),dt);
            scores[t]=dot/sqrt((double)d);max=fmax(max,scores[t]);
        }
        for(int t=0;t<len;t++){scores[t]=exp(scores[t]-max);sum+=scores[t];}
        for(int j=0;j<d;j++){
            double out=0;for(int t=0;t<len;t++)out+=scores[t]/sum*nv_unpack(c->h_pages,kv_offset(c,b,t,kh,j,1),dt);
            c->reference[((size_t)b*heads+h)*d+j]=nv_round((float)out,dt);
        }
        free(scores);
    }
}
static Error check(AttentionCase*c,int offset){
    size_t bytes=nv_bytes(c->dtype);CHECK(cudaStreamSynchronize(c->stream));
    CHECK(cudaMemcpy(c->h_y,c->d_yraw,(c->n+11)*bytes,cudaMemcpyDeviceToHost));
    int error;CHECK(cudaMemcpy(&error,c->a.d_error,4,cudaMemcpyDeviceToHost));REQUIRE(error==0);
    Error e={0};double atol=c->dtype==MI_F32?2.e-5:c->dtype==MI_F16?1.e-3:8.e-3,rtol=c->dtype==MI_F32?3.e-4:c->dtype==MI_F16?2.e-3:1.6e-2;
    for(size_t i=0;i<c->n+11;i++){
        float x=nv_unpack(c->h_y,i,c->dtype);
        if(i<(size_t)offset||i>=c->n+offset)REQUIRE(x==-123.f);
        else compare(&e,x,c->reference[i-offset],atol,rtol,"paged attention");
    }
    CHECK(cudaMemcpy(c->h_work,c->work_raw,c->work_bytes+32,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<c->work_prefix;i++)REQUIRE(c->h_work[i]==-123.f);
    for(size_t i=c->work_bytes/4+c->work_prefix;i<(c->work_bytes+32)/4;i++)REQUIRE(c->h_work[i]==-123.f);
    return e;
}
static void run(AttentionCase*c,int opt){CHECK(nv_paged_attention(c->dtype,c->d_q,c->d_y,c->a,c->work,c->work_bytes,opt,c->stream));}
static void destroy(AttentionCase*c){
    CHECK(cudaStreamSynchronize(c->stream));void*ptrs[]={c->d_arena,(void*)c->a.d_pages,(void*)c->a.d_block_table,(void*)c->a.d_lengths,c->a.d_error,c->d_qraw,c->d_yraw,c->work_raw};
    for(size_t i=0;i<sizeof(ptrs)/sizeof(*ptrs);i++)CHECK(cudaFree(ptrs[i]));
    free(c->table);free(c->lengths);free(c->h_pages);free(c->h_pointers);free(c->h_q);free(c->h_y);free(c->reference);free(c->h_work);CHECK(cudaStreamDestroy(c->stream));
}
int attention_tests(void){
    const int lengths[]={128,1024,1037,4096},ratios[]={1,4,8};Error total[3]={{0}};int cases=0;
    for(int dt=0;dt<3;dt++)for(int bi=0;bi<2;bi++)for(int r=0;r<3;r++)for(int l=0;l<4;l++){
        AttentionCase c;create(&c,(mi_dtype)dt,bi?3:1,8,8/ratios[r],64,lengths[l],1);
        for(int opt=0;opt<3;opt++){run(&c,opt);Error e=check(&c,1);total[dt].abs=fmax(total[dt].abs,e.abs);total[dt].rel=fmax(total[dt].rel,e.rel);total[dt].n+=e.n;}
        destroy(&c);cases++;
    }
    const int dims[]={1,31,65,127,128,256},lens[]={0,1,15,16,17,33};
    for(int dt=0;dt<3;dt++)for(int j=0;j<6;j++){
        AttentionCase c;create(&c,(mi_dtype)dt,2,6,2,dims[j],lens[j],j&1);
        for(int opt=0;opt<3;opt++){run(&c,opt);(void)check(&c,j&1);}
        destroy(&c);cases++;
    }
    AttentionCase c;create(&c,MI_F32,1,8,2,65,1037,1);
    REQUIRE(nv_paged_attention(MI_F32,c.d_q,c.d_y,c.a,NULL,0,1,c.stream)==cudaErrorInvalidValue);
    REQUIRE(nv_paged_attention(MI_F32,c.d_q,c.d_y,c.a,NULL,0,3,c.stream)==cudaErrorInvalidValue);
    CHECK(nv_paged_attention(MI_F32,c.d_q,c.d_y,c.a,NULL,0,2,c.stream));(void)check(&c,1);
    int bad=-1,flag;CHECK(cudaMemcpy((void*)c.a.d_block_table,&bad,4,cudaMemcpyHostToDevice));run(&c,1);CHECK(cudaStreamSynchronize(c.stream));CHECK(cudaMemcpy(&flag,c.a.d_error,4,cudaMemcpyDeviceToHost));REQUIRE(flag&2);
    CHECK(cudaMemset(c.a.d_error,0,4));bad=c.a.max_blocks*16+1;CHECK(cudaMemcpy((void*)c.a.d_lengths,&bad,4,cudaMemcpyHostToDevice));run(&c,1);CHECK(cudaStreamSynchronize(c.stream));CHECK(cudaMemcpy(&flag,c.a.d_error,4,cudaMemcpyDeviceToHost));REQUIRE(flag&1);destroy(&c);
    fprintf(stderr,"PASS attention cases=%d + invalid-metadata guards, seed=0x82a6371\n",cases);
    for(int dt=0;dt<3;dt++)fprintf(stderr,"attention %s comparisons=%zu max_abs_err=%.9g max_rel_err=%.9g\n",dtype_name(dt),total[dt].n,total[dt].abs,total[dt].rel);
    return 0;
}
static size_t measured_flush_bytes=0;
static void timing(AttentionCase*c,int opt,int cold,int repeats,double*p50,double*p95){
    enum{INNER=5};void*flush=NULL;size_t flush_bytes=0;
    if(cold)CHECK(nv_bench_cache_init(&flush,&flush_bytes));
    measured_flush_bytes=flush_bytes;
    cudaEvent_t start[INNER],end[INNER];cudaGraph_t graph;cudaGraphExec_t exec;
    double *times=nv_alloc((size_t)INNER*repeats,sizeof(double));
    for(int i=0;i<10;i++)run(c,opt);
    CHECK(cudaStreamSynchronize(c->stream));
    for(int i=0;i<INNER;i++){CHECK(cudaEventCreate(&start[i]));CHECK(cudaEventCreate(&end[i]));}
    CHECK(cudaStreamBeginCapture(c->stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<INNER;i++){if(cold)CHECK(nv_bench_cache_flush(flush,flush_bytes,c->stream));CHECK(cudaEventRecordWithFlags(start[i],c->stream,cudaEventRecordExternal));run(c,opt);CHECK(cudaEventRecordWithFlags(end[i],c->stream,cudaEventRecordExternal));}
    CHECK(cudaStreamEndCapture(c->stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
    for(int r=0;r<repeats;r++){CHECK(cudaGraphLaunch(exec,c->stream));CHECK(cudaStreamSynchronize(c->stream));for(int i=0;i<INNER;i++){float ms;CHECK(cudaEventElapsedTime(&ms,start[i],end[i]));times[r*INNER+i]=ms*1000.;}}
    *p50=percentile(times,INNER*repeats,.5);*p95=percentile(times,INNER*repeats,.95);
    free(times);if(flush)CHECK(cudaFree(flush));CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));for(int i=0;i<INNER;i++){CHECK(cudaEventDestroy(start[i]));CHECK(cudaEventDestroy(end[i]));}
}
void attention_benchmark(int profile,int quick,int selected){
    int lengths[]={128,1024,1037,4096},ratios[]={1,4,16};
    if(!profile)puts("dtype,batch,heads,kv_heads,head_dim,seq_len,implementation,p50_us,p95_us,ideal_bytes,effective_GiB_s,max_abs_err,max_rel_err,samples,cache_policy,flush_bytes");
    for(int dt=profile?selected:0;dt<(profile?selected+1:3);dt++)for(int bi=0;bi<(quick?1:2);bi++)for(int ri=0;ri<(quick?1:3);ri++)for(int li=0;li<(quick?1:4);li++){
        AttentionCase c;int len=quick?4096:lengths[li];create(&c,dt,bi?4:1,16,quick?4:16/ratios[ri],128,len,0);
        for(int opt=profile?1:0;opt<2;opt++){
            run(&c,opt);Error e=check(&c,0);double p50=0,p95=0;
            if(!profile)timing(&c,opt,1,60,&p50,&p95);else for(int i=0;i<4;i++)run(&c,opt);
            (void)check(&c,0);size_t sum=0;for(int b=0;b<c.a.batch;b++)sum+=c.lengths[b];
            size_t bytes=nv_bytes(dt)*(2*sum*c.a.kv_heads*c.a.head_dim+2*c.n);
            if(!profile)printf("%s,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%zu,%.6f,%.9g,%.9g,300,cold,%zu\n",dtype_name(dt),c.a.batch,c.a.heads,c.a.kv_heads,c.a.head_dim,len,opt==2?"unsplit_vec4":opt?"split_online_vec4":"serial_online",p50,p95,bytes,bytes/(p50*1.e-6)/1073741824.,e.abs,e.rel,measured_flush_bytes);
            else fprintf(stderr,"PROFILE validated %s split-online, abs=%.9g rel=%.9g\n",dtype_name(dt),e.abs,e.rel);
            fflush(stdout);
        }
        destroy(&c);
    }
}

/* Explicit shape/cache/method entry point for audit and profiler. Method -1
 * compares all three; method 0 is one warp/head, four warps/CTA, NO sequence split. */
void attention_audit(mi_dtype dt,int length,int batch,int cold,int samples,int method,int profile){
    REQUIRE(length>=1&&length<=65536&&batch>=1&&batch<=64&&samples>=5&&samples%5==0&&samples<=10000&&method>=-1&&method<=2);
    AttentionCase c;create(&c,dt,batch,16,4,128,length,0);
    puts("dtype,batch,heads,kv_heads,head_dim,seq_len,implementation,p50_us,p95_us,ideal_bytes,effective_GiB_s,max_abs_err,max_rel_err,samples,cache_policy,flush_bytes");
    for(int opt=0;opt<3;opt++){
        if(method>=0&&opt!=method)continue;
        run(&c,opt);Error e=check(&c,0);double p50=0,p95=0;
        if(profile){
            void*flush=NULL;size_t bytes=0;if(cold)CHECK(nv_bench_cache_init(&flush,&bytes));measured_flush_bytes=bytes;
            for(int i=0;i<10;i++)run(&c,opt);
            if(cold)CHECK(nv_bench_cache_flush(flush,bytes,c.stream));
            CHECK(cudaStreamSynchronize(c.stream));CHECK(cudaProfilerStart());run(&c,opt);
            CHECK(cudaStreamSynchronize(c.stream));CHECK(cudaProfilerStop());if(flush)CHECK(cudaFree(flush));
        }else timing(&c,opt,cold,samples/5,&p50,&p95);
        e=check(&c,0);size_t sum=0;for(int b=0;b<batch;b++)sum+=c.lengths[b];
        size_t bytes=nv_bytes(dt)*(2*sum*c.a.kv_heads*c.a.head_dim+2*c.n);
        printf("%s,%d,16,4,128,%d,%s,%.6f,%.6f,%zu,%.6f,%.9g,%.9g,%d,%s,%zu\n",dtype_name(dt),batch,length,opt==2?"unsplit_vec4":opt?"split_online_vec4":"serial_online",p50,p95,bytes,profile?0:bytes/(p50*1.e-6)/1073741824.,e.abs,e.rel,profile?1:samples,cold?"cold":"hot",measured_flush_bytes);
    }
    destroy(&c);
}
