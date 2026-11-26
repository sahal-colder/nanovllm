#include "common.h"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
/* Independent FP64 arithmetic reference with explicit storage-rounding points.
 * nano-vLLM's unchanged CPU forward + Python dense oracle are also tested. */
static void norm(float*out,const float*x,const float*w,int n,mi_dtype t){
    double sum=0;for(int j=0;j<n;j++)sum+=(double)x[j]*x[j];double scale=1./sqrt(sum/n+1.e-5);
    for(int j=0;j<n;j++)out[j]=nv_round((float)(x[j]*scale*nv_round(w[j],t)),t);
}
static void mat(float*out,const float*x,const float*w,int in,int rows,mi_dtype t){
    for(int i=0;i<rows;i++){double sum=0;for(int j=0;j<in;j++)sum+=(double)nv_round(w[(size_t)i*in+j],t)*x[j];out[i]=nv_round((float)sum,t);}
}
static float *reference(NvModel*m,NvCache*c,const int*table,int token,int pos,mi_dtype t){
    NvConfig*p=&m->config;NvWeights*w=&m->weights;int d=p->dim,f=p->hidden_dim,hs=d/p->n_heads,kv=hs*p->n_kv_heads;
    size_t dd=(size_t)d*d,fd=(size_t)f*d,kd=(size_t)kv*d;
    for(int i=0;i<d;i++)m->x[i]=nv_round(w->embedding[(size_t)token*d+i],t);
    double*scores=nv_alloc(pos+1,sizeof(double));
    for(int l=0;l<p->n_layers;l++){
        norm(m->xb,m->x,w->att_norm+(size_t)l*d,d,t);
        float*k=nv_cache_key(c,table[pos/c->block_size],l,pos%c->block_size),*v=nv_cache_value(c,table[pos/c->block_size],l,pos%c->block_size);
        mat(m->q,m->xb,w->wq+l*dd,d,d,t);mat(k,m->xb,w->wk+l*kd,d,kv,t);mat(v,m->xb,w->wv+l*kd,d,kv,t);
        for(int i=0;i<d;i+=2){
            double angle=pos/pow(10000.,(double)(i%hs)/hs),cs=cos(angle),sn=sin(angle),x=m->q[i],y=m->q[i+1];
            m->q[i]=nv_round((float)(x*cs-y*sn),t);m->q[i+1]=nv_round((float)(x*sn+y*cs),t);
            if(i<kv){x=k[i];y=k[i+1];k[i]=nv_round((float)(x*cs-y*sn),t);k[i+1]=nv_round((float)(x*sn+y*cs),t);}
        }
        for(int h=0;h<p->n_heads;h++){
            int kh=h/(p->n_heads/p->n_kv_heads);double max=-INFINITY,sum=0;
            for(int posk=0;posk<=pos;posk++){
                const float*key=nv_cache_key(c,table[posk/c->block_size],l,posk%c->block_size)+kh*hs;
                double dot=0;for(int j=0;j<hs;j++)dot+=(double)m->q[h*hs+j]*key[j];scores[posk]=dot/sqrt((double)hs);max=fmax(max,scores[posk]);
            }
            for(int posk=0;posk<=pos;posk++){scores[posk]=exp(scores[posk]-max);sum+=scores[posk];}
            for(int j=0;j<hs;j++){
                double result=0;for(int posk=0;posk<=pos;posk++)result+=scores[posk]/sum*nv_cache_value(c,table[posk/c->block_size],l,posk%c->block_size)[kh*hs+j];
                m->xb[h*hs+j]=nv_round((float)result,t);
            }
        }
        mat(m->xb2,m->xb,w->wo+l*dd,d,d,t);for(int i=0;i<d;i++)m->x[i]=nv_round(m->x[i]+m->xb2[i],t);
        norm(m->xb,m->x,w->ffn_norm+(size_t)l*d,d,t);mat(m->hb,m->xb,w->w1+l*fd,d,f,t);mat(m->hb2,m->xb,w->w3+l*fd,d,f,t);
        for(int j=0;j<f;j++){double x=m->hb[j];m->hb[j]=nv_round((float)(x/(1.+exp(-x))*m->hb2[j]),t);}
        mat(m->xb,m->hb,w->w2+l*fd,f,d,t);for(int j=0;j<d;j++)m->x[j]=nv_round(m->x[j]+m->xb[j],t);
    }
    norm(m->x,m->x,w->final_norm,d,t);mat(m->logits,m->x,w->classifier,d,p->vocab_size,t);free(scores);return m->logits;
}
static void tolerances(mi_dtype t,double*abs,double*rel){*abs=t==MI_F32?1.e-4:t==MI_F16?3.e-3:2.e-2;*rel=t==MI_F32?3.e-4:t==MI_F16?8.e-3:5.e-2;}
static void check_kv(NvCuda*g,NvCache*c,Error*err){
    int kv=c->kv_dim,layers=c->layers,t=c->block_size;size_t count=(size_t)2*layers*t*kv,bytes=nv_bytes(g->dtype);
    void*h=nv_alloc(count,bytes);double atol,rtol;tolerances(g->dtype,&atol,&rtol);
    for(int p=0;p<c->num_blocks;p++){
        CHECK(cudaMemcpy(h,g->h_page_allocations[p],count*bytes,cudaMemcpyDeviceToHost));
        for(int l=0;l<layers;l++)for(int v=0;v<2;v++)for(int off=0;off<t;off++)for(int j=0;j<kv;j++){
            const float*ref=v?nv_cache_value(c,p,l,off):nv_cache_key(c,p,l,off);
            size_t idx=(((size_t)l*2+v)*t+off)*kv+j;
            compare(err,nv_unpack(h,idx,g->dtype),ref[j],atol,rtol,"all physical KV slots including unused tail");
        }
    }
    free(h);
}
static void shuffled(int32_t*table,int n){for(int i=0;i<n;i++)table[i]=i;for(int i=n-1;i>0;i--){int j=(int)((rnd()+1)*.4999f*(i+1));int tmp=table[i];table[i]=table[j];table[j]=tmp;}}
static void model_tests(const char*path,int page_tokens,int batch,mi_dtype dtype){
    NvModel m;nv_model_load(&m,path);int maxblocks=(m.config.seq_len-1)/page_tokens+1,pages=maxblocks*batch;
    NvCache cpu;nv_cache_init(&cpu,&m.config,(NvOptions){pages,page_tokens,batch,1,0});
    NvCuda g;nv_cuda_init(&g,&m,pages,page_tokens,batch,dtype,1);
    int32_t*table=nv_alloc(pages,4),*tokens=nv_alloc(batch,4),*lengths=nv_alloc(batch,4);shuffled(table,pages);
    float*expected=nv_alloc((size_t)batch*m.config.vocab_size,4);Error output={0},kv_error={0};double atol,rtol;tolerances(dtype,&atol,&rtol);
    cudaGraphExec_t original=NULL;
    int steps=m.config.n_layers>1?83:21;
    for(int pos=0;pos<steps;pos++){
        if(pos>0&&pos%page_tokens==0){ /* Rebind an unused current page; preserve all previous tokens. */
            for(int b=0;b<batch;b++){int logical=pos/page_tokens,next=logical+1;if(next<maxblocks){int tmp=table[b*maxblocks+logical];table[b*maxblocks+logical]=table[b*maxblocks+next];table[b*maxblocks+next]=tmp;}}
        }
        for(int b=0;b<batch;b++){
            tokens[b]=3+(int)((rnd()+1)*.4999f*(m.config.vocab_size-3));lengths[b]=pos+1;
            float*ref=reference(&m,&cpu,table+b*maxblocks,tokens[b],pos,dtype);
            memcpy(expected+(size_t)b*m.config.vocab_size,ref,(size_t)m.config.vocab_size*4);
        }
        CHECK(nv_cuda_upload(&g,tokens,lengths,table));
        if(pos==0){
            CHECK(nv_cuda_capture(&g));original=g.replay;size_t count=0;CHECK(cudaGraphGetNodes(g.graph,NULL,&count));
            REQUIRE(count==(size_t)(4+m.config.n_layers*(g.max_blocks*page_tokens>64?12:11)));
            cudaGraphNode_t*nodes=nv_alloc(count,sizeof(*nodes));CHECK(cudaGraphGetNodes(g.graph,nodes,&count));
            for(size_t n=0;n<count;n++){enum cudaGraphNodeType type;CHECK(cudaGraphNodeGetType(nodes[n],&type));REQUIRE(type==cudaGraphNodeTypeKernel||type==cudaGraphNodeTypeMemset);}
            free(nodes);
        }
        if(pos&1)CHECK(nv_cuda_step(&g));else CHECK(cudaGraphLaunch(g.replay,g.stream));
        REQUIRE(original==g.replay);CHECK(nv_cuda_download(&g));
        for(size_t i=0;i<(size_t)batch*m.config.vocab_size;i++)compare(&output,g.h_logits[i],expected[i],atol,rtol,"multi-step full decoder logits");
    }
    check_kv(&g,&cpu,&kv_error);
    fprintf(stderr,"PASS decode layers=%d batch=%d dtype=%s page=%d steps=%d logits=%zu abs=%.9g rel=%.9g KV=%zu KV_abs=%.9g KV_rel=%.9g graph_identity_unchanged=1 no_allocation_or_host_nodes=1\n",
        m.config.n_layers,batch,dtype_name(dtype),page_tokens,steps,output.n,output.abs,output.rel,kv_error.n,kv_error.abs,kv_error.rel);
    /* Upload rejects bad host metadata and never silently wraps context. */
    lengths[0]=m.config.seq_len+1;REQUIRE(nv_cuda_upload(&g,tokens,lengths,table)==cudaErrorInvalidValue);
    nv_cuda_free(&g);nv_cache_free(&cpu);nv_model_free(&m);free(table);free(tokens);free(lengths);free(expected);
}
static void drain(NvEngine*e){int limit=10000;while(nv_engine_step(e)&&--limit);REQUIRE(limit>0);}
static void engine_test(const char*path,int graph,int tight){
    NvModel m;nv_model_load(&m,path);NvEngine cpu,gpu;NvOptions opts={tight?4:16,4,3,9,1};
    nv_engine_init(&cpu,&m,opts);nv_engine_init(&gpu,&m,opts);REQUIRE(nv_engine_use_cuda(&gpu,0,graph));
    int prompt[]={1,259,260,261,262,260,261,262,260};NvSampling sample={0,1,123,1};
    for(int i=0;i<3;i++){REQUIRE(nv_engine_add(&cpu,prompt,9,8,sample)==i);REQUIRE(nv_engine_add(&gpu,prompt,9,8,sample)==i);}
    drain(&cpu);drain(&gpu);
    for(int i=0;i<3;i++){REQUIRE(cpu.requests[i]->num_tokens==gpu.requests[i]->num_tokens);REQUIRE(!memcmp(cpu.requests[i]->tokens,gpu.requests[i]->tokens,(size_t)cpu.requests[i]->num_tokens*4));}
    if(tight)REQUIRE(gpu.preemptions>0);
    fprintf(stderr,"PASS existing scheduler GPU/CPU token identity graph=%d tight=%d generated=%llu preemptions=%llu hits=%llu\n",graph,tight,(unsigned long long)gpu.generated_tokens,(unsigned long long)gpu.preemptions,(unsigned long long)gpu.cache.hits);
    nv_engine_free(&cpu);nv_engine_free(&gpu);nv_model_free(&m);
}
int decode_tests(const char*root){
    char path[4096];
    for(int dt=0;dt<3;dt++)for(int model=0;model<2;model++){
        snprintf(path,sizeof(path),"%s/decode%d.bin",root,model?3:1);
        model_tests(path,model?3:16,model?2:1,dt);
    }
    snprintf(path,sizeof(path),"%s/gqa.bin",root);for(int graph=0;graph<2;graph++)for(int tight=0;tight<2;tight++)engine_test(path,graph,tight);
    return 0;
}
static void fill_prefix(NvCuda*g,NvCache*c){
    size_t count=(size_t)2*c->layers*c->block_size*c->kv_dim;void*h=nv_alloc(count,nv_bytes(g->dtype));
    for(int p=0;p<c->num_blocks;p++){
        for(int l=0;l<c->layers;l++)for(int v=0;v<2;v++)for(int t=0;t<c->block_size;t++)for(int j=0;j<c->kv_dim;j++){
            float x=nv_round(rnd()*.1f,g->dtype);size_t idx=(((size_t)l*2+v)*c->block_size+t)*c->kv_dim+j;
            nv_pack(h,idx,g->dtype,x);(v?nv_cache_value(c,p,l,t):nv_cache_key(c,p,l,t))[j]=x;
        }
        CHECK(cudaMemcpy(g->h_page_allocations[p],h,count*nv_bytes(g->dtype),cudaMemcpyHostToDevice));
    }free(h);
}
void decode_benchmark(const char*path,int trace){
    puts("dtype,mode,hidden,layers,heads,kv_heads,batch,seq_len,graph_nodes,samples,p50_wall_us,p95_wall_us,p50_device_us,p95_device_us,max_abs_err,max_rel_err");
    for(int dt=0;dt<(trace?1:3);dt++){
        NvModel m;nv_model_load(&m,path);int t=16,blocks=(m.config.seq_len-1)/t+1;
        NvCuda g;nv_cuda_init(&g,&m,blocks,t,1,dt,1);NvCache cpu;nv_cache_init(&cpu,&m.config,(NvOptions){blocks,t,1,1,0});fill_prefix(&g,&cpu);
        int32_t*table=nv_alloc(blocks,4);shuffled(table,blocks);int32_t token=7,len=129;
        float*ref=reference(&m,&cpu,table,token,len-1,dt);float*expected=nv_alloc(m.config.vocab_size,4);memcpy(expected,ref,(size_t)m.config.vocab_size*4);
        CHECK(nv_cuda_upload(&g,&token,&len,table));CHECK(nv_cuda_capture(&g));CHECK(cudaGraphLaunch(g.replay,g.stream));CHECK(nv_cuda_download(&g));
        Error error={0};double atol,rtol;tolerances(dt,&atol,&rtol);for(int i=0;i<m.config.vocab_size;i++)compare(&error,g.h_logits[i],expected[i],atol,rtol,"benchmark full decoder");
        size_t nodes=0;CHECK(cudaGraphGetNodes(g.graph,NULL,&nodes));
        for(int i=0;i<20;i++){CHECK(nv_cuda_step(&g));CHECK(cudaGraphLaunch(g.replay,g.stream));}CHECK(cudaStreamSynchronize(g.stream));
        if(trace)CHECK(cudaProfilerStart());
        for(int graph=0;graph<2;graph++){
            int samples=trace?5:100;double wall[100],device_time[100];cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
            nvtxRangePushA(graph?"decode_graph":"decode_eager");
            for(int i=0;i<samples;i++){
                double begin=now_us();CHECK(cudaEventRecord(start,g.stream));
                if(graph)CHECK(cudaGraphLaunch(g.replay,g.stream));else CHECK(nv_cuda_step(&g));
                CHECK(cudaEventRecord(end,g.stream));CHECK(cudaEventSynchronize(end));wall[i]=now_us()-begin;
                float ms;CHECK(cudaEventElapsedTime(&ms,start,end));device_time[i]=ms*1000.;
            }
            nvtxRangePop();
            double p50=percentile(wall,samples,.5),p95=percentile(wall,samples,.95),gpu50=percentile(device_time,samples,.5),gpu95=percentile(device_time,samples,.95);
            printf("%s,%s,%d,%d,%d,%d,1,%d,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.9g,%.9g\n",dtype_name(dt),graph?"graph":"eager",m.config.dim,m.config.n_layers,m.config.n_heads,m.config.n_kv_heads,len,nodes,samples,p50,p95,gpu50,gpu95,error.abs,error.rel);fflush(stdout);
            CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));CHECK(nv_cuda_download(&g));for(int i=0;i<m.config.vocab_size;i++)compare(&error,g.h_logits[i],expected[i],atol,rtol,"post-benchmark decoder");
        }
        if(trace)CHECK(cudaProfilerStop());
        nv_cuda_free(&g);nv_cache_free(&cpu);nv_model_free(&m);free(table);free(expected);
    }
}
