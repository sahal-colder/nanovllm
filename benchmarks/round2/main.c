#include "common.h"
#include <errno.h>
static int integer(const char*s,int lo,int hi){
    char*end;errno=0;long n=strtol(s,&end,10);REQUIRE(!errno&&end!=s&&!*end&&n>=lo&&n<=hi);return (int)n;
}
int main(int argc,char**argv){
    int n=0;cudaError_t e=cudaGetDeviceCount(&n);
    if(e!=cudaSuccess||!n){fprintf(stderr,"SKIP no CUDA GPU: %s\n",cudaGetErrorString(e));return 77;}
    if((argc==8||argc==9)&&!strcmp(argv[1],"--audit-attention")){
        int dt=!strcmp(argv[2],"f32")?0:!strcmp(argv[2],"f16")?1:!strcmp(argv[2],"bf16")?2:-1;
        REQUIRE(dt>=0&&(!strcmp(argv[5],"cold")||!strcmp(argv[5],"hot")));
        REQUIRE(argc==8||!strcmp(argv[8],"--profile"));
        attention_audit(dt,integer(argv[3],1,65536),integer(argv[4],1,64),!strcmp(argv[5],"cold"),integer(argv[6],5,10000),integer(argv[7],-1,2),argc==9);return 0;
    }
    if(argc==5&&!strcmp(argv[1],"--audit-decode")){
        int dt=!strcmp(argv[3],"f32")?0:!strcmp(argv[3],"f16")?1:!strcmp(argv[3],"bf16")?2:-1;REQUIRE(dt>=0);
        decode_audit(argv[2],dt,integer(argv[4],5,1000));return 0;
    }
    if(argc==2&&!strcmp(argv[1],"--test-attention"))return attention_tests();
    if(argc==3&&!strcmp(argv[1],"--test-decode"))return decode_tests(argv[2]);
    if(argc==3&&!strcmp(argv[1],"--test-all")){attention_tests();return decode_tests(argv[2]);}
    if(argc==2&&!strcmp(argv[1],"--attention")){attention_benchmark(0,0,-1);return 0;}
    if(argc==2&&!strcmp(argv[1],"--quick-attention")){attention_benchmark(0,1,-1);return 0;}
    if(argc==3&&!strcmp(argv[1],"--profile-attention")){
        int d=!strcmp(argv[2],"f32")?0:!strcmp(argv[2],"f16")?1:!strcmp(argv[2],"bf16")?2:-1;
        REQUIRE(d>=0);attention_benchmark(1,1,d);return 0;
    }
    if(argc==4&&!strcmp(argv[1],"--trace-decode")){
        int dt=!strcmp(argv[3],"f32")?0:!strcmp(argv[3],"f16")?1:!strcmp(argv[3],"bf16")?2:-1;REQUIRE(dt>=0);
        decode_benchmark(argv[2],1,dt);return 0;
    }
    if(argc==3&&(!strcmp(argv[1],"--decode")||!strcmp(argv[1],"--trace"))){decode_benchmark(argv[2],!strcmp(argv[1],"--trace"),!strcmp(argv[1],"--trace")?0:-1);return 0;}
    fprintf(stderr,"usage: %s --test-attention | --test-decode FIXTURES | --attention | --decode MODEL | --trace MODEL | --trace-decode MODEL DT | --profile-attention DT | --audit-attention DT SEQ B cold|hot SAMPLES METHOD(-1/0/1/2) [--profile] | --audit-decode MODEL DT SAMPLES\n",argv[0]);return 2;
}
