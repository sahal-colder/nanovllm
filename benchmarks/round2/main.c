#include "common.h"
int main(int argc,char**argv){
    int n=0;cudaError_t e=cudaGetDeviceCount(&n);
    if(e!=cudaSuccess||!n){fprintf(stderr,"SKIP no CUDA GPU: %s\n",cudaGetErrorString(e));return 77;}
    if(argc==2&&!strcmp(argv[1],"--test-attention"))return attention_tests();
    if(argc==3&&!strcmp(argv[1],"--test-decode"))return decode_tests(argv[2]);
    if(argc==3&&!strcmp(argv[1],"--test-all")){attention_tests();return decode_tests(argv[2]);}
    if(argc==2&&!strcmp(argv[1],"--attention")){attention_benchmark(0,0,-1);return 0;}
    if(argc==2&&!strcmp(argv[1],"--quick-attention")){attention_benchmark(0,1,-1);return 0;}
    if(argc==3&&!strcmp(argv[1],"--profile-attention")){
        int d=!strcmp(argv[2],"f32")?0:!strcmp(argv[2],"f16")?1:!strcmp(argv[2],"bf16")?2:-1;
        REQUIRE(d>=0);attention_benchmark(1,1,d);return 0;
    }
    if(argc==3&&(!strcmp(argv[1],"--decode")||!strcmp(argv[1],"--trace"))){decode_benchmark(argv[2],!strcmp(argv[1],"--trace"));return 0;}
    fprintf(stderr,"usage: %s --test-attention | --test-decode FIXTURES | --attention | --decode MODEL | --trace MODEL | --profile-attention\n",argv[0]);return 2;
}
