/* No model weights: compare native operators against a separately built,
 * pinned ds4 library. This is diagnostic/test linking only, never production. */
#include "myds_v41_moe.h"
#include "ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

bool myds_trace_f32(const char *s,uint32_t l,uint32_t p,const myds_metal_buffer_t *b,size_t n) {
    (void)s;(void)l;(void)p;(void)b;(void)n; return true;
}
static int equal(const char *name, const void *expected, const void *actual, size_t count, int integers) {
    if (!memcmp(expected,actual,count*4)) return 1;
    unsigned differences=0; float maximum=0;
    for(size_t i=0;i<count;i++) if(memcmp((const char*)expected+4*i,(const char*)actual+4*i,4)) {
        if(differences++<3) {
            if(integers) fprintf(stderr,"%s[%zu] ref=%d actual=%d\n",name,i,((const int*)expected)[i],((const int*)actual)[i]);
            else fprintf(stderr,"%s[%zu] ref=%.9g actual=%.9g\n",name,i,((const float*)expected)[i],((const float*)actual)[i]);
        }
        if(!integers) maximum=fmaxf(maximum,fabsf(((const float*)expected)[i]-((const float*)actual)[i]));
    }
    fprintf(stderr,"%s: %u differing words, max_abs=%.9g\n",name,differences,maximum);
    return 0;
}
int main(int argc,char **argv) {
    if(argc!=2) { fprintf(stderr,"Usage: test_v41_moe_oracle /absolute/myds/metal (run in pinned ds4 source directory)\n"); return 2; }
    if(!ds4_gpu_init() || !myds_metal_init(argv[1])) return 1;
    enum {N=384,K=6,W=2304};
    float *bias=mmap(NULL,16384,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(bias==MAP_FAILED) return 1;
    float logits[N],refprobs[N],actualprobs[N],refweights[K],actualweights[K];
    int refids[K],actualids[K];
    ds4_gpu_tensor *rl=ds4_gpu_tensor_alloc(N*4),*rp=ds4_gpu_tensor_alloc(N*4),
      *ri=ds4_gpu_tensor_alloc(K*4),*rw=ds4_gpu_tensor_alloc(K*4);
    myds_metal_buffer_t *nl=myds_metal_buffer_alloc(N*4),*np=myds_metal_buffer_alloc(N*4),
      *ns=myds_metal_buffer_alloc(N*4),*ni=myds_metal_buffer_alloc(K*4),
      *nw=myds_metal_buffer_alloc(K*4),*nb=myds_metal_buffer_alloc(N*4);
    int all=1;
    for(int sample=0;sample<8;sample++) {
        for(int i=0;i<N;i++) {
            logits[i] = sample==0 ? 0 : sample==1 ? (float)(i%13-6) :
                sample==2 ? -100.0f+(float)i*.25f : sample==3 ? (float)i*.25f :
                (float)(sin((double)(i*37+sample*11))*(sample-3)*5.0);
            bias[i] = sample<4 ? 0 : (float)(cos((double)(i*23+sample))*0.3);
        }
        float scale= sample<4 ? 1.5f : 2.5f;
        if(!ds4_gpu_tensor_write(rl,0,logits,sizeof(logits)) ||
           !ds4_gpu_router_select_tensor(ri,rw,rp,bias,16384,0,0,0,35,N,K,scale,0,0,true,false,rl) ||
           !ds4_gpu_tensor_read(rp,0,refprobs,sizeof(refprobs)) ||
           !ds4_gpu_tensor_read(ri,0,refids,sizeof(refids)) ||
           !ds4_gpu_tensor_read(rw,0,refweights,sizeof(refweights)) ||
           !myds_metal_buffer_write(nl,0,logits,sizeof(logits)) ||
           !myds_metal_buffer_write(nb,0,bias,N*4) ||
           !myds_v41_route(ni,nw,np,ns,nl,nb,N,K,scale) ||
           !myds_metal_buffer_read(np,0,actualprobs,sizeof(actualprobs)) ||
           !myds_metal_buffer_read(ni,0,actualids,sizeof(actualids)) ||
           !myds_metal_buffer_read(nw,0,actualweights,sizeof(actualweights))) {all=0;break;}
        int ok=equal("probabilities",refprobs,actualprobs,N,0) &
               equal("selected",refids,actualids,K,1) & equal("weights",refweights,actualweights,K,0);
        printf("router sample %d: %s\n",sample,ok?"EXACT":"FAIL"); fflush(stdout); all&=ok;
    }
    float gate[W],up[W],ref[W],actual[W];
    ds4_gpu_tensor *rg=ds4_gpu_tensor_alloc(W*4),*ru=ds4_gpu_tensor_alloc(W*4),*ro=ds4_gpu_tensor_alloc(W*4);
    myds_metal_buffer_t *ng=myds_metal_buffer_alloc(W*4),*nu=myds_metal_buffer_alloc(W*4),*no=myds_metal_buffer_alloc(W*4);
    for(int sample=0;sample<3;sample++) {
        for(int i=0;i<W;i++) {gate[i]=(float)(sin((double)i)*20);up[i]=(float)(cos((double)i)*20);}
        float clamp=sample==0?0:sample==1?10:1;
        if(!ds4_gpu_tensor_write(rg,0,gate,sizeof(gate)) || !ds4_gpu_tensor_write(ru,0,up,sizeof(up)) ||
           !ds4_gpu_swiglu_round_tensor(ro,rg,ru,W,clamp,1) ||
           !ds4_gpu_tensor_read(ro,0,ref,sizeof(ref)) ||
           !myds_metal_buffer_write(ng,0,gate,sizeof(gate)) || !myds_metal_buffer_write(nu,0,up,sizeof(up)) ||
           !myds_v41_shared_swiglu(no,ng,nu,W,clamp) || !myds_metal_buffer_read(no,0,actual,sizeof(actual))) {all=0;break;}
        int ok=equal("shared_swiglu",ref,actual,W,0);
        printf("shared SwiGLU clamp %.0f: %s\n",clamp,ok?"EXACT":"FAIL"); fflush(stdout); all&=ok;
    }
    ds4_gpu_tensor_free(rl);ds4_gpu_tensor_free(rp);ds4_gpu_tensor_free(ri);ds4_gpu_tensor_free(rw);
    ds4_gpu_tensor_free(rg);ds4_gpu_tensor_free(ru);ds4_gpu_tensor_free(ro);
    myds_metal_buffer_free(nl);myds_metal_buffer_free(np);myds_metal_buffer_free(ns);
    myds_metal_buffer_free(ni);myds_metal_buffer_free(nw);myds_metal_buffer_free(nb);
    myds_metal_buffer_free(ng);myds_metal_buffer_free(nu);myds_metal_buffer_free(no);
    ds4_gpu_shutdown();myds_metal_cleanup();munmap(bias,16384);
    return all?0:1;
}
