/* Isolated exact-indexer bit-mask skip probe.  No model weights are loaded. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GUARD 64u
static NSMutableDictionary *manifest;
static void need(BOOL yes,const char *message) { if(!yes){fprintf(stderr,"FAILED: %s\n",message);exit(2);} }
static NSString *canon(NSString *s) { return [s.stringByStandardizingPath stringByResolvingSymlinksInPath]; }
static NSString *sha(const void *p,size_t n) {
    need(n<=UINT32_MAX,"hash length"); unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(p,(CC_LONG)n,d); NSMutableString *s=[NSMutableString stringWithCapacity:64];
    for(unsigned i=0;i<sizeof(d);i++) [s appendFormat:@"%02x",d[i]]; return s;
}
static NSString *filesha(NSString *s) { NSData *d=[NSData dataWithContentsOfFile:s];need(d!=nil,"file SHA");return sha(d.bytes,d.length); }
static int cmp_double(const void *a,const void *b) { double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y); }
static BOOL guards(const void *p,size_t n) { const uint8_t *b=p;for(unsigned i=0;i<GUARD;i++)if(b[i]!=0xa5||b[GUARD+n+i]!=0xa5)return NO;return YES; }
static void emit(NSDictionary *d) {
    NSData *x=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:nil];
    need(x!=nil,"JSON serialization"); fwrite(x.bytes,1,x.length,stdout);fputc('\n',stdout);fflush(stdout);
}
#include "indexer_mask_skip_topk.inc"

struct exact_args {
    uint32_t n_rows,n_head,head_dim,cache_f16;
    float scale;
    uint32_t n_tokens,causal_start,causal_ratio,window,mask_stride;
};
_Static_assert(sizeof(struct exact_args)==40,"candidate ABI");
static NSData *filter_production(const void *input,const uint32_t *words,uint32_t nc,uint32_t nt,uint32_t start,uint32_t ratio,uint32_t stride) {
    size_t bytes=(size_t)nc*nt*4u, mask_bytes=(size_t)stride*nt*4u;
    myds_metal_buffer_t *scores=myds_metal_buffer_alloc(bytes),*mask=myds_metal_buffer_alloc(mask_bytes);
    need(scores && mask,"filter buffers");
    need(myds_metal_buffer_write(scores,0,input,bytes)&&myds_metal_buffer_write(mask,0,words,mask_bytes),"filter inputs");
    need(myds_metal_begin_batch()&&myds_v41_gpu_filter_rows_bits(scores,mask,nc,nt,start,ratio,stride)&&
         myds_metal_commit_batch(),"production mask filter GPU completion");
    NSData *result=[NSData dataWithBytes:myds_metal_buffer_data(scores) length:bytes];
    myds_metal_buffer_free(scores);myds_metal_buffer_free(mask);return result;
}

static NSData *runtime_mask_skip(NSData * __strong *data,const uint32_t *words,
        uint32_t nc,uint32_t nt,uint32_t start,uint32_t ratio,uint32_t stride,BOOL produce,BOOL fallback_test) {
    const size_t qb=(size_t)nt*32u*128u*4u,wb=(size_t)nt*32u*4u,kb=(size_t)nc*128u*2u,
        sb=(size_t)nc*nt*4u,mb=(size_t)stride*nt*4u;
    myds_metal_buffer_t *q=myds_metal_buffer_alloc(qb),*w=myds_metal_buffer_alloc(wb),
        *k=myds_metal_buffer_alloc(kb),*scores=myds_metal_buffer_alloc(sb),*mask=myds_metal_buffer_alloc(mb);
    need(q&&w&&k&&scores&&mask,"runtime candidate buffers");
    need(myds_metal_buffer_write(q,0,(const uint8_t *)data[0].bytes+GUARD,qb)&&
         myds_metal_buffer_write(w,0,(const uint8_t *)data[1].bytes+GUARD,wb)&&
         myds_metal_buffer_write(k,0,(const uint8_t *)data[2].bytes+GUARD,kb)&&
         myds_metal_buffer_write(mask,0,words,mb),"runtime candidate inputs");
    setenv("MYDS_INDEXER_TILED","0",1);setenv("MYDS_INDEXER_CHUNK_EXACT","1",1);
    setenv("MYDS_INDEXER_MASK_SKIP","1",1);myds_metal_set_auto_barrier(1);
    myds_metal_buffer_t *ids=NULL;
    if(produce){
        ids=myds_metal_buffer_alloc((size_t)nt*4u);need(ids!=NULL,"producer ids");
        int32_t *ip=myds_metal_buffer_data(ids);
        for(uint32_t t=0;t<nt;t++)ip[t]=(int32_t)(t%((nc+7u)/8u));
    }
    myds_v41_gpu_mask_skip_counters(NULL,true);
    need(myds_metal_begin_batch(),"candidate batch");
    if(produce)need(myds_v41_gpu_maskbits_rows(mask,ids,(nc+7u)/8u,1u,nt,stride),"same batch production mask producer");
    need(myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,ratio,stride)==1,
         "actual scoped candidate encoded");
    need(myds_v41_gpu_filter_rows_bits(scores,mask,nc,nt,start,ratio,stride)&&myds_metal_commit_batch(),
         "candidate plus production filter batch completion");
    myds_v41_gpu_mask_skip_counters_t count;myds_v41_gpu_mask_skip_counters(&count,false);
    need(count.requests==1&&count.encoded==1&&!count.unavailable&&!count.failures,"provider witness counters");
    NSData *result=[NSData dataWithBytes:myds_metal_buffer_data(scores) length:sb];
    memset(myds_metal_buffer_data(scores),0xa5,sb);
    need(myds_metal_begin_batch(),"scope negative batch");
    setenv("MYDS_INDEXER_MASK_SKIP","0",1);
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,ratio,stride),"default off scope");
    setenv("MYDS_INDEXER_MASK_SKIP","1",1);setenv("MYDS_INDEXER_TILED","1",1);
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,ratio,stride),"tiled scope");
    setenv("MYDS_INDEXER_TILED","0",1);setenv("MYDS_INDEXER_CHUNK_EXACT","0",1);
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,ratio,stride),"direct scope");
    setenv("MYDS_INDEXER_CHUNK_EXACT","1",1);
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,2u,stride),"ratio scope");
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,16384u,nt,start,ratio,stride),"short width scope");
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,ratio,0u),"stride rejection");
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,scores,nc,nt,start,ratio,stride),"writer alias rejection");
    myds_metal_set_auto_barrier(0);
    need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,ratio,stride),"barrier-off scope");
    myds_metal_set_auto_barrier(1);need(myds_metal_commit_batch(),"scope negative completion");
    const uint8_t *z=myds_metal_buffer_data(scores);for(size_t i=0;i<sb;i++)need(z[i]==0xa5,"negative scope did not write");
    if(fallback_test){
        /* Capture direct output independently; the injected missing exact PSO
         * makes the real selector take that same fallback. */
        setenv("MYDS_INDEXER_CHUNK_EXACT","0",1);
        need(myds_metal_begin_batch()&&myds_v41_gpu_indexscores_chunk_bf16(scores,q,w,k,nc,32u,nt,start,ratio)&&
             myds_v41_gpu_filter_rows_bits(scores,mask,nc,nt,start,ratio,stride)&&myds_metal_commit_batch(),"independent direct reference");
        NSData *direct=[NSData dataWithBytes:myds_metal_buffer_data(scores) length:sb];
        setenv("MYDS_INDEXER_CHUNK_EXACT","1",1);
        for(uint32_t which=1u;which<=2u;which++){
            myds_v41_gpu_inject_mask_skip_unavailable(which);myds_v41_gpu_mask_skip_counters(NULL,true);
            need(myds_metal_begin_batch(),"missing PSO fallback batch");
            need(!myds_v41_gpu_indexscores_mask_skip_prefill(scores,q,w,k,mask,nc,nt,start,ratio,stride),"missing PSO rejected before encoding");
            need(myds_v41_gpu_indexscores_chunk_bf16(scores,q,w,k,nc,32u,nt,start,ratio)&&
                 myds_v41_gpu_filter_rows_bits(scores,mask,nc,nt,start,ratio,stride)&&myds_metal_commit_batch(),
                 "original selector and filter complete after missing PSO");
            myds_v41_gpu_mask_skip_counters(&count,false);
            need(count.requests==1&&count.unavailable==1&&!count.encoded&&!count.failures,"missing PSO counter");
            NSData *expected=which==1u?direct:result;
            need(!memcmp(myds_metal_buffer_data(scores),expected.bytes,sb),"fallback score exact for selected provider");
        }
        myds_v41_gpu_inject_mask_skip_unavailable(0u);
    }
    myds_metal_buffer_free(q);myds_metal_buffer_free(w);myds_metal_buffer_free(k);
    myds_metal_buffer_free(scores);myds_metal_buffer_free(mask);if(ids)myds_metal_buffer_free(ids);return result;
}

static uint32_t word(uint64_t i,uint32_t seed) {
    uint32_t x=(uint32_t)(i+1u)*747796405u+seed*2891336453u;
    x=((x>>((x>>28)+4))^x)*277803737u;return(x>>22)^x;
}
static float value(uint64_t i,uint32_t seed,unsigned profile) {
    float f=((int32_t)(word(i,seed)%8191u)-4095)/4096.f;
    if(profile==0){uint32_t u;memcpy(&u,&f,4);u&=0xffff0000u;memcpy(&f,&u,4);}
    if(profile==1 && i%193u==0){uint32_t u=(i/193u)%2u?0x80000000u:0;memcpy(&f,&u,4);}
    return f;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,id<MTLLibrary> lib,NSString *name) {
    id<MTLFunction> fn=[lib newFunctionWithName:name];need(fn!=nil,"pipeline symbol");NSError *error=nil;
    id<MTLComputePipelineState> state=[device newComputePipelineStateWithFunction:fn error:&error];
    if(!state) fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(state && state.threadExecutionWidth==32 && state.maxTotalThreadsPerThreadgroup>=128,"pipeline geometry");
    return state;
}
static void encode(id<MTLCommandBuffer> cb,id<MTLComputePipelineState> state,
                   const struct exact_args *a,id<MTLBuffer> q,id<MTLBuffer> w,
                   id<MTLBuffer> k,id<MTLBuffer> dst,id<MTLBuffer> mask,uint32_t mask_stride,BOOL reference) {
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];
    pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"encoder");
    [e setComputePipelineState:state];[e setBytes:a length:reference?36u:sizeof(*a) atIndex:0];
    [e setBuffer:q offset:GUARD atIndex:1];[e setBuffer:w offset:GUARD atIndex:2];
    [e setBuffer:k offset:GUARD atIndex:3];[e setBuffer:dst offset:GUARD atIndex:4];
    if(!reference){[e setBuffer:mask offset:GUARD atIndex:5];}
    (void)mask_stride;
    [e setThreadgroupMemoryLength:5120 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((a->n_rows+15)/16,(a->n_tokens+7)/8,1)
      threadsPerThreadgroup:MTLSizeMake(32,4,1)];[e endEncoding];
}
static double dispatch(id<MTLCommandQueue> queue,id<MTLComputePipelineState> state,
                       const struct exact_args *a,id<MTLBuffer> q,id<MTLBuffer> w,
                       id<MTLBuffer> k,id<MTLBuffer> __strong *dst,id<MTLBuffer> mask,uint32_t mask_stride,unsigned repeats,BOOL reference) {
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"command buffer");
    for(unsigned i=0;i<repeats;i++) encode(cb,state,a,q,w,k,dst[i],mask,mask_stride,reference);
    [cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU error: %s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU command buffer completed");
    double elapsed=cb.GPUEndTime-cb.GPUStartTime;need(elapsed>0,"GPU timestamps available");return elapsed;
}

int main(int argc,char **argv) {@autoreleasepool {
    need(argc==3 || (argc==4 && !strcmp(argv[3],"--correctness-only")),
         "probe candidate.metallib deployed.metallib [--correctness-only]");
    const BOOL correctness_only=argc==4;
    NSString *candidate=canon([NSString stringWithUTF8String:argv[1]]);
    NSString *deployed=canon([NSString stringWithUTF8String:argv[2]]);
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();need(d!=nil,"device");id<MTLCommandQueue> queue=[d newCommandQueue];need(queue!=nil,"queue");
    NSError *err=nil;id<MTLLibrary> libs[2];
    libs[0]=[d newLibraryWithURL:[NSURL fileURLWithPath:deployed] error:&err];need(libs[0]!=nil,"deployed library");
    libs[1]=[d newLibraryWithURL:[NSURL fileURLWithPath:candidate] error:&err];need(libs[1]!=nil,"candidate library");
    id<MTLComputePipelineState> ps[3]={pipeline(d,libs[0],@"kernel_myds_indexer_score_chunk_exact"),
        pipeline(d,libs[1],@"kernel_myds_indexer_score_chunk_exact"),pipeline(d,libs[1],@"probe_indexer_mask_skip")};
    emit(@{@"kind":@"identity",@"device":d.name,@"deployed_path":deployed,@"deployed_sha256":filesha(deployed),
           @"candidate_library_path":candidate,@"candidate_library_sha256":filesha(candidate),
           @"symbols":@[@"kernel_myds_indexer_score_chunk_exact (deployed)",@"kernel_myds_indexer_score_chunk_exact (same TU)",@"probe_indexer_mask_skip"]});
    manifest=[NSMutableDictionary new];init_topk(candidate);
    const uint32_t cases[][5]={
        {16385,1,32762,0,2},{16385,7,32762,1,2},{16385,8,32762,2,2},
        {32768,9,65519,3,2},{48169,15,48154,4,1},{48169,15,48154,6,1},
        {99333,21,99312,5,1},{99333,21,99312,7,1},{65536,32,131040,8,2},
        {49358,41,49317,9,1},{99333,21,99312,10,1},{16385,9,16376,11,1},{16385,1,16384,12,1}};
    const unsigned repeats=3,samples=7;
    BOOL ok=YES;
    const char *case_filter=getenv("MYDS_MASK_PROBE_CASE");
    for(unsigned ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){@autoreleasepool {
        if(case_filter && (unsigned)strtoul(case_filter,NULL,10)!=ci)continue;
        const uint32_t nc=cases[ci][0],nt=cases[ci][1],start=cases[ci][2],profile=cases[ci][3],ratio=cases[ci][4];
        struct exact_args a={nc,32,128,2,1.0f/64.0f,nt,start,ratio,0,0};
        const size_t sizes[]={ (size_t)nt*32*128*4,(size_t)nt*32*4,(size_t)nc*128*2,(size_t)nt*nc*4};
        NSMutableData *data[3];id<MTLBuffer> inputs[3];
        for(unsigned i=0;i<3;i++){data[i]=[NSMutableData dataWithLength:sizes[i]+2*GUARD];memset(data[i].mutableBytes,0xa5,data[i].length);}
        float *q=(float *)((uint8_t *)data[0].mutableBytes+GUARD),*w=(float *)((uint8_t *)data[1].mutableBytes+GUARD);
        uint16_t *k=(uint16_t *)((uint8_t *)data[2].mutableBytes+GUARD);
        for(size_t i=0;i<sizes[0]/4;i++)q[i]=value(i,17+ci,ci%2);
        for(size_t i=0;i<sizes[1]/4;i++)w[i]=value(i,29+ci,ci%2);
        for(size_t i=0;i<sizes[2]/2;i++){float f=value(i,37+ci,0);uint32_t u;memcpy(&u,&f,4);k[i]=(uint16_t)(u>>16);}
        for(unsigned i=0;i<3;i++){inputs[i]=[d newBufferWithBytes:data[i].bytes length:data[i].length options:MTLResourceStorageModeShared];need(inputs[i]!=nil,"input buffer");}
        const char *sample_dir=getenv("MYDS_MASK_PROBE_SAMPLES");
        if((profile==9u||profile==10u) && !sample_dir) continue;
        const uint32_t stride=profile==9u?198u:profile==10u?392u:(((nc+7u)/8u)+31u)/32u+1u;
        a.mask_stride=stride;
        const size_t mask_size=(size_t)stride*nt*4u;
        NSMutableData *mask_data=[NSMutableData dataWithLength:mask_size+2u*GUARD];
        memset(mask_data.mutableBytes,0xa5,mask_data.length);
        uint32_t *words=(uint32_t *)((uint8_t *)mask_data.mutableBytes+GUARD);
        for(uint32_t t=0;t<nt;t++)for(uint32_t j=0;j<stride;j++){
            uint32_t v=0u;
            if(profile==1u||profile==12u) v=~0u;
            else if(profile==2u) v=j%2u?0xaaaaaaaau:0x55555555u;
            else if(profile==3u) v=1u<<((t+j)%32u);
            else if(profile>=4u && profile<=10u) {
                /* Per-bit deterministic Bernoulli approximation to the
                 * observed warm mask rejection rates. This is synthetic, not
                 * a replay of the real mask distribution. */
                const unsigned rejected=(profile==4u||profile==6u)?67u:(profile==5u||profile==7u)?84u:50u;
                const uint32_t mask_row=(profile==6u||profile==7u)?0u:t;
                for(uint32_t b=0;b<32u;b++)
                    if(word(((uint64_t)mask_row*stride+j)*32u+b,91u)%100u>=rejected) v|=1u<<b;
            }
            if(profile==11u){ uint32_t block=t%((nc+7u)/8u); v=j==(block>>5)?1u<<(block&31u):0u; }
            words[(size_t)t*stride+j]=v;
        }
        if(profile==9u||profile==10u){
            NSString *name=profile==9u?@"c48-l24-case1345":@"c96-l24-case4417";
            NSString *dir=[NSString stringWithUTF8String:sample_dir];
            NSString *jp=[dir stringByAppendingPathComponent:[name stringByAppendingString:@".json"]];
            NSString *bp=[dir stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]];
            NSData *md=[NSData dataWithContentsOfFile:jp],*raw=[NSData dataWithContentsOfFile:bp];
            need(md&&raw,"captured mask and metadata");
            NSDictionary *meta=[NSJSONSerialization JSONObjectWithData:md options:0 error:nil];
            NSDictionary *rd=meta[@"raw_mask"];
            NSString *expected=profile==9u?@"eabd2f126ebebcd737ed9310d53aef223d329d6618b521d4fbbba2eedb9798ad":@"e55c015bcf3cea6f6bdbe2d38beb2e93d60a42a2984c66ad0287dfb6edb29284";
            need([sha(raw.bytes,raw.length) isEqualToString:expected]&&[rd[@"sha256"] isEqualToString:expected]&&raw.length==mask_size&&
                 [meta[@"nc"] unsignedIntValue]==nc&&[meta[@"glen"] unsignedIntValue]==nt&&
                 [meta[@"g_start"] unsignedIntValue]==start&&[meta[@"ratio"] unsignedIntValue]==ratio&&
                 [meta[@"mask_stride_words"] unsignedIntValue]==stride&&[meta[@"layer"] unsignedIntValue]==24u&&
                 [meta[@"phase"] unsignedIntValue]==1u&&[meta[@"mask_bits"] boolValue],"captured mask identity and tuple");
            memcpy(words,raw.bytes,mask_size);
            emit(@{@"kind":@"historical_mask",@"case":@(ci),@"name":name,@"mask_sha256":expected,
                   @"metadata_sha256":sha(md.bytes,md.length),@"note":@"historical mask, synthetic q/w/k"});
        }
        id<MTLBuffer> mask=[d newBufferWithBytes:mask_data.bytes length:mask_data.length options:MTLResourceStorageModeShared];need(mask!=nil,"mask buffer");
        id<MTLBuffer> outputs[3];NSMutableData *poison=[NSMutableData dataWithLength:sizes[3]+2*GUARD];memset(poison.mutableBytes,0xa5,poison.length);
        for(unsigned arm=0;arm<3;arm++){
            outputs[arm]=[d newBufferWithBytes:poison.bytes length:poison.length options:MTLResourceStorageModeShared];need(outputs[arm]!=nil,"output buffer");
            id<MTLBuffer> dest[]={outputs[arm]};dispatch(queue,ps[arm],&a,inputs[0],inputs[1],inputs[2],dest,mask,stride,1,arm!=2);
        }
        BOOL guard=guards(mask.contents,mask_size),inputs_intact=!memcmp(mask.contents,mask_data.bytes,mask_data.length);
        size_t native_diff=0,candidate_diff=0; NSData *filtered[3];
        for(unsigned i=0;i<3;i++){
            inputs_intact &= !memcmp(inputs[i].contents,data[i].bytes,data[i].length);
            guard &= guards(outputs[i].contents,sizes[3]);
            const void *raw=(const uint8_t *)outputs[i].contents+GUARD;
            filtered[i]=filter_production(raw,words,nc,nt,start,ratio,stride);
        }
        const uint32_t *f0=filtered[0].bytes,*f1=filtered[1].bytes,*f2=filtered[2].bytes;
        BOOL causal=YES;
        for(uint32_t t=0;t<nt;t++)for(uint32_t key=0;key<nc;key++){
            size_t i=(size_t)t*nc+key;native_diff+=f0[i]!=f1[i];candidate_diff+=f0[i]!=f2[i];
            uint32_t block=key>>3; BOOL allowed=(words[(size_t)t*stride+(block>>5)]&(1u<<(block&31u)))!=0;
            if(!allowed || key>=(start+t+1u)/ratio)causal &= f0[i]==0xff800000u && f2[i]==0xff800000u;
        }
        NSData *runtime=ratio==1u?runtime_mask_skip(data,words,nc,nt,start,ratio,stride,profile==11u,profile==12u):nil;
        BOOL runtime_ok=!runtime || [runtime isEqualToData:filtered[0]];
        NSDictionary *top=topk_against_exact(filtered[0],runtime?runtime:filtered[2],nc,nt,start,ratio);
        BOOL correct=runtime_ok&&guard&&causal&&inputs_intact&&!native_diff&&!candidate_diff &&
            [top[@"guards_and_causal_ids_valid"]boolValue] && ![top[@"ordered_id_mismatches"]unsignedLongLongValue] &&
            ![top[@"exact_aa_ordered_id_mismatches"]unsignedLongLongValue];
        emit(@{@"kind":@"correctness",@"case":@(ci),@"keys":@(nc),@"tokens":@(nt),@"start":@(start),
               @"mask_stride":@(stride),@"ratio":@(ratio),@"mask_profile":@(profile),@"mask_seed":@91,@"deployed_vs_same_tu_bit_mismatches":@(native_diff),
               @"deployed_vs_mask_skip_filtered_bit_mismatches":@(candidate_diff),
               @"guards":@(guard),@"causal_and_mask":@(causal),@"inputs_intact":@(inputs_intact),@"simulated_pso_fallback":@(profile==12u),@"same_batch_mask_producer":@(profile==11u),@"actual_runtime_mask_skip":@(runtime!=nil),@"runtime_filtered_equal":@(runtime_ok),@"top512":top,@"pass":@(correct)});
        if(!correct){ok=NO;break;}
        if(correctness_only)continue;
        /* Multiple disjoint outputs allow a concurrent pass without write races.
         * Every timed CB submits identical geometry/work; no wait or CPU time is included. */
        id<MTLBuffer> perf[3];for(unsigned i=0;i<repeats;i++){perf[i]=[d newBufferWithLength:sizes[3]+2*GUARD options:MTLResourceStorageModeShared];need(perf[i]!=nil,"timing output");}
        for(unsigned j=0;j<3;j++)dispatch(queue,ps[j==1?2:1],&a,inputs[0],inputs[1],inputs[2],perf,mask,stride,repeats,j!=1);
        /* A/A calibration precedes B timing; a 2% median drift gate is fixed
         * before launch and rejects the case without searching for a retry. */
        double ca[5],cb[5];
        for(unsigned i=0;i<5;i++){
            ca[i]=dispatch(queue,ps[1],&a,inputs[0],inputs[1],inputs[2],perf,mask,stride,repeats,YES);
            cb[i]=dispatch(queue,ps[1],&a,inputs[0],inputs[1],inputs[2],perf,mask,stride,repeats,YES);
            emit(@{@"kind":@"control_timing",@"case":@(ci),@"sample":@(i),@"a1_seconds":@(ca[i]),@"a2_seconds":@(cb[i])});
        }
        qsort(ca,5,sizeof(double),cmp_double);qsort(cb,5,sizeof(double),cmp_double);
        double drift=cb[2]/ca[2]-1.0;BOOL control_pass=fabs(drift)<=0.02;
        emit(@{@"kind":@"control_gate",@"case":@(ci),@"median_a1_seconds":@(ca[2]),@"median_a2_seconds":@(cb[2]),@"drift":@(drift),@"limit":@0.02,@"pass":@(control_pass)});
        if(!control_pass) continue;
        for(unsigned i=0;i<samples;i++){
            unsigned order[3]={1,2,1}; /* same-TU A1 / candidate B / same-TU A2 */
            for(unsigned j=0;j<3;j++){
                unsigned arm=order[j];double elapsed=dispatch(queue,ps[arm],&a,inputs[0],inputs[1],inputs[2],perf,mask,stride,repeats,arm==1);
                emit(@{@"kind":@"timing",@"case":@(ci),@"sample":@(i),@"order":@(j),@"arm":j==0?@"A1":j==1?@"B":@"A2",
                       @"gpu_seconds":@(elapsed),@"dispatches":@(repeats),@"scope":@"concurrent_encoder_cb_gpu_timestamps"});
            }
        }
    }}
    myds_v41_gpu_scratch_free(topk_scratch);myds_metal_cleanup();emit(@{@"kind":@"complete",@"correctness_passed":@(ok),@"correctness_only":@(correctness_only),@"production_adoption":@NO});
    return ok?0:2;
}}
