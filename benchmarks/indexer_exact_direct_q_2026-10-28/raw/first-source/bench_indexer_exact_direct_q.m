/* Isolated exact-indexer direct-Q probe.  No model weights are loaded. */
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
static BOOL guards(const void *p,size_t n) { const uint8_t *b=p;for(unsigned i=0;i<GUARD;i++)if(b[i]!=0xa5||b[GUARD+n+i]!=0xa5)return NO;return YES; }
static void emit(NSDictionary *d) {
    NSData *x=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:nil];
    need(x!=nil,"JSON serialization"); fwrite(x.bytes,1,x.length,stdout);fputc('\n',stdout);fflush(stdout);
}
#include "indexer_long_topk_probe.inc"

struct exact_args {
    uint32_t n_rows,n_head,head_dim,cache_f16;
    float scale;
    uint32_t n_tokens,causal_start,causal_ratio,window;
};
_Static_assert(sizeof(struct exact_args)==36,"native exact ABI");
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
                   id<MTLBuffer> k,id<MTLBuffer> dst,BOOL reference) {
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];
    pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"encoder");
    [e setComputePipelineState:state];[e setBytes:a length:sizeof(*a) atIndex:0];
    [e setBuffer:q offset:GUARD atIndex:1];[e setBuffer:w offset:GUARD atIndex:2];
    [e setBuffer:k offset:GUARD atIndex:3];[e setBuffer:dst offset:GUARD atIndex:4];
    if(reference)[e setThreadgroupMemoryLength:5120 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((a->n_rows+15)/16,(a->n_tokens+7)/8,1)
      threadsPerThreadgroup:MTLSizeMake(32,4,1)];[e endEncoding];
}
static double dispatch(id<MTLCommandQueue> queue,id<MTLComputePipelineState> state,
                       const struct exact_args *a,id<MTLBuffer> q,id<MTLBuffer> w,
                       id<MTLBuffer> k,id<MTLBuffer> __strong *dst,unsigned repeats,BOOL reference) {
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"command buffer");
    for(unsigned i=0;i<repeats;i++) encode(cb,state,a,q,w,k,dst[i],reference);
    [cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU error: %s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU command buffer completed");
    double elapsed=cb.GPUEndTime-cb.GPUStartTime;need(elapsed>0,"GPU timestamps available");return elapsed;
}

int main(int argc,char **argv) {@autoreleasepool {
    need(argc==3,"probe candidate.metallib deployed.metallib");
    NSString *candidate=canon([NSString stringWithUTF8String:argv[1]]);
    NSString *deployed=canon([NSString stringWithUTF8String:argv[2]]);
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();need(d!=nil,"device");id<MTLCommandQueue> queue=[d newCommandQueue];need(queue!=nil,"queue");
    NSError *err=nil;id<MTLLibrary> libs[2];
    libs[0]=[d newLibraryWithURL:[NSURL fileURLWithPath:deployed] error:&err];need(libs[0]!=nil,"deployed library");
    libs[1]=[d newLibraryWithURL:[NSURL fileURLWithPath:candidate] error:&err];need(libs[1]!=nil,"candidate library");
    id<MTLComputePipelineState> ps[3]={pipeline(d,libs[0],@"kernel_myds_indexer_score_chunk_exact"),
        pipeline(d,libs[1],@"kernel_myds_indexer_score_chunk_exact"),pipeline(d,libs[1],@"probe_indexer_exact_direct_q")};
    emit(@{@"kind":@"identity",@"device":d.name,@"deployed_path":deployed,@"deployed_sha256":filesha(deployed),
           @"candidate_library_path":candidate,@"candidate_library_sha256":filesha(candidate),
           @"symbols":@[@"kernel_myds_indexer_score_chunk_exact (deployed)",@"kernel_myds_indexer_score_chunk_exact (same TU)",@"probe_indexer_exact_direct_q"]});
    manifest=[NSMutableDictionary new];init_topk(deployed);
    const uint32_t cases[][3]={{16385,8,32762},{32768,31,65505},{65536,32,131040}};
    const unsigned repeats=3,samples=7;
    BOOL ok=YES;
    for(unsigned ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){@autoreleasepool {
        const uint32_t nc=cases[ci][0],nt=cases[ci][1],start=cases[ci][2];
        struct exact_args a={nc,32,128,2,1.0f/64.0f,nt,start,2,0};
        const size_t sizes[]={ (size_t)nt*32*128*4,(size_t)nt*32*4,(size_t)nc*128*2,(size_t)nt*nc*4};
        NSMutableData *data[3];id<MTLBuffer> inputs[3];
        for(unsigned i=0;i<3;i++){data[i]=[NSMutableData dataWithLength:sizes[i]+2*GUARD];memset(data[i].mutableBytes,0xa5,data[i].length);}
        float *q=(float *)((uint8_t *)data[0].mutableBytes+GUARD),*w=(float *)((uint8_t *)data[1].mutableBytes+GUARD);
        uint16_t *k=(uint16_t *)((uint8_t *)data[2].mutableBytes+GUARD);
        for(size_t i=0;i<sizes[0]/4;i++)q[i]=value(i,17+ci,ci%2);
        for(size_t i=0;i<sizes[1]/4;i++)w[i]=value(i,29+ci,ci%2);
        for(size_t i=0;i<sizes[2]/2;i++){float f=value(i,37+ci,0);uint32_t u;memcpy(&u,&f,4);k[i]=(uint16_t)(u>>16);}
        for(unsigned i=0;i<3;i++){inputs[i]=[d newBufferWithBytes:data[i].bytes length:data[i].length options:MTLResourceStorageModeShared];need(inputs[i]!=nil,"input buffer");}
        id<MTLBuffer> outputs[3];NSMutableData *poison=[NSMutableData dataWithLength:sizes[3]+2*GUARD];memset(poison.mutableBytes,0xa5,poison.length);
        for(unsigned arm=0;arm<3;arm++){
            outputs[arm]=[d newBufferWithBytes:poison.bytes length:poison.length options:MTLResourceStorageModeShared];need(outputs[arm]!=nil,"output buffer");
            id<MTLBuffer> dest[]={outputs[arm]};dispatch(queue,ps[arm],&a,inputs[0],inputs[1],inputs[2],dest,1,arm!=2);
        }
        BOOL guard=YES,causal=YES,inputs_intact=YES;size_t native_diff=0,candidate_diff=0;
        const uint32_t *score[3];
        for(unsigned i=0;i<3;i++){
            inputs_intact &= !memcmp(inputs[i].contents,data[i].bytes,data[i].length);
            guard &= guards(outputs[i].contents,sizes[3]);score[i]=(uint32_t *)((uint8_t *)outputs[i].contents+GUARD);
        }
        for(uint32_t t=0;t<nt;t++)for(uint32_t key=0;key<nc;key++){
            size_t i=(size_t)t*nc+key;native_diff+=score[0][i]!=score[1][i];candidate_diff+=score[0][i]!=score[2][i];
            const uint32_t visible=(start+t+1)/2;
            if(key>=visible)causal &= score[0][i]==0xff800000u && score[2][i]==0xff800000u;
            else causal &= (score[0][i]&0x7f800000u)!=0x7f800000u && (score[2][i]&0x7f800000u)!=0x7f800000u;
        }
        NSData *expected=[NSData dataWithBytes:score[0] length:sizes[3]],*actual=[NSData dataWithBytes:score[2] length:sizes[3]];
        NSDictionary *top=topk_against_exact(expected,actual,nc,nt,start);
        BOOL correct=guard&&causal&&inputs_intact&&!native_diff&&!candidate_diff &&
            [top[@"guards_and_causal_ids_valid"]boolValue] && ![top[@"ordered_id_mismatches"]unsignedLongLongValue] &&
            ![top[@"exact_aa_ordered_id_mismatches"]unsignedLongLongValue];
        emit(@{@"kind":@"correctness",@"case":@(ci),@"keys":@(nc),@"tokens":@(nt),@"start":@(start),
               @"deployed_vs_same_tu_bit_mismatches":@(native_diff),@"deployed_vs_direct_q_bit_mismatches":@(candidate_diff),
               @"guards":@(guard),@"causal":@(causal),@"inputs_intact":@(inputs_intact),@"top512":top,@"pass":@(correct)});
        if(!correct){ok=NO;break;}
        /* Multiple disjoint outputs allow a concurrent pass without write races.
         * Every timed CB submits identical geometry/work; no wait or CPU time is included. */
        id<MTLBuffer> perf[3];for(unsigned i=0;i<repeats;i++){perf[i]=[d newBufferWithLength:sizes[3]+2*GUARD options:MTLResourceStorageModeShared];need(perf[i]!=nil,"timing output");}
        for(unsigned j=0;j<3;j++)dispatch(queue,ps[j==1?2:1],&a,inputs[0],inputs[1],inputs[2],perf,repeats,j!=1);
        for(unsigned i=0;i<samples;i++){
            unsigned order[3]={1,2,1}; /* same-TU A1 / candidate B / same-TU A2 */
            for(unsigned j=0;j<3;j++){
                unsigned arm=order[j];double elapsed=dispatch(queue,ps[arm],&a,inputs[0],inputs[1],inputs[2],perf,repeats,arm==1);
                emit(@{@"kind":@"timing",@"case":@(ci),@"sample":@(i),@"order":@(j),@"arm":j==0?@"A1":j==1?@"B":@"A2",
                       @"gpu_seconds":@(elapsed),@"dispatches":@(repeats),@"scope":@"concurrent_encoder_cb_gpu_timestamps"});
            }
        }
    }}
    myds_v41_gpu_scratch_free(topk_scratch);myds_metal_cleanup();emit(@{@"kind":@"complete",@"correctness_passed":@(ok),@"production_adoption":@NO});
    return ok?0:2;
}}
