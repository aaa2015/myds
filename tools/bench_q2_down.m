// Independent routed Q2_K down screening. Does not initialize myds or its server.
// Build shader: xcrun -sdk macosx metal -O3 -ffast-math -c metal/probe_q2_down.metal -o /tmp/q2.air
//               xcrun -sdk macosx metallib /tmp/q2.air -o /tmp/q2.metallib
// Build harness: cc -O2 -fobjc-arc -framework Foundation -framework Metal tools/bench_q2_down.m -o /tmp/bench_q2_down
// Run: /tmp/bench_q2_down /tmp/q2.metallib NEW_OUTPUT_DIR [nsg=4] [trials=8] [samples=9] [candidate CSV] [golden production.metallib]
// 384 experts / Top-6 / 5120 output / 2304 MID / 40 disjoint expert sets.
// These are seeded legal blocks, NOT captured GGUF weights or a full-model benchmark.
// Disjoint expert sets span ~929 MB per arm; physically cold DRAM is NOT proven.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(v) do { if (!(v)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#v); exit(1); } } while (0)
struct args_mul_mv_id {
    int32_t nei0, nei1; uint64_t nbi1; int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13; uint64_t nb10, nb11, nb12; int32_t ne0, ne1; uint64_t nb1;
    int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base;
};
_Static_assert(sizeof(struct args_mul_mv_id) == 136, "Metal argument ABI changed");
enum { DIM=5120, WIDTH=2304, EXPERTS=384, ACTIVE=6, LAYERS=40, GUARD=64 };
static uint64_t rng_s = 88172645463325252ull;
static uint64_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static NSString *sha256(const void *p, size_t n) {
    CC_SHA256_CTX ctx; CC_SHA256_Init(&ctx);
    const uint8_t *src=p;
    while(n){CC_LONG take=(CC_LONG)(n > 0x10000000u ? 0x10000000u : n); CC_SHA256_Update(&ctx,src,take);src+=take;n-=take;}
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(digest,&ctx);
    char out[65]; for(unsigned i=0;i<32;++i)snprintf(out+2*i,3,"%02x",digest[i]);
    return [NSString stringWithUTF8String:out];
}
static void emit(NSDictionary *d) {
    NSError *e=nil; NSData *data=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:&e];
    REQUIRE(data); fwrite(data.bytes,1,data.length,stdout); putchar('\n'); fflush(stdout);
}
static void save_raw(NSString *directory, NSMutableArray *manifest, NSString *name,const void *p,size_t n) {
    NSError *e=nil; NSData *data=[NSData dataWithBytes:p length:n];
    REQUIRE([data writeToFile:[directory stringByAppendingPathComponent:name] options:NSDataWritingWithoutOverwriting error:&e]);
    [manifest addObject:@{@"path":name,@"bytes":@(n),@"sha256":sha256(p,n)}];
}
static void fill_q2(void *ptr, size_t n) {
    uint8_t *p=ptr; REQUIRE(n%84==0);
    for(size_t b=0;b<n/84;++b){
        uint8_t *q=p+b*84;
        for(size_t j=0;j<80;j+=8){uint64_t value=rnd();memcpy(q+j,&value,8);}
        // The Q2_K header has scales[16], qs[64], then d/dmin at byte 80/82.
        uint16_t d=0x2000u+(rnd()%0x0800u), m=0x1800u+(rnd()%0x0800u);
        if(b%257u==0)d=0; if(b%263u==0)m=0;
        memcpy(q+80,&d,2);memcpy(q+82,&m,2);
    }
}
static void fill_inputs(id<MTLBuffer> mid,id<MTLBuffer> ids,unsigned trial) {
    float *m=mid.contents;
    for(unsigned i=0;i<LAYERS*ACTIVE*WIDTH;++i){
        float v=((int32_t)(rnd()%2000001)-1000000)/1000000.f;
        if(trial%4==1)v*=16.f;
        if(trial%4==2)v=ldexpf(v,(int)(i%17)-8);
        if(trial%4==3)v=(i%2 ? 1.f:-1.f)*((i%23)/23.f);
        m[i]=v;
    }
    int32_t order[EXPERTS];for(unsigned i=0;i<EXPERTS;++i)order[i]=i;
    for(unsigned i=EXPERTS-1;i>0;--i){unsigned j=rnd()%(i+1);int32_t t=order[i];order[i]=order[j];order[j]=t;}
    memcpy(ids.contents,order,LAYERS*ACTIVE*sizeof(int32_t));
}
static BOOL guards(id<MTLBuffer> out,size_t bytes) {
    const uint8_t *p=out.contents;for(unsigned i=0;i<GUARD;++i)if(p[i]!=0xa5||p[GUARD+bytes+i]!=0xa5)return NO;return YES;
}
static BOOL finite_values(const float *p,size_t n){for(size_t i=0;i<n;++i)if(!isfinite(p[i]))return NO;return YES;}
static int cmpd(const void *a,const void *b){double x=*(const double *)a,y=*(const double *)b;return(x>y)-(x<y);}
static double median(double *v,unsigned n){double *p=malloc(n*sizeof(double));REQUIRE(p);memcpy(p,v,n*sizeof(double));qsort(p,n,sizeof(double),cmpd);double r=n%2?p[n/2]:(p[n/2-1]+p[n/2])/2;free(p);return r;}
static unsigned number(const char *s,unsigned lo,unsigned hi){char *end=NULL;unsigned long n=strtoul(s,&end,10);REQUIRE(*s&&end&&!*end&&n>=lo&&n<=hi);return(unsigned)n;}

int main(int argc,char **argv){
    @autoreleasepool{
        if(argc<3||argc>8){fprintf(stderr,"usage: %s probe.metallib NEW_OUTPUT_DIR [nsg 2|4] [trials 1..64] [samples 3..64] [candidate CSV] [golden production.metallib]\n",argv[0]);return 2;}
        const short nsg=argc>3?(short)number(argv[3],2,4):4; REQUIRE(nsg==2||nsg==4);
        const unsigned trials=argc>4?number(argv[4],1,64):8,samples=argc>5?number(argv[5],3,64):9;
        NSString *dir=[NSString stringWithUTF8String:argv[2]];NSError *err=nil;
        REQUIRE(![[NSFileManager defaultManager] fileExistsAtPath:dir]);
        REQUIRE([[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:&err]);
        NSArray<NSString *> *candidates=@[@"probe_q2_down_yvec_f32",@"probe_q2_down_qvec_f32",@"probe_q2_down_nr1_f32",@"probe_q2_down_nr4_f32"];
        if(argc>6)candidates=[[NSString stringWithUTF8String:argv[6]] componentsSeparatedByString:@","];
        REQUIRE(candidates.count>=1&&candidates.count<=5);
        NSSet *allowed=[NSSet setWithArray:@[@"probe_q2_down_yvec_f32",@"probe_q2_down_qvec_f32",@"probe_q2_down_nr1_f32",@"probe_q2_down_nr4_f32",@"probe_q2_down_v41_f32"]];
        REQUIRE([NSSet setWithArray:candidates].count==candidates.count);
        for(NSString *s in candidates)REQUIRE([allowed containsObject:s]);
        NSMutableArray<NSString *> *names=[NSMutableArray arrayWithObject:@"probe_q2_down_ref_f32"];[names addObjectsFromArray:candidates];
        // Fixed-shape screening always requires the actual production PSO.
        if([candidates containsObject:@"probe_q2_down_v41_f32"])REQUIRE(argc>7);
        const NSUInteger candidate_limit=names.count;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();REQUIRE(device);
        NSString *libpath=[NSString stringWithUTF8String:argv[1]];
        NSData *libdata=[NSData dataWithContentsOfFile:libpath];REQUIRE(libdata);
        id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:libpath] error:&err];REQUIRE(library);
        MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];[constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
        NSMutableArray<id<MTLComputePipelineState>> *psos=[NSMutableArray new];NSMutableArray *identity=[NSMutableArray new];
        for(NSString *s in names){
            id<MTLFunction> f=[library newFunctionWithName:s constantValues:constants error:&err];REQUIRE(f);
            id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&err];REQUIRE(p);
            REQUIRE(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=32u*(unsigned)nsg);
            [psos addObject:p];
            unsigned nr=[s containsString:@"nr1"]?1:([s containsString:@"nr4"]?4:2);
            [identity addObject:@{@"kernel":s,@"nr0":@(nr),@"groups":@((DIM+nr*nsg-1)/(nr*nsg)),@"pso_max_threads":@(p.maxTotalThreadsPerThreadgroup),@"static_tg_bytes":@(p.staticThreadgroupMemoryLength)}];
        }
        // The optional golden library supplies the real production PSO. Its
        // full F32 outputs must agree with the frozen copy before it can be a
        // timing control. With no golden library, this remains a copy-only probe.
        NSData *golden_data=nil;
        NSUInteger control_index=0;
        if(argc>7){
            NSString *golden_path=[NSString stringWithUTF8String:argv[7]];
            golden_data=[NSData dataWithContentsOfFile:golden_path];REQUIRE(golden_data);
            id<MTLLibrary> golden_library=[device newLibraryWithURL:[NSURL fileURLWithPath:golden_path] error:&err];REQUIRE(golden_library);
            NSString *actual_kernel=@"kernel_mul_mv_id_q2_K_sum6_f32";
            id<MTLFunction> f=[golden_library newFunctionWithName:actual_kernel constantValues:constants error:&err];REQUIRE(f);
            id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&err];REQUIRE(p);
            REQUIRE(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=32u*(unsigned)nsg);
            control_index=names.count;[names addObject:actual_kernel];[psos addObject:p];
            [identity addObject:@{@"kernel":actual_kernel,@"nr0":@2,@"groups":@((DIM+2*nsg-1)/(2*nsg)),@"pso_max_threads":@(p.maxTotalThreadsPerThreadgroup),@"static_tg_bytes":@(p.staticThreadgroupMemoryLength),@"golden_library":@YES}];
        }
        const size_t row_bytes=(WIDTH/256)*84u,expert_bytes=DIM*row_bytes,tensor_bytes=EXPERTS*expert_bytes;
        const size_t mid_layer_bytes=ACTIVE*WIDTH*sizeof(float),out_layer_bytes=DIM*sizeof(float),out_bytes=LAYERS*out_layer_bytes;
        REQUIRE(tensor_bytes<=device.maxBufferLength);
        id<MTLBuffer> weights=[device newBufferWithLength:tensor_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> mid=[device newBufferWithLength:LAYERS*mid_layer_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids=[device newBufferWithLength:LAYERS*ACTIVE*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> add=[device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        REQUIRE(weights&&mid&&ids&&add);memset(add.contents,0,out_bytes);
        NSMutableArray<id<MTLBuffer>> *outputs=[NSMutableArray new];
        for(NSUInteger i=0;i<names.count;++i){id<MTLBuffer>b=[device newBufferWithLength:out_bytes+2*GUARD options:MTLResourceStorageModeShared];REQUIRE(b);[outputs addObject:b];}
        fill_q2(weights.contents,tensor_bytes);
        NSString *weight_sha=sha256(weights.contents,tensor_bytes);
        id<MTLCommandQueue> queue=[device newCommandQueue];REQUIRE(queue);
        const struct args_mul_mv_id base={.nei0=ACTIVE,.nei1=1,.nbi1=ACTIVE*4u,.ne00=WIDTH,.ne01=DIM,.ne02=EXPERTS,.nb00=84,.nb01=row_bytes,.nb02=expert_bytes,.ne10=WIDTH,.ne11=ACTIVE,.ne12=1,.ne13=1,.nb10=4,.nb11=WIDTH*4u,.nb12=mid_layer_bytes,.ne0=DIM,.ne1=1,.nb1=out_layer_bytes,.nr0=2,.tp_world=1};
        void(^encode)(id<MTLComputeCommandEncoder>,NSUInteger,unsigned)=^(id<MTLComputeCommandEncoder> e,NSUInteger candidate,unsigned layer){
            unsigned nr=[names[candidate] containsString:@"nr1"]?1:([names[candidate] containsString:@"nr4"]?4:2);
            struct args_mul_mv_id a=base;a.nr0=nr;
            [e setComputePipelineState:psos[candidate]];[e setBytes:&a length:sizeof(a) atIndex:0];
            [e setBuffer:weights offset:0 atIndex:1];[e setBuffer:mid offset:layer*mid_layer_bytes atIndex:2];
            [e setBuffer:outputs[candidate] offset:GUARD+layer*out_layer_bytes atIndex:3];
            [e setBuffer:ids offset:layer*ACTIVE*4u atIndex:4];[e setBuffer:add offset:layer*out_layer_bytes atIndex:5];
            [e setThreadgroupMemoryLength:16 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((DIM+nr*nsg-1)/(nr*nsg),1,1) threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
        };
        id<MTLCommandBuffer>(^run)(NSUInteger)=^id<MTLCommandBuffer>(NSUInteger candidate){
            id<MTLCommandBuffer>cb=[queue commandBuffer];REQUIRE(cb);
            id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];REQUIRE(e);
            for(unsigned layer=0;layer<LAYERS;++layer)encode(e,candidate,layer);
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU failed: %s\n",cb.error.localizedDescription.UTF8String);
            REQUIRE(cb.status==MTLCommandBufferStatusCompleted);return cb;
        };
        emit(@{@"kind":@"identity",@"device":device.name,@"nsg":@(nsg),@"threads":@(32*nsg),@"trials":@(trials),@"samples":@(samples),@"dim":@(DIM),@"width":@(WIDTH),@"experts":@(EXPERTS),@"active":@(ACTIVE),@"layers":@(LAYERS),@"weight_bytes":@(tensor_bytes),@"weight_sha256":weight_sha,@"working_set_bytes_per_arm":@(LAYERS*ACTIVE*expert_bytes),@"metallib_sha256":sha256(libdata.bytes,libdata.length),@"reference_source_sha256":@"bac66b725d05728eeaaa6f4a2840df817a28c965a883083ee822330f50641e85",@"psos":identity,@"golden_library_sha256":golden_data?(id)sha256(golden_data.bytes,golden_data.length):[NSNull null],@"actual_production_kernel":golden_data?(id)@"kernel_mul_mv_id_q2_K_sum6_f32":[NSNull null],@"frozen_reference_symbol":@"probe_q2_down_ref_f32",@"timing_control_kernel":names[control_index],@"seeded_fixture_not_gguf":@YES,@"fullmodel":@NO,@"cold_dram_proven":@NO,@"disjoint_experts_within_arm":@YES,@"encoder_dispatch_type":@"serial"});
        NSMutableArray *manifest=[NSMutableArray new];BOOL all_exact=YES,golden_exact=YES;
        NSMutableArray<NSNumber *> *parity_order=[NSMutableArray arrayWithObject:@0];
        if(golden_data)[parity_order addObject:@(control_index)];
        for(NSUInteger c=1;c<candidate_limit;++c)[parity_order addObject:@(c)];
        for(unsigned trial=0;trial<trials;++trial){
            fill_inputs(mid,ids,trial);
            NSString *prefix=[NSString stringWithFormat:@"parity-%02u",trial];
            save_raw(dir,manifest,[prefix stringByAppendingString:@"-mid.f32"],mid.contents,mid.length);
            save_raw(dir,manifest,[prefix stringByAppendingString:@"-ids.i32"],ids.contents,ids.length);
            for(NSNumber *candidate_index in parity_order){
                const NSUInteger c=candidate_index.unsignedIntegerValue;
                memset(outputs[c].contents,0xa5,outputs[c].length);(void)run(c);
                const uint8_t *p=(const uint8_t *)outputs[c].contents+GUARD;
                save_raw(dir,manifest,[NSString stringWithFormat:@"%@-%@.f32",prefix,names[c]],p,out_bytes);
                const BOOL g=guards(outputs[c],out_bytes),fin=finite_values((const float *)p,out_bytes/4u);
                const uint32_t *ref=(const uint32_t *)((const uint8_t *)outputs[0].contents+GUARD),*cand=(const uint32_t *)p;
                size_t mismatch=0,first=SIZE_MAX;for(size_t i=0;i<out_bytes/4u;++i)if(ref[i]!=cand[i]){if(first==SIZE_MAX)first=i;++mismatch;}
                BOOL exact=!mismatch&&g&&fin;all_exact&=exact;
                if(golden_data&&c==control_index)golden_exact&=exact;
                emit(@{@"kind":@"parity",@"trial":@(trial),@"kernel":names[c],@"floats":@(out_bytes/4u),@"mismatches":@(mismatch),@"first_mismatch":first==SIZE_MAX?(id)[NSNull null]:@(first),@"full_f32_bit_exact":@(exact),@"finite":@(fin),@"guards":@(g),@"completed_cb":@YES,@"sha256":sha256(p,out_bytes),@"actual_production_pso":@(golden_data&&c==control_index),@"comparison_reference_symbol":@"probe_q2_down_ref_f32"});
                if(!golden_exact)break;
            }
            if(!golden_exact)break;
        }
        // Refuse to time any mismatching candidate. Raw files stay available on failure.
        if(!all_exact){NSData *data=[NSJSONSerialization dataWithJSONObject:@{@"files":manifest} options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&err];REQUIRE(data);REQUIRE([data writeToFile:[dir stringByAppendingPathComponent:@"artifact-manifest.json"] options:NSDataWritingWithoutOverwriting error:&err]);return 1;}
        NSData *frozen_reference=[NSData dataWithBytes:(const uint8_t *)outputs[0].contents+GUARD length:out_bytes];
        REQUIRE(frozen_reference);
        save_raw(dir,manifest,@"timing-mid.f32",mid.contents,mid.length);save_raw(dir,manifest,@"timing-ids.i32",ids.contents,ids.length);
        for(NSUInteger c=1;c<candidate_limit;++c){
            double medians[3];
            for(unsigned arm=0;arm<3;++arm){NSUInteger index=arm==1?c:control_index;double times[64];
                NSString *armname=arm==0?@"A1":(arm==1?@"B":@"A2");
                for(unsigned sample=0;sample<=samples;++sample){id<MTLCommandBuffer>cb=run(index);
                    REQUIRE(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime);
                    double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/LAYERS;
                    const uint8_t *p=(const uint8_t *)outputs[index].contents+GUARD;
                    REQUIRE(guards(outputs[index],out_bytes)&&finite_values((const float*)p,out_bytes/4u));
                    // Check both controls and candidates against the frozen last-trial output.
                    REQUIRE(!memcmp(frozen_reference.bytes,p,out_bytes));
                    emit(@{@"kind":@"timing",@"candidate":names[c],@"kernel":names[index],@"arm":armname,@"sample":@(sample),@"warmup":@(sample==0),@"dispatches":@(LAYERS),@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"us_per_dispatch":@(us),@"completed_cb":@YES,@"full_f32_bit_exact":@YES,@"production_math_oracle_validated":@(golden_data!=nil),@"actual_production_pso":@(golden_data&&index==control_index)});
                    if(sample)times[sample-1]=us;
                }
                medians[arm]=median(times,samples);
                const uint8_t *p=(const uint8_t *)outputs[index].contents+GUARD;
                save_raw(dir,manifest,[NSString stringWithFormat:@"timing-%@-%@.f32",names[c],armname],p,out_bytes);
            }
            double control=(medians[0]+medians[2])/2;
            emit(@{@"kind":@"summary",@"candidate":names[c],@"a1_us":@(medians[0]),@"b_us":@(medians[1]),@"a2_us":@(medians[2]),@"control_drift_percent":@((medians[2]/medians[0]-1)*100),@"time_saved_percent":@((control-medians[1])/control*100),@"estimated_down_ms_saved_per_40_layers":@((control-medians[1])*LAYERS/1000),@"fullmodel_speedup_proven":@NO,@"production_math_oracle_validated":@(golden_data!=nil),@"control_kernel":names[control_index]});
        }
        NSData *manifest_data=[NSJSONSerialization dataWithJSONObject:@{@"schema_version":@1,@"files":manifest,@"weight_sha256":weight_sha,@"golden_library_sha256":golden_data?(id)sha256(golden_data.bytes,golden_data.length):[NSNull null]} options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&err];REQUIRE(manifest_data);
        REQUIRE([manifest_data writeToFile:[dir stringByAppendingPathComponent:@"artifact-manifest.json"] options:NSDataWritingWithoutOverwriting error:&err]);
        emit(@{@"kind":@"complete",@"full_f32_bit_exact":@YES,@"raw_files":@(manifest.count),@"gpu_benchmark_executed":@YES,@"production_math_oracle_validated":@(golden_data!=nil)});
    }
    return 0;
}
