// Exact-topology router probe. No myds initialization or service interaction.
// Shader: xcrun -sdk macosx metal -O3 -ffast-math -c metal/probe_v41_route.metal -o /tmp/route.air
//         xcrun -sdk macosx metallib /tmp/route.air -o /tmp/route.metallib
// Host: cc -O2 -fobjc-arc -framework Foundation -framework Metal tools/bench_v41_route.m -o /tmp/bench_v41_route
// Run: /tmp/bench_v41_route probe.metallib production.metallib NEW_OUTPUT_DIR [trials=16] [samples=9]
// Golden production PSO is REQUIRED. Each trial exports full probs, scores,
// ordered IDs and weights for 40 separate single-token dispatches. Inputs are
// seeded fixtures including equal scores, signed zero, large finite values,
// transform threshold neighborhoods and the production 128 padding entries.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(v) do {if(!(v)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#v);exit(1);}}while(0)
struct route_args{uint32_t count,top_k,op;float scale;uint32_t rows,row,row4;};
_Static_assert(sizeof(struct route_args)==28,"route argument ABI changed");
enum{E=384,K=6,LAYERS=40,GUARD=64};
static uint64_t rng_s=88172645463325252ull;
static uint64_t rnd(void){rng_s^=rng_s<<13;rng_s^=rng_s>>7;rng_s^=rng_s<<17;return rng_s;}
static NSString *sha256(const void*p,size_t n){unsigned char hash[32];REQUIRE(n<=UINT32_MAX);CC_SHA256(p,(CC_LONG)n,hash);char s[65];for(unsigned i=0;i<32;++i)snprintf(s+2*i,3,"%02x",hash[i]);return[NSString stringWithUTF8String:s];}
static void emit(NSDictionary*d){NSError*e=nil;NSData*x=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:&e];REQUIRE(x);fwrite(x.bytes,1,x.length,stdout);putchar('\n');fflush(stdout);}
static void save(NSString*dir,NSMutableArray*manifest,NSString*name,const void*p,size_t n){NSError*e=nil;NSData*x=[NSData dataWithBytes:p length:n];REQUIRE([x writeToFile:[dir stringByAppendingPathComponent:name] options:NSDataWritingWithoutOverwriting error:&e]);[manifest addObject:@{@"path":name,@"bytes":@(n),@"sha256":sha256(p,n)}];}
static unsigned number(const char*s,unsigned lo,unsigned hi){char*end=NULL;unsigned long n=strtoul(s,&end,10);REQUIRE(*s&&end&&!*end&&n>=lo&&n<=hi);return(unsigned)n;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static double median(double*p,unsigned n){double t[64];REQUIRE(n<=64);memcpy(t,p,n*sizeof(double));qsort(t,n,sizeof(double),cmpd);return n%2?t[n/2]:(t[n/2-1]+t[n/2])/2;}
static BOOL finite(const float*p,size_t n){for(size_t i=0;i<n;++i)if(!isfinite(p[i]))return NO;return YES;}
static BOOL guard(id<MTLBuffer>b,size_t payload){const uint8_t*p=b.contents;for(unsigned i=0;i<GUARD;++i)if(p[i]!=0xa5||p[GUARD+payload+i]!=0xa5)return NO;return YES;}
static BOOL ids_valid(const int32_t*p){for(unsigned l=0;l<LAYERS;++l)for(unsigned i=0;i<K;++i){if(p[l*K+i]<0||p[l*K+i]>=E)return NO;for(unsigned j=0;j<i;++j)if(p[l*K+i]==p[l*K+j])return NO;}return YES;}
static NSString*case_name(unsigned t){NSArray*s=@[@"random",@"all_scores_equal",@"repeated_bias_ties",@"zero_probs_signed_zero_bias",@"large_finite_logits",@"softplus_thresholds",@"small_scale",@"large_positive_bias_ties",@"large_negative_bias_ties",@"equal_score_bands",@"zero_prob_score_bands",@"near_ties"];return s[t%s.count];}
static float fill(id<MTLBuffer>logits,id<MTLBuffer>bias,unsigned trial){
    float*x=logits.contents,*b=bias.contents;
    const unsigned mode=trial%12;
    for(unsigned l=0;l<LAYERS;++l)for(unsigned i=0;i<E;++i){
        const size_t p=l*E+i;
        x[p]=((int32_t)(rnd()%2000001)-1000000)/100000.f;
        b[p]=((int32_t)(rnd()%2000001)-1000000)/1000000.f;
        switch(mode){
            case 1:x[p]=0;b[p]=0;break;
            case 2:x[p]=0;b[p]=((i+l)%7)*0.01f;break;
            case 3:x[p]=-1000;b[p]=(i%2)?0.f:-0.f;break;
            case 4:x[p]=((i+l)%3==0)?FLT_MAX:(((i+l)%3==1)?-FLT_MAX:20.f);b[p]=0;break;
            case 5:{const float edge[]={-3.465736f,nextafterf(-3.465736f,-INFINITY),nextafterf(-3.465736f,INFINITY),20.f,nextafterf(20.f,-INFINITY),nextafterf(20.f,INFINITY),-104.f,-87.f};x[p]=edge[(i+l)%8];b[p]=(i%5)*1e-6f;break;}
            case 6:x[p]*=0.25f;break;
            case 7:x[p]=0;b[p]=FLT_MAX;break;
            case 8:x[p]=0;b[p]=-FLT_MAX;break;
            case 9:x[p]=0;b[p]=((i+l)%16==0)?1.f:0.f;break;
            case 10:x[p]=-1000;b[p]=((i+l)%8==0)?1.f:((i%2)?0.f:-0.f);break;
            case 11:x[p]=0;b[p]=((i+l)%11)*FLT_EPSILON;break;
            default:break;
        }
    }
    return mode==6?0.00001f:((trial%2)?2.5f:1.25f);
}
int main(int argc,char**argv){@autoreleasepool{
    if(argc<4||argc>6){fprintf(stderr,"usage: %s probe.metallib production.metallib NEW_OUTPUT_DIR [trials 12..64] [samples 3..64]\n",argv[0]);return 2;}
    unsigned trials=argc>4?number(argv[4],12,64):16,samples=argc>5?number(argv[5],3,64):9;
    NSString*dir=[NSString stringWithUTF8String:argv[3]];NSError*error=nil;
    REQUIRE(![[NSFileManager defaultManager] fileExistsAtPath:dir]);REQUIRE([[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:&error]);
    id<MTLDevice>device=MTLCreateSystemDefaultDevice();REQUIRE(device);id<MTLCommandQueue>q=[device newCommandQueue];REQUIRE(q);
    NSString*probe_path=[NSString stringWithUTF8String:argv[1]],*golden_path=[NSString stringWithUTF8String:argv[2]];
    NSData*probe_data=[NSData dataWithContentsOfFile:probe_path],*golden_data=[NSData dataWithContentsOfFile:golden_path];REQUIRE(probe_data&&golden_data);
    id<MTLLibrary>probe=[device newLibraryWithURL:[NSURL fileURLWithPath:probe_path] error:&error];REQUIRE(probe);
    id<MTLLibrary>golden=[device newLibraryWithURL:[NSURL fileURLWithPath:golden_path] error:&error];REQUIRE(golden);
    NSArray<NSString*>*names=@[@"kernel_myds_v41_route_fused",@"probe_v41_route_ref",@"probe_v41_route_shuffle"];
    NSMutableArray<id<MTLComputePipelineState>>*psos=[NSMutableArray new];NSMutableArray*identities=[NSMutableArray new];
    for(NSUInteger c=0;c<names.count;++c){id<MTLFunction>f=[c==0?golden:probe newFunctionWithName:names[c]];REQUIRE(f);id<MTLComputePipelineState>p=[device newComputePipelineStateWithFunction:f error:&error];REQUIRE(p);REQUIRE(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=512);[psos addObject:p];[identities addObject:@{@"kernel":names[c],@"max_threads":@(p.maxTotalThreadsPerThreadgroup),@"static_tg_bytes":@(p.staticThreadgroupMemoryLength),@"golden_library":@(c==0)}];}
    const size_t full_bytes=LAYERS*E*4u,top_bytes=LAYERS*K*4u;
    id<MTLBuffer>logits=[device newBufferWithLength:full_bytes options:MTLResourceStorageModeShared],bias=[device newBufferWithLength:full_bytes options:MTLResourceStorageModeShared];REQUIRE(logits&&bias);
    NSArray<NSString*>*fields=@[@"probs",@"scores",@"ordered_ids",@"weights"];
    NSMutableArray<NSArray<id<MTLBuffer>>*>*buffers=[NSMutableArray new];
    for(NSUInteger c=0;c<names.count;++c){NSMutableArray*out=[NSMutableArray new];for(unsigned f=0;f<4;++f){size_t n=f<2?full_bytes:top_bytes;id<MTLBuffer>b=[device newBufferWithLength:n+2*GUARD options:MTLResourceStorageModeShared];REQUIRE(b);[out addObject:b];}[buffers addObject:out];}
    __block struct route_args args={.count=E,.top_k=K,.scale=2.5f,.rows=1,.row=E,.row4=E/4};
    id<MTLCommandBuffer>(^run)(NSUInteger)=^id<MTLCommandBuffer>(NSUInteger c){
        id<MTLCommandBuffer>cb=[q commandBuffer];REQUIRE(cb);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];REQUIRE(e);
        [e setComputePipelineState:psos[c]];[e setBytes:&args length:sizeof(args) atIndex:0];[e setThreadgroupMemoryLength:5760 atIndex:0];
        for(unsigned l=0;l<LAYERS;++l){[e setBuffer:logits offset:l*E*4u atIndex:1];[e setBuffer:bias offset:l*E*4u atIndex:2];for(unsigned f=0;f<4;++f)[e setBuffer:buffers[c][f] offset:GUARD+l*(f<2?E:K)*4u atIndex:3+f];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(512,1,1)];}
        [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU failed %s\n",cb.error.localizedDescription.UTF8String);REQUIRE(cb.status==MTLCommandBufferStatusCompleted);return cb;
    };
    emit(@{@"kind":@"identity",@"device":device.name,@"experts":@(E),@"top_k":@(K),@"layers":@(LAYERS),@"threads":@512,@"dynamic_tg_bytes":@5760,@"probe_library_sha256":sha256(probe_data.bytes,probe_data.length),@"golden_library_sha256":sha256(golden_data.bytes,golden_data.length),@"golden_kernel":names[0],@"frozen_symbol":names[1],@"reference_source_sha256":@"2e0818694abc7dea45ea7b3aee0869fb69daea9e0bd89b4ae1fb7823413a53c0",@"psos":identities,@"fullmodel":@NO,@"seeded_fixture":@YES,@"encoder_dispatch_type":@"serial"});
    NSMutableArray*manifest=[NSMutableArray new];BOOL exact=YES;
    for(unsigned t=0;t<trials;++t){args.scale=fill(logits,bias,t);NSString*prefix=[NSString stringWithFormat:@"parity-%02u",t];save(dir,manifest,[prefix stringByAppendingString:@"-logits.f32"],logits.contents,full_bytes);save(dir,manifest,[prefix stringByAppendingString:@"-bias.f32"],bias.contents,full_bytes);
        for(NSUInteger c=0;c<names.count;++c){for(id<MTLBuffer>b in buffers[c])memset(b.contents,0xa5,b.length);(void)run(c);
            BOOL kernel_exact=YES;NSMutableArray*results=[NSMutableArray new];
            for(unsigned f=0;f<4;++f){size_t n=f<2?full_bytes:top_bytes;const uint8_t*p=(const uint8_t*)buffers[c][f].contents+GUARD,*r=(const uint8_t*)buffers[0][f].contents+GUARD;size_t mismatch=0,first=SIZE_MAX;for(size_t i=0;i<n/4;++i)if(((const uint32_t*)p)[i]!=((const uint32_t*)r)[i]){if(first==SIZE_MAX)first=i;++mismatch;}BOOL good=guard(buffers[c][f],n)&&(f==2?ids_valid((const int32_t*)p):finite((const float*)p,n/4));kernel_exact&=!mismatch&&good;save(dir,manifest,[NSString stringWithFormat:@"%@-%@-%@.%@",prefix,names[c],fields[f],f==2?@"i32":@"f32"],p,n);[results addObject:@{@"field":fields[f],@"elements":@(n/4),@"mismatches":@(mismatch),@"first_mismatch":first==SIZE_MAX?(id)[NSNull null]:@(first),@"finite_or_ids_valid":@(good),@"guards":@(guard(buffers[c][f],n)),@"sha256":sha256(p,n)}];}
            exact&=kernel_exact;emit(@{@"kind":@"parity",@"trial":@(t),@"case":case_name(t),@"scale":@(args.scale),@"kernel":names[c],@"fields":results,@"all_fields_bit_exact":@(kernel_exact),@"completed_cb":@YES,@"actual_production_pso":@(c==0)});
            if(!kernel_exact)break;
        }
        if(!exact)break;
    }
    if(!exact){NSData*d=[NSJSONSerialization dataWithJSONObject:@{@"files":manifest} options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];REQUIRE(d);REQUIRE([d writeToFile:[dir stringByAppendingPathComponent:@"artifact-manifest.json"] options:NSDataWritingWithoutOverwriting error:&error]);return 1;}
    // Time ordinary random finite inputs; all corner-case contracts were checked above.
    args.scale=fill(logits,bias,0);for(id<MTLBuffer>b in buffers[0])memset(b.contents,0xa5,b.length);(void)run(0);
    NSMutableArray<NSData*>*frozen=[NSMutableArray new];for(unsigned f=0;f<4;++f){size_t n=f<2?full_bytes:top_bytes;[frozen addObject:[NSData dataWithBytes:(const uint8_t*)buffers[0][f].contents+GUARD length:n]];}
    save(dir,manifest,@"timing-logits.f32",logits.contents,full_bytes);save(dir,manifest,@"timing-bias.f32",bias.contents,full_bytes);
    double medians[3];
    for(unsigned arm=0;arm<3;++arm){NSUInteger c=arm==1?2:0;NSString*armname=arm==0?@"A1":(arm==1?@"B":@"A2");double times[64];
        for(unsigned sample=0;sample<=samples;++sample){id<MTLCommandBuffer>cb=run(c);REQUIRE(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime);for(unsigned f=0;f<4;++f){size_t n=f<2?full_bytes:top_bytes;const uint8_t*p=(const uint8_t*)buffers[c][f].contents+GUARD;REQUIRE(guard(buffers[c][f],n));REQUIRE(!memcmp(frozen[f].bytes,p,n));}double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/LAYERS;emit(@{@"kind":@"timing",@"arm":armname,@"kernel":names[c],@"sample":@(sample),@"warmup":@(sample==0),@"dispatches":@(LAYERS),@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"us_per_dispatch":@(us),@"all_fields_bit_exact":@YES,@"completed_cb":@YES,@"actual_production_pso":@(c==0)});if(sample)times[sample-1]=us;}
        medians[arm]=median(times,samples);for(unsigned f=0;f<4;++f){size_t n=f<2?full_bytes:top_bytes;save(dir,manifest,[NSString stringWithFormat:@"timing-%@-%@.%@",armname,fields[f],f==2?@"i32":@"f32"],(const uint8_t*)buffers[c][f].contents+GUARD,n);}
    }
    double control=(medians[0]+medians[2])/2;emit(@{@"kind":@"summary",@"candidate":names[2],@"control_kernel":names[0],@"a1_us":@(medians[0]),@"b_us":@(medians[1]),@"a2_us":@(medians[2]),@"control_drift_percent":@((medians[2]/medians[0]-1)*100),@"time_saved_percent":@((control-medians[1])/control*100),@"estimated_route_ms_saved_per_40_layers":@((control-medians[1])*LAYERS/1000),@"fullmodel_speedup_proven":@NO});
    NSData*d=[NSJSONSerialization dataWithJSONObject:@{@"schema_version":@1,@"files":manifest,@"golden_library_sha256":sha256(golden_data.bytes,golden_data.length),@"probe_library_sha256":sha256(probe_data.bytes,probe_data.length)} options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];REQUIRE(d);REQUIRE([d writeToFile:[dir stringByAppendingPathComponent:@"artifact-manifest.json"] options:NSDataWritingWithoutOverwriting error:&error]);emit(@{@"kind":@"complete",@"all_fields_bit_exact":@YES,@"raw_files":@(manifest.count),@"fullmodel":@NO});
}return 0;}
