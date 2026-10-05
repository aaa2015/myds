/* Standalone Metal calibration, deliberately independent of myds kernels.
 * Build (CPU compilation/linking only):
 *   clang -O3 -Wall -Wextra -fobjc-arc tools/bench_roofline.m \
 *     -framework Foundation -framework Metal -lm -o /tmp/bench_roofline
 * Run in an exclusive GPU window; defaults require about 2.1 GiB:
 *   /tmp/bench_roofline --mib 1024 --samples 9 --warmup 3 > roofline.json
 * Logical bytes/FLOPs are counted from kernel work, NOT hardware counters.
 * The read probe includes a retained integer checksum; compute probes reuse
 * register operands. These are synthetic calibrations, not model throughput.
 * GPU timestamps cover the original CB, including GPU dispatch/barrier costs;
 * CPU allocation, initialization, compilation and validation are outside it.
 * Every output element and both allocation guards are checked before accepting
 * a result. No engine, weights, residency defaults or fleet state are modified.
 * GPU core count is null: public MTLDevice exposes no such property. Obtain
 * the actual SKU/core count separately with system_profiler SPDisplaysDataType.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const size_t GUARD = 4096;
static const uint8_t GUARD_VALUE = 0xa7;
static const char *SHADER =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Args { uint nvec; uint threads; uint iters; uint groups; float a; float b; };\n"
"uint pattern(uint i) { return (i * 1664525u + 1013904223u) ^ (i >> 13); }\n"
"uint4 value(uint j) { uint i=j*4u; return uint4(pattern(i),pattern(i+1),pattern(i+2),pattern(i+3)); }\n"
"kernel void probe_read(device const uint4 *src [[buffer(0)]], device uint4 *dst [[buffer(1)]],\n"
"                       constant Args &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t >= p.threads) return; uint4 sum=0;\n"
"  for (uint j=t; j<p.nvec; j+=p.threads) sum+=src[j]; dst[t]=sum;\n"
"}\n"
"kernel void probe_write(device const uint4 *src [[buffer(0)]], device uint4 *dst [[buffer(1)]],\n"
"                        constant Args &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
"  (void)src; if (t >= p.threads) return;\n"
"  for (uint j=t; j<p.nvec; j+=p.threads) dst[j]=value(j);\n"
"}\n"
"kernel void probe_copy(device const uint4 *src [[buffer(0)]], device uint4 *dst [[buffer(1)]],\n"
"                       constant Args &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t >= p.threads) return; for (uint j=t; j<p.nvec; j+=p.threads) dst[j]=src[j];\n"
"}\n"
"kernel void probe_fma(device const float4 *src [[buffer(0)]], device float4 *dst [[buffer(1)]],\n"
"                      constant Args &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t >= p.threads) return;\n"
"  float4 x0=src[t*4], x1=src[t*4+1], x2=src[t*4+2], x3=src[t*4+3];\n"
"  for (uint i=0; i<p.iters; ++i) {\n"
"    x0=fma(x0,p.a,p.b); x1=fma(x1,p.a,p.b); x2=fma(x2,p.a,p.b); x3=fma(x3,p.a,p.b);\n"
"  }\n"
"  dst[t*4]=x0; dst[t*4+1]=x1; dst[t*4+2]=x2; dst[t*4+3]=x3;\n"
"}\n"
"kernel void probe_half_mm(device const half *src [[buffer(0)]], device float *dst [[buffer(1)]],\n"
"                          constant Args &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
"  uint s=t/32u; if (s>=p.groups) return;\n"
"  simdgroup_half8x8 a,b; simdgroup_load(a,src+s*128u,8,ulong2(0),false);\n"
"  simdgroup_load(b,src+s*128u+64u,8,ulong2(0),false);\n"
"  simdgroup_float8x8 c0(0.0f),c1(0.25f),c2(-0.5f),c3(1.0f);\n"
"  for (uint i=0; i<p.iters; ++i) {\n"
"    simdgroup_multiply_accumulate(c0,a,b,c0); simdgroup_multiply_accumulate(c1,a,b,c1);\n"
"    simdgroup_multiply_accumulate(c2,a,b,c2); simdgroup_multiply_accumulate(c3,a,b,c3);\n"
"  }\n"
"  simdgroup_store(c0,dst+s*256u,8,ulong2(0),false);\n"
"  simdgroup_store(c1,dst+s*256u+64u,8,ulong2(0),false);\n"
"  simdgroup_store(c2,dst+s*256u+128u,8,ulong2(0),false);\n"
"  simdgroup_store(c3,dst+s*256u+192u,8,ulong2(0),false);\n"
"}\n";

typedef struct { uint32_t nvec, threads, iters, groups; float a, b; } Args;
typedef struct { unsigned mib, samples, warmup, inner, fma_iters, threads, sg_count, mm_iters; } Config;

static void fail(const char *message) { fprintf(stderr,"bench_roofline: %s\n",message); exit(2); }
static void require(bool condition, const char *message) { if (!condition) fail(message); }
static double wall_seconds(void) {
    struct timespec t; require(clock_gettime(CLOCK_MONOTONIC,&t)==0,"clock_gettime failed");
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}
static uint32_t pattern(uint32_t i) { return (i*UINT32_C(1664525)+UINT32_C(1013904223))^(i>>13); }
static uint64_t fnv(const void *data, size_t bytes) {
    uint64_t h=UINT64_C(14695981039346656037); const uint8_t *p=data;
    for (size_t i=0;i<bytes;++i) h=(h^p[i])*UINT64_C(1099511628211); return h;
}
static NSString *hash_string(const void *data, size_t bytes) {
    return [NSString stringWithFormat:@"%016" PRIx64,fnv(data,bytes)];
}
static id<MTLBuffer> buffer(id<MTLDevice> device, size_t bytes) {
    require(bytes<=SIZE_MAX-2*GUARD && bytes+2*GUARD<=device.maxBufferLength,"buffer exceeds device maxBufferLength");
    id<MTLBuffer> b=[device newBufferWithLength:bytes+2*GUARD options:MTLResourceStorageModeShared];
    require(b!=nil,"buffer allocation failed"); memset(b.contents,GUARD_VALUE,b.length); return b;
}
static void guards(id<MTLBuffer> b) {
    const uint8_t *v=b.contents;
    for (size_t i=0;i<GUARD;++i) require(v[i]==GUARD_VALUE && v[b.length-GUARD+i]==GUARD_VALUE,"allocation guard overwritten");
}
static void *data(id<MTLBuffer> b) { return (uint8_t *)b.contents+GUARD; }
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,id<MTLLibrary> lib,NSString *name) {
    NSError *error=nil; id<MTLFunction> f=[lib newFunctionWithName:name]; require(f!=nil,"shader function absent");
    id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&error];
    if (!p) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    require(p!=nil,"pipeline creation failed"); return p;
}
static NSDictionary *submit(id<MTLCommandQueue> queue,id<MTLComputePipelineState> p,
                            id<MTLBuffer> src,id<MTLBuffer> dst,Args args,
                            NSUInteger threads,unsigned inner,size_t copy_bytes) {
    @autoreleasepool {
        const double t0=wall_seconds(); id<MTLCommandBuffer> cb=[queue commandBuffer]; require(cb!=nil,"command buffer unavailable");
        if (copy_bytes) {
            id<MTLBlitCommandEncoder> e=[cb blitCommandEncoder]; require(e!=nil,"blit encoder unavailable");
            for (unsigned i=0;i<inner;++i) [e copyFromBuffer:src sourceOffset:GUARD toBuffer:dst destinationOffset:GUARD size:copy_bytes];
            [e endEncoding];
        } else {
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder]; require(e!=nil,"compute encoder unavailable");
            [e setComputePipelineState:p]; [e setBuffer:src offset:GUARD atIndex:0];
            [e setBuffer:dst offset:GUARD atIndex:1]; [e setBytes:&args length:sizeof(args) atIndex:2];
            NSUInteger tg=MIN((NSUInteger)256,p.maxTotalThreadsPerThreadgroup);
            require(tg>=p.threadExecutionWidth,"invalid pipeline threadgroup limit"); tg-=tg%p.threadExecutionWidth;
            for (unsigned i=0;i<inner;++i) {
                [e dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
                if (i+1<inner) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [e endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted]; const double wall=wall_seconds()-t0;
        if (cb.status!=MTLCommandBufferStatusCompleted) fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
        require(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");
        require(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime,"GPU timestamps unavailable");
        guards(src); guards(dst);
        return @{@"gpu_s":@(cb.GPUEndTime-cb.GPUStartTime),@"wall_s":@(wall)};
    }
}
static double median(NSArray<NSNumber *> *values) {
    NSArray *sorted=[values sortedArrayUsingSelector:@selector(compare:)]; NSUInteger n=sorted.count;
    return n%2 ? [sorted[n/2] doubleValue] : ([sorted[n/2-1] doubleValue]+[sorted[n/2] doubleValue])*0.5;
}
typedef NSDictionary *(^Validation)(void);
static NSDictionary *bench(NSString *name,id<MTLCommandQueue> queue,id<MTLComputePipelineState> p,
                           id<MTLBuffer> src,id<MTLBuffer> dst,Args args,NSUInteger threads,
                           Config cfg,uint64_t reads,uint64_t writes,uint64_t flops,
                           size_t copy_bytes,NSString *scope,Validation validate) {
    fprintf(stderr,"bench_roofline: %s warmup=%u samples=%u inner=%u\n",name.UTF8String,cfg.warmup,cfg.samples,cfg.inner);
    for (unsigned i=0;i<cfg.warmup;++i) (void)submit(queue,p,src,dst,args,threads,cfg.inner,copy_bytes);
    (void)validate(); /* Gate timing on a complete validation, not a canary alone. */
    NSMutableArray *gpu=[NSMutableArray new],*wall=[NSMutableArray new];
    for (unsigned i=0;i<cfg.samples;++i) {
        NSDictionary *t=submit(queue,p,src,dst,args,threads,cfg.inner,copy_bytes);
        [gpu addObject:t[@"gpu_s"]]; [wall addObject:t[@"wall_s"]];
    }
    NSDictionary *validation=validate(); const double med=median(gpu)/cfg.inner;
    NSArray *sorted=[gpu sortedArrayUsingSelector:@selector(compare:)];
    return @{@"name":name,@"scope":scope,@"warmup_command_buffers":@(cfg.warmup),
        @"formal_command_buffers":@(cfg.samples),@"dispatches_per_command_buffer":@(cfg.inner),
        @"gpu_times_s_per_command_buffer":gpu,@"host_wall_times_s_per_command_buffer":wall,
        @"median_gpu_ms_per_dispatch":@(med*1e3),
        @"min_gpu_ms_per_dispatch":@([sorted.firstObject doubleValue]*1e3/cfg.inner),
        @"max_gpu_ms_per_dispatch":@([sorted.lastObject doubleValue]*1e3/cfg.inner),
        @"median_host_wall_ms_per_dispatch":@(median(wall)*1e3/cfg.inner),
        @"logical_read_bytes_per_dispatch":@(reads),@"logical_write_bytes_per_dispatch":@(writes),
        @"logical_flops_per_dispatch":@(flops),@"logical_GB_s":@((reads+writes)/med/1e9),
        @"logical_TFLOP_s":@(flops/med/1e12),@"validation":validation,
        @"threads_per_dispatch":@(threads),@"pipeline_max_threads_per_threadgroup":p?@(p.maxTotalThreadsPerThreadgroup):(id)[NSNull null],
        @"pipeline_simd_width":p?@(p.threadExecutionWidth):(id)[NSNull null]};
}
static void bandwidth(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> lib,
                       Config cfg,NSMutableArray *results) {
    @autoreleasepool {
        const size_t bytes=(size_t)cfg.mib*1024*1024,checksum_bytes=(size_t)cfg.threads*16;
        require(bytes/16<=UINT32_MAX && cfg.threads<=bytes/16,"bandwidth workload indices invalid");
        id<MTLBuffer> src=buffer(device,bytes),dst=buffer(device,bytes),sum=buffer(device,checksum_bytes);
        uint32_t *v=data(src),*expected=calloc(1,checksum_bytes); require(expected!=NULL,"checksum reference allocation failed");
        for (size_t i=0;i<bytes/4;++i) { v[i]=pattern((uint32_t)i); expected[((i/4)%cfg.threads)*4+i%4]+=v[i]; }
        Args args={.nvec=(uint32_t)(bytes/16),.threads=cfg.threads};
        [results addObject:bench(@"stream_read_uint4_checksum",queue,pipeline(device,lib,@"probe_read"),src,sum,args,cfg.threads,cfg,
            bytes,checksum_bytes,0,0,@"large shared buffer; sequential vector loads plus retained uint4 sums; logical bytes are not physical DRAM counters",^{
                require(memcmp(data(sum),expected,checksum_bytes)==0,"read checksum differs from complete CPU reference");
                guards(src); guards(sum); return @{@"passed":@YES,@"checked_uint32_elements":@(checksum_bytes/4),@"output_fnv1a64":hash_string(data(sum),checksum_bytes)};
            })];
        [results addObject:bench(@"stream_write_uint4",queue,pipeline(device,lib,@"probe_write"),src,dst,args,cfg.threads,cfg,
            0,bytes,0,0,@"large shared buffer; generated nonzero uint4 stores plus integer pattern arithmetic",^{
                const uint32_t *o=data(dst); for (size_t i=0;i<bytes/4;++i) require(o[i]==pattern((uint32_t)i),"write output differs from CPU reference");
                guards(src); guards(dst); return @{@"passed":@YES,@"checked_uint32_elements":@(bytes/4),@"output_fnv1a64":hash_string(data(dst),bytes)};
            })];
        memset(data(dst),0,bytes);
        [results addObject:bench(@"stream_copy_uint4",queue,pipeline(device,lib,@"probe_copy"),src,dst,args,cfg.threads,cfg,
            bytes,bytes,0,0,@"large shared source and destination; logical read+write bytes",^{
                require(memcmp(data(src),data(dst),bytes)==0,"compute copy differs from complete source");
                guards(src); guards(dst); return @{@"passed":@YES,@"checked_bytes":@(bytes),@"output_fnv1a64":hash_string(data(dst),bytes)};
            })];
        memset(data(dst),0,bytes);
        [results addObject:bench(@"blit_copy",queue,nil,src,dst,args,0,cfg,bytes,bytes,0,bytes,
            @"Metal blit engine; logical read+write bytes, not a compute-kernel model ceiling",^{
                require(memcmp(data(src),data(dst),bytes)==0,"blit copy differs from complete source");
                guards(src); guards(dst); return @{@"passed":@YES,@"checked_bytes":@(bytes),@"output_fnv1a64":hash_string(data(dst),bytes)};
            })];
        free(expected);
    }
}
static void compute(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> lib,
                    Config cfg,NSMutableArray *results) {
    @autoreleasepool {
        const size_t bytes=(size_t)cfg.threads*16*sizeof(float);
        id<MTLBuffer> src=buffer(device,bytes),dst=buffer(device,bytes);
        float *v=data(src); for (size_t i=0;i<bytes/4;++i) v[i]=(float)(i%257+1)/1024.0f;
        Args args={.threads=cfg.threads,.iters=cfg.fma_iters,.a=0x1.000002p0f,.b=0x1p-18f};
        float reference[257];
        for (unsigned j=0;j<257;++j) { float x=(float)(j+1)/1024.0f; for (unsigned i=0;i<cfg.fma_iters;++i) x=fmaf(x,args.a,args.b); reference[j]=x; }
        const float *reference_values=reference;
        [results addObject:bench(@"fp32_fma_16_chains",queue,pipeline(device,lib,@"probe_fma"),src,dst,args,cfg.threads,cfg,
            bytes,bytes,(uint64_t)cfg.threads*16*cfg.fma_iters*2,0,
            @"16 independent nonconstant FP32 FMA chains per thread; 2 FLOPs/FMA; safe math; register-reused operands; synthetic compute calibration",^{
                const float *o=data(dst); for (size_t i=0;i<bytes/4;++i) require(memcmp(&o[i],&reference_values[i%257],sizeof(float))==0,"FP32 FMA differs from exact CPU fmaf reference");
                guards(src); guards(dst); return @{@"passed":@YES,@"checked_f32_elements":@(bytes/4),@"comparison":@"bit exact CPU fmaf",@"output_fnv1a64":hash_string(data(dst),bytes)};
            })];
    }
    @autoreleasepool {
        const size_t input_bytes=(size_t)cfg.sg_count*128*sizeof(_Float16),output_bytes=(size_t)cfg.sg_count*256*sizeof(float);
        id<MTLBuffer> src=buffer(device,input_bytes),dst=buffer(device,output_bytes);
        _Float16 *v=data(src);
        for (unsigned s=0;s<cfg.sg_count;++s) for (unsigned row=0;row<8;++row) for (unsigned col=0;col<8;++col) {
            v[(size_t)s*128+row*8+col]=(_Float16)((s+row+col)%4+1)/16;
            v[(size_t)s*128+64+row*8+col]=(_Float16)((s*3+row+col)%4+1)/32;
        }
        id<MTLComputePipelineState> p=pipeline(device,lib,@"probe_half_mm"); require(p.threadExecutionWidth==32,"half MM requires 32-lane SIMDgroups");
        Args args={.iters=cfg.mm_iters,.groups=cfg.sg_count};
        [results addObject:bench(@"half_8x8_mm_float_accum_4_chains",queue,p,src,dst,args,(NSUInteger)cfg.sg_count*32,cfg,
            input_bytes,output_bytes,(uint64_t)cfg.sg_count*4*cfg.mm_iters*2*8*8*8,0,
            @"4 independent half-input 8x8 SIMDgroup matrix multiply chains with F32 accumulation; operands retained in registers; 2 FLOPs/multiply-add; exact dyadic inputs; synthetic compute calibration",^{
                const float initial[4]={0,0.25f,-0.5f,1}; const float *o=data(dst);
                float dots[4][64];
                for (unsigned s=0;s<4;++s) for (unsigned row=0;row<8;++row) for (unsigned col=0;col<8;++col) {
                    float dot=0; for (unsigned k=0;k<8;++k) dot+=((float)((s+row+k)%4+1)/16)*((float)((s*3+k+col)%4+1)/32);
                    dots[s][row*8+col]=dot;
                }
                for (unsigned s=0;s<cfg.sg_count;++s) for (unsigned c=0;c<4;++c) for (unsigned j=0;j<64;++j) {
                    /* A scalar simdgroup_matrix constructor initializes the
                     * diagonal (Metal stdlib _simdgroup_init_diag_impl). */
                    const float expected=(j/8==j%8?initial[c]:0)+dots[s%4][j]*cfg.mm_iters;
                    require(o[(size_t)s*256+c*64+j]==expected,"half MM differs from complete exact dyadic CPU reference");
                }
                guards(src); guards(dst); return @{@"passed":@YES,@"checked_f32_elements":@(output_bytes/4),@"comparison":@"exact dyadic CPU matrix products",@"output_fnv1a64":hash_string(data(dst),output_bytes)};
            })];
    }
}
static unsigned positive(const char *text) {
    char *end=NULL; unsigned long n=strtoul(text,&end,10); require(text[0] && end && !*end && n>0 && n<=UINT32_MAX,"invalid positive integer argument"); return (unsigned)n;
}
int main(int argc,char **argv) {
    Config cfg={.mib=1024,.samples=9,.warmup=3,.inner=1,.fma_iters=1024,.threads=65536,.sg_count=8192,.mm_iters=512};
    const char *only="all";
    for (int i=1;i<argc;++i) {
        const char *opt=argv[i];
        if (!strcmp(opt,"--help")) {
            puts("usage: bench_roofline [--only all|bw|compute] [--mib 1024] [--samples 9] [--warmup 3] [--inner 1] [--compute-threads 65536] [--fma-iters 1024] [--sg-count 8192] [--mm-iters 512]"); return 0;
        }
        require(i+1<argc,"option requires a value"); const char *value=argv[++i];
        if (!strcmp(opt,"--only")) only=value;
        else if (!strcmp(opt,"--mib")) cfg.mib=positive(value);
        else if (!strcmp(opt,"--samples")) cfg.samples=positive(value);
        else if (!strcmp(opt,"--warmup")) cfg.warmup=positive(value);
        else if (!strcmp(opt,"--inner")) cfg.inner=positive(value);
        else if (!strcmp(opt,"--compute-threads")) cfg.threads=positive(value);
        else if (!strcmp(opt,"--fma-iters")) cfg.fma_iters=positive(value);
        else if (!strcmp(opt,"--sg-count")) cfg.sg_count=positive(value);
        else if (!strcmp(opt,"--mm-iters")) cfg.mm_iters=positive(value);
        else fail("unknown option");
    }
    require(!strcmp(only,"all")||!strcmp(only,"bw")||!strcmp(only,"compute"),"--only must be all, bw or compute");
    require(cfg.mib>=128 && cfg.mib<=4096,"--mib must be 128..4096; use large working sets and compare multiple sizes");
    require(cfg.samples>=3 && cfg.samples<=99 && cfg.warmup<=100 && cfg.inner<=64,"sample/warmup/inner count out of range");
    require(cfg.threads>=256 && cfg.threads<=1048576 && cfg.threads%256==0,"compute threads must be 256..1048576 and a multiple of 256");
    require(cfg.sg_count>=256 && cfg.sg_count<=131072 && cfg.sg_count%8==0,"SIMDgroup count out of range or not a multiple of 8");
    require(cfg.fma_iters<=8192 && cfg.mm_iters<=8192,"iteration limit is 8192");
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice(); require(device!=nil,"Metal device unavailable");
        id<MTLCommandQueue> queue=[device newCommandQueue]; require(queue!=nil,"queue unavailable");
        MTLCompileOptions *options=[MTLCompileOptions new];
        if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled=NO;
#pragma clang diagnostic pop
        }
        NSError *error=nil; id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:SHADER] options:options error:&error];
        if (!lib) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String); require(lib!=nil,"Metal shader compilation failed");
        NSMutableArray *results=[NSMutableArray new];
        const NSInteger thermal_before=NSProcessInfo.processInfo.thermalState;
        if (strcmp(only,"compute")) bandwidth(device,queue,lib,cfg,results);
        if (strcmp(only,"bw")) compute(device,queue,lib,cfg,results);
        NSDictionary *report=@{@"schema":@"myds-roofline-calibration-v1",@"device":@{
            @"name":device.name,@"registry_id":@(device.registryID),@"has_unified_memory":@(device.hasUnifiedMemory),
            @"max_buffer_length":@(device.maxBufferLength),@"recommended_max_working_set_size":@(device.recommendedMaxWorkingSetSize),
            @"gpu_core_count":[NSNull null],@"gpu_core_count_source":@"not exposed by public MTLDevice; record system_profiler SPDisplaysDataType separately"},
            @"host":@{@"os":NSProcessInfo.processInfo.operatingSystemVersionString,@"cpu_logical_processors":@(NSProcessInfo.processInfo.activeProcessorCount),
                @"physical_memory_bytes":@(NSProcessInfo.processInfo.physicalMemory),@"thermal_state_before":@(thermal_before),@"thermal_state_after":@(NSProcessInfo.processInfo.thermalState)},
            @"configuration":@{@"only":[NSString stringWithUTF8String:only],@"bandwidth_payload_mib":@(cfg.mib),@"compute_threads":@(cfg.threads),
                @"fma_iterations":@(cfg.fma_iters),@"matrix_simdgroups":@(cfg.sg_count),@"matrix_iterations":@(cfg.mm_iters),@"math_mode":@"safe"},
            @"boundaries":@[@"Synthetic calibration of the current device/software path; not an architectural peak, not model throughput.",
                @"Bytes and FLOPs are logical kernel work, not physical DRAM or hardware operation counters.",
                @"Read checksum and write-pattern integer arithmetic can limit bandwidth; compare all streams and working-set sizes.",
                @"Compute calibration reuses operands in registers; dequantization, gather, attention and real shapes need separate replay measurements.",
                @"GPU timestamps include GPU dispatch/barrier costs but exclude CPU initialization, encoding, completion waits and full output validation.",
                @"Exclusive fleet/process control, temperature/frequency stabilization and input provenance are the caller's responsibility."],@"results":results};
        NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
        require(json!=nil,"JSON serialization failed"); require(fwrite(json.bytes,1,json.length,stdout)==json.length,"stdout write failed"); putchar('\n');
    }
    return 0;
}
