/* tests/bench_hc_mix_step16.m
 *
 * Standalone M4 / M2 Ultra hardware-gate microbench for [P-H1] (Step 1 + Step 6
 * HC-mix F16 GEMV):
 *   Production: kernel_mul_mv_f16_f32_4_b4_scaled from bin/myds.metallib
 *               NR0=2, B=4, NSG=8, grid=(12, 512, 1) -> 12x X-TG input reload
 *   Prototype 1: NR0=24, B=1, NSG=8, grid=(1, 2048, 1) -> 1x single-read input
 *   Prototype 2: NR0=24, B=2, NSG=8, grid=(1, 1024, 1) -> 1x single-read input
 *
 * Workload: real Step 1 / Step 6 prefill chunk shape:
 *   input  src1:   2048 x (4 * 5120) F32 = 167,772,160 B (167.8 MB > 96 MB M2U SLC)
 *   weight src0:   24   x 20480      F16 =     983,040 B (983 KB)
 *   scales:        2048              F32 =       8,192 B
 *   output dst:    2048 x 24         F32 =     196,608 B
 *
 * Gates:
 *   1. Strict 0-ULP bit-identity (memcmp == 0) vs production on BOTH n_tok=2048
 *      and odd tail n_tok=123.
 *   2. 7-run median GPU time (GPUStartTime..GPUEndTime, warmup + interleaved trials)
 *      measuring real DRAM/SLC behavior and speedup.
 *
 * Usage:
 *   bin/bench_hc_mix_step16 [bin/myds.metallib] [--batch 4] [--runs 7]
 */
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Exact ABI match to metal/dense.metal & src/myds_metal.m */
struct ds4_metal_args_mul_mv {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
};

static uint32_t query_gpu_core_count(void) {
    io_iterator_t iter = IO_OBJECT_NULL;
    kern_return_t kr = IOServiceGetMatchingServices(
        kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iter);
    if (kr != KERN_SUCCESS || iter == IO_OBJECT_NULL) return 0;
    uint32_t cores = 0;
    io_object_t service = IO_OBJECT_NULL;
    while ((service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        CFTypeRef prop = IORegistryEntryCreateCFProperty(
            service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (prop) {
            if (CFGetTypeID(prop) == CFNumberGetTypeID()) {
                int64_t v = 0;
                if (CFNumberGetValue((CFNumberRef)prop, kCFNumberSInt64Type, &v) && v > 0) {
                    cores = (uint32_t)v;
                }
            }
            CFRelease(prop);
        }
        IOObjectRelease(service);
        if (cores > 0) break;
    }
    IOObjectRelease(iter);
    return cores;
}

static int detect_apple_gpu_family(id<MTLDevice> dev) {
    int best = 0;
    for (int fam = 1; fam <= 14; ++fam) {
        MTLGPUFamily f = (MTLGPUFamily)(1000 + fam);
        @try {
            if ([dev supportsFamily:f]) best = fam;
        } @catch (__unused NSException *ex) {
            break;
        }
    }
    return best;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da < db) ? -1 : (da > db ? 1 : 0);
}

static uint64_t g_rng = 0x2026112000178ULL;
static uint32_t rng_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}
static float rng_signed(float amp) {
    /* [-amp, +amp] */
    return ((float)(rng_u32() & 0xFFFFu) / 32767.5f - 1.0f) * amp;
}

static NSString *const kProtoShaderSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"#define N_SIMDWIDTH 32\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"\n"
"constant short FC_mul_mv_nsg [[function_constant(600)]];\n"
"\n"
"struct ds4_metal_args_mul_mv {\n"
"    int   ne00;\n"
"    int   ne01;\n"
"    int   ne02;\n"
"    ulong nb00;\n"
"    ulong nb01;\n"
"    ulong nb02;\n"
"    ulong nb03;\n"
"    int   ne10;\n"
"    int   ne11;\n"
"    int   ne12;\n"
"    ulong nb10;\n"
"    ulong nb11;\n"
"    ulong nb12;\n"
"    ulong nb13;\n"
"    int   ne0;\n"
"    int   ne1;\n"
"    int   nr0;\n"
"    short r2;\n"
"    short r3;\n"
"};\n"
"\n"
"template<short NR0, short B>\n"
"static inline void proto_mul_mv_f16_f32_scaled_impl(\n"
"        constant ds4_metal_args_mul_mv & args,\n"
"        device const char  * src0,\n"
"        device const char  * src1,\n"
"        device       char  * dst,\n"
"        device const float * scales,\n"
"        threadgroup  char  * shmem,\n"
"        uint3  tgpig,\n"
"        ushort tiisg,\n"
"        ushort sgitg) {\n"
"    const short NSG = FC_mul_mv_nsg;\n"
"    constexpr short NW  = N_SIMDWIDTH;\n"
"    constexpr short NB  = 32;\n"
"    constexpr short NF  = 16;\n"
"    constexpr short NF4 = NF / 4;\n"
"\n"
"    const int nb = args.ne00 / NB;\n"
"    const int r0 = (int)tgpig.x * NR0;\n"
"    const int t0 = (int)tgpig.y * B;\n"
"    const int im = (int)tgpig.z;\n"
"\n"
"    if (t0 >= args.ne11) return;\n"
"\n"
"    const uint i12 = (uint)im % (uint)args.ne12;\n"
"    const uint i13 = (uint)im / (uint)args.ne12;\n"
"\n"
"    device const float  * y  [B];\n"
"    device const float4 * yb4[B];\n"
"    float scale[B];\n"
"\n"
"    const short ix = tiisg / (NW / NF);\n"
"    const short il = tiisg % (NW / NF);\n"
"    const int ib0  = (int)sgitg * NF + ix;\n"
"\n"
"    FOR_UNROLL (short k = 0; k < B; ++k) {\n"
"        if (t0 + k < args.ne11) {\n"
"            const uint64_t offset1 = (uint64_t)(t0 + k) * args.nb11\n"
"                                   + (uint64_t)i12 * args.nb12\n"
"                                   + (uint64_t)i13 * args.nb13;\n"
"            y[k] = (device const float *)(src1 + offset1);\n"
"            device const float4 * y4 = (device const float4 *)(src1 + offset1);\n"
"            yb4[k] = y4 + (ib0 * NB + il * NF) / 4;\n"
"            scale[k] = scales[t0 + k];\n"
"        } else {\n"
"            y[k] = nullptr;\n"
"            yb4[k] = nullptr;\n"
"            scale[k] = 0.0f;\n"
"        }\n"
"    }\n"
"\n"
"    const uint64_t base0 = (uint64_t)r0 * args.nb01\n"
"                         + (uint64_t)(i12 / (uint)args.r2) * args.nb02\n"
"                         + (uint64_t)(i13 / (uint)args.r3) * args.nb03;\n"
"    device const char * src0_base = src0 + base0;\n"
"\n"
"    float sumf[B][NR0];\n"
"    FOR_UNROLL (short k = 0; k < B; ++k) {\n"
"        FOR_UNROLL (short row = 0; row < NR0; ++row) {\n"
"            sumf[k][row] = 0.0f;\n"
"        }\n"
"    }\n"
"\n"
"    for (int ib = ib0; ib < nb; ib += NSG * NF) {\n"
"        float4 yl4[B][NF4];\n"
"        FOR_UNROLL (short k = 0; k < B; ++k) {\n"
"            if (yb4[k]) {\n"
"                FOR_UNROLL (short i = 0; i < NF4; ++i) {\n"
"                    yl4[k][i] = yb4[k][i] * scale[k];\n"
"                }\n"
"            }\n"
"        }\n"
"\n"
"        const uint64_t col_byte_off = (uint64_t)(ib * NB + il * NF) * sizeof(half);\n"
"        device const char * w_row = src0_base + col_byte_off;\n"
"\n"
"        #pragma clang loop unroll(disable)\n"
"        for (short row = 0; row < NR0; ++row) {\n"
"            device const half4 * xb4 = (device const half4 *)w_row;\n"
"            w_row += args.nb01;\n"
"\n"
"            FOR_UNROLL (short k = 0; k < B; ++k) {\n"
"                if (yb4[k]) {\n"
"                    float sumq = 0.0f;\n"
"                    FOR_UNROLL (short i = 0; i < NF4; ++i) {\n"
"                        sumq += dot(float4(xb4[i]), yl4[k][i]);\n"
"                    }\n"
"                    sumf[k][row] += sumq;\n"
"                }\n"
"            }\n"
"        }\n"
"\n"
"        FOR_UNROLL (short k = 0; k < B; ++k) {\n"
"            if (yb4[k]) {\n"
"                yb4[k] += NSG * NF * NW / 4;\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    for (int i = nb * NB + (int)sgitg * NW + (int)tiisg; i < args.ne00; i += NW * NSG) {\n"
"        device const char * w_row = src0_base + (uint64_t)i * sizeof(half);\n"
"        for (short row = 0; row < NR0; ++row) {\n"
"            float w = (float)(*(device const half *)w_row);\n"
"            w_row += args.nb01;\n"
"            FOR_UNROLL (short k = 0; k < B; ++k) {\n"
"                if (y[k]) {\n"
"                    sumf[k][row] += w * (y[k][i] * scale[k]);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    /* Exact reduction tree matching helper_mv_batch_reduce_and_write<NR0, B, false>. */\n"
"    for (short k = 0; k < B; ++k) {\n"
"        for (short row = 0; row < NR0; ++row) {\n"
"            short slot = k * NR0 + row;\n"
"            threadgroup float * sm = (threadgroup float *)shmem + NW * slot;\n"
"            if (sgitg == 0) {\n"
"                sm[tiisg] = 0.0f;\n"
"            }\n"
"            sumf[k][row] = simd_sum(sumf[k][row]);\n"
"        }\n"
"    }\n"
"\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    for (short k = 0; k < B; ++k) {\n"
"        for (short row = 0; row < NR0; ++row) {\n"
"            short slot = k * NR0 + row;\n"
"            threadgroup float * sm = (threadgroup float *)shmem + NW * slot;\n"
"            if (tiisg == 0) {\n"
"                sm[sgitg] = sumf[k][row];\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    device float * dst_f32_base = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1;\n"
"    for (short k = 0; k < B; ++k) {\n"
"        if (t0 + k < args.ne11) {\n"
"            device float * dst_k = dst_f32_base + (uint64_t)(t0 + k) * (uint64_t)args.ne0;\n"
"            for (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {\n"
"                short slot = k * NR0 + row;\n"
"                if ((slot % NSG) == (short)sgitg) {\n"
"                    threadgroup float * sm = (threadgroup float *)shmem + NW * slot;\n"
"                    float tot = simd_sum(sm[tiisg]);\n"
"                    if (tiisg == 0) {\n"
"                        dst_k[r0 + row] = tot;\n"
"                    }\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void kernel_proto_hc_mix_nr24_b1(\n"
"        constant ds4_metal_args_mul_mv & args [[buffer(0)]],\n"
"        device const char  * src0             [[buffer(1)]],\n"
"        device const char  * src1             [[buffer(2)]],\n"
"        device       char  * dst              [[buffer(3)]],\n"
"        device const float * scales           [[buffer(4)]],\n"
"        threadgroup  char  * shmem            [[threadgroup(0)]],\n"
"        uint3  tgpig [[threadgroup_position_in_grid]],\n"
"        ushort tiisg [[thread_index_in_simdgroup]],\n"
"        ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    proto_mul_mv_f16_f32_scaled_impl<24, 1>(args, src0, src1, dst, scales, shmem, tgpig, tiisg, sgitg);\n"
"}\n"
"\n"
"kernel void kernel_proto_hc_mix_nr24_b2(\n"
"        constant ds4_metal_args_mul_mv & args [[buffer(0)]],\n"
"        device const char  * src0             [[buffer(1)]],\n"
"        device const char  * src1             [[buffer(2)]],\n"
"        device       char  * dst              [[buffer(3)]],\n"
"        device const float * scales           [[buffer(4)]],\n"
"        threadgroup  char  * shmem            [[threadgroup(0)]],\n"
"        uint3  tgpig [[threadgroup_position_in_grid]],\n"
"        ushort tiisg [[thread_index_in_simdgroup]],\n"
"        ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    proto_mul_mv_f16_f32_scaled_impl<24, 2>(args, src0, src1, dst, scales, shmem, tgpig, tiisg, sgitg);\n"
"}\n";

static struct ds4_metal_args_mul_mv make_args(uint32_t out_dim, uint32_t in_dim, uint32_t n_tok, int32_t nr0) {
    struct ds4_metal_args_mul_mv a;
    memset(&a, 0, sizeof(a));
    a.ne00 = (int32_t)in_dim;
    a.ne01 = (int32_t)out_dim;
    a.ne02 = 1;
    a.nb00 = 2u;
    a.nb01 = (uint64_t)in_dim * 2u;
    a.nb02 = a.nb01 * out_dim;
    a.nb03 = a.nb02;
    a.ne10 = (int32_t)in_dim;
    a.ne11 = (int32_t)n_tok;
    a.ne12 = 1;
    a.nb10 = sizeof(float);
    a.nb11 = (uint64_t)in_dim * sizeof(float);
    a.nb12 = (uint64_t)in_dim * n_tok * sizeof(float);
    a.nb13 = a.nb12;
    a.ne0  = (int32_t)out_dim;
    a.ne1  = (int32_t)n_tok;
    a.nr0  = nr0;
    a.r2   = 1;
    a.r3   = 1;
    return a;
}

static double dispatch_arm(
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pso,
    const struct ds4_metal_args_mul_mv *args,
    id<MTLBuffer> buf_w,
    id<MTLBuffer> buf_x,
    id<MTLBuffer> buf_y,
    id<MTLBuffer> buf_s,
    MTLSize grid,
    MTLSize tpg,
    NSUInteger shmem_bytes,
    uint32_t n_batch)
{
    @autoreleasepool {
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBytes:args length:sizeof(*args) atIndex:0];
        [enc setBuffer:buf_w offset:0 atIndex:1];
        [enc setBuffer:buf_x offset:0 atIndex:2];
        [enc setBuffer:buf_y offset:0 atIndex:3];
        [enc setBuffer:buf_s offset:0 atIndex:4];
        [enc setThreadgroupMemoryLength:shmem_bytes atIndex:0];
        for (uint32_t b = 0; b < n_batch; ++b) {
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
            if (b + 1 < n_batch) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "ERROR: CB failed: %s\n",
                    cb.error.localizedDescription.UTF8String ?: "unknown");
            exit(1);
        }
        double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
        return ((gpu_s > 0.0) ? gpu_s : (t1 - t0)) * 1000.0 / (double)n_batch;
    }
}

static int verify_zero_ulp(const char *label, const float *ref, const float *got, size_t n_elems) {
    if (memcmp(ref, got, n_elems * sizeof(float)) == 0) {
        return 1;
    }
    size_t diffs = 0;
    int32_t max_ulp = 0;
    double max_abs = 0.0;
    size_t first_idx = 0;
    for (size_t i = 0; i < n_elems; ++i) {
        uint32_t ur, ug;
        memcpy(&ur, &ref[i], 4);
        memcpy(&ug, &got[i], 4);
        if (ur != ug) {
            if (diffs == 0) first_idx = i;
            ++diffs;
            double ad = fabs((double)ref[i] - (double)got[i]);
            if (ad > max_abs) max_abs = ad;
            int32_t u = (int32_t)(ur > ug ? ur - ug : ug - ur);
            if (u > max_ulp) max_ulp = u;
        }
    }
    uint32_t ur0, ug0;
    memcpy(&ur0, &ref[first_idx], 4);
    memcpy(&ug0, &got[first_idx], 4);
    fprintf(stderr, "FAIL [0-ULP] %s: %zu/%zu elems differ; first idx=%zu ref=%g (0x%08x) got=%g (0x%08x) max_ulp=%d max_abs=%g\n",
            label, diffs, n_elems, first_idx, ref[first_idx], ur0, got[first_idx], ug0, max_ulp, max_abs);
    return 0;
}

int main(int argc, char **argv) {
    const char *metallib_path = "bin/myds.metallib";
    uint32_t n_batch = 4;
    uint32_t n_runs  = 7;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) n_batch = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) n_runs = (uint32_t)atoi(argv[++i]);
        else if (argv[i][0] != '-') metallib_path = argv[i];
        else {
            fprintf(stderr, "usage: %s [bin/myds.metallib] [--batch 4] [--runs 7]\n", argv[0]);
            return 2;
        }
    }
    if (!n_batch || !n_runs || n_runs > 101) {
        fprintf(stderr, "invalid parameters\n");
        return 2;
    }

    const uint32_t N_TOK   = 2048u;
    const uint32_t IN_DIM  = 4u * 5120u; /* 20,480 */
    const uint32_t OUT_DIM = 24u;
    const uint32_t NSG     = 8u;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "ERROR: no Metal device\n"); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) { fprintf(stderr, "ERROR: no command queue\n"); return 1; }

        NSError *err = nil;
        NSURL *lib_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        id<MTLLibrary> prod_lib = [dev newLibraryWithURL:lib_url error:&err];
        if (!prod_lib) {
            fprintf(stderr, "ERROR: cannot load %s: %s\n",
                    metallib_path, err.localizedDescription.UTF8String ?: "unknown");
            return 1;
        }

        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeFast;
        id<MTLLibrary> proto_lib = [dev newLibraryWithSource:kProtoShaderSource options:opts error:&err];
        if (!proto_lib) {
            fprintf(stderr, "ERROR: prototype shader compile failed: %s\n",
                    err.localizedDescription.UTF8String ?: "unknown");
            return 1;
        }

        MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
        short short_nsg = (short)NSG;
        [fc setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];

        id<MTLFunction> fn_prod = [prod_lib newFunctionWithName:@"kernel_mul_mv_f16_f32_4_b4_scaled"
                                                 constantValues:fc error:&err];
        if (!fn_prod) {
            fprintf(stderr, "ERROR: missing kernel_mul_mv_f16_f32_4_b4_scaled in %s: %s\n",
                    metallib_path, err.localizedDescription.UTF8String ?: "unknown");
            return 1;
        }
        id<MTLComputePipelineState> pso_prod = [dev newComputePipelineStateWithFunction:fn_prod error:&err];
        if (!pso_prod) { fprintf(stderr, "ERROR: prod PSO failed\n"); return 1; }

        id<MTLFunction> fn_b1 = [proto_lib newFunctionWithName:@"kernel_proto_hc_mix_nr24_b1"
                                                constantValues:fc error:&err];
        id<MTLComputePipelineState> pso_b1 = fn_b1 ? [dev newComputePipelineStateWithFunction:fn_b1 error:&err] : nil;
        if (!pso_b1) {
            fprintf(stderr, "ERROR: NR0=24 B=1 PSO failed: %s\n",
                    err.localizedDescription.UTF8String ?: "unknown");
            return 1;
        }

        id<MTLFunction> fn_b2 = [proto_lib newFunctionWithName:@"kernel_proto_hc_mix_nr24_b2"
                                                constantValues:fc error:&err];
        id<MTLComputePipelineState> pso_b2 = fn_b2 ? [dev newComputePipelineStateWithFunction:fn_b2 error:&err] : nil;
        if (!pso_b2) {
            fprintf(stderr, "ERROR: NR0=24 B=2 PSO failed: %s\n",
                    err.localizedDescription.UTF8String ?: "unknown");
            return 1;
        }

        const size_t x_bytes = (size_t)N_TOK * IN_DIM * sizeof(float);       /* 167,772,160 B */
        const size_t w_bytes = (size_t)OUT_DIM * IN_DIM * sizeof(__fp16);    /*     983,040 B */
        const size_t s_bytes = (size_t)N_TOK * sizeof(float);                /*       8,192 B */
        const size_t y_bytes = (size_t)N_TOK * OUT_DIM * sizeof(float);      /*     196,608 B */

        id<MTLBuffer> buf_x      = [dev newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_w      = [dev newBufferWithLength:w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_s      = [dev newBufferWithLength:s_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_y_ref  = [dev newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_y_b1   = [dev newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_y_b2   = [dev newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
        if (!buf_x || !buf_w || !buf_s || !buf_y_ref || !buf_y_b1 || !buf_y_b2) {
            fprintf(stderr, "ERROR: buffer allocation failed\n");
            return 1;
        }

        /* Deterministic non-trivial data: varied signs, magnitudes, and per-token inv-RMS scales */
        float  *xp = (float  *)buf_x.contents;
        __fp16 *wp = (__fp16 *)buf_w.contents;
        float  *sp = (float  *)buf_s.contents;
        for (size_t i = 0; i < (size_t)OUT_DIM * IN_DIM; ++i) wp[i] = (__fp16)rng_signed(0.25f);
        for (uint32_t t = 0; t < N_TOK; ++t) sp[t] = 0.05f + 0.0005f * (float)(t & 255u) + fabsf(rng_signed(0.02f));
        /* Fast deterministic fill for 41.94M floats */
        uint32_t lcg = 0x12345678u;
        for (size_t i = 0; i < (size_t)N_TOK * IN_DIM; ++i) {
            lcg = lcg * 1664525u + 1013904223u;
            xp[i] = ((float)(int32_t)(lcg >> 16) - 32768.0f) * (1.0f / 32768.0f);
        }

        const uint32_t gpu_cores = query_gpu_core_count();
        const int apple_fam = detect_apple_gpu_family(dev);
        printf("=== [P-H1] Step 1+6 HC-Mix F16 GEMV Hardware Gate ===\n");
        printf("device=%s  apple_family=Apple%d  gpu_cores=%u  metallib=%s\n",
               dev.name.UTF8String, apple_fam, gpu_cores, metallib_path);
        printf("shape: n_tok=%u x in_dim=%u (%.2f MB F32) * %u x %u (%.2f KB F16) -> %u x %u (%.2f KB F32)\n",
               N_TOK, IN_DIM, (double)x_bytes / 1e6,
               OUT_DIM, IN_DIM, (double)w_bytes / 1024.0,
               N_TOK, OUT_DIM, (double)y_bytes / 1024.0);
        printf("PSO maxTotalThreadsPerThreadgroup: prod=%lu  nr24_b1=%lu  nr24_b2=%lu  (tpg=256)\n\n",
               (unsigned long)pso_prod.maxTotalThreadsPerThreadgroup,
               (unsigned long)pso_b1.maxTotalThreadsPerThreadgroup,
               (unsigned long)pso_b2.maxTotalThreadsPerThreadgroup);

        MTLSize tpg = MTLSizeMake(32, NSG, 1);
        const NSUInteger shmem_prod = 32u * 2u  * 4u * sizeof(float); /* 1024 B */
        const NSUInteger shmem_b1   = 32u * 24u * 1u * sizeof(float); /* 3072 B */
        const NSUInteger shmem_b2   = 32u * 24u * 2u * sizeof(float); /* 6144 B */

        /* 1. Strict 0-ULP Verification on n_tok=2048 and odd tail n_tok=123 */
        const uint32_t check_toks[2] = { N_TOK, 123u };
        for (int ci = 0; ci < 2; ++ci) {
            const uint32_t nt = check_toks[ci];
            const size_t n_out = (size_t)nt * OUT_DIM;
            memset(buf_y_ref.contents, 0xAA, n_out * sizeof(float));
            memset(buf_y_b1.contents,  0x55, n_out * sizeof(float));
            memset(buf_y_b2.contents,  0x55, n_out * sizeof(float));

            struct ds4_metal_args_mul_mv a_prod = make_args(OUT_DIM, IN_DIM, nt, 2);
            struct ds4_metal_args_mul_mv a_nr24 = make_args(OUT_DIM, IN_DIM, nt, 24);

            (void)dispatch_arm(queue, pso_prod, &a_prod, buf_w, buf_x, buf_y_ref, buf_s,
                               MTLSizeMake((OUT_DIM + 1u) / 2u, (nt + 3u) / 4u, 1), tpg, shmem_prod, 1);
            (void)dispatch_arm(queue, pso_b1, &a_nr24, buf_w, buf_x, buf_y_b1, buf_s,
                               MTLSizeMake(1, nt, 1), tpg, shmem_b1, 1);
            (void)dispatch_arm(queue, pso_b2, &a_nr24, buf_w, buf_x, buf_y_b2, buf_s,
                               MTLSizeMake(1, (nt + 1u) / 2u, 1), tpg, shmem_b2, 1);

            const float *yref = (const float *)buf_y_ref.contents;
            const float *yb1  = (const float *)buf_y_b1.contents;
            const float *yb2  = (const float *)buf_y_b2.contents;

            /* CPU FP64 spot-check on row 0..3 of token 0 and last token */
            double max_cpu_err = 0.0;
            const uint32_t spot_t[2] = { 0u, nt - 1u };
            for (int st = 0; st < 2; ++st) {
                uint32_t t = spot_t[st];
                for (uint32_t r = 0; r < 4; ++r) {
                    double acc = 0.0;
                    const float *xt = xp + (size_t)t * IN_DIM;
                    const __fp16 *wr = wp + (size_t)r * IN_DIM;
                    for (uint32_t k = 0; k < IN_DIM; ++k) {
                        acc += (double)(float)wr[k] * (double)(xt[k] * sp[t]);
                    }
                    double err_v = fabs(acc - (double)yref[(size_t)t * OUT_DIM + r]);
                    if (err_v > max_cpu_err) max_cpu_err = err_v;
                }
            }

            if (!verify_zero_ulp("NR0=24 B=1 vs prod", yref, yb1, n_out)) return 1;
            if (!verify_zero_ulp("NR0=24 B=2 vs prod", yref, yb2, n_out)) return 1;
            printf("[0-ULP PASS] n_tok=%-4u: memcmp==0 for BOTH NR0=24 B=1 and B=2 (%zu floats; CPU FP64 spot max_err=%.3e)\n",
                   nt, n_out, max_cpu_err);
        }

        /* 2. Warmup + 7-run interleaved GPU timing on full n_tok=2048 */
        struct ds4_metal_args_mul_mv a_prod = make_args(OUT_DIM, IN_DIM, N_TOK, 2);
        struct ds4_metal_args_mul_mv a_nr24 = make_args(OUT_DIM, IN_DIM, N_TOK, 24);
        MTLSize grid_prod = MTLSizeMake(12, 512, 1);
        MTLSize grid_b1   = MTLSizeMake(1, 2048, 1);
        MTLSize grid_b2   = MTLSizeMake(1, 1024, 1);

        for (int w = 0; w < 2; ++w) {
            (void)dispatch_arm(queue, pso_prod, &a_prod, buf_w, buf_x, buf_y_ref, buf_s, grid_prod, tpg, shmem_prod, n_batch);
            (void)dispatch_arm(queue, pso_b1,   &a_nr24, buf_w, buf_x, buf_y_b1,  buf_s, grid_b1,   tpg, shmem_b1,   n_batch);
            (void)dispatch_arm(queue, pso_b2,   &a_nr24, buf_w, buf_x, buf_y_b2,  buf_s, grid_b2,   tpg, shmem_b2,   n_batch);
        }

        double ms_prod[101], ms_b1[101], ms_b2[101];
        for (uint32_t r = 0; r < n_runs; ++r) {
            ms_prod[r] = dispatch_arm(queue, pso_prod, &a_prod, buf_w, buf_x, buf_y_ref, buf_s, grid_prod, tpg, shmem_prod, n_batch);
            ms_b1[r]   = dispatch_arm(queue, pso_b1,   &a_nr24, buf_w, buf_x, buf_y_b1,  buf_s, grid_b1,   tpg, shmem_b1,   n_batch);
            ms_b2[r]   = dispatch_arm(queue, pso_b2,   &a_nr24, buf_w, buf_x, buf_y_b2,  buf_s, grid_b2,   tpg, shmem_b2,   n_batch);
        }

        qsort(ms_prod, n_runs, sizeof(double), cmp_double);
        qsort(ms_b1,   n_runs, sizeof(double), cmp_double);
        qsort(ms_b2,   n_runs, sizeof(double), cmp_double);

        const double med_prod = ms_prod[n_runs / 2];
        const double med_b1   = ms_b1[n_runs / 2];
        const double med_b2   = ms_b2[n_runs / 2];

        /* Minimum physical bytes per dispatch = input + weights + scales + output */
        const double min_bytes = (double)(x_bytes + w_bytes + s_bytes + y_bytes);
        const double prod_logical_bytes = (double)(12ull * x_bytes + w_bytes + s_bytes + y_bytes);

        printf("\n=== GPU Timing (batch=%u/CB, %u interleaved runs, per-dispatch GPU ms) ===\n", n_batch, n_runs);
        printf("%-30s  %14s  %9s  %9s  %9s  %10s  %8s\n",
               "Arm", "Grid", "med_ms", "min_ms", "max_ms", "eff_GB/s", "speedup");
        printf("--------------------------------------------------------------------------------------------------------\n");
        printf("%-30s  %14s  %9.3f  %9.3f  %9.3f  %10.1f  %7.2fx\n",
               "prod_b4 (NR0=2, B=4)", "(12, 512, 1)",
               med_prod, ms_prod[0], ms_prod[n_runs - 1],
               (min_bytes / (med_prod * 1e-3)) / 1e9, 1.0);
        printf("%-30s  %14s  %9.3f  %9.3f  %9.3f  %10.1f  %7.2fx\n",
               "proto_nr24_b1 (NR0=24, B=1)", "(1, 2048, 1)",
               med_b1, ms_b1[0], ms_b1[n_runs - 1],
               (min_bytes / (med_b1 * 1e-3)) / 1e9, med_prod / med_b1);
        printf("%-30s  %14s  %9.3f  %9.3f  %9.3f  %10.1f  %7.2fx\n",
               "proto_nr24_b2 (NR0=24, B=2)", "(1, 1024, 1)",
               med_b2, ms_b2[0], ms_b2[n_runs - 1],
               (min_bytes / (med_b2 * 1e-3)) / 1e9, med_prod / med_b2);

        const double best_ms = med_b1 < med_b2 ? med_b1 : med_b2;
        const char *best_name = med_b1 < med_b2 ? "NR0=24 B=1" : "NR0=24 B=2";
        const double saved_ms = med_prod - best_ms;
        printf("\nImplied prod logical traffic = %.1f MB (12x input) vs min physical = %.1f MB (1x input)\n",
               prod_logical_bytes / 1e6, min_bytes / 1e6);
        printf("Best single-read prototype: %s (%.3f ms vs prod %.3f ms => %.2fx speedup, -%.3f ms/dispatch)\n",
               best_name, best_ms, med_prod, med_prod / best_ms, saved_ms);
        printf("Projected Step 1+6 savings at this delta: cold_32k (674 disp) = -%.3f s | cold_128k (2629 disp) = -%.3f s\n",
               saved_ms * 674.0 / 1000.0, saved_ms * 2629.0 / 1000.0);
        printf("PASS: [P-H1] strict 0-ULP verified and GPU timing complete.\n");
    }
    return 0;
}
