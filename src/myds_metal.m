#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "myds_metal.h"
#include "myds_log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <TargetConditionals.h>

struct myds_metal_buffer {
    id<MTLBuffer> mtl_buf;
    size_t        size;
    size_t        offset;
    void         *ptr;
    bool          is_view;
};

static id<MTLDevice>               g_device = nil;
static id<MTLCommandQueue>          g_queue = nil;
static id<MTLLibrary>              g_library = nil;
static id<MTLComputePipelineState> g_pso_rms_norm = nil;
static id<MTLComputePipelineState> g_pso_v41_norm_plain = nil;
static id<MTLComputePipelineState> g_pso_v41_bf16 = nil;
static id<MTLComputePipelineState> g_pso_v41_hc_split = nil;
static id<MTLComputePipelineState> g_pso_v41_hc_sum = nil;
static id<MTLComputePipelineState> g_pso_v41_hc_expand = nil;
static id<MTLComputePipelineState> g_pso_v41_engram_add = nil;
static id<MTLComputePipelineState> g_pso_swiglu = nil;
static id<MTLComputePipelineState> g_pso_add = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_f32 = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_f16 = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_q4_K = nil;
static id<MTLComputePipelineState> g_pso_softmax = nil;
static id<MTLComputePipelineState> g_pso_moe_mv_id_iq2_xxs = nil;
static id<MTLComputePipelineState> g_pso_moe_mv_id_q2_k = nil;
static id<MTLComputePipelineState> g_pso_moe_swiglu_weight = nil;
static id<MTLComputePipelineState> g_pso_moe_sum6 = nil;
static id<MTLComputePipelineState> g_pso_moe_pair_swiglu_iq2_xxs = nil;
static id<MTLComputePipelineState> g_pso_moe_sum6_q2_k = nil;
static id<MTLComputePipelineState> g_pso_moe_mm_map0_ne20_6 = nil;
static id<MTLComputePipelineState> g_pso_moe_mm_iq2_pair_swiglu_f16 = nil;
static id<MTLComputePipelineState> g_pso_moe_mm_q2_k_f16 = nil;
static id<MTLComputePipelineState> g_pso_moe_mm_q2_k_f32 = nil;
static id<MTLBuffer>               g_moe_id_map_buffer = nil;
static NSUInteger                  g_moe_id_map_bytes = 0;
static id<MTLComputePipelineState> g_pso_attn_out_low_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q8_0_bco = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f16 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f16_bco = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f32 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f32_bco = nil;
static id<MTLComputePipelineState> g_pso_compute_copy_u8 = nil;
static id<MTLComputePipelineState> g_pso_compute_copy_u32 = nil;
static id<MTLComputePipelineState> g_pso_compute_copy_u32x4 = nil;
static id<MTLSharedEvent>          g_shared_event = nil;
static uint64_t                    g_shared_event_val = 0;
static bool                        g_initialized = false;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_native_pipelines = nil;
struct native_kernel_cache_entry {
    char name[64];
    id<MTLComputePipelineState> pso;
};
static struct native_kernel_cache_entry g_native_pso_cache[64];
static size_t g_native_pso_cache_count = 0;


/* Batch command buffer state */
static id<MTLCommandBuffer>         g_batch_cb = nil;
static id<MTLComputeCommandEncoder> g_batch_enc = nil;
/* An event can wake the CPU before the rest of its command buffer finishes. */
static NSMutableArray<id<MTLCommandBuffer>> *g_inflight_cbs = nil;
static uint64_t                     g_batch_commit_count = 0;
static int                          g_auto_barrier = 1;

/* ══════════════════════════════════════════════════════════════════════════
 * Residency keep-alive —— llama.cpp PR #17766 (ggerganov) 的移植
 * ══════════════════════════════════════════════════════════════════════════
 *
 * 上游结论（PR 正文原文）：
 *   "something changed in MacOS recently ... the memory wiring/unwiring
 *    (a.k.a. throttling) after 1 second of being idle is back."
 *   "Just attaching the residency sets to the Metal queue is not enough.
 *    Ended up implementing a background thread that periodically calls
 *    requestResidency() in order to keep the memory buffers wired."
 *
 * 本舰队的实测（macstudio, 151 GB, 2026-09-29）：
 *   无驻留            空闲 170 s 后首请求 5.56 s（热态 0.11 s）
 *   只挂 queue        6.99 s → 每次请求都付 ~2.9 s（= 心跳已停的等价状态）
 *   挂 queue + 5 ms 心跳   0.19 s
 *   心跳保活 30 s < 空闲   2.92 s（因果对照：心跳停 ⇒ 又冷）
 *   → 详见 docs/verification/2026-09-29-per-request-latency-root-cause.md §14
 *
 * 开关：
 *   MYDS_NO_RESIDENCY_HB=1        整体关闭（应急回退，无需重编译）
 *   MYDS_RESIDENCY_KEEP_ALIVE_S   空闲多久后心跳停止（默认 180）
 *   MYDS_RESIDENCY_HB_MS          心跳周期（默认 5 ms，与上游一致）
 * ══════════════════════════════════════════════════════════════════════════ */
static id<MTLResidencySet> g_res_set = nil;
static NSLock             *g_res_lock = nil;
static int                 g_res_count = 0;
static int                 g_res_keep_alive_s = 180;
static int                 g_res_hb_ms = 5;
static bool                g_res_off = false;   /* 环境变量或建集失败 */
static bool                g_res_ready = false; /* 已 requestResidency + 挂队 */
static atomic_bool         g_res_stop = false;
static atomic_int          g_res_ticks = 0;     /* 剩余心跳次数 */

static bool myds_metal_residency_disabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("MYDS_NO_RESIDENCY_HB") ? 1 : 0;
    return cached == 1;
}

/* 每个按字节包裹的模型 view 登记进同一个 residency set。
 * 必须在 g_res_ready 之前完成（即模型加载期间）。 */
static void myds_metal_residency_register(id<MTLBuffer> buf) {
    if (!buf || g_res_off || myds_metal_residency_disabled()) return;
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (!g_res_lock) g_res_lock = [[NSLock alloc] init];
        [g_res_lock lock];
        if (!g_res_set) {
            const char *ka = getenv("MYDS_RESIDENCY_KEEP_ALIVE_S");
            if (ka && atoi(ka) > 0) g_res_keep_alive_s = atoi(ka);
            const char *ms = getenv("MYDS_RESIDENCY_HB_MS");
            if (ms && atoi(ms) > 0) g_res_hb_ms = atoi(ms);
            MTLResidencySetDescriptor *d = [[MTLResidencySetDescriptor alloc] init];
            d.label = @"myds_model";
            d.initialCapacity = 4096;
            NSError *err = nil;
            g_res_set = [g_device newResidencySetWithDescriptor:d error:&err];
            if (!g_res_set) {
                MYDS_LOG_E("residency set creation failed (%s) — keep-alive disabled",
                           [[err localizedDescription] UTF8String]);
                g_res_off = true;
                [g_res_lock unlock];
                return;
            }
        }
        [g_res_set addAllocation:buf];
        [g_res_set commit];
        g_res_count++;
        [g_res_lock unlock];
    }
#endif
}

void myds_metal_residency_activate(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_res_off || g_res_ready || !g_res_set ||
            myds_metal_residency_disabled()) return;

        [g_res_lock lock];
        if (g_res_ready) { [g_res_lock unlock]; return; }

        /* ① 一次性把「逐页校验」付在这里（加载期），而不是留给第一个请求。
         *    实测这就是那 ~2.9 s。 */
        [g_res_set requestResidency];
        /* ② 挂到队列：此后每次提交都自带该驻留集（上游：单靠这一步不够） */
        if (g_queue && [g_queue respondsToSelector:@selector(addResidencySet:)]) {
            [g_queue addResidencySet:g_res_set];
        }
        g_res_ready = true;
        atomic_store_explicit(&g_res_ticks,
                              g_res_keep_alive_s * (1000 / g_res_hb_ms),
                              memory_order_relaxed);
        [g_res_lock unlock];

        /* ③ llama.cpp #26082：从未被任何 GPU 工作引用过的 residency set
         *    会直到进程退出才释放其 wired 内存。一次极小的 blit 即可规避。 */
        if (g_queue && g_device) {
            id<MTLBuffer> tmp = [g_device newBufferWithLength:1
                                                     options:MTLResourceStorageModePrivate];
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            if (tmp && cb) {
                id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
                [enc fillBuffer:tmp range:NSMakeRange(0, 1) value:0];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }

        MYDS_LOG_I("Residency keep-alive: %d allocations, queue-attached, keep_alive=%ds, period=%dms",
                   g_res_count, g_res_keep_alive_s, g_res_hb_ms);

        /* ④ 心跳线程：周期性 requestResidency，把 wired 状态钉住。
         *    （等价于上游循环；不占用主线程，不产生 GPU 负载） */
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
            while (!atomic_load_explicit(&g_res_stop, memory_order_relaxed)) {
                if (atomic_load_explicit(&g_res_ticks, memory_order_relaxed) > 0) {
                    [g_res_lock lock];
                    [g_res_set requestResidency];
                    [g_res_lock unlock];
                    atomic_fetch_sub_explicit(&g_res_ticks, 1, memory_order_relaxed);
                }
                usleep((useconds_t)g_res_hb_ms * 1000);
            }
        });
    }
#endif
}

/* 每次向 GPU 提交真实工作时重置保活倒计时
 * （对应上游 ggml_metal_device_rsets_keep_alive） */
void myds_metal_residency_keep_alive(void) {
    if (g_res_ready) {
        atomic_store_explicit(&g_res_ticks,
                              g_res_keep_alive_s * (1000 / g_res_hb_ms),
                              memory_order_relaxed);
    }
}

void myds_metal_residency_stop(void) {
    atomic_store_explicit(&g_res_stop, true, memory_order_relaxed);
}

void myds_metal_set_auto_barrier(int enable) {
    g_auto_barrier = enable;
}

static id<MTLComputeCommandEncoder> get_compute_encoder(id<MTLCommandBuffer> cb) {
    if (cb == g_batch_cb && g_batch_enc) {
        return g_batch_enc;
    }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (cb == g_batch_cb) {
        g_batch_enc = enc;
    }
    return enc;
}

void myds_metal_barrier(void) {
    if (g_batch_cb) {
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(g_batch_cb);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
}

static void close_compute_encoder(void) {
    if (g_batch_enc) {
        [g_batch_enc endEncoding];
        g_batch_enc = nil;
    }
}

static int command_buffer_result(id<MTLCommandBuffer> cb) {
    if (cb && cb.status == MTLCommandBufferStatusCompleted) return 1;
    fprintf(stderr, "myds_metal: command buffer failed: %s\n",
            cb.error ? [[cb.error localizedDescription] UTF8String] : "command buffer did not complete");
    return 0;
}

static int wait_inflight_commands(void) {
    int ok = 1;
    for (id<MTLCommandBuffer> cb in g_inflight_cbs) {
        [cb waitUntilCompleted];
        if (!command_buffer_result(cb)) ok = 0;
    }
    [g_inflight_cbs removeAllObjects];
    return ok;
}

static int wait_command_buffer(id<MTLCommandBuffer> cb) {
    [cb waitUntilCompleted];
    int ok = command_buffer_result(cb);
    /* Keep errors from earlier asynchronous submissions observable, too. */
    if (!wait_inflight_commands()) ok = 0;
    return ok;
}

static const char *g_metal_base_preamble =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#ifndef QK_K\n"
"#define QK_K 256\n"
"#endif\n"
"#define N_SIMDWIDTH 32\n"
"#ifndef N_R0_Q8_0\n"
"#define N_R0_Q8_0 2\n"
"#endif\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"struct block_q8_K {\n"
"    float d;\n"
"    int8_t qs[QK_K];\n"
"    int16_t bsums[QK_K / 16];\n"
"};\n"
"\n";

int myds_metal_init(const char *shader_dir) {
    if (g_initialized) return 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "myds_metal: Metal device not available on this system\n");
            return 0;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "myds_metal: failed to create Metal command queue\n");
            g_device = nil;
            return 0;
        }

        const char *dir = shader_dir ? shader_dir : "metal";
        NSArray<NSString *> *shader_files = @[
            @"flash_attn.metal",
            @"dense.metal",
            @"moe.metal",
            @"dsv4_hc.metal",
            @"unary.metal",
            @"dsv4_kv.metal",
            @"dsv41.metal",
            @"myds_v41_moe.metal",
            @"dsv4_rope.metal",
            @"dsv4_misc.metal",
            @"argsort.metal",
            @"cpy.metal",
            @"concat.metal",
            @"get_rows.metal",
            @"sum_rows.metal",
            @"softmax.metal",
            @"repeat.metal",
            @"glu.metal",
            @"norm.metal",
            @"bin.metal",
            @"set_rows.metal"
        ];

        NSMutableString *full_source = [NSMutableString stringWithUTF8String:g_metal_base_preamble];
        NSFileManager *fm = [NSFileManager defaultManager];

        for (NSString *filename in shader_files) {
            NSString *path = [NSString stringWithFormat:@"%s/%@", dir, filename];
            if (![fm fileExistsAtPath:path]) {
                path = [NSString stringWithFormat:@"./metal/%@", filename];
            }
            if (![fm fileExistsAtPath:path]) {
                path = [NSString stringWithFormat:@"~/Projects/myds/metal/%@", filename];
            }

            NSError *read_err = nil;
            NSString *content = [NSString stringWithContentsOfFile:path
                                                          encoding:NSUTF8StringEncoding
                                                             error:&read_err];
            if (!content) {
                fprintf(stderr, "myds_metal: failed reading shader '%s': %s\n",
                        [path UTF8String], [[read_err localizedDescription] UTF8String]);
                return 0;
            }
            [full_source appendFormat:@"\n#line 1 \"%@\"\n%@", filename, content];
        }

        [full_source appendString:@"\n"
            "struct ds4_compute_copy_args {\n"
            "    uint64_t src_offset;\n"
            "    uint64_t dst_offset;\n"
            "    uint64_t count;\n"
            "};\n"
            "kernel void kernel_compute_copy_u8(\n"
            "    device const uchar *src [[buffer(0)]],\n"
            "    device uchar *dst [[buffer(1)]],\n"
            "    constant ds4_compute_copy_args &args [[buffer(2)]],\n"
            "    uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid < args.count) {\n"
            "        (dst + args.dst_offset)[gid] = (src + args.src_offset)[gid];\n"
            "    }\n"
            "}\n"
            "kernel void kernel_compute_copy_u32(\n"
            "    device const uchar *src [[buffer(0)]],\n"
            "    device uchar *dst [[buffer(1)]],\n"
            "    constant ds4_compute_copy_args &args [[buffer(2)]],\n"
            "    uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid < args.count) {\n"
            "        device const uint *s = (device const uint *)(src + args.src_offset);\n"
            "        device uint *d = (device uint *)(dst + args.dst_offset);\n"
            "        d[gid] = s[gid];\n"
            "    }\n"
            "}\n"
            "kernel void kernel_compute_copy_u32x4(\n"
            "    device const uchar *src [[buffer(0)]],\n"
            "    device uchar *dst [[buffer(1)]],\n"
            "    constant ds4_compute_copy_args &args [[buffer(2)]],\n"
            "    uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid < args.count) {\n"
            "        device const uint4 *s = (device const uint4 *)(src + args.src_offset);\n"
            "        device uint4 *d = (device uint4 *)(dst + args.dst_offset);\n"
            "        d[gid] = s[gid];\n"
            "    }\n"
            "}\n"
        ];

        MTLCompileOptions *options = [MTLCompileOptions new];
        /* Match the pinned reference defaults for standalone HC sigmoid and
         * normalization. Existing generic operators do not use these macros. */
        options.preprocessorMacros = @{@"DS4_METAL_HC_STABLE": @"1",
                                       @"DS4_METAL_NORM_RSQRT_DISABLE": @"1"};
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }

        NSError *compile_err = nil;
        g_library = [g_device newLibraryWithSource:full_source options:options error:&compile_err];
        if (!g_library) {
            fprintf(stderr, "myds_metal: Metal shader compilation failed: %s\n",
                    [[compile_err localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        /* 1. Build RMSNorm pipeline */
        id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_rms_norm_mul_f32_4"];
        if (fn) {
            g_pso_rms_norm = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }
        fn = [g_library newFunctionWithName:@"kernel_rms_norm_f32_4"];
        if (fn) g_pso_v41_norm_plain = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_dsv41_bf16_linear"];
        if (fn) g_pso_v41_bf16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_dsv4_hc_split_sinkhorn"];
        if (fn) g_pso_v41_hc_split = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum"];
        if (fn) g_pso_v41_hc_sum = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_dsv4_hc_expand4"];
        if (fn) g_pso_v41_hc_expand = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_dsv41_engram_add"];
        if (fn) g_pso_v41_engram_add = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_compute_copy_u8"];
        if (fn) g_pso_compute_copy_u8 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_compute_copy_u32"];
        if (fn) g_pso_compute_copy_u32 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_compute_copy_u32x4"];
        if (fn) g_pso_compute_copy_u32x4 = [g_device newComputePipelineStateWithFunction:fn error:nil];

        /* 2. Build SwiGLU pipeline */
        fn = [g_library newFunctionWithName:@"kernel_swiglu_flat_f32"];
        if (fn) {
            g_pso_swiglu = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 3. Build Residual Add pipeline */
        fn = [g_library newFunctionWithName:@"kernel_add2_f32"];
        if (fn) {
            g_pso_add = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 4. Build Softmax pipeline */
        fn = [g_library newFunctionWithName:@"kernel_soft_max_f32"];
        if (fn) {
            g_pso_softmax = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 5. Build Q8_0 MatVec pipeline with function constants */
        MTLFunctionConstantValues *mv_constants = [[MTLFunctionConstantValues alloc] init];
        short nsg = 4;
        [mv_constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_q8_0_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_mul_mv_q8_0 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 6. Build F32 MatVec pipeline */
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_f32_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_mul_mv_f32 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 7. Build F16 MatVec pipeline */
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_f16_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_mul_mv_f16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 8. Build Q4_K MatVec pipeline */
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_q4_K_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_mul_mv_q4_K = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 9. Build Routed MoE pipelines */
        MTLFunctionConstantValues *moe_mv_id_constants = [[MTLFunctionConstantValues alloc] init];
        short moe_id_nsg = 2;
        [moe_mv_id_constants setConstantValue:&moe_id_nsg type:MTLDataTypeShort atIndex:600];

        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_f32" constantValues:moe_mv_id_constants error:nil];
        if (fn) {
            g_pso_moe_mv_id_iq2_xxs = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_f32" constantValues:moe_mv_id_constants error:nil];
        if (fn) {
            g_pso_moe_mv_id_q2_k = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_dsv4_moe_swiglu_weight"];
        if (fn) {
            g_pso_moe_swiglu_weight = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_dsv4_moe_sum6_f32"];
        if (fn) {
            g_pso_moe_sum6 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        NSError *pso_err = nil;
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32" constantValues:moe_mv_id_constants error:&pso_err];
        if (fn) {
            g_pso_moe_pair_swiglu_iq2_xxs = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_pair_swiglu_iq2_xxs) {
                printf("myds_metal: kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 PSO error: %s\n", [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 fn error: %s\n", [[pso_err localizedDescription] UTF8String]);
        }

        pso_err = nil;
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32" constantValues:moe_mv_id_constants error:&pso_err];
        if (fn) {
            g_pso_moe_sum6_q2_k = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_sum6_q2_k) {
                printf("myds_metal: kernel_mul_mv_id_q2_K_sum6_f32 PSO error: %s\n", [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: kernel_mul_mv_id_q2_K_sum6_f32 fn error: %s\n", [[pso_err localizedDescription] UTF8String]);
        }

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_id_map0_ne20_6"];
        if (fn) {
            g_pso_moe_mm_map0_ne20_6 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16"];
        if (fn) {
            g_pso_moe_mm_iq2_pair_swiglu_f16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 11. Build Batched MatMul (GEMM) pipelines with function constants 700/701 */
        MTLFunctionConstantValues *mm_c_bco0 = [[MTLFunctionConstantValues alloc] init];
        bool b_false = false;
        bool b_true = true;
        [mm_c_bco0 setConstantValue:&b_false type:MTLDataTypeBool atIndex:700];
        [mm_c_bco0 setConstantValue:&b_false type:MTLDataTypeBool atIndex:701];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_id_q2_K_f16" constantValues:mm_c_bco0 error:nil];
        if (fn) {
            g_pso_moe_mm_q2_k_f16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_id_q2_K_f32" constantValues:mm_c_bco0 error:nil];
        if (fn) {
            g_pso_moe_mm_q2_k_f32 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_dsv4_attn_out_low_q8_0_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_attn_out_low_q8_0 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        MTLFunctionConstantValues *mm_c_bco1 = [[MTLFunctionConstantValues alloc] init];
        [mm_c_bco1 setConstantValue:&b_false type:MTLDataTypeBool atIndex:700];
        [mm_c_bco1 setConstantValue:&b_true type:MTLDataTypeBool atIndex:701];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_q8_0_f32" constantValues:mm_c_bco0 error:nil];
        if (fn) g_pso_mul_mm_q8_0 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_mul_mm_q8_0_f32" constantValues:mm_c_bco1 error:nil];
        if (fn) g_pso_mul_mm_q8_0_bco = [g_device newComputePipelineStateWithFunction:fn error:nil];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f16_f32" constantValues:mm_c_bco0 error:nil];
        if (fn) g_pso_mul_mm_f16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f16_f32" constantValues:mm_c_bco1 error:nil];
        if (fn) g_pso_mul_mm_f16_bco = [g_device newComputePipelineStateWithFunction:fn error:nil];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f32_f32_full" constantValues:mm_c_bco0 error:nil];
        if (fn) g_pso_mul_mm_f32 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f32_f32_full" constantValues:mm_c_bco1 error:nil];
        if (fn) g_pso_mul_mm_f32_bco = [g_device newComputePipelineStateWithFunction:fn error:nil];

        g_initialized = true;
        printf("myds_metal: Initialized successfully on GPU: %s\n", [[g_device name] UTF8String]);
        return 1;
    }
}

void myds_metal_cleanup(void) {
    if (!g_initialized) return;
    (void)myds_metal_synchronize();
    g_pso_rms_norm = nil;
    g_pso_v41_norm_plain = nil;
    g_pso_v41_bf16 = nil;
    g_pso_v41_hc_split = nil;
    g_pso_v41_hc_sum = nil;
    g_pso_v41_hc_expand = nil;
    g_pso_v41_engram_add = nil;
    g_pso_swiglu = nil;
    g_pso_add = nil;
    g_pso_mul_mv_q8_0 = nil;
    g_pso_mul_mv_f32 = nil;
    g_pso_mul_mv_f16 = nil;
    g_pso_mul_mv_q4_K = nil;
    g_pso_softmax = nil;
    g_pso_moe_mv_id_iq2_xxs = nil;
    g_pso_moe_mv_id_q2_k = nil;
    g_pso_moe_swiglu_weight = nil;
    g_pso_moe_sum6 = nil;
    g_pso_moe_pair_swiglu_iq2_xxs = nil;
    g_pso_moe_sum6_q2_k = nil;
    g_pso_moe_mm_map0_ne20_6 = nil;
    g_pso_moe_mm_iq2_pair_swiglu_f16 = nil;
    g_pso_moe_mm_q2_k_f16 = nil;
    g_pso_moe_mm_q2_k_f32 = nil;
    g_moe_id_map_buffer = nil;
    g_moe_id_map_bytes = 0;
    g_pso_attn_out_low_q8_0 = nil;
    g_pso_mul_mm_q8_0 = nil;
    g_pso_mul_mm_q8_0_bco = nil;
    g_pso_mul_mm_f16 = nil;
    g_pso_mul_mm_f16_bco = nil;
    g_pso_mul_mm_f32 = nil;
    g_pso_mul_mm_f32_bco = nil;
    g_pso_compute_copy_u8 = nil;
    g_pso_compute_copy_u32 = nil;
    g_pso_compute_copy_u32x4 = nil;
    g_inflight_cbs = nil;
    g_shared_event = nil;
    g_shared_event_val = 0;
    g_library = nil;
    g_native_pipelines = nil;
    g_queue = nil;
    for (size_t i = 0; i < g_native_pso_cache_count; i++) {
        g_native_pso_cache[i].pso = nil;
        g_native_pso_cache[i].name[0] = '\0';
    }
    g_native_pso_cache_count = 0;
    g_device = nil;
    g_initialized = false;
}


int myds_metal_begin_batch(void) {
    if (!g_queue && !myds_metal_init(NULL)) return 0;
    if (g_batch_cb) return 1;
    g_batch_cb = [g_queue commandBuffer];
    g_batch_enc = nil;
    return g_batch_cb != nil;
}

int myds_metal_commit_batch(void) {
    if (!g_batch_cb) return wait_inflight_commands();
    close_compute_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    myds_metal_residency_keep_alive();
    [cb commit];
    g_batch_commit_count++;
    return wait_command_buffer(cb);
}

int myds_metal_flush_batch(void) {
    if (!g_batch_cb) return 1;
    close_compute_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    if (!g_inflight_cbs) g_inflight_cbs = [NSMutableArray new];
    [g_inflight_cbs addObject:cb];
    myds_metal_residency_keep_alive();
    [cb commit];
    g_batch_commit_count++;
    return 1;
}

int myds_metal_signal_event(uint64_t *out_val) {
    if (!g_batch_cb) return 0;
    close_compute_encoder();
    if (!g_shared_event) {
        g_shared_event = [g_device newSharedEvent];
        if (!g_shared_event) return 0;
    }
    uint64_t val = ++g_shared_event_val;
    [g_batch_cb encodeSignalEvent:g_shared_event value:val];
    if (out_val) *out_val = val;
    return 1;
}

int myds_metal_wait_event(uint64_t val) {
    if (!g_batch_cb || !g_shared_event || val == 0 || val > g_shared_event_val) return 0;
    close_compute_encoder();
    [g_batch_cb encodeWaitForEvent:g_shared_event value:val];
    return 1;
}

int myds_metal_commit_and_wait_event(uint64_t val) {
    if (!g_batch_cb || !g_shared_event || val == 0 || val > g_shared_event_val) return 0;
    close_compute_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    if (!g_inflight_cbs) g_inflight_cbs = [NSMutableArray new];
    [g_inflight_cbs addObject:cb];
    myds_metal_residency_keep_alive();
    [cb commit];
    g_batch_commit_count++;
    BOOL signaled = [g_shared_event waitUntilSignaledValue:val timeoutMS:60000];
    if (!signaled) {
        fprintf(stderr, "myds_metal: timeout waiting for shared event value %llu\n", (unsigned long long)val);
        return 0;
    }
    return 1;
}

uint64_t myds_metal_get_commit_count(void) {
    return g_batch_commit_count;
}

int myds_metal_synchronize(void) {
    if (g_batch_cb) {
        return myds_metal_commit_batch();
    }
    return wait_inflight_commands();
}

myds_metal_buffer_t *myds_metal_buffer_alloc(size_t bytes) {
    if (!g_device && !myds_metal_init(NULL)) return NULL;
    if (bytes == 0) bytes = 16;

    @autoreleasepool {
        id<MTLBuffer> mtl_buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!mtl_buf) return NULL;

        myds_metal_buffer_t *buf = (myds_metal_buffer_t *)calloc(1, sizeof(myds_metal_buffer_t));
        buf->mtl_buf = mtl_buf;
        buf->size = bytes;
        buf->offset = 0;
        buf->ptr = [mtl_buf contents];
        buf->is_view = false;
        return buf;
    }
}

myds_metal_buffer_t *myds_metal_buffer_from_ptr(const void *ptr, size_t bytes) {
    if (!g_device && !myds_metal_init(NULL)) return NULL;
    if (!ptr || bytes == 0) return NULL;

    @autoreleasepool {
        size_t page_sz = (size_t)getpagesize();
        uintptr_t target_ptr = (uintptr_t)ptr;
        uintptr_t page_aligned_ptr = target_ptr & ~(page_sz - 1);
        size_t page_offset = target_ptr - page_aligned_ptr;
        size_t page_len = ((bytes + page_offset + page_sz - 1) / page_sz) * page_sz;

        id<MTLBuffer> mtl_buf = [g_device newBufferWithBytesNoCopy:(void *)page_aligned_ptr
                                                            length:page_len
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
        if (!mtl_buf) {
            return NULL;
        }
        myds_metal_residency_register(mtl_buf);

        myds_metal_buffer_t *buf = (myds_metal_buffer_t *)calloc(1, sizeof(myds_metal_buffer_t));
        buf->mtl_buf = mtl_buf;
        buf->size = bytes;
        buf->offset = page_offset;
        buf->ptr = (void *)((uintptr_t)[mtl_buf contents] + page_offset);
        buf->is_view = true;
        return buf;
    }
}

myds_metal_buffer_t *myds_metal_buffer_sub_view(const myds_metal_buffer_t *parent, size_t offset, size_t bytes) {
    if (!parent || !parent->mtl_buf || (offset + bytes) > parent->size) return NULL;
    myds_metal_buffer_t *buf = (myds_metal_buffer_t *)calloc(1, sizeof(myds_metal_buffer_t));
    if (!buf) return NULL;
    buf->mtl_buf = parent->mtl_buf;
    buf->size = bytes;
    buf->offset = parent->offset + offset;
    buf->ptr = (void *)((uintptr_t)parent->ptr + offset);
    buf->is_view = true;
    return buf;
}

myds_metal_buffer_t *myds_metal_buffer_create_view(void) {
    myds_metal_buffer_t *buf = (myds_metal_buffer_t *)calloc(1, sizeof(myds_metal_buffer_t));
    if (buf) buf->is_view = true;
    return buf;
}

int myds_metal_buffer_update_sub_view(myds_metal_buffer_t *view, const myds_metal_buffer_t *parent, size_t offset, size_t bytes) {
    if (!view || !parent || !parent->mtl_buf || (offset + bytes) > parent->size) return 0;
    view->mtl_buf = parent->mtl_buf;
    view->size = bytes;
    view->offset = parent->offset + offset;
    view->ptr = (void *)((uintptr_t)parent->ptr + offset);
    view->is_view = true;
    return 1;
}

void myds_metal_buffer_free(myds_metal_buffer_t *buf) {
    if (!buf) return;
    buf->mtl_buf = nil;
    free(buf);
}

void *myds_metal_buffer_data(myds_metal_buffer_t *buf) {
    return buf ? buf->ptr : NULL;
}

size_t myds_metal_buffer_size(const myds_metal_buffer_t *buf) {
    return buf ? buf->size : 0;
}

size_t myds_metal_buffer_offset(const myds_metal_buffer_t *buf) {
    return buf ? buf->offset : 0;
}

bool myds_metal_buffer_is_view(const myds_metal_buffer_t *buf) {
    return buf ? buf->is_view : false;
}

int myds_metal_buffer_write(myds_metal_buffer_t *buf, size_t offset, const void *src, size_t bytes) {
    if (!buf || !src || offset + bytes > buf->size) return 0;
    memcpy((uint8_t *)buf->ptr + offset, src, bytes);
    return 1;
}

int myds_metal_buffer_read(const myds_metal_buffer_t *buf, size_t offset, void *dst, size_t bytes) {
    if (!buf || !dst || offset + bytes > buf->size) return 0;
    memcpy(dst, (const uint8_t *)buf->ptr + offset, bytes);
    return 1;
}

/* =========================================================================
 * Kernel Dispatches
 * ========================================================================= */

struct ds4_metal_args_norm {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
    int32_t  round_bf16;
};

int myds_metal_rmsnorm(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *weight,
    uint32_t                   dim,
    float                      eps) {
    if (!out || !x || !weight || dim == 0) return 0;

    if (g_pso_rms_norm && (dim % 4 == 0)) {
        @autoreleasepool {
            struct ds4_metal_args_norm args;
            memset(&args, 0, sizeof(args));
            args.ne00 = dim;
            args.ne00_t = dim / 4;
            args.nb1 = dim * sizeof(float);
            args.eps = eps;
            args.round_bf16 = 0;
            for (int i = 0; i < 3; i++) {
                args.nef1[i] = args.nef2[i] = args.nef3[i] = 1;
            }

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_rms_norm];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:x->mtl_buf offset:(x ? x->offset : 0) atIndex:1];
            [enc setBuffer:weight->mtl_buf offset:(weight ? weight->offset : 0) atIndex:2];
            [enc setBuffer:weight->mtl_buf offset:(weight ? weight->offset : 0) atIndex:3];
            [enc setBuffer:out->mtl_buf offset:(out ? out->offset : 0) atIndex:4];
            [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];

            NSUInteger threads = (dim / 4 < 32) ? 32 : (dim / 4 > 256 ? 256 : dim / 4);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    }

    /* CPU Fallback */
    const float *xv = (const float *)x->ptr;
    const float *wv = (const float *)weight->ptr;
    float *ov = (float *)out->ptr;

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum_sq += xv[i] * xv[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    for (uint32_t i = 0; i < dim; i++) {
        ov[i] = xv[i] * scale * wv[i];
    }
    return 1;
}

/* V4.1 uses the already vendored reference kernels, with their original
 * layouts and dispatch reductions. Keep these contracts separate from the
 * generic one-stream operators above. */
static bool v41_floats(const myds_metal_buffer_t *b, uint64_t n) {
    return b && b->mtl_buf && n && n <= b->size / sizeof(float) &&
           (b->offset % sizeof(float)) == 0;
}

static int v41_finish(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (g_batch_cb) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
    [enc endEncoding];
    [cb commit];
    return wait_command_buffer(cb);
}

/* 诊断用（参谋长 2026-09-30）：统计 GPU dispatch 次数。
 * 用途：判定 prefill 的 GPU 时间是被"启动开销 x 海量小 kernel"主导，还是被算力/带宽主导。 */
static uint64_t g_dispatch_count = 0;
uint64_t myds_metal_dispatch_count(void) { return g_dispatch_count; }
void myds_metal_dispatch_count_reset(void) { g_dispatch_count = 0; }

int myds_metal_dispatch_kernel(const char *name, const void *args, size_t args_size,
    const myds_metal_buffer_t *const *buffers, size_t n_buffers,
    const uint32_t groups[3], const uint32_t threads[3], size_t threadgroup_memory) {
    return myds_metal_dispatch_kernel_constants(name,args,args_size,buffers,n_buffers,
        groups,threads,threadgroup_memory,NULL,0);
}

uint32_t myds_metal_kernel_max_threads(const char *name) {
    if (!g_initialized || !name) return 0;
    @autoreleasepool {
        NSString *key=[NSString stringWithUTF8String:name];
        if (!key) return 0;
        if (!g_native_pipelines) g_native_pipelines=[NSMutableDictionary new];
        id<MTLComputePipelineState> pipeline=g_native_pipelines[key];
        if (!pipeline) {
            id<MTLFunction> fn=[g_library newFunctionWithName:key];
            if (!fn) return 0;
            pipeline=[g_device newComputePipelineStateWithFunction:fn error:nil];
            if (!pipeline) return 0;
            g_native_pipelines[key]=pipeline;
        }
        return (uint32_t)pipeline.maxTotalThreadsPerThreadgroup;
    }
}

int myds_metal_dispatch_kernel_constants(const char *name, const void *args, size_t args_size,
    const myds_metal_buffer_t *const *buffers, size_t n_buffers,
    const uint32_t groups[3], const uint32_t threads[3], size_t threadgroup_memory,
    const myds_metal_constant_t *constants, size_t n_constants) {
    if (!g_initialized || !name || !*name || !args || !args_size || args_size > 4096 ||
        !buffers || !n_buffers || n_buffers > 30 || !groups || !threads ||
        (n_constants && !constants) || n_constants>64) return 0;
    uint64_t n_threads = 1;
    const MTLSize maximum = g_device.maxThreadsPerThreadgroup;
    const NSUInteger limits[3] = {maximum.width, maximum.height, maximum.depth};
    for (unsigned i = 0; i < 3; ++i) {
        if (!groups[i] || !threads[i] || threads[i] > limits[i] ||
            n_threads > UINT64_MAX / threads[i]) return 0;
        n_threads *= threads[i];
    }
    for (size_t i = 0; i < n_buffers; ++i)
        if (!buffers[i] || !buffers[i]->mtl_buf || !buffers[i]->size) return 0;
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        if (n_constants == 0) {
            for (size_t i = 0; i < g_native_pso_cache_count; i++) {
                if (strcmp(g_native_pso_cache[i].name, name) == 0) {
                    pipeline = g_native_pso_cache[i].pso;
                    break;
                }
            }
        }
        if (!pipeline) {
            NSString *function_name = [NSString stringWithUTF8String:name];
            if (!function_name) return 0;
            NSMutableString *key = [function_name mutableCopy];
            for (size_t i=0;i<n_constants;i++)
                [key appendFormat:@":%u:%d:%d",constants[i].index,constants[i].is_bool,constants[i].value];
            if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
            pipeline = g_native_pipelines[key];
            if (!pipeline) {
                MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
                for (size_t i=0;i<n_constants;i++) {
                    bool bv=constants[i].value!=0;
                    int32_t iv=constants[i].value;
                    [values setConstantValue:constants[i].is_bool ? (const void *)&bv : (const void *)&iv
                        type:constants[i].is_bool ? MTLDataTypeBool : MTLDataTypeInt atIndex:constants[i].index];
                }
                id<MTLFunction> fn = n_constants ? [g_library newFunctionWithName:function_name constantValues:values error:nil]
                                               : [g_library newFunctionWithName:function_name];
                if (!fn) return 0;
                pipeline = [g_device newComputePipelineStateWithFunction:fn error:nil];
                if (!pipeline) return 0;
                g_native_pipelines[key] = pipeline;
            }
            if (n_constants == 0 && g_native_pso_cache_count < 64 && strlen(name) < 64) {
                strncpy(g_native_pso_cache[g_native_pso_cache_count].name, name, 63);
                g_native_pso_cache[g_native_pso_cache_count].name[63] = '\0';
                g_native_pso_cache[g_native_pso_cache_count].pso = pipeline;
                g_native_pso_cache_count++;
            }
        }
        if (n_threads > pipeline.maxTotalThreadsPerThreadgroup ||
            threadgroup_memory > g_device.maxThreadgroupMemoryLength ||
            pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - threadgroup_memory ||
            threadgroup_memory % 16) return 0;
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:args length:args_size atIndex:0];
        for (size_t i = 0; i < n_buffers; ++i)
            [enc setBuffer:buffers[i]->mtl_buf offset:buffers[i]->offset atIndex:i+1];
        if (threadgroup_memory) [enc setThreadgroupMemoryLength:threadgroup_memory atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
            threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
        g_dispatch_count++;
        return v41_finish(cb, enc);
    }
}

int myds_metal_buffer_copy(myds_metal_buffer_t *dst, size_t dst_offset,
    const myds_metal_buffer_t *src, size_t src_offset, size_t bytes) {
    if (!g_initialized || !dst || !src || !dst->mtl_buf || !src->mtl_buf ||
        dst_offset > dst->size || bytes > dst->size-dst_offset ||
        src_offset > src->size || bytes > src->size-src_offset) return 0;
    if (!bytes) return 1;
    size_t a = dst->offset+dst_offset, b = src->offset+src_offset;
    if (dst->mtl_buf == src->mtl_buf) {
        if (a == b) return 1;
        if (a < b ? b-a < bytes : a-b < bytes) return 0;
    }
    if (g_batch_cb && g_pso_compute_copy_u8) {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(g_batch_cb);
            if (!enc) return 0;
            id<MTLComputePipelineState> pso = g_pso_compute_copy_u8;
            uint64_t count = bytes;
            if ((a % 16 == 0) && (b % 16 == 0) && (bytes % 16 == 0) && g_pso_compute_copy_u32x4) {
                pso = g_pso_compute_copy_u32x4;
                count = bytes / 16;
            } else if ((a % 4 == 0) && (b % 4 == 0) && (bytes % 4 == 0) && g_pso_compute_copy_u32) {
                pso = g_pso_compute_copy_u32;
                count = bytes / 4;
            }
            struct {
                uint64_t src_offset;
                uint64_t dst_offset;
                uint64_t count;
            } args = { (uint64_t)b, (uint64_t)a, count };

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src->mtl_buf offset:0 atIndex:0];
            [enc setBuffer:dst->mtl_buf offset:0 atIndex:1];
            [enc setBytes:&args length:sizeof(args) atIndex:2];

            NSUInteger nth = 256;
            if (count < 256) {
                nth = ((count + 31) / 32) * 32;
                if (nth == 0) nth = 32;
            }
            NSUInteger ngr = (count + nth - 1) / nth;
            [enc dispatchThreadgroups:MTLSizeMake(ngr, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            return 1;
        }
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        if (g_batch_cb) close_compute_encoder();
        id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        if (!enc) return 0;
        [enc copyFromBuffer:src->mtl_buf sourceOffset:b toBuffer:dst->mtl_buf destinationOffset:a size:bytes];
        [enc endEncoding];
        if (g_batch_cb) return 1;
        [cb commit];
        return wait_command_buffer(cb);
    }
}

int myds_metal_v41_bf16(myds_metal_buffer_t *x, uint32_t count) {
    if (!g_initialized || !g_pso_v41_bf16 || !v41_floats(x, count) || x->offset % 16) return 0;
    @autoreleasepool {
        const uint64_t n = count;
        const NSUInteger work = ((NSUInteger)count + 3u) / 4u;
        const NSUInteger nth = MIN((NSUInteger)256, work);
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_bf16];
        [enc setBytes:&n length:sizeof(n) atIndex:0];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((work + nth - 1u) / nth, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_rmsnorm(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *x, const myds_metal_buffer_t *weight,
    uint32_t dim, float eps, bool round_bf16) {
    if (!g_initialized || !dim || dim > INT32_MAX || dim % 128 ||
        !isfinite(eps) || eps < 0 || !v41_floats(out, dim) || !v41_floats(x, dim) ||
        out->offset % 16 || x->offset % 16 ||
        (weight && (!v41_floats(weight, dim) || weight->offset % 16))) return 0;
    id<MTLComputePipelineState> pipeline = weight ? g_pso_rms_norm : g_pso_v41_norm_plain;
    if (!pipeline) { fprintf(stderr, "myds_metal: V4.1 RMSNorm pipeline unavailable (weight=%d)\n", weight != NULL); return 0; }
    NSUInteger nth = 32;
    while (nth < dim / 4u && nth < 1024u) nth *= 2;
    if (nth > dim / 4u) nth = dim / 4u;
    if (nth > pipeline.maxTotalThreadsPerThreadgroup) return 0;
    @autoreleasepool {
        const uint64_t bytes = (uint64_t)dim * sizeof(float);
        struct ds4_metal_args_norm args = {
            .ne00 = (int32_t)dim, .ne00_t = (int32_t)(dim / 4),
            .nb1 = bytes, .nb2 = bytes, .nb3 = bytes, .eps = eps,
            .nef1 = {1,1,1}, .nef2 = {1,1,1}, .nef3 = {1,1,1},
            .nbf1 = {bytes,bytes,bytes}, .nbf2 = {bytes,bytes,bytes},
            .nbf3 = {bytes,bytes,bytes}, .round_bf16 = round_bf16,
        };
        const myds_metal_buffer_t *w = weight ? weight : x;
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:1];
        [enc setBuffer:w->mtl_buf offset:w->offset atIndex:2];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:3];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:4];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_rmsnorm_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *x, const myds_metal_buffer_t *weight,
    uint32_t dim, uint32_t n_tok, float eps, bool round_bf16) {
    if (!n_tok) return 1;
    if (n_tok == 1) return myds_metal_v41_rmsnorm(out, x, weight, dim, eps, round_bf16);
    if (!g_initialized || !dim || dim > INT32_MAX || dim % 128 ||
        !isfinite(eps) || eps < 0 || !v41_floats(out, (uint64_t)dim * n_tok) ||
        !v41_floats(x, (uint64_t)dim * n_tok) ||
        out->offset % 16 || x->offset % 16 ||
        (weight && (!v41_floats(weight, dim) || weight->offset % 16))) return 0;
    id<MTLComputePipelineState> pipeline = weight ? g_pso_rms_norm : g_pso_v41_norm_plain;
    if (!pipeline) return 0;
    NSUInteger nth = 32;
    while (nth < dim / 4u && nth < 1024u) nth *= 2;
    if (nth > dim / 4u) nth = dim / 4u;
    if (nth > pipeline.maxTotalThreadsPerThreadgroup) return 0;
    @autoreleasepool {
        const uint64_t bytes = (uint64_t)dim * sizeof(float);
        struct ds4_metal_args_norm args = {
            .ne00 = (int32_t)dim, .ne00_t = (int32_t)(dim / 4),
            .nb1 = bytes, .nb2 = bytes * n_tok, .nb3 = bytes * n_tok, .eps = eps,
            .nef1 = {(int32_t)n_tok, 1, 1}, .nef2 = {1, 1, 1}, .nef3 = {1, 1, 1},
            .nbf1 = {bytes, 0, 0}, .nbf2 = {bytes * n_tok, 0, 0},
            .nbf3 = {bytes * n_tok, 0, 0}, .round_bf16 = round_bf16,
        };
        const myds_metal_buffer_t *w = weight ? weight : x;
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:1];
        [enc setBuffer:w->mtl_buf offset:w->offset atIndex:2];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:3];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:4];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

struct v41_hc_split_args {
    int32_t n_hc, sinkhorn_iters;
    int64_t n_rows, mix_hc;
    uint64_t nb01, nb1;
    float eps;
};

int myds_metal_v41_hc_mix(myds_metal_buffer_t *split,
    const myds_metal_buffer_t *residual, const myds_metal_buffer_t *fn,
    ggml_type fn_type, const myds_metal_buffer_t *scale,
    const myds_metal_buffer_t *base, myds_metal_buffer_t *flat_norm,
    myds_metal_buffer_t *mix, uint32_t dim, float eps, float hc_eps) {
    if (!g_initialized || !g_pso_v41_hc_split || !dim || dim > INT32_MAX / 4 || dim % 128 ||
        !isfinite(hc_eps) || hc_eps < 0 || !v41_floats(split, 24) || !v41_floats(mix, 24) ||
        !v41_floats(scale, 3) || !v41_floats(base, 24) ||
        split->offset % 16 || mix->offset % 16 || base->offset % 16) return 0;
    if (!myds_metal_v41_rmsnorm(flat_norm, residual, NULL, 4u * dim, eps, false) ||
        !(fn_type == GGML_TYPE_F16
            ? myds_metal_v41_gemv(mix, fn, flat_norm, 24, 4u * dim, fn_type)
            : myds_metal_gemv(mix, fn, flat_norm, 24, 4u * dim, fn_type))) return 0;
    @autoreleasepool {
        struct v41_hc_split_args args = {4, 20, 1, 24, 24 * sizeof(float), 24 * sizeof(float), hc_eps};
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_split];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mix->mtl_buf offset:mix->offset atIndex:1];
        [enc setBuffer:scale->mtl_buf offset:scale->offset atIndex:2];
        [enc setBuffer:base->mtl_buf offset:base->offset atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_hc_split_batch(myds_metal_buffer_t *split,
    const myds_metal_buffer_t *mix, const myds_metal_buffer_t *scale,
    const myds_metal_buffer_t *base, uint32_t n_tokens, float hc_eps) {
    if (!g_initialized || !g_pso_v41_hc_split || !n_tokens ||
        !v41_floats(split, 24ull * n_tokens) || !v41_floats(mix, 24ull * n_tokens) ||
        !v41_floats(scale, 3) || !v41_floats(base, 24) ||
        split->offset % 16 || mix->offset % 16 || base->offset % 16) return 0;
    @autoreleasepool {
        struct v41_hc_split_args args = {4, 20, (int64_t)n_tokens, 24, 24 * sizeof(float), 24 * sizeof(float), hc_eps};
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_split];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mix->mtl_buf offset:mix->offset atIndex:1];
        [enc setBuffer:scale->mtl_buf offset:scale->offset atIndex:2];
        [enc setBuffer:base->mtl_buf offset:base->offset atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset atIndex:4];
        NSUInteger nth = 32;
        NSUInteger ngr = (n_tokens + nth - 1) / nth;
        [enc dispatchThreadgroups:MTLSizeMake(ngr, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

struct v41_hc_sum_args {
    int64_t n_embd, n_hc, n_tokens;
    uint64_t nb_x0, nb_x1, nb_x2, nb_w0, nb_w1, nb0, nb1;
    int32_t round_bf16;
};

int myds_metal_v41_hc_weighted_sum(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *residual, const myds_metal_buffer_t *weights,
    uint32_t dim) {
    if (!g_initialized || !g_pso_v41_hc_sum || !v41_floats(out, dim) ||
        !v41_floats(residual, 4ull * dim) || !v41_floats(weights, 4)) return 0;
    @autoreleasepool {
        struct v41_hc_sum_args args = {
            .n_embd=dim, .n_hc=4, .n_tokens=1,
            .nb_x0=4, .nb_x1=4ull*dim, .nb_x2=16ull*dim,
            .nb_w0=4, .nb_w1=16, .nb0=4, .nb1=4ull*dim, .round_bf16=1,
        };
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_sum];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:1];
        [enc setBuffer:weights->mtl_buf offset:weights->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        [enc dispatchThreadgroups:MTLSizeMake((dim+nth-1)/nth, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

struct v41_hc_expand_args {
    int64_t n_embd, n_hc, n_tokens;
    uint64_t nb_block0, nb_block1, nb_add0, nb_add1, nb_res0, nb_res1, nb_res2;
    uint64_t nb_post0, nb_post1, nb_comb0, nb_comb1, nb_comb2, nb0, nb1, nb2;
    int32_t has_add, round_bf16;
};

int myds_metal_v41_hc_expand(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *block, const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *split, uint32_t dim) {
    if (!g_initialized || !g_pso_v41_hc_expand || !v41_floats(out, 4ull*dim) ||
        !v41_floats(block, dim) || !v41_floats(residual, 4ull*dim) || !v41_floats(split, 24)) return 0;
    @autoreleasepool {
        struct v41_hc_expand_args args = {
            .n_embd=dim, .n_hc=4, .n_tokens=1,
            .nb_block0=4, .nb_block1=4ull*dim, .nb_add0=4, .nb_add1=4ull*dim,
            .nb_res0=4, .nb_res1=4ull*dim, .nb_res2=16ull*dim,
            .nb_post0=4, .nb_post1=96, .nb_comb0=4, .nb_comb1=16, .nb_comb2=96,
            .nb0=4, .nb1=4ull*dim, .nb2=16ull*dim, .has_add=0, .round_bf16=1,
        };
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_expand];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:block->mtl_buf offset:block->offset atIndex:1];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:2];
        [enc setBuffer:split->mtl_buf offset:split->offset+16 atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset+32 atIndex:4];
        [enc setBuffer:block->mtl_buf offset:block->offset atIndex:5];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:6];
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        [enc dispatchThreadgroups:MTLSizeMake((dim+nth-1)/nth, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_hc_weighted_sum_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *residual, const myds_metal_buffer_t *weights,
    uint32_t dim, uint32_t n_tok, uint64_t weight_stride_bytes) {
    if (!n_tok) return 1;
    if (n_tok == 1) return myds_metal_v41_hc_weighted_sum(out, residual, weights, dim);
    if (!g_initialized || !g_pso_v41_hc_sum || !v41_floats(out, (uint64_t)dim * n_tok) ||
        !v41_floats(residual, 4ull * dim * n_tok) || !weights || !weights->mtl_buf) return 0;
    @autoreleasepool {
        struct v41_hc_sum_args args = {
            .n_embd = dim, .n_hc = 4, .n_tokens = n_tok,
            .nb_x0 = 4, .nb_x1 = 4ull * dim, .nb_x2 = 16ull * dim,
            .nb_w0 = 4, .nb_w1 = weight_stride_bytes,
            .nb0 = 4, .nb1 = 4ull * dim, .round_bf16 = 1,
        };
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_sum];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:1];
        [enc setBuffer:weights->mtl_buf offset:weights->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        uint64_t total_elements = (uint64_t)dim * n_tok;
        [enc dispatchThreadgroups:MTLSizeMake((total_elements + nth - 1) / nth, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_hc_expand_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *block, const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *split, uint32_t dim, uint32_t n_tok) {
    if (!n_tok) return 1;
    if (n_tok == 1) return myds_metal_v41_hc_expand(out, block, residual, split, dim);
    if (!g_initialized || !g_pso_v41_hc_expand || !v41_floats(out, 4ull * dim * n_tok) ||
        !v41_floats(block, (uint64_t)dim * n_tok) || !v41_floats(residual, 4ull * dim * n_tok) ||
        !v41_floats(split, 24ull * n_tok)) return 0;
    @autoreleasepool {
        struct v41_hc_expand_args args = {
            .n_embd = dim, .n_hc = 4, .n_tokens = n_tok,
            .nb_block0 = 4, .nb_block1 = 4ull * dim,
            .nb_add0 = 4, .nb_add1 = 4ull * dim,
            .nb_res0 = 4, .nb_res1 = 4ull * dim, .nb_res2 = 16ull * dim,
            .nb_post0 = 4, .nb_post1 = 96,
            .nb_comb0 = 4, .nb_comb1 = 16, .nb_comb2 = 96,
            .nb0 = 4, .nb1 = 4ull * dim, .nb2 = 16ull * dim,
            .has_add = 0, .round_bf16 = 1,
        };
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_expand];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:block->mtl_buf offset:block->offset atIndex:1];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:2];
        [enc setBuffer:split->mtl_buf offset:split->offset + 16 atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset + 32 atIndex:4];
        [enc setBuffer:block->mtl_buf offset:block->offset atIndex:5];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:6];
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        uint64_t total_elements = (uint64_t)dim * n_tok;
        [enc dispatchThreadgroups:MTLSizeMake((total_elements + nth - 1) / nth, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_engram_add(myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *projected, const myds_metal_buffer_t *q_weight,
    const myds_metal_buffer_t *k_weight, uint32_t dim, float eps) {
    if (!g_initialized || !g_pso_v41_engram_add || !isfinite(eps) || eps <= 0 ||
        !v41_floats(residual, 4ull*dim) || !v41_floats(projected, 5ull*dim) ||
        !v41_floats(q_weight, 4ull*dim) || !v41_floats(k_weight, 4ull*dim)) return 0;
    @autoreleasepool {
        struct {uint32_t width, rows; float eps; uint32_t masked;} args = {dim,1,eps,0};
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_engram_add];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:1];
        [enc setBuffer:projected->mtl_buf offset:projected->offset atIndex:2];
        [enc setBuffer:q_weight->mtl_buf offset:q_weight->offset atIndex:3];
        [enc setBuffer:k_weight->mtl_buf offset:k_weight->offset atIndex:4];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 4, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_rope(
    myds_metal_buffer_t *q_rope,
    myds_metal_buffer_t *k_rope,
    uint32_t             n_heads,
    uint32_t             rope_dim,
    uint32_t             pos,
    float                freq_base,
    float                freq_scale) {
    if (!q_rope && !k_rope) return 0;
    if (rope_dim == 0 || (rope_dim % 2 != 0)) return 0;

    /* Compute decoupled RoPE on query and key vectors */
    float *q = q_rope ? (float *)q_rope->ptr : NULL;
    float *k = k_rope ? (float *)k_rope->ptr : NULL;

    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < rope_dim; i += 2) {
            float theta = (float)pos * freq_scale * powf(freq_base, -((float)i / (float)rope_dim));
            float cos_th = cosf(theta);
            float sin_th = sinf(theta);

            if (q) {
                size_t idx0 = h * rope_dim + i;
                size_t idx1 = h * rope_dim + i + 1;
                float q0 = q[idx0];
                float q1 = q[idx1];
                q[idx0] = q0 * cos_th - q1 * sin_th;
                q[idx1] = q0 * sin_th + q1 * cos_th;
            }

            if (k && h == 0) { /* For decoupled MLA key, typically 1 head (dim = 64) */
                size_t idx0 = i;
                size_t idx1 = i + 1;
                float k0 = k[idx0];
                float k1 = k[idx1];
                k[idx0] = k0 * cos_th - k1 * sin_th;
                k[idx1] = k0 * sin_th + k1 * cos_th;
            }
        }
    }
    return 1;
}

struct ds4_metal_args_glu {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
    int32_t  round_bf16;
};

int myds_metal_swiglu(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *gate,
    const myds_metal_buffer_t *up,
    uint32_t                   dim) {
    if (!out || !gate || !up || dim == 0 || dim > INT32_MAX) return 0;
    const size_t bytes = (size_t)dim * sizeof(float);
    if (out->size < bytes || gate->size < bytes || up->size < bytes) return 0;

    if (g_pso_swiglu) {
        @autoreleasepool {
            struct ds4_metal_args_glu args;
            memset(&args, 0, sizeof(args));
            args.ne00 = dim;
            args.nb01 = dim * sizeof(float);
            args.ne10 = dim;
            args.nb11 = dim * sizeof(float);
            args.ne0 = dim;
            args.nb1 = dim * sizeof(float);
            args.alpha = 1.0f;
            args.limit = 0.0f;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_swiglu];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:gate->mtl_buf offset:(gate ? gate->offset : 0) atIndex:1];
            [enc setBuffer:up->mtl_buf offset:(up ? up->offset : 0) atIndex:2];
            [enc setBuffer:out->mtl_buf offset:(out ? out->offset : 0) atIndex:3];

            NSUInteger threads = MIN((NSUInteger)dim, MIN((NSUInteger)256, g_pso_swiglu.maxTotalThreadsPerThreadgroup));
            [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    }

    /* CPU Fallback */
    const float *g = (const float *)gate->ptr;
    const float *u = (const float *)up->ptr;
    float *o = (float *)out->ptr;

    for (uint32_t i = 0; i < dim; i++) {
        float x0 = g[i];
        float silu = x0 / (1.0f + expf(-x0));
        o[i] = silu * u[i];
    }
    return 1;
}

static inline float fp16_to_fp32(uint16_t h) {
#if defined(__arm64__) || defined(__aarch64__)
    return (float)(*(_Float16 *)&h);
#else
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    int32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) {
            float zero;
            memcpy(&zero, &sign, sizeof(float));
            return zero;
        }
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp += (127 - 15);
    }
    uint32_t f = sign | ((uint32_t)exp << 23) | (mant << 13);
    float val;
    memcpy(&val, &f, sizeof(float));
    return val;
#endif
}

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

int myds_metal_gemv_batch(
    myds_metal_buffer_t       *y,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *x,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    uint32_t                   n_tok,
    ggml_type                  weight_type) {
    if (!y || !W || !x || !out_dim || !in_dim || !n_tok ||
        out_dim > INT32_MAX || in_dim > INT32_MAX || n_tok > INT32_MAX) return 0;

    id<MTLComputePipelineState> pipeline = nil;
    uint64_t row_bytes = 0;
    uint64_t block_bytes = 0;
    uint32_t nsg = 4u;
    if (weight_type == GGML_TYPE_Q8_0) {
        if (in_dim % 32u) return 0;
        block_bytes = 34;
        row_bytes = (uint64_t)(in_dim / 32u) * block_bytes;
        if (out_dim > 65536u) nsg = 8u;
        NSString *key = [NSString stringWithFormat:@"q8_0_%u", nsg];
        pipeline = g_native_pipelines[key];
        if (!pipeline) {
            MTLFunctionConstantValues *mv_constants = [MTLFunctionConstantValues new];
            short short_nsg = (short)nsg;
            [mv_constants setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
            id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_mul_mv_q8_0_f32" constantValues:mv_constants error:nil];
            if (fn) {
                pipeline = [g_device newComputePipelineStateWithFunction:fn error:nil];
                if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
                g_native_pipelines[key] = pipeline;
            }
        }
    } else if (weight_type == GGML_TYPE_F32) {
        pipeline = g_pso_mul_mv_f32;
        block_bytes = sizeof(float);
        row_bytes = (uint64_t)in_dim * block_bytes;
    } else if (weight_type == GGML_TYPE_F16) {
        pipeline = g_pso_mul_mv_f16;
        block_bytes = sizeof(uint16_t);
        row_bytes = (uint64_t)in_dim * block_bytes;
    } else {
        MYDS_LOG_E("myds_metal_gemv: unsupported GGML quantization type %d (%s) for GEMV out_dim=%u, in_dim=%u",
                   (int)weight_type, ggml_type_name(weight_type), out_dim, in_dim);
        return 0;
    }
    /* Division-based bounds checks also avoid overflowing dimension products. */
    if (W->size / row_bytes < out_dim ||
        x->size / sizeof(float) / in_dim < n_tok ||
        y->size / sizeof(float) / out_dim < n_tok) return 0;

    if (pipeline && W->mtl_buf && x->mtl_buf && y->mtl_buf) {
        @autoreleasepool {
            struct ds4_metal_args_mul_mv args;
            memset(&args, 0, sizeof(args));
            args.ne00 = (int32_t)in_dim;
            args.ne01 = (int32_t)out_dim;
            args.ne02 = 1;
            args.nb00 = block_bytes;
            args.nb01 = row_bytes;
            args.nb02 = row_bytes * out_dim;
            args.nb03 = row_bytes * out_dim;
            args.ne10 = (int32_t)in_dim;
            args.ne11 = (int32_t)n_tok;
            args.ne12 = 1;
            args.nb10 = sizeof(float);
            args.nb11 = (uint64_t)in_dim * sizeof(float);
            args.nb12 = (uint64_t)in_dim * n_tok * sizeof(float);
            args.nb13 = args.nb12;
            args.ne0 = (int32_t)out_dim;
            args.ne1 = (int32_t)n_tok;
            args.nr0 = 2;
            args.r2 = 1;
            args.r3 = 1;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:W->mtl_buf offset:W->offset atIndex:1];
            [enc setBuffer:x->mtl_buf offset:x->offset atIndex:2];
            [enc setBuffer:y->mtl_buf offset:y->offset atIndex:3];
            [enc setThreadgroupMemoryLength:32 * 2 * sizeof(float) atIndex:0];
            /* Each token retains the single-vector lane assignment/reduction. */
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + 1) / 2, n_tok, 1)
                threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
            if (g_batch_cb) {
                if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    }

    /* A missing pipeline may require CPU access to GPU-produced activations. */
    const bool resume_batch = g_batch_cb != nil;
    if (!myds_metal_synchronize()) return 0;
    typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0_t;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *xv = (const float *)x->ptr + (size_t)t * in_dim;
        float *yv = (float *)y->ptr + (size_t)t * out_dim;
        for (uint32_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const uint8_t *w_row = (const uint8_t *)W->ptr + (size_t)row * row_bytes;
            if (weight_type == GGML_TYPE_Q8_0) {
                const block_q8_0_t *blocks = (const block_q8_0_t *)w_row;
                for (uint32_t b = 0; b < in_dim / 32u; b++) {
                    float dot = 0.0f;
                    for (uint32_t i = 0; i < 32; i++) dot += (float)blocks[b].qs[i] * xv[b * 32u + i];
                    sum += dot * fp16_to_fp32(blocks[b].d);
                }
            } else if (weight_type == GGML_TYPE_F32) {
                const float *weights = (const float *)w_row;
                for (uint32_t col = 0; col < in_dim; col++) sum += weights[col] * xv[col];
            } else {
                const uint16_t *weights = (const uint16_t *)w_row;
                for (uint32_t col = 0; col < in_dim; col++) sum += fp16_to_fp32(weights[col]) * xv[col];
            }
            yv[row] = sum;
        }
    }
    return resume_batch ? myds_metal_begin_batch() : 1;
}

int myds_metal_gemv(
    myds_metal_buffer_t       *y,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *x,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    ggml_type                  weight_type) {
    return myds_metal_gemv_batch(y, W, x, out_dim, in_dim, 1, weight_type);
}

int myds_metal_v41_gemv_batch(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim, uint32_t n_tok, ggml_type weight_type) {
    if (!out || !weight || !input || !out_dim || !in_dim || !n_tok ||
        (weight_type != GGML_TYPE_F16 && weight_type != GGML_TYPE_F32) ||
        out_dim > INT32_MAX || in_dim > INT32_MAX || n_tok > INT32_MAX ||
        in_dim % 4u ||
        input->size / sizeof(float) / in_dim < n_tok ||
        out->size / sizeof(float) / out_dim < n_tok)
        return 0;
    const uint64_t elem_bytes = weight_type == GGML_TYPE_F32 ? sizeof(float) : 2u;
    if (weight->size / ((uint64_t)in_dim * elem_bytes) < out_dim) return 0;
    struct ds4_metal_args_mul_mv args;
    memset(&args, 0, sizeof(args));
    const uint32_t nsg = (in_dim + 127u) / 128u > 8u ? 8u : (in_dim + 127u) / 128u;
    args.ne00 = (int32_t)in_dim;
    args.ne01 = (int32_t)out_dim;
    args.ne02 = 1;
    args.nb00 = elem_bytes;
    args.nb01 = (uint64_t)in_dim * elem_bytes;
    args.nb02 = args.nb01 * out_dim;
    args.nb03 = args.nb02;
    args.ne10 = (int32_t)in_dim;
    args.ne11 = (int32_t)n_tok;
    args.ne12 = 1;
    args.nb10 = sizeof(float);
    args.nb11 = (uint64_t)in_dim * sizeof(float);
    args.nb12 = (uint64_t)in_dim * n_tok * sizeof(float);
    args.nb13 = args.nb12;
    args.ne0 = (int32_t)out_dim;
    args.ne1 = (int32_t)n_tok;
    args.nr0 = 2;
    args.r2 = 1;
    args.r3 = 1;
    @autoreleasepool {
        NSString *funcName = weight_type == GGML_TYPE_F32 ? @"kernel_mul_mv_f32_f32_4" : @"kernel_mul_mv_f16_f32_4";
        NSString *key = [NSString stringWithFormat:@"%@_%u", funcName, nsg];
        id<MTLComputePipelineState> pipeline = g_native_pipelines[key];
        if (!pipeline) {
            MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
            short short_nsg = (short)nsg;
            [values setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
            id<MTLFunction> fn = [g_library newFunctionWithName:funcName
                                                   constantValues:values error:nil];
            if (!fn) { fprintf(stderr, "myds_metal: v41 %s function unavailable\n", [funcName UTF8String]); return 0; }
            pipeline = [g_device newComputePipelineStateWithFunction:fn error:nil];
            if (!pipeline) { fprintf(stderr, "myds_metal: v41 %s pipeline unavailable\n", [funcName UTF8String]); return 0; }
            if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
            g_native_pipelines[key] = pipeline;
        }
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc || 32u * nsg > pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "myds_metal: v41 %s encoder unavailable (nsg=%u)\n", [funcName UTF8String], nsg);
            return 0;
        }
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weight->mtl_buf offset:weight->offset atIndex:1];
        [enc setBuffer:input->mtl_buf offset:input->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((out_dim + 1u) / 2u, n_tok, 1)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_gemv(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim, ggml_type weight_type) {
    return myds_metal_v41_gemv_batch(out, weight, input, out_dim, in_dim, 1, weight_type);
}

struct ds4_metal_args_mul_mm {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

int myds_metal_mul_mm(
    myds_metal_buffer_t       *dst,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *src1,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    uint32_t                   n_tok,
    ggml_type                  weight_type) {
    if (!dst || !W || !src1 || out_dim == 0 || in_dim == 0 || n_tok == 0) return 0;
    if (n_tok == 1) {
        return myds_metal_gemv(dst, W, src1, out_dim, in_dim, weight_type);
    }

    const bool bc_inp = (in_dim % 32u) != 0;
    const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;

    id<MTLComputePipelineState> pipeline = nil;
    uint64_t row_bytes = 0;

    if (weight_type == GGML_TYPE_Q8_0) {
        pipeline = bc_out ? g_pso_mul_mm_q8_0_bco : g_pso_mul_mm_q8_0;
        row_bytes = (in_dim / 32u) * 34u;
    } else if (weight_type == GGML_TYPE_F16) {
        pipeline = bc_out ? g_pso_mul_mm_f16_bco : g_pso_mul_mm_f16;
        row_bytes = in_dim * sizeof(uint16_t);
    } else if (weight_type == GGML_TYPE_F32) {
        pipeline = bc_out ? g_pso_mul_mm_f32_bco : g_pso_mul_mm_f32;
        row_bytes = in_dim * sizeof(float);
    }

    if (pipeline && W->mtl_buf && src1->mtl_buf && dst->mtl_buf && !bc_inp) {
        @autoreleasepool {
            struct ds4_metal_args_mul_mm args;
            memset(&args, 0, sizeof(args));
            args.ne00 = (int32_t)in_dim;
            args.ne02 = 1;
            args.nb01 = row_bytes;
            args.nb02 = row_bytes * out_dim;
            args.nb03 = row_bytes * out_dim;
            args.ne12 = 1;
            args.nb10 = sizeof(float);
            args.nb11 = in_dim * sizeof(float);
            args.nb12 = (uint64_t)in_dim * n_tok * sizeof(float);
            args.nb13 = (uint64_t)in_dim * n_tok * sizeof(float);
            args.ne0 = (int32_t)out_dim;
            args.ne1 = (int32_t)n_tok;
            args.r2 = 1;
            args.r3 = 1;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:W->mtl_buf offset:(W ? W->offset : 0) atIndex:1];
            [enc setBuffer:src1->mtl_buf offset:(src1 ? src1->offset : 0) atIndex:2];
            [enc setBuffer:dst->mtl_buf offset:(dst ? dst->offset : 0) atIndex:3];
            uint32_t tg_mem = (weight_type == GGML_TYPE_F32) ? 12288u : (bc_out ? 8192u : 6144u);
            [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                                  ((NSUInteger)out_dim + 63u) / 64u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    }

    /* Fallback to per-vector GEMV */
    for (uint32_t t = 0; t < n_tok; t++) {
        myds_metal_buffer_t x_t = *src1;
        x_t.offset += (size_t)t * in_dim * sizeof(float);
        x_t.ptr = (char *)src1->ptr + (size_t)t * in_dim * sizeof(float);
        x_t.size = in_dim * sizeof(float);

        myds_metal_buffer_t y_t = *dst;
        y_t.offset += (size_t)t * out_dim * sizeof(float);
        y_t.ptr = (char *)dst->ptr + (size_t)t * out_dim * sizeof(float);
        y_t.size = out_dim * sizeof(float);

        if (!myds_metal_gemv(&y_t, W, &x_t, out_dim, in_dim, weight_type)) {
            return 0;
        }
    }
    return 1;
}

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_addend;
    int32_t  tp_expert_base;
};

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

struct ds4_metal_dsv4_moe_sum6_args {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
};

struct ds4_metal_args_mul_mm_id_map0 {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
};

struct ds4_metal_args_mul_mm_id {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_expert_base;
};

static int ensure_moe_id_map_buffer(NSUInteger total_bytes) {
    if (g_moe_id_map_buffer && g_moe_id_map_bytes >= total_bytes) {
        return 1;
    }
    g_moe_id_map_buffer = nil;
    NSUInteger alloc_bytes = 1048576u;
    while (alloc_bytes < total_bytes) {
        alloc_bytes *= 2;
    }
    g_moe_id_map_buffer = [g_device newBufferWithLength:alloc_bytes options:MTLResourceStorageModeShared];
    if (!g_moe_id_map_buffer) {
        g_moe_id_map_bytes = 0;
        return 0;
    }
    g_moe_id_map_bytes = alloc_bytes;
    return 1;
}

static int get_moe_mm_min_tokens(void) {
    static int min_tokens = -2;
    if (min_tokens == -2) {
        const char *disable = getenv("MYDS_DISABLE_MOE_MM");
        if (disable && atoi(disable) != 0) {
            min_tokens = -1;
        } else {
            const char *env = getenv("MYDS_MOE_MM_MIN_TOKENS");
            min_tokens = env ? atoi(env) : 32;
        }
    }
    return min_tokens;
}

static inline float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void topk_desc(const float *score, int n, int k, int32_t *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = (int32_t)i;
                break;
            }
        }
    }
}

int myds_metal_moe_route_biased(
    int32_t                   *out_expert_ids,
    float                     *out_expert_weights,
    const myds_metal_buffer_t *router_logits,
    const myds_metal_buffer_t *router_bias,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    float                      routed_scaling_factor) {
    if (!out_expert_ids || !out_expert_weights || !router_logits) return 0;
    if (n_routed_experts == 0 || n_activated_experts == 0 || n_activated_experts > n_routed_experts) return 0;

    const float *logits = (const float *)router_logits->ptr;
    const float *bias = router_bias ? (const float *)router_bias->ptr : NULL;

    float probs_stack[512];
    float selection_stack[512];
    float *probs = (n_routed_experts <= 512) ? probs_stack : (float *)malloc(n_routed_experts * sizeof(float));
    float *selection = (n_routed_experts <= 512) ? selection_stack : (float *)malloc(n_routed_experts * sizeof(float));
    if (!probs || !selection) {
        if (probs && probs != probs_stack) free(probs);
        if (selection && selection != selection_stack) free(selection);
        return 0;
    }

    for (uint32_t i = 0; i < n_routed_experts; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
        selection[i] = probs[i] + (bias ? bias[i] : 0.0f);
    }

    topk_desc(selection, (int)n_routed_experts, (int)n_activated_experts, out_expert_ids);

    float sum = 0.0f;
    for (uint32_t k = 0; k < n_activated_experts; k++) {
        int32_t exp_id = out_expert_ids[k];
        out_expert_weights[k] = probs[exp_id];
        sum += probs[exp_id];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    float norm_factor = (routed_scaling_factor / sum);
    for (uint32_t k = 0; k < n_activated_experts; k++) {
        out_expert_weights[k] *= norm_factor;
    }

    if (selection != selection_stack) free(selection);
    if (probs != probs_stack) free(probs);
    return 1;
}

int myds_metal_moe_route(
    int32_t                   *out_expert_ids,
    float                     *out_expert_weights,
    const myds_metal_buffer_t *router_logits,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    float                      routed_scaling_factor) {
    return myds_metal_moe_route_biased(out_expert_ids, out_expert_weights, router_logits, NULL,
                                       n_routed_experts, n_activated_experts, routed_scaling_factor);
}

int myds_metal_routed_moe_batch(
    myds_metal_buffer_t       *moe_out,
    const myds_metal_buffer_t *x_in,
    const myds_metal_buffer_t *gate_exps,
    const myds_metal_buffer_t *up_exps,
    const myds_metal_buffer_t *down_exps,
    const myds_metal_buffer_t *selected_ids,
    const myds_metal_buffer_t *selected_weights,
    myds_metal_buffer_t       *scratch_gate,
    myds_metal_buffer_t       *scratch_up,
    myds_metal_buffer_t       *scratch_mid,
    myds_metal_buffer_t       *scratch_down,
    uint32_t                   dim,
    uint32_t                   expert_dim,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    uint32_t                   n_tok,
    ggml_type                  gate_type,
    ggml_type                  up_type,
    ggml_type                  down_type,
    float                      clamp_val) {
    if (!moe_out || !x_in || !gate_exps || !up_exps || !down_exps ||
        !selected_ids || !selected_weights || !scratch_gate || !scratch_up ||
        !scratch_mid || !scratch_down || n_tok == 0) {
        return 0;
    }
    if (n_activated_experts != 6) {
        return 0;
    }
    if (!g_pso_moe_mv_id_iq2_xxs || !g_pso_moe_mv_id_q2_k ||
        !g_pso_moe_swiglu_weight || !g_pso_moe_sum6) {
        return 0;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);

        uint64_t gate_row_bytes = (dim / 256u) * 66u;
        uint64_t gate_exp_bytes = (uint64_t)expert_dim * gate_row_bytes;
        uint64_t down_row_bytes = (expert_dim / 256u) * 84u;
        uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;

        int min_mm_tokens = get_moe_mm_min_tokens();
        bool can_use_mm = (min_mm_tokens > 0) &&
                          ((int)n_tok >= min_mm_tokens) &&
                          (gate_type == GGML_TYPE_IQ2_XXS) &&
                          (up_type == GGML_TYPE_IQ2_XXS) &&
                          (down_type == GGML_TYPE_Q2_K) &&
                          g_pso_moe_mm_map0_ne20_6 &&
                          g_pso_moe_mm_iq2_pair_swiglu_f16 &&
                          g_pso_moe_mm_q2_k_f16 &&
                          g_pso_moe_sum6;

        if (can_use_mm) {
            const NSUInteger tpe_bytes = (NSUInteger)n_routed_experts * sizeof(int32_t);
            const NSUInteger hids_bytes = (NSUInteger)n_routed_experts * (NSUInteger)n_tok * sizeof(int32_t);
            const NSUInteger work_offset = (tpe_bytes + hids_bytes + 7u) & ~7u;
            const uint64_t pair_rows = (uint64_t)n_activated_experts * (uint64_t)n_tok;
            const uint64_t work_cap = (pair_rows + 31u * (uint32_t)n_routed_experts + 31u) / 32u;
            const NSUInteger work_item_bytes = 2u * sizeof(uint32_t);
            const NSUInteger total_bytes = work_offset + 8u + (NSUInteger)work_cap * work_item_bytes;

            if (!ensure_moe_id_map_buffer(total_bytes)) {
                return 0;
            }

            static bool logged_moe_mm = false;
            if (!logged_moe_mm) {
                MYDS_LOG_I("MoE batch path: MM (kernel_mul_mm_id pair_swiglu + q2_k_f16 + sum6) — active for n_tok >= %d", min_mm_tokens);
                logged_moe_mm = true;
            }

            struct ds4_metal_args_mul_mm_id_map0 map_args = {
                .ne02 = (int32_t)n_routed_experts,
                .ne10 = (int32_t)dim,
                .ne11 = 1,
                .nb11 = (uint64_t)dim * sizeof(float),
                .nb12 = (uint64_t)dim * sizeof(float),
                .ne21 = (int32_t)n_tok,
                .ne20 = (int32_t)n_activated_experts,
                .nb21 = (uint64_t)n_activated_experts * sizeof(int32_t),
            };

            const uint64_t x_row_bytes = (uint64_t)dim * sizeof(float);
            struct ds4_metal_args_mul_mm_id gate_mm_args = {
                .ne00 = (int32_t)dim,
                .ne02 = (int32_t)n_routed_experts,
                .nb01 = gate_row_bytes,
                .nb02 = gate_exp_bytes,
                .nb03 = (uint64_t)n_routed_experts * gate_exp_bytes,
                .ne11 = 1,
                .nb10 = sizeof(float),
                .nb11 = x_row_bytes,
                .nb12 = x_row_bytes,
                .nb13 = (uint64_t)n_tok * x_row_bytes,
                .ne20 = (int32_t)n_activated_experts,
                .ne21 = (int32_t)n_tok,
                .ne0 = (int32_t)expert_dim,
                .ne1 = (int32_t)n_activated_experts,
                .r2 = 1,
                .r3 = 1,
                .tp_rank = 0,
                .tp_world = 1,
                .tp_expert_base = 0,
            };

            struct ds4_metal_dsv4_moe_swiglu_weight_args swiglu_mm_args = {
                .width = expert_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_dim * sizeof(uint16_t),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp_val,
            };

            const uint64_t mid_row_bytes = (uint64_t)expert_dim * sizeof(uint16_t);
            struct ds4_metal_args_mul_mm_id down_mm_args = {
                .ne00 = (int32_t)expert_dim,
                .ne02 = (int32_t)n_routed_experts,
                .nb01 = down_row_bytes,
                .nb02 = down_exp_bytes,
                .nb03 = (uint64_t)n_routed_experts * down_exp_bytes,
                .ne11 = (int32_t)n_activated_experts,
                .nb10 = sizeof(uint16_t),
                .nb11 = mid_row_bytes,
                .nb12 = (uint64_t)n_activated_experts * mid_row_bytes,
                .nb13 = (uint64_t)n_tok * (uint64_t)n_activated_experts * mid_row_bytes,
                .ne20 = (int32_t)n_activated_experts,
                .ne21 = (int32_t)n_tok,
                .ne0 = (int32_t)dim,
                .ne1 = (int32_t)n_activated_experts,
                .r2 = 1,
                .r3 = 1,
                .tp_rank = 0,
                .tp_world = 1,
                .tp_expert_base = 0,
            };

            struct ds4_metal_dsv4_moe_sum6_args sum_args = {
                .width = dim,
                .tokens = n_tok,
                .src_token_stride = 6u * (uint64_t)dim * sizeof(float),
                .dst_token_stride = (uint64_t)dim * sizeof(float),
            };

            /* Step 1: Map dispatch */
            [enc setComputePipelineState:g_pso_moe_mm_map0_ne20_6];
            [enc setBytes:&map_args length:sizeof(map_args) atIndex:0];
            [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:1];
            [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:2];
            [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:3];
            [enc setBuffer:g_moe_id_map_buffer offset:work_offset atIndex:4];
            [enc setThreadgroupMemoryLength:(NSUInteger)n_routed_experts * (NSUInteger)n_activated_experts * sizeof(uint16_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)n_routed_experts, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* Step 2: Gate + Up Pair SwiGLU dispatch */
            [enc setComputePipelineState:g_pso_moe_mm_iq2_pair_swiglu_f16];
            [enc setBytes:&gate_mm_args length:sizeof(gate_mm_args) atIndex:0];
            [enc setBytes:&swiglu_mm_args length:sizeof(swiglu_mm_args) atIndex:1];
            [enc setBuffer:gate_exps->mtl_buf offset:gate_exps->offset atIndex:2];
            [enc setBuffer:up_exps->mtl_buf offset:up_exps->offset atIndex:3];
            [enc setBuffer:x_in->mtl_buf offset:x_in->offset atIndex:4];
            [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:5];
            [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:6];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:7];
            [enc setBuffer:selected_weights->mtl_buf offset:selected_weights->offset atIndex:8];
            [enc setBuffer:g_moe_id_map_buffer offset:work_offset atIndex:9];
            [enc setThreadgroupMemoryLength:16384u atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)work_cap,
                                                  ((NSUInteger)expert_dim + 63u) / 64u,
                                                  1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* Step 3: Down Projection dispatch */
            [enc setComputePipelineState:g_pso_moe_mm_q2_k_f16];
            [enc setBytes:&down_mm_args length:sizeof(down_mm_args) atIndex:0];
            [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
            [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
            [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
            [enc setBuffer:scratch_down->mtl_buf offset:scratch_down->offset atIndex:5];
            [enc setBuffer:g_moe_id_map_buffer offset:work_offset atIndex:6];
            [enc setThreadgroupMemoryLength:8192u atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)work_cap,
                                                  ((NSUInteger)dim + 63u) / 64u,
                                                  1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* Step 4: Sum Experts dispatch */
            NSUInteger nth = g_pso_moe_sum6.maxTotalThreadsPerThreadgroup;
            if (nth > 256u) nth = 256u;
            if (nth > dim) nth = dim;
            if (nth == 0) nth = 1u;
            [enc setComputePipelineState:g_pso_moe_sum6];
            [enc setBytes:&sum_args length:sizeof(sum_args) atIndex:0];
            [enc setBuffer:scratch_down->mtl_buf offset:scratch_down->offset atIndex:1];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tok, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }

        /* 1. Gate experts dispatch (IQ2_XXS) */
        struct ds4_metal_args_mul_mv_id gate_args = {
            .nei0 = (int32_t)n_activated_experts,
            .nei1 = (int32_t)n_tok,
            .nbi1 = (uint64_t)n_activated_experts * sizeof(int32_t),
            .ne00 = (int32_t)dim,
            .ne01 = (int32_t)expert_dim,
            .ne02 = (int32_t)n_routed_experts,
            .nb00 = 66u,
            .nb01 = gate_row_bytes,
            .nb02 = gate_exp_bytes,
            .ne10 = (int32_t)dim,
            .ne11 = 1,
            .ne12 = 1,
            .ne13 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)dim * sizeof(float),
            .nb12 = (uint64_t)dim * sizeof(float),
            .ne0 = (int32_t)expert_dim,
            .ne1 = (int32_t)n_activated_experts,
            .nb1 = (uint64_t)expert_dim * sizeof(float),
            .nr0 = 4,
            .tp_rank = 0,
            .tp_world = 0,
            .tp_addend = 0,
            .tp_expert_base = 0,
        };

        const NSUInteger nr0_gate = 4;
        const NSUInteger nsg_gate = 2;
        const NSUInteger rows_per_group_gate = nr0_gate * nsg_gate; /* 8 */
        const NSUInteger row_groups_gate = (expert_dim + rows_per_group_gate - 1) / rows_per_group_gate;
        const NSUInteger pairs = (NSUInteger)n_activated_experts * (NSUInteger)n_tok;
        const NSUInteger smem_iq2 = 256u * sizeof(uint64_t) + 128u * sizeof(uint8_t); /* 2176 */
        static bool logged_moe_mode = false;
        if (!logged_moe_mode) {
            /* 2026-09-29 参谋长:
             *   原标签把「两个 PSO 都不可用」叫成 "FALLBACK (5-step)", 但代码里
             *   根本没有 5-step 回退路径 —— 那条分支直接 `return 0` 失败。
             *   SLOTS6 备用路径已于同日【删除】: 实测它能运行但【输出错误】
             *   (15.94 t/s + 乱码; 对比 FUSED 27.07 t/s + 逐位正确)。它是条会
             *   静默产出错误结果的路径 —— 宁可响亮失败, 也不要静默出错。
             *   依据: docs/verification/2026-09-29-gpu-bottleneck-ablation.md §3/§9 */
            if (g_pso_moe_pair_swiglu_iq2_xxs && g_pso_moe_sum6_q2_k) {
                MYDS_LOG_I("MoE path: FUSED (2-step) — verified bit-exact");
            } else {
                MYDS_LOG_E("MoE path: FUSED kernels unavailable; refusing MoE rather than "
                           "silently producing wrong output (no fallback path exists).");
            }
            logged_moe_mode = true;
        }

        struct ds4_metal_args_mul_mv_id down_args = {
            .nei0 = (int32_t)n_activated_experts,
            .nei1 = (int32_t)n_tok,
            .nbi1 = (uint64_t)n_activated_experts * sizeof(int32_t),
            .ne00 = (int32_t)expert_dim,
            .ne01 = (int32_t)dim,
            .ne02 = (int32_t)n_routed_experts,
            .nb00 = 84u,
            .nb01 = down_row_bytes,
            .nb02 = down_exp_bytes,
            .ne10 = (int32_t)expert_dim,
            .ne11 = (int32_t)n_activated_experts,
            .ne12 = 1,
            .ne13 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)expert_dim * sizeof(float),
            .nb12 = (uint64_t)n_activated_experts * (uint64_t)expert_dim * sizeof(float),
            .ne0 = (int32_t)dim,
            .ne1 = (int32_t)n_tok,
            .nb1 = (uint64_t)dim * sizeof(float),
            .nr0 = 2,
            .tp_rank = 0,
            .tp_world = 1,
            .tp_addend = 0,
            .tp_expert_base = 0,
        };

        const NSUInteger nr0_down = 2;
        const NSUInteger nsg_down = 2;
        const NSUInteger rows_per_group_down = nr0_down * nsg_down; /* 4 */
        const NSUInteger row_groups_down = (dim + rows_per_group_down - 1) / rows_per_group_down;

        struct ds4_metal_dsv4_moe_swiglu_weight_args swiglu_args = {
            .width = expert_dim,
            .rows = (uint64_t)n_activated_experts * (uint64_t)n_tok,
            .gate_row_stride = (uint64_t)expert_dim * sizeof(float),
            .up_row_stride = (uint64_t)expert_dim * sizeof(float),
            .mid_row_stride = (uint64_t)expert_dim * sizeof(float),
            .weight_stride = sizeof(float),
            .write_clamped = 0,
            .clamp_value = clamp_val,
        };

        if (g_pso_moe_pair_swiglu_iq2_xxs && g_pso_moe_sum6_q2_k) {
            /* Fused Gate + Up + SwiGLU + Scaling dispatch (IQ2_XXS) across all tokens */
            [enc setComputePipelineState:g_pso_moe_pair_swiglu_iq2_xxs];
            [enc setBytes:&gate_args length:sizeof(gate_args) atIndex:0];
            [enc setBytes:&swiglu_args length:sizeof(swiglu_args) atIndex:1];
            [enc setBuffer:gate_exps->mtl_buf offset:gate_exps->offset atIndex:2];
            [enc setBuffer:up_exps->mtl_buf offset:up_exps->offset atIndex:3];
            [enc setBuffer:x_in->mtl_buf offset:x_in->offset atIndex:4];
            [enc setBuffer:scratch_gate->mtl_buf offset:scratch_gate->offset atIndex:5];
            [enc setBuffer:scratch_up->mtl_buf offset:scratch_up->offset atIndex:6];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:7];
            [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:8];
            [enc setBuffer:selected_weights->mtl_buf offset:selected_weights->offset atIndex:9];
            [enc setThreadgroupMemoryLength:smem_iq2 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_gate, 1, pairs)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_gate, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* Fused Down + Sum6 direct accumulation dispatch (Q2_K) across all tokens */
            [enc setComputePipelineState:g_pso_moe_sum6_q2_k];
            [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
            [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:3];
            [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_down, (NSUInteger)n_tok, 1)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_down, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        } else {
            return 0;
        }

        if (g_batch_cb) {
            if (g_auto_barrier) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        } else {
            [enc endEncoding];
            [cb commit];
            if (!wait_command_buffer(cb)) return 0;
        }
        return 1;
    }
}

int myds_metal_routed_moe(
    myds_metal_buffer_t       *moe_out,
    const myds_metal_buffer_t *x_in,
    const myds_metal_buffer_t *gate_exps,
    const myds_metal_buffer_t *up_exps,
    const myds_metal_buffer_t *down_exps,
    const myds_metal_buffer_t *selected_ids,
    const myds_metal_buffer_t *selected_weights,
    myds_metal_buffer_t       *scratch_gate,
    myds_metal_buffer_t       *scratch_up,
    myds_metal_buffer_t       *scratch_mid,
    myds_metal_buffer_t       *scratch_down,
    uint32_t                   dim,
    uint32_t                   expert_dim,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    ggml_type                  gate_type,
    ggml_type                  up_type,
    ggml_type                  down_type,
    float                      clamp_val) {
    return myds_metal_routed_moe_batch(
        moe_out, x_in, gate_exps, up_exps, down_exps,
        selected_ids, selected_weights,
        scratch_gate, scratch_up, scratch_mid, scratch_down,
        dim, expert_dim, n_routed_experts, n_activated_experts, 1,
        gate_type, up_type, down_type, clamp_val);
}

int myds_metal_attn_output_a_batch(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    uint32_t                   n_tok,
    ggml_type                  weight_type) {
    if (!low || !out_a || !heads || n_groups == 0 || group_dim == 0 || rank == 0 || n_tok == 0) return 0;

    if (weight_type == GGML_TYPE_Q8_0 && g_pso_attn_out_low_q8_0 && out_a->mtl_buf && heads->mtl_buf && low->mtl_buf && (group_dim % 32 == 0)) {
        @autoreleasepool {
            const uint64_t a_group_bytes = (uint64_t)rank * (group_dim / 32u) * 34u;
            struct ds4_metal_args_mul_mv_id args;
            memset(&args, 0, sizeof(args));
            args.nei0 = (int32_t)n_groups;
            args.nei1 = (int32_t)n_tok;
            args.nbi1 = (uint64_t)n_groups * sizeof(int32_t);
            args.ne00 = (int32_t)group_dim;
            args.ne01 = (int32_t)rank;
            args.ne02 = (int32_t)n_groups;
            args.nb00 = 34u;
            args.nb01 = (group_dim / 32u) * 34u;
            args.nb02 = a_group_bytes;
            args.ne10 = (int32_t)group_dim;
            args.ne11 = (int32_t)n_groups;
            args.ne12 = 1;
            args.ne13 = (int32_t)n_tok;
            args.nb10 = sizeof(float);
            args.nb11 = (uint64_t)group_dim * sizeof(float);
            args.nb12 = (uint64_t)n_groups * (uint64_t)group_dim * sizeof(float);
            args.ne0 = (int32_t)rank;
            args.ne1 = (int32_t)n_groups;
            args.nb1 = (uint64_t)rank * sizeof(float);
            args.nr0 = 2;

            const NSUInteger rows_per_group = 2; // For Q8_0, nr0 = 2
            const NSUInteger row_groups = (rank + rows_per_group - 1) / rows_per_group;
            const NSUInteger pairs = (NSUInteger)n_groups * (NSUInteger)n_tok;
            const NSUInteger smem_bytes = 256;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_attn_out_low_q8_0];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:out_a->mtl_buf offset:out_a->offset atIndex:1];
            [enc setBuffer:heads->mtl_buf offset:heads->offset atIndex:2];
            [enc setBuffer:low->mtl_buf offset:low->offset atIndex:3];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    }

    /* CPU Fallback for mock buffers / tests */
    typedef struct {
        uint16_t d;
        int8_t   qs[32];
    } engine_block_q8_0_t;

    float *out = (float *)low->ptr;
    const float *x = (const float *)heads->ptr;
    uint64_t total_rows = (uint64_t)n_groups * rank;

    for (uint32_t tok = 0; tok < n_tok; tok++) {
        float *tok_out = out + (size_t)tok * total_rows;
        const float *tok_x = x + (size_t)tok * n_groups * group_dim;

        if (weight_type == GGML_TYPE_Q8_0) {
            const engine_block_q8_0_t *blocks = (const engine_block_q8_0_t *)out_a->ptr;
            uint64_t blocks_per_row = group_dim / 32;

            for (uint64_t r = 0; r < total_rows; r++) {
                uint32_t group = (uint32_t)(r / rank);
                const float *xg = tok_x + (size_t)group * group_dim;
                const engine_block_q8_0_t *row_blocks = blocks + r * blocks_per_row;

                float sum = 0.0f;
                for (uint64_t b = 0; b < blocks_per_row; b++) {
                    float d = fp16_to_fp32(row_blocks[b].d);
                    const int8_t *qs = row_blocks[b].qs;
                    const float *xb = xg + b * 32;
                    for (int k = 0; k < 32; k++) {
                        sum += d * (float)qs[k] * xb[k];
                    }
                }
                tok_out[r] = sum;
            }
        } else if (weight_type == GGML_TYPE_F32) {
            const float *w_f32 = (const float *)out_a->ptr;
            for (uint64_t r = 0; r < total_rows; r++) {
                uint32_t group = (uint32_t)(r / rank);
                const float *xg = tok_x + (size_t)group * group_dim;
                const float *row = w_f32 + r * group_dim;
                float sum = 0.0f;
                for (uint64_t i = 0; i < group_dim; i++) sum += row[i] * xg[i];
                tok_out[r] = sum;
            }
        } else {
            return 0;
        }
    }
    return 1;
}

int myds_metal_attn_output_a(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    ggml_type                  weight_type) {
    return myds_metal_attn_output_a_batch(low, out_a, heads, n_groups, group_dim, rank, 1, weight_type);
}

int myds_metal_moe_sum(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *expert_outs,
    const float               *weights,
    uint32_t                   n_activated,
    uint32_t                   dim) {
    if (!out || !expert_outs || !weights || n_activated == 0 || dim == 0) return 0;

    float *ov = (float *)out->ptr;
    const float *eo = (const float *)expert_outs->ptr;

    memset(ov, 0, dim * sizeof(float));
    for (uint32_t k = 0; k < n_activated; k++) {
        float w = weights[k];
        const float *src = eo + k * dim;
        for (uint32_t i = 0; i < dim; i++) {
            ov[i] += w * src[i];
        }
    }
    return 1;
}

struct ds4_metal_args_add3 {
    uint32_t n;
    uint32_t r2;
    uint32_t r3;
};

int myds_metal_add(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b,
    uint32_t                   count) {
    if (!out || !a || !b || count == 0) return 0;

    if (g_pso_add && a->mtl_buf && b->mtl_buf && out->mtl_buf) {
        @autoreleasepool {
            struct ds4_metal_args_add3 args;
            args.n = count;
            args.r2 = 1;
            args.r3 = 1;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_add];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:a->mtl_buf offset:(a ? a->offset : 0) atIndex:1];
            [enc setBuffer:b->mtl_buf offset:(b ? b->offset : 0) atIndex:2];
            [enc setBuffer:out->mtl_buf offset:(out ? out->offset : 0) atIndex:3];

            NSUInteger threads = (count < 256) ? count : 256;
            if (threads == 0) threads = 1;
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    }

    /* CPU Fallback */
    float *ov = (float *)out->ptr;
    const float *av = (const float *)a->ptr;
    const float *bv = (const float *)b->ptr;

    for (uint32_t i = 0; i < count; i++) {
        ov[i] = av[i] + bv[i];
    }
    return 1;
}

int myds_metal_softmax(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *x,
    uint32_t                   count) {
    if (!out || !x || count == 0) return 0;

    const float *xv = (const float *)x->ptr;
    float *ov = (float *)out->ptr;

    float max_val = -INFINITY;
    for (uint32_t i = 0; i < count; i++) {
        if (xv[i] > max_val) max_val = xv[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float exp_v = expf(xv[i] - max_val);
        ov[i] = exp_v;
        sum += exp_v;
    }

    float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < count; i++) {
        ov[i] *= inv_sum;
    }
    return 1;
}

void myds_metal_warm_buffer(const myds_metal_buffer_t *buf) {
    if (!buf || !buf->ptr || buf->size == 0) return;
    const size_t page_sz = (size_t)getpagesize();
    const uint8_t *p = (const uint8_t *)buf->ptr;
    volatile uint64_t dummy = 0;
    for (size_t off = 0; off < buf->size; off += page_sz) {
        dummy += p[off];
    }
    (void)dummy;
}

void myds_metal_warm_memory(const void *addr, size_t size) {
    if (!addr || size == 0) return;
    uint64_t memsize = 0;
    size_t mem_len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &mem_len, NULL, 0) != 0 || memsize == 0) {
        memsize = 16ull * 1024ull * 1024ull * 1024ull;
    }
    uint64_t max_warm = (uint64_t)(memsize * 0.85);
    if (size > max_warm) size = (size_t)max_warm;

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)addr, size, POSIX_MADV_WILLNEED);
#endif
    const size_t page_sz = (size_t)getpagesize();
    const uint8_t *p = (const uint8_t *)addr;
    const size_t chunk_size = 64 * 1024 * 1024; /* 64 MB */
    const size_t n_chunks = (size + chunk_size - 1) / chunk_size;

    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t chunk_idx) {
        size_t start = chunk_idx * chunk_size;
        size_t end = start + chunk_size;
        if (end > size) end = size;
        volatile uint64_t dummy = 0;
        for (size_t off = start; off < end; off += page_sz) {
            dummy += p[off];
        }
        (void)dummy;
    });
}
