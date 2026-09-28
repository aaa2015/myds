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
static id<MTLComputePipelineState> g_pso_moe_slots6_pair_swiglu_iq2_xxs = nil;
static id<MTLComputePipelineState> g_pso_moe_slots6_sum6_q2_k = nil;
static id<MTLComputePipelineState> g_pso_attn_out_low_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q8_0_bco = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f16 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f16_bco = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f32 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_f32_bco = nil;
static id<MTLSharedEvent>          g_shared_event = nil;
static uint64_t                    g_shared_event_val = 0;
static bool                        g_initialized = false;
static NSCache                    *g_exact_view_cache = nil;

static id<MTLBuffer> get_exact_buffer_view(const void *base_ptr, uint64_t offset, uint64_t len, NSUInteger *out_offset) {
    if (!g_device || !base_ptr || len == 0) return nil;
    const size_t page_sz = (size_t)getpagesize();
    const uintptr_t target_ptr = (uintptr_t)base_ptr + (uintptr_t)offset;
    const uintptr_t page_aligned_ptr = target_ptr & ~(page_sz - 1);
    const NSUInteger leading = (NSUInteger)(target_ptr - page_aligned_ptr);
    const NSUInteger page_len = (NSUInteger)(((len + leading + page_sz - 1) / page_sz) * page_sz);

    if (out_offset) *out_offset = leading;

    if (!g_exact_view_cache) {
        g_exact_view_cache = [[NSCache alloc] init];
        [g_exact_view_cache setCountLimit:16384];
    }

    NSString *key = [NSString stringWithFormat:@"%lx:%lx", (unsigned long)page_aligned_ptr, (unsigned long)page_len];
    id<MTLBuffer> buf = [g_exact_view_cache objectForKey:key];
    if (!buf) {
        buf = [g_device newBufferWithBytesNoCopy:(void *)page_aligned_ptr
                                          length:page_len
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
        if (buf) {
            [g_exact_view_cache setObject:buf forKey:key];
        }
    }
    return buf;
}

/* Batch command buffer state */
static id<MTLCommandBuffer>         g_batch_cb = nil;
static id<MTLComputeCommandEncoder> g_batch_enc = nil;
static uint64_t                     g_batch_commit_count = 0;
static int                          g_auto_barrier = 1;

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

        MTLCompileOptions *options = [MTLCompileOptions new];
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

        /* 2. Build SwiGLU pipeline */
        fn = [g_library newFunctionWithName:@"kernel_swiglu_f32"];
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

        pso_err = nil;
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32" constantValues:moe_mv_id_constants error:&pso_err];
        if (fn) {
            g_pso_moe_slots6_pair_swiglu_iq2_xxs = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_slots6_pair_swiglu_iq2_xxs) {
                printf("myds_metal: kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32 PSO error: %s\n", [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32 fn error: %s\n", [[pso_err localizedDescription] UTF8String]);
        }

        pso_err = nil;
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_slots6_q2_K_sum6_f32" constantValues:moe_mv_id_constants error:&pso_err];
        if (fn) {
            g_pso_moe_slots6_sum6_q2_k = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_slots6_sum6_q2_k) {
                printf("myds_metal: kernel_mul_mv_slots6_q2_K_sum6_f32 PSO error: %s\n", [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: kernel_mul_mv_slots6_q2_K_sum6_f32 fn error: %s\n", [[pso_err localizedDescription] UTF8String]);
        }

        fn = [g_library newFunctionWithName:@"kernel_dsv4_attn_out_low_q8_0_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_attn_out_low_q8_0 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 11. Build Batched MatMul (GEMM) pipelines with function constants 700/701 */
        MTLFunctionConstantValues *mm_c_bco0 = [[MTLFunctionConstantValues alloc] init];
        bool b_false = false;
        bool b_true = true;
        [mm_c_bco0 setConstantValue:&b_false type:MTLDataTypeBool atIndex:700];
        [mm_c_bco0 setConstantValue:&b_false type:MTLDataTypeBool atIndex:701];

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
    if (g_batch_enc) {
        [g_batch_enc endEncoding];
        g_batch_enc = nil;
    }
    if (g_batch_cb) {
        [g_batch_cb commit];
        [g_batch_cb waitUntilCompleted];
        g_batch_cb = nil;
    }
    g_pso_rms_norm = nil;
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
    g_pso_moe_slots6_pair_swiglu_iq2_xxs = nil;
    g_pso_moe_slots6_sum6_q2_k = nil;
    g_pso_attn_out_low_q8_0 = nil;
    g_shared_event = nil;
    g_shared_event_val = 0;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_exact_view_cache = nil;
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
    if (!g_batch_cb) return 1;
    close_compute_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    [cb commit];
    [cb waitUntilCompleted];
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
    if (!g_batch_cb || !g_shared_event || val == 0) return 0;
    close_compute_encoder();
    [g_batch_cb encodeWaitForEvent:g_shared_event value:val];
    return 1;
}

int myds_metal_commit_and_wait_event(uint64_t val) {
    if (!g_batch_cb) return 0;
    close_compute_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    [cb commit];
    g_batch_commit_count++;
    if (!g_shared_event) return 0;
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
    return 1;
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
                [cb waitUntilCompleted];
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
    if (!out || !gate || !up || dim == 0) return 0;

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

            NSUInteger threads = (dim < 256) ? dim : 256;
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
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

int myds_metal_gemv(
    myds_metal_buffer_t       *y,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *x,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    ggml_type                  weight_type) {
    if (!y || !W || !x || out_dim == 0 || in_dim == 0) return 0;

    /* 1. Handle Q8_0 weights */
    if (weight_type == GGML_TYPE_Q8_0) {
        if (g_pso_mul_mv_q8_0 && W->mtl_buf && x->mtl_buf && y->mtl_buf && (in_dim % 32 == 0)) {
            @autoreleasepool {
                const uint64_t row_bytes = (in_dim / 32u) * 34u;
                struct ds4_metal_args_mul_mv args;
                memset(&args, 0, sizeof(args));
                args.ne00 = (int32_t)in_dim;
                args.ne01 = (int32_t)out_dim;
                args.ne02 = 1;
                args.nb00 = 34;
                args.nb01 = row_bytes;
                args.nb02 = row_bytes * out_dim;
                args.nb03 = row_bytes * out_dim;
                args.ne10 = (int32_t)in_dim;
                args.ne11 = 1;
                args.ne12 = 1;
                args.nb10 = sizeof(float);
                args.nb11 = in_dim * sizeof(float);
                args.nb12 = in_dim * sizeof(float);
                args.nb13 = in_dim * sizeof(float);
                args.ne0 = (int32_t)out_dim;
                args.ne1 = 1;
                args.nr0 = 2;
                args.r2 = 1;
                args.r3 = 1;

                id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
                [enc setComputePipelineState:g_pso_mul_mv_q8_0];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:W->mtl_buf offset:(W ? W->offset : 0) atIndex:1];
                [enc setBuffer:x->mtl_buf offset:(x ? x->offset : 0) atIndex:2];
                [enc setBuffer:y->mtl_buf offset:(y ? y->offset : 0) atIndex:3];
                [enc setThreadgroupMemoryLength:(32 * 2 * sizeof(float)) atIndex:0];

                NSUInteger grid_x = (out_dim + 1) / 2;
                [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

                if (g_batch_cb) {
                    if (g_auto_barrier) {
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                } else {
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
                return 1;
            }
        }

        /* CPU Fallback for Q8_0 */
        typedef struct {
            uint16_t d; /* half */
            int8_t   qs[32];
        } block_q8_0_t;

        const float *xv = (const float *)x->ptr;
        float *yv = (float *)y->ptr;
        const block_q8_0_t *blocks = (const block_q8_0_t *)W->ptr;
        uint32_t n_blocks_per_row = in_dim / 32;

        for (uint32_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const block_q8_0_t *row_blocks = blocks + row * n_blocks_per_row;
            for (uint32_t b = 0; b < n_blocks_per_row; b++) {
                float scale = fp16_to_fp32(row_blocks[b].d);
                const int8_t *qs = row_blocks[b].qs;
                const float *xb = xv + b * 32;
                float dot = 0.0f;
                for (int i = 0; i < 32; i++) {
                    dot += (float)qs[i] * xb[i];
                }
                sum += dot * scale;
            }
            yv[row] = sum;
        }
        return 1;
    }

    /* 2. Handle F32 weights */
    if (weight_type == GGML_TYPE_F32) {
        if (g_pso_mul_mv_f32 && W->mtl_buf && x->mtl_buf && y->mtl_buf) {
            @autoreleasepool {
                const uint64_t row_bytes = in_dim * sizeof(float);
                struct ds4_metal_args_mul_mv args;
                memset(&args, 0, sizeof(args));
                args.ne00 = (int32_t)in_dim;
                args.ne01 = (int32_t)out_dim;
                args.ne02 = 1;
                args.nb00 = sizeof(float);
                args.nb01 = row_bytes;
                args.nb02 = row_bytes * out_dim;
                args.nb03 = row_bytes * out_dim;
                args.ne10 = (int32_t)in_dim;
                args.ne11 = 1;
                args.ne12 = 1;
                args.nb10 = sizeof(float);
                args.nb11 = in_dim * sizeof(float);
                args.nb12 = in_dim * sizeof(float);
                args.nb13 = in_dim * sizeof(float);
                args.ne0 = (int32_t)out_dim;
                args.ne1 = 1;
                args.nr0 = 2;
                args.r2 = 1;
                args.r3 = 1;

                id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
                [enc setComputePipelineState:g_pso_mul_mv_f32];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:W->mtl_buf offset:(W ? W->offset : 0) atIndex:1];
                [enc setBuffer:x->mtl_buf offset:(x ? x->offset : 0) atIndex:2];
                [enc setBuffer:y->mtl_buf offset:(y ? y->offset : 0) atIndex:3];
                [enc setThreadgroupMemoryLength:(32 * 2 * sizeof(float)) atIndex:0];

                NSUInteger grid_x = (out_dim + 1) / 2;
                [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

                if (g_batch_cb) {
                    if (g_auto_barrier) {
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                } else {
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
                return 1;
            }
        }

        /* CPU Fallback for F32 */
        const float *xv = (const float *)x->ptr;
        float *yv = (float *)y->ptr;
        const float *weights = (const float *)W->ptr;
        for (uint32_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const float *w_row = weights + row * in_dim;
            for (uint32_t col = 0; col < in_dim; col++) {
                sum += w_row[col] * xv[col];
            }
            yv[row] = sum;
        }
        return 1;
    }

    /* 3. Handle F16 weights */
    if (weight_type == GGML_TYPE_F16) {
        if (g_pso_mul_mv_f16 && W->mtl_buf && x->mtl_buf && y->mtl_buf) {
            @autoreleasepool {
                const uint64_t row_bytes = in_dim * sizeof(uint16_t);
                struct ds4_metal_args_mul_mv args;
                memset(&args, 0, sizeof(args));
                args.ne00 = (int32_t)in_dim;
                args.ne01 = (int32_t)out_dim;
                args.ne02 = 1;
                args.nb00 = sizeof(uint16_t);
                args.nb01 = row_bytes;
                args.nb02 = row_bytes * out_dim;
                args.nb03 = row_bytes * out_dim;
                args.ne10 = (int32_t)in_dim;
                args.ne11 = 1;
                args.ne12 = 1;
                args.nb10 = sizeof(float);
                args.nb11 = in_dim * sizeof(float);
                args.nb12 = in_dim * sizeof(float);
                args.nb13 = in_dim * sizeof(float);
                args.ne0 = (int32_t)out_dim;
                args.ne1 = 1;
                args.nr0 = 2;
                args.r2 = 1;
                args.r3 = 1;

                id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
                [enc setComputePipelineState:g_pso_mul_mv_f16];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:W->mtl_buf offset:(W ? W->offset : 0) atIndex:1];
                [enc setBuffer:x->mtl_buf offset:(x ? x->offset : 0) atIndex:2];
                [enc setBuffer:y->mtl_buf offset:(y ? y->offset : 0) atIndex:3];
                [enc setThreadgroupMemoryLength:(32 * 2 * sizeof(float)) atIndex:0];

                NSUInteger grid_x = (out_dim + 1) / 2;
                [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

                if (g_batch_cb) {
                    if (g_auto_barrier) {
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                } else {
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
                return 1;
            }
        }

        /* CPU Fallback for F16 */
        const float *xv = (const float *)x->ptr;
        float *yv = (float *)y->ptr;
        const uint16_t *weights = (const uint16_t *)W->ptr;
        for (uint32_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const uint16_t *w_row = weights + row * in_dim;
            for (uint32_t col = 0; col < in_dim; col++) {
                float wf = fp16_to_fp32(w_row[col]);
                sum += wf * xv[col];
            }
            yv[row] = sum;
        }
        return 1;
    }

    MYDS_LOG_E("myds_metal_gemv: unsupported GGML quantization type %d (%s) for GEMV out_dim=%u, in_dim=%u",
               (int)weight_type, ggml_type_name(weight_type), out_dim, in_dim);
    return 0;
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
            [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
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
                [cb waitUntilCompleted];
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
    (void)gate_type;
    (void)up_type;
    (void)down_type;

    if (!moe_out || !x_in || !gate_exps || !up_exps || !down_exps ||
        !selected_ids || !selected_weights || !scratch_gate || !scratch_up ||
        !scratch_mid || !scratch_down) {
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

        /* 1. Gate experts dispatch (IQ2_XXS) */
        uint64_t gate_row_bytes = (dim / 256u) * 66u;
        uint64_t gate_exp_bytes = (uint64_t)expert_dim * gate_row_bytes;
        struct ds4_metal_args_mul_mv_id gate_args = {
            .nei0 = (int32_t)n_activated_experts,
            .nei1 = 1,
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
        const NSUInteger pairs = n_activated_experts; /* 6 */
        const NSUInteger smem_iq2 = 256u * sizeof(uint64_t) + 128u * sizeof(uint8_t); /* 2176 */
        static bool logged_moe_mode = false;
        if (!logged_moe_mode) {
            printf("myds_metal: routed_moe mode = %s\n",
                   (g_pso_moe_slots6_pair_swiglu_iq2_xxs && g_pso_moe_slots6_sum6_q2_k) ? "SLOTS6 (2-step)" :
                   (g_pso_moe_pair_swiglu_iq2_xxs && g_pso_moe_sum6_q2_k) ? "FUSED (2-step)" : "FALLBACK (5-step)");
            fflush(stdout);
            logged_moe_mode = true;
        }

        uint64_t down_row_bytes = (expert_dim / 256u) * 84u;
        uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;
        struct ds4_metal_args_mul_mv_id down_args = {
            .nei0 = (int32_t)n_activated_experts,
            .nei1 = 1,
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
            .ne1 = 1,
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
            .rows = n_activated_experts,
            .gate_row_stride = (uint64_t)expert_dim * sizeof(float),
            .up_row_stride = (uint64_t)expert_dim * sizeof(float),
            .mid_row_stride = (uint64_t)expert_dim * sizeof(float),
            .weight_stride = sizeof(float),
            .write_clamped = 0,
            .clamp_value = clamp_val,
        };

        if (g_pso_moe_slots6_pair_swiglu_iq2_xxs && g_pso_moe_slots6_sum6_q2_k) {
            const int32_t *sel_ids = (const int32_t *)selected_ids->ptr;

            /* 1. Slots6 Fused Gate + Up + SwiGLU + Scaling dispatch (IQ2_XXS) */
            [enc setComputePipelineState:g_pso_moe_slots6_pair_swiglu_iq2_xxs];
            [enc setBytes:&gate_args length:sizeof(gate_args) atIndex:0];
            [enc setBytes:&swiglu_args length:sizeof(swiglu_args) atIndex:1];
            for (uint32_t s = 0; s < 6; s++) {
                int32_t exp_id = sel_ids[s];
                NSUInteger inner_gate = 0, inner_up = 0;
                id<MTLBuffer> buf_gate = (exp_id >= 0 && (uint32_t)exp_id < n_routed_experts) ?
                    get_exact_buffer_view(gate_exps->ptr, (uint64_t)exp_id * gate_exp_bytes, gate_exp_bytes, &inner_gate) : nil;
                id<MTLBuffer> buf_up   = (exp_id >= 0 && (uint32_t)exp_id < n_routed_experts) ?
                    get_exact_buffer_view(up_exps->ptr,   (uint64_t)exp_id * gate_exp_bytes, gate_exp_bytes, &inner_up) : nil;
                if (buf_gate && buf_up) {
                    [enc setBuffer:buf_gate offset:inner_gate atIndex:2 + s];
                    [enc setBuffer:buf_up   offset:inner_up   atIndex:8 + s];
                } else {
                    uint64_t off_gate = gate_exps->offset + (uint64_t)exp_id * gate_exp_bytes;
                    uint64_t off_up   = up_exps->offset   + (uint64_t)exp_id * gate_exp_bytes;
                    [enc setBuffer:gate_exps->mtl_buf offset:off_gate atIndex:2 + s];
                    [enc setBuffer:up_exps->mtl_buf   offset:off_up   atIndex:8 + s];
                }
            }
            [enc setBuffer:x_in->mtl_buf offset:x_in->offset atIndex:14];
            [enc setBuffer:scratch_gate->mtl_buf offset:scratch_gate->offset atIndex:15];
            [enc setBuffer:scratch_up->mtl_buf offset:scratch_up->offset atIndex:16];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:17];
            [enc setBuffer:selected_weights->mtl_buf offset:selected_weights->offset atIndex:18];
            [enc setThreadgroupMemoryLength:smem_iq2 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_gate, 1, pairs)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_gate, 1)];

            /* Inter-kernel barrier within same encoder: ensure scratch_mid is written before sum6 reads */
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* 2. Slots6 Fused Down + Sum6 direct accumulation dispatch (Q2_K) */
            [enc setComputePipelineState:g_pso_moe_slots6_sum6_q2_k];
            [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
            for (uint32_t s = 0; s < 6; s++) {
                int32_t exp_id = sel_ids[s];
                NSUInteger inner_down = 0;
                id<MTLBuffer> buf_down = (exp_id >= 0 && (uint32_t)exp_id < n_routed_experts) ?
                    get_exact_buffer_view(down_exps->ptr, (uint64_t)exp_id * down_exp_bytes, down_exp_bytes, &inner_down) : nil;
                if (buf_down) {
                    [enc setBuffer:buf_down offset:inner_down atIndex:1 + s];
                } else {
                    uint64_t off_down = down_exps->offset + (uint64_t)exp_id * down_exp_bytes;
                    [enc setBuffer:down_exps->mtl_buf offset:off_down atIndex:1 + s];
                }
            }
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:7];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_down, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_down, 1)];

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            return 1;
        } else if (g_pso_moe_pair_swiglu_iq2_xxs && g_pso_moe_sum6_q2_k) {
            /* Fused Gate + Up + SwiGLU + Scaling dispatch (IQ2_XXS) */
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

            /* Fused Down + Sum6 direct accumulation dispatch (Q2_K) */
            [enc setComputePipelineState:g_pso_moe_sum6_q2_k];
            [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
            [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:3];
            [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_down, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_down, 1)];

            if (g_batch_cb) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            return 1;
        } else {
            /* 1. Gate experts dispatch (IQ2_XXS) */
            [enc setComputePipelineState:g_pso_moe_mv_id_iq2_xxs];
            [enc setBytes:&gate_args length:sizeof(gate_args) atIndex:0];
            [enc setBuffer:gate_exps->mtl_buf offset:gate_exps->offset atIndex:1];
            [enc setBuffer:x_in->mtl_buf offset:x_in->offset atIndex:2];
            [enc setBuffer:scratch_gate->mtl_buf offset:scratch_gate->offset atIndex:3];
            [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
            [enc setThreadgroupMemoryLength:smem_iq2 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_gate, 1, pairs)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_gate, 1)];

            /* 2. Up experts dispatch (IQ2_XXS) */
            struct ds4_metal_args_mul_mv_id up_args = gate_args;
            [enc setComputePipelineState:g_pso_moe_mv_id_iq2_xxs];
            [enc setBytes:&up_args length:sizeof(up_args) atIndex:0];
            [enc setBuffer:up_exps->mtl_buf offset:up_exps->offset atIndex:1];
            [enc setBuffer:x_in->mtl_buf offset:x_in->offset atIndex:2];
            [enc setBuffer:scratch_up->mtl_buf offset:scratch_up->offset atIndex:3];
            [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
            [enc setThreadgroupMemoryLength:smem_iq2 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_gate, 1, pairs)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_gate, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* 3. SwiGLU + weight scaling dispatch */
            struct ds4_metal_dsv4_moe_swiglu_weight_args swiglu_args = {
                .width = expert_dim,
                .rows = n_activated_experts,
                .gate_row_stride = (uint64_t)expert_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp_val,
            };

            NSUInteger nth_swiglu = 256;
            if (nth_swiglu > expert_dim) nth_swiglu = expert_dim;
            [enc setComputePipelineState:g_pso_moe_swiglu_weight];
            [enc setBytes:&swiglu_args length:sizeof(swiglu_args) atIndex:0];
            [enc setBuffer:scratch_gate->mtl_buf offset:scratch_gate->offset atIndex:1];
            [enc setBuffer:scratch_up->mtl_buf offset:scratch_up->offset atIndex:2];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:3];
            [enc setBuffer:selected_weights->mtl_buf offset:selected_weights->offset atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(n_activated_experts, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(nth_swiglu, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* 4. Down experts dispatch (Q2_K) */
            uint64_t down_row_bytes = (expert_dim / 256u) * 84u;
            uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;
            struct ds4_metal_args_mul_mv_id down_args = {
                .nei0 = (int32_t)n_activated_experts,
                .nei1 = 1,
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
                .ne1 = (int32_t)n_activated_experts,
                .nb1 = (uint64_t)dim * sizeof(float),
                .nr0 = 2,
                .tp_rank = 0,
                .tp_world = 0,
                .tp_addend = 0,
                .tp_expert_base = 0,
            };

            const NSUInteger nr0_down = 2;
            const NSUInteger nsg_down = 2;
            const NSUInteger rows_per_group_down = nr0_down * nsg_down; /* 4 */
            const NSUInteger row_groups_down = (dim + rows_per_group_down - 1) / rows_per_group_down;

            [enc setComputePipelineState:g_pso_moe_mv_id_q2_k];
            [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
            [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
            [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
            [enc setBuffer:scratch_down->mtl_buf offset:scratch_down->offset atIndex:3];
            [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(row_groups_down, 1, pairs)
                threadsPerThreadgroup:MTLSizeMake(32, nsg_down, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* 5. Sum6 dispatch */
            struct ds4_metal_dsv4_moe_sum6_args sum_args = {
                .width = dim,
                .tokens = 1,
                .src_token_stride = (uint64_t)n_activated_experts * (uint64_t)dim * sizeof(float),
                .dst_token_stride = (uint64_t)dim * sizeof(float),
            };

            NSUInteger nth_sum = 256;
            if (nth_sum > dim) nth_sum = dim;
            [enc setComputePipelineState:g_pso_moe_sum6];
            [enc setBytes:&sum_args length:sizeof(sum_args) atIndex:0];
            [enc setBuffer:scratch_down->mtl_buf offset:scratch_down->offset atIndex:1];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(nth_sum, 1, 1)];
        }

        if (g_batch_cb) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        return 1;
    }
}

int myds_metal_attn_output_a(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    ggml_type                  weight_type) {
    if (!low || !out_a || !heads || n_groups == 0 || group_dim == 0 || rank == 0) return 0;

    if (weight_type == GGML_TYPE_Q8_0 && g_pso_attn_out_low_q8_0 && out_a->mtl_buf && heads->mtl_buf && low->mtl_buf && (group_dim % 32 == 0)) {
        @autoreleasepool {
            const uint64_t a_group_bytes = (uint64_t)rank * (group_dim / 32u) * 34u;
            struct ds4_metal_args_mul_mv_id args;
            memset(&args, 0, sizeof(args));
            args.nei0 = (int32_t)n_groups;
            args.nei1 = 1;
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
            args.ne13 = 1;
            args.nb10 = sizeof(float);
            args.nb11 = (uint64_t)group_dim * sizeof(float);
            args.nb12 = (uint64_t)n_groups * (uint64_t)group_dim * sizeof(float);
            args.ne0 = (int32_t)rank;
            args.ne1 = (int32_t)n_groups;
            args.nb1 = (uint64_t)rank * sizeof(float);
            args.nr0 = 2;

            const NSUInteger rows_per_group = 2; // For Q8_0, nr0 = 2
            const NSUInteger row_groups = (rank + rows_per_group - 1) / rows_per_group;
            const NSUInteger pairs = n_groups;
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
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
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

    if (weight_type == GGML_TYPE_Q8_0) {
        const engine_block_q8_0_t *blocks = (const engine_block_q8_0_t *)out_a->ptr;
        uint64_t blocks_per_row = group_dim / 32;

        for (uint64_t r = 0; r < total_rows; r++) {
            uint32_t group = (uint32_t)(r / rank);
            const float *xg = x + (size_t)group * group_dim;
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
            out[r] = sum;
        }
        return 1;
    } else if (weight_type == GGML_TYPE_F32) {
        const float *w_f32 = (const float *)out_a->ptr;
        for (uint64_t r = 0; r < total_rows; r++) {
            uint32_t group = (uint32_t)(r / rank);
            const float *xg = x + (size_t)group * group_dim;
            const float *row = w_f32 + r * group_dim;
            float sum = 0.0f;
            for (uint64_t i = 0; i < group_dim; i++) sum += row[i] * xg[i];
            out[r] = sum;
        }
        return 1;
    }
    return 0;
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
                [cb waitUntilCompleted];
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

