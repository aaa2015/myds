#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "myds_metal.h"
#include "myds_log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
static id<MTLComputePipelineState> g_pso_mul_mv_q4_K = nil;
static id<MTLComputePipelineState> g_pso_softmax = nil;
static bool                        g_initialized = false;

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
        fn = [g_library newFunctionWithName:@"kernel_soft_max"];
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

        /* 6. Build Q4_K MatVec pipeline */
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_q4_K_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_mul_mv_q4_K = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        g_initialized = true;
        printf("myds_metal: Initialized successfully on GPU: %s\n", [[g_device name] UTF8String]);
        return 1;
    }
}

void myds_metal_cleanup(void) {
    if (!g_initialized) return;
    g_pso_rms_norm = nil;
    g_pso_swiglu = nil;
    g_pso_add = nil;
    g_pso_mul_mv_q8_0 = nil;
    g_pso_mul_mv_q4_K = nil;
    g_pso_softmax = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_initialized = false;
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

            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
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
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
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

            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_pso_swiglu];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:gate->mtl_buf offset:(gate ? gate->offset : 0) atIndex:1];
            [enc setBuffer:up->mtl_buf offset:(up ? up->offset : 0) atIndex:2];
            [enc setBuffer:out->mtl_buf offset:(out ? out->offset : 0) atIndex:3];

            NSUInteger threads = (dim < 256) ? dim : 256;
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
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

int myds_metal_gemv(
    myds_metal_buffer_t       *y,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *x,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    ggml_type                  weight_type) {
    if (!y || !W || !x || out_dim == 0 || in_dim == 0) return 0;

    const float *xv = (const float *)x->ptr;
    float *yv = (float *)y->ptr;

    /* Handle Q8_0 weights */
    if (weight_type == GGML_TYPE_Q8_0) {
        typedef struct {
            uint16_t d; /* half */
            int8_t   qs[32];
        } block_q8_0_t;

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

    /* Handle F32 weights */
    if (weight_type == GGML_TYPE_F32) {
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

    /* Handle F16 weights */
    if (weight_type == GGML_TYPE_F16) {
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

int myds_metal_moe_route(
    int32_t                   *out_expert_ids,
    float                     *out_expert_weights,
    const myds_metal_buffer_t *router_logits,
    uint32_t                   n_routed_experts,
    uint32_t                   n_activated_experts,
    float                      routed_scaling_factor) {
    if (!out_expert_ids || !out_expert_weights || !router_logits) return 0;
    if (n_routed_experts == 0 || n_activated_experts == 0 || n_activated_experts > n_routed_experts) return 0;

    const float *logits = (const float *)router_logits->ptr;

    /* Compute sigmoid gating scores for all routed experts */
    float *scores = (float *)malloc(n_routed_experts * sizeof(float));
    for (uint32_t e = 0; e < n_routed_experts; e++) {
        scores[e] = 1.0f / (1.0f + expf(-logits[e]));
    }

    /* Top-K selection */
    for (uint32_t k = 0; k < n_activated_experts; k++) {
        float max_s = -1e9f;
        int32_t best_e = -1;
        for (uint32_t e = 0; e < n_routed_experts; e++) {
            bool already_picked = false;
            for (uint32_t j = 0; j < k; j++) {
                if (out_expert_ids[j] == (int32_t)e) {
                    already_picked = true;
                    break;
                }
            }
            if (!already_picked && scores[e] > max_s) {
                max_s = scores[e];
                best_e = (int32_t)e;
            }
        }
        out_expert_ids[k] = best_e;
        out_expert_weights[k] = max_s;
    }

    /* Normalize top-k weights and apply routed scaling factor */
    float sum = 0.0f;
    for (uint32_t k = 0; k < n_activated_experts; k++) {
        sum += out_expert_weights[k];
    }
    float norm_factor = (sum > 1e-12f) ? (routed_scaling_factor / sum) : 1.0f;
    for (uint32_t k = 0; k < n_activated_experts; k++) {
        out_expert_weights[k] *= norm_factor;
    }

    free(scores);
    return 1;
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

int myds_metal_add(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b,
    uint32_t                   count) {
    if (!out || !a || !b || count == 0) return 0;

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

int myds_metal_synchronize(void) {
    if (!g_queue) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
        return 1;
    }
}
