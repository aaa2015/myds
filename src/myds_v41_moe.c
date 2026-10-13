#include "myds_v41_moe.h"
#include "myds_v41_decode.h"
#include "myds_v41.h"
#include "myds_log.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

/* 必须与 metal/myds_v41_moe.metal 的 myds_v41_route_args 逐字段一致 */
struct route_args { uint32_t count, top_k, op; float scale; uint32_t rows, row, row4; };
struct sort_args {
    int32_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne0, ne1, ne2, ne3, top_k;
    uint32_t causal_start, causal_ratio;
};
struct glu_args {
    int32_t ne00; uint64_t nb01;
    int32_t ne10; uint64_t nb11;
    int32_t ne0; uint64_t nb1;
    int32_t i00, i10; float alpha, limit; int32_t round_bf16;
};
/* 必须与 metal/myds_v41_moe.metal 的 myds_v41_shared_swiglu_args 逐字段一致 */
struct myds_v41_shared_swiglu_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t row_bytes;
    float clamp;
};

static int sized(const myds_metal_buffer_t *buf, size_t count) {
    return buf && count <= myds_metal_buffer_size(buf) / sizeof(float);
}
static int dispatch(const char *kernel, const void *args, size_t size,
    const myds_metal_buffer_t *const *buffers, size_t count,
    uint32_t groups, uint32_t threads, size_t scratch) {
    const uint32_t grid[3] = {groups, 1, 1}, block[3] = {threads, 1, 1};
    return myds_metal_dispatch_kernel(kernel, args, size, buffers, count, grid, block, scratch);
}

static inline int get_route_fuse_level(void) {
    const char *v = getenv("MYDS_MOE_ROUTE_FUSE");
    if (!v) return 2; /* 默认启用全部融合 (all-in-one 单 kernel 派发) */
    if (strcmp(v, "0") == 0) return 0; /* 禁用融合 (7 个小 kernel) */
    if (strcmp(v, "1") == 0) return 1; /* 3 个 kernel (prepare + argsort + finish) */
    return atoi(v); /* >= 2: 全融合 */
}

static _Atomic uint64_t route_shuffle_requests, route_shuffle_encoded, route_shuffle_failures;
int myds_v41_route_simd_shuffle_mode(void) {
    const char *v = getenv("MYDS_ROUTE_SIMD_SHUFFLE");
    if (!v || !strcmp(v, "0")) return 0;
    if (!strcmp(v, "1")) return 1;
    MYDS_LOG_E("MYDS_ROUTE_SIMD_SHUFFLE must be exactly 0 or 1");
    return -1;
}
void myds_v41_route_simd_shuffle_counters(
    myds_v41_route_simd_shuffle_counters_t *out, bool reset) {
#define ROUTE_SHUFFLE_COUNT(field) (reset ? atomic_exchange_explicit(&route_shuffle_##field, 0, memory_order_relaxed) \
                                        : atomic_load_explicit(&route_shuffle_##field, memory_order_relaxed))
    const myds_v41_route_simd_shuffle_counters_t value = {
        .requests = ROUTE_SHUFFLE_COUNT(requests), .encoded = ROUTE_SHUFFLE_COUNT(encoded),
        .failures = ROUTE_SHUFFLE_COUNT(failures)};
    if (out) *out = value;
#undef ROUTE_SHUFFLE_COUNT
}

int myds_v41_route_batch(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probs, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t experts, uint32_t top_k, float scale, uint32_t rows) {
    if (experts != 384 || top_k != 6 || !isfinite(scale) || scale <= 0 || rows == 0 ||
        !sized(selected, (size_t)top_k * rows) || !sized(weights, (size_t)top_k * rows) ||
        !sized(probs, (size_t)experts * rows) || !sized(scores, (size_t)experts * rows) ||
        !sized(logits, (size_t)experts * rows) || !sized(bias, experts)) return 0;
    const uint64_t total = (uint64_t)experts * (uint64_t)rows;
    struct route_args args = {(uint32_t)total, top_k, 0, scale, rows, experts, experts / 4u};

    const int shuffle_mode = myds_v41_route_simd_shuffle_mode();
    if (shuffle_mode < 0) return 0;
    if (shuffle_mode == 1) {
        atomic_fetch_add_explicit(&route_shuffle_requests, 1, memory_order_relaxed);
        const myds_metal_buffer_t *fused_bufs[] = {logits, bias, probs, scores, selected, weights};
        const size_t shmem_bytes = 512 * sizeof(int32_t) + 512 * sizeof(float) +
                                   experts * sizeof(float) + 32 * sizeof(float);
        if (!dispatch("kernel_myds_v41_route_simd_shuffle", &args, sizeof(args), fused_bufs, 6,
                      rows, 512, shmem_bytes)) {
            atomic_fetch_add_explicit(&route_shuffle_failures, 1, memory_order_relaxed);
            MYDS_LOG_E("V4.1 SIMD shuffle route dispatch failed; refusing fallback");
            return 0;
        }
        atomic_fetch_add_explicit(&route_shuffle_encoded, 1, memory_order_relaxed);
        const myds_metal_buffer_t *out_bufs[] = {selected, weights};
        myds_metal_barrier_buffers(out_bufs, 2);
        return 1;
    }

    const int fuse_level = get_route_fuse_level();
    if (fuse_level >= 2) {
        const myds_metal_buffer_t *fused_bufs[] = {logits, bias, probs, scores, selected, weights};
        /* 共享内存: 512 ints + 512 floats + 384 floats + 32 floats = 5760 字节 */
        const size_t shmem_bytes = 512 * sizeof(int32_t) + 512 * sizeof(float) +
                                   experts * sizeof(float) + 32 * sizeof(float);
        if (dispatch("kernel_myds_v41_route_fused", &args, sizeof(args), fused_bufs, 6,
                     rows, 512, shmem_bytes)) {
            const myds_metal_buffer_t *out_bufs[] = {selected, weights};
            myds_metal_barrier_buffers(out_bufs, 2);
            return 1;
        }
        /* 若全融合派发失败，平滑退回到分阶段路径 */
    }

    const uint32_t tgroups = (uint32_t)((total / 4u + 31u) / 32u);

    int fused_prep_ok = 0;
    if (fuse_level >= 1) {
        const myds_metal_buffer_t *prep[] = {logits, bias, probs, scores};
        fused_prep_ok = dispatch("kernel_myds_v41_route_prepare", &args, sizeof(args), prep, 4,
                                 tgroups, 32, 0);
    }

    if (!fused_prep_ok) {
        const myds_metal_buffer_t *softplus[] = {logits, bias, probs};
        args.op = 0;
        if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), softplus, 3,
                      tgroups, 32, 0)) {
            MYDS_LOG_E("V4.1 route transform dispatch failed");
            return 0;
        }
        myds_metal_barrier_buffer(probs);
        const myds_metal_buffer_t *sqrt_buffers[] = {probs, bias, probs};
        args.op = 1;
        if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), sqrt_buffers, 3,
                      tgroups, 32, 0)) return 0;
        myds_metal_barrier_buffer(probs);
        const myds_metal_buffer_t *add[] = {probs, bias, scores};
        args.op = 2;
        if (!dispatch("kernel_myds_v41_route_transform", &args, sizeof(args), add, 3,
                      tgroups, 32, 0)) {
            MYDS_LOG_E("V4.1 route bias dispatch failed");
            return 0;
        }
    }

    myds_metal_barrier_buffer(scores);

    /* One row per threadgroup: the kernel already indexes rows by tgpig[0] % ne01. */
    const uint64_t row_bytes = (uint64_t)experts * sizeof(float);
    struct sort_args sort = {.ne00=(int32_t)experts,.ne01=(int32_t)rows,.ne02=1,.ne03=1,
        .nb00=4,.nb01=row_bytes,.nb02=row_bytes,.nb03=row_bytes,.ne0=(int32_t)top_k,
        .ne1=1,.ne2=1,.ne3=1,.top_k=(int32_t)top_k};
    const myds_metal_buffer_t *sort_buffers[] = {scores, selected};
    if (!dispatch("kernel_argsort_f32_i32_desc", &sort, sizeof(sort), sort_buffers, 2,
                  rows, 512, 512 * (sizeof(int32_t) + sizeof(float)))) {
        MYDS_LOG_E("V4.1 route top-k dispatch failed");
        return 0;
    }

    const myds_metal_buffer_t *finish_in[] = {probs, selected};
    myds_metal_barrier_buffers(finish_in, 2);

    if (fuse_level >= 1) {
        const myds_metal_buffer_t *finish[] = {probs, selected, weights, scores};
        if (dispatch("kernel_myds_v41_route_finish", &args, sizeof(args), finish, 4, rows, top_k, 128)) {
            const myds_metal_buffer_t *out_bufs[] = {selected, weights};
            myds_metal_barrier_buffers(out_bufs, 2);
            return 1;
        }
    }

    const myds_metal_buffer_t *sum[] = {probs, selected, weights, scores};
    const myds_metal_buffer_t *divide[] = {weights, scores};
    const myds_metal_buffer_t *multiply[] = {weights};
    int ok = dispatch("kernel_myds_v41_route_sum", &args, sizeof(args), sum, 4, rows, top_k, 128);
    const myds_metal_buffer_t *div_in[] = {weights, scores};
    myds_metal_barrier_buffers(div_in, 2);
    ok = ok && dispatch("kernel_myds_v41_route_divide", &args, sizeof(args), divide, 2, rows, top_k, 0);
    myds_metal_barrier_buffer(weights);
    ok = ok && dispatch("kernel_myds_v41_route_scale", &args, sizeof(args), multiply, 1, rows, top_k, 0);
    const myds_metal_buffer_t *out_bufs[] = {selected, weights};
    myds_metal_barrier_buffers(out_bufs, 2);
    return ok;
}

int myds_v41_route(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probs, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t experts, uint32_t top_k, float scale) {
    return myds_v41_route_batch(selected, weights, probs, scores, logits, bias,
                                experts, top_k, scale, 1);
}

int myds_v41_shared_swiglu(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *gate, const myds_metal_buffer_t *up,
    uint32_t width, float clamp) {
    if (!width || width > INT32_MAX || !isfinite(clamp) || clamp < 0 ||
        !sized(out,width) || !sized(gate,width) || !sized(up,width)) return 0;
    struct glu_args args = {.ne00=(int32_t)width,.nb01=(uint64_t)width*4,
        .ne10=(int32_t)width,.nb11=(uint64_t)width*4,.ne0=(int32_t)width,
        .nb1=(uint64_t)width*4,.alpha=1,.limit=clamp,.round_bf16=1};
    const myds_metal_buffer_t *buffers[] = {gate,up,out};
    return dispatch("kernel_swiglu_flat_f32", &args, sizeof(args), buffers, 3,
                    (width+255)/256,256,0);
}

int myds_dspark_shared_swiglu(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *gate, const myds_metal_buffer_t *up,
    uint32_t width, float clamp_val) {
    if (!width || width > INT32_MAX || !isfinite(clamp_val) || clamp_val < 0 ||
        !sized(out, width) || !sized(gate, width) || !sized(up, width)) return 0;
    struct glu_args args = {.ne00=(int32_t)width,.nb01=(uint64_t)width*4,
        .ne10=(int32_t)width,.nb11=(uint64_t)width*4,.ne0=(int32_t)width,
        .nb1=(uint64_t)width*4,.alpha=1,.limit=clamp_val,.round_bf16=0};
    const myds_metal_buffer_t *buffers[] = {gate, up, out};
    if (!dispatch("kernel_swiglu_flat_f32", &args, sizeof(args), buffers, 3,
                   (width+255)/256, 256, 0)) return 0;
    myds_metal_barrier_buffer(out);
    return 1;
}

static inline bool is_shared_fuse_enabled(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        const char *v = getenv("MYDS_SHARED_FUSE");
        cached = (v && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

int myds_v41_shared_gate_up_swiglu(myds_metal_buffer_t *out,
                                  const myds_metal_buffer_t *gate,
                                  const myds_metal_buffer_t *up,
                                  const myds_metal_buffer_t *input,
                                  uint32_t out_dim, uint32_t in_dim,
                                  ggml_type gate_type, ggml_type up_type,
                                  float clamp) {
    if (!out || !gate || !up || !input || !out_dim || !in_dim ||
        out_dim > INT32_MAX || in_dim > INT32_MAX ||
        !isfinite(clamp) || clamp < 0 ||
        myds_metal_buffer_offset(out) % 16u != 0) return 0;

    if (gate_type == GGML_TYPE_Q8_0 && up_type == GGML_TYPE_Q8_0) {
        if (in_dim % 32u) return 0;
        const uint32_t row_bytes = (in_dim / 32u) * 34u;
        if (myds_metal_buffer_size(gate) / row_bytes < out_dim ||
            myds_metal_buffer_size(up) / row_bytes < out_dim ||
            myds_metal_buffer_size(input) / 4u < in_dim ||
            myds_metal_buffer_size(out) / 4u < out_dim) return 0;

        struct myds_v41_shared_swiglu_args args = {
            .in_dim = in_dim,
            .out_dim = out_dim,
            .row_bytes = row_bytes,
            .clamp = clamp
        };

        static int nr0_cached = 0;
        if (!nr0_cached) {
            const char *v = getenv("MYDS_SHARED_NR0");
            nr0_cached = (v && strcmp(v, "2") == 0) ? 2 : 4;
        }
        const uint32_t nr0 = (uint32_t)nr0_cached;
        const char *kernel_name = (nr0 == 4) ? "kernel_myds_v41_shared_gate_up_swiglu_r4"
                                             : "kernel_myds_v41_shared_gate_up_swiglu";

        const myds_metal_buffer_t *buffers[] = { gate, up, input, out };
        const uint32_t groups = (out_dim + nr0 - 1u) / nr0;
        const uint32_t threads = 128u; // 32 * NSG (NSG=4)
        const size_t shmem_bytes = 2u * nr0 * 32u * sizeof(float);

        return dispatch(kernel_name, &args, sizeof(args), buffers, 4,
                        groups, threads, shmem_bytes);
    } else if (gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K) {
        if (in_dim % 256u) return 0;
        const uint32_t row_bytes = (in_dim / 256u) * 144u;
        if (myds_metal_buffer_size(gate) / row_bytes < out_dim ||
            myds_metal_buffer_size(up) / row_bytes < out_dim ||
            myds_metal_buffer_size(input) / 4u < in_dim ||
            myds_metal_buffer_size(out) / 4u < out_dim) return 0;

        struct myds_v41_shared_swiglu_args args = {
            .in_dim = in_dim,
            .out_dim = out_dim,
            .row_bytes = row_bytes,
            .clamp = clamp
        };

        const char *kernel_name = "kernel_myds_v41_shared_gate_up_swiglu_q4";
        const myds_metal_buffer_t *buffers[] = { gate, up, input, out };
        const uint32_t groups = (out_dim + 8u - 1u) / 8u; // NSG=4 * NR0=2 = 8 rows per threadgroup
        const uint32_t threads = 128u; // 32 * NSG (NSG=4)
        const size_t shmem_bytes = 0;

        return dispatch(kernel_name, &args, sizeof(args), buffers, 4,
                        groups, threads, shmem_bytes);
    }
    return 0;
}

static bool shared_pair_overlap(const myds_metal_buffer_t *a, uint64_t a_bytes,
                                const myds_metal_buffer_t *b, uint64_t b_bytes) {
    const uintptr_t first = (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)a);
    const uintptr_t second = (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)b);
    return first <= second ? second - first < a_bytes : first - second < b_bytes;
}

/* Shared3 is a distinct candidate provider, never an implicit retry. Its
 * caller owns transaction restoration after any partial encoding failure. */
static _Atomic uint64_t routed_shared3_requests, routed_shared3_encoded, routed_shared3_failures;
void myds_v41_depth2_routed_shared3_counters(
    myds_v41_depth2_routed_shared3_counters_t *out, bool reset) {
#define ROUTED_COUNT(field) (reset ? atomic_exchange_explicit(&routed_shared3_##field, 0, memory_order_relaxed) \
                                  : atomic_load_explicit(&routed_shared3_##field, memory_order_relaxed))
    const myds_v41_depth2_routed_shared3_counters_t value = {
        .requests = ROUTED_COUNT(requests), .encoded = ROUTED_COUNT(encoded), .failures = ROUTED_COUNT(failures)};
    if (out) *out = value;
#undef ROUTED_COUNT
}

struct myds_moe_shared3_args { uint32_t dim, width, experts; float clamp_value; };
static bool routed_shared3_buffer(const myds_metal_buffer_t *b, uint64_t bytes, uint32_t alignment) {
    return b && bytes <= myds_metal_buffer_size(b) &&
        (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)b) % alignment == 0;
}

int myds_v41_routed_moe_shared3(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *input, const myds_metal_buffer_t *gate,
    const myds_metal_buffer_t *up, const myds_metal_buffer_t *down,
    const myds_metal_buffer_t *ids, const myds_metal_buffer_t *weights,
    myds_metal_buffer_t *mid, uint32_t dim, uint32_t width, uint32_t experts,
    ggml_type gate_type, ggml_type up_type, ggml_type down_type, float clamp_value) {
    atomic_fetch_add_explicit(&routed_shared3_requests, 1, memory_order_relaxed);
    if (!dim || dim > INT32_MAX || dim % 256u || !width || width > 2304u ||
        width % 256u || experts < 6u || experts > 384u || !isfinite(clamp_value) ||
        clamp_value < 0 || gate_type != GGML_TYPE_IQ2_XXS || up_type != GGML_TYPE_IQ2_XXS ||
        down_type != GGML_TYPE_Q2_K) goto fail;
    const uint64_t gate_bytes = (uint64_t)experts * width * (dim / 256u) * 66u;
    const uint64_t down_bytes = (uint64_t)experts * dim * (width / 256u) * 84u;
    const uint64_t input_bytes = (uint64_t)3u * dim * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)18u * width * sizeof(float);
    const myds_metal_buffer_t *read[] = {input, gate, up, down, ids, weights};
    const uint64_t read_bytes[] = {input_bytes, gate_bytes, gate_bytes, down_bytes,
                                  18u * sizeof(int32_t), 18u * sizeof(float)};
    const uint32_t alignment[] = {16u, 2u, 2u, 2u, 4u, 4u};
    if (!routed_shared3_buffer(out, input_bytes, 4u) ||
        !routed_shared3_buffer(mid, mid_bytes, 4u) ||
        shared_pair_overlap(out, input_bytes, mid, mid_bytes)) goto fail;
    for (uint32_t i = 0; i < 6; ++i)
        if (!routed_shared3_buffer(read[i], read_bytes[i], alignment[i]) ||
            shared_pair_overlap(out, input_bytes, read[i], read_bytes[i]) ||
            shared_pair_overlap(mid, mid_bytes, read[i], read_bytes[i])) goto fail;
    const struct myds_moe_shared3_args args = {dim, width, experts, clamp_value};
    /* Dynamic selected IDs are trusted route output, not host-readable while
     * their producer is active. Kernels also bounds-check before blob reads. */
    const myds_metal_buffer_t *inputs[] = {input, ids, weights};
    myds_metal_barrier_buffers(inputs, 3);
    const myds_metal_buffer_t *gu[] = {gate, up, input, ids, weights, mid};
    const uint32_t gu_grid[3] = {(width + 15u) / 16u, 1u, 18u};
    const uint32_t gu_threads[3] = {128u, 1u, 1u};
    if (!myds_metal_dispatch_kernel("kernel_myds_v41_iq2_gu_shared3", &args, sizeof(args),
        gu, 6, gu_grid, gu_threads, 2176u)) goto fail;
    myds_metal_barrier_buffer(mid);
    const myds_metal_buffer_t *down_buffers[] = {down, mid, ids, out};
    const uint32_t down_grid[3] = {dim / 2u, 1u, 1u}, down_threads[3] = {32u, 1u, 1u};
    const size_t cache_bytes = (size_t)18u * 2u * (width / 256u) * 84u;
    if (!myds_metal_dispatch_kernel("kernel_myds_v41_q2_down_shared3", &args, sizeof(args),
        down_buffers, 4, down_grid, down_threads, (cache_bytes + 15u) & ~(size_t)15u)) goto fail;
    myds_metal_barrier_buffer(out);
    atomic_fetch_add_explicit(&routed_shared3_encoded, 1, memory_order_relaxed);
    return 1;
fail:
    atomic_fetch_add_explicit(&routed_shared3_failures, 1, memory_order_relaxed);
    return 0;
}

int myds_v41_shared_gate_up_swiglu_pair(myds_metal_buffer_t *const out[2],
                                      const myds_metal_buffer_t *gate,
                                      const myds_metal_buffer_t *up,
                                      myds_metal_buffer_t *const input[2],
                                      uint32_t out_dim, uint32_t in_dim,
                                      ggml_type gate_type, ggml_type up_type,
                                      float clamp) {
    if (!out || !input || !out[0] || !out[1] || !input[0] || !input[1] ||
        !gate || !up || !out_dim || !in_dim ||
        out_dim > INT32_MAX || in_dim > INT32_MAX ||
        !isfinite(clamp) || clamp < 0 || gate_type != up_type) return 0;
    const uint32_t block = gate_type == GGML_TYPE_Q4_K ? 256u : 32u;
    const uint32_t bytes = gate_type == GGML_TYPE_Q4_K ? 144u : 34u;
    if ((gate_type != GGML_TYPE_Q4_K && gate_type != GGML_TYPE_Q8_0) ||
        in_dim % block) return 0;
    const uint64_t row_bytes = (uint64_t)(in_dim / block) * bytes;
    if (myds_metal_buffer_size(gate) / row_bytes < out_dim ||
        myds_metal_buffer_size(up) / row_bytes < out_dim) return 0;
    const uint64_t output_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t input_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t weight_bytes = row_bytes * out_dim;
    if (!myds_metal_buffer_data((myds_metal_buffer_t *)gate) ||
        !myds_metal_buffer_data((myds_metal_buffer_t *)up)) return 0;
    for (uint32_t t = 0; t < 2; ++t)
        if (!sized(out[t], out_dim) || !sized(input[t], in_dim) ||
            !myds_metal_buffer_data(out[t]) || !myds_metal_buffer_data(input[t]) ||
            myds_metal_buffer_offset(out[t]) % 16u ||
            myds_metal_buffer_offset(input[t]) % sizeof(float)) return 0;
    if (shared_pair_overlap(out[0], output_bytes, out[1], output_bytes)) return 0;
    for (uint32_t t = 0; t < 2; ++t) {
        if (shared_pair_overlap(out[t], output_bytes, gate, weight_bytes) ||
            shared_pair_overlap(out[t], output_bytes, up, weight_bytes)) return 0;
        for (uint32_t s = 0; s < 2; ++s)
            if (shared_pair_overlap(out[t], output_bytes, input[s], input_bytes)) return 0;
    }
    /* Keep the existing scalar kernel and its compiled arithmetic. The
     * shared2-only real-model arm first differs at layer-38 Q1 raw KV, even
     * though the previous clamp=0 component fixture passed. The intermediate
     * GU output has not been tapped, so no compiler root cause is asserted. */
    for (uint32_t t = 0; t < 2; ++t)
        if (!myds_v41_shared_gate_up_swiglu(out[t], gate, up, input[t],
            out_dim, in_dim, gate_type, up_type, clamp)) return 0;
    return 1;
}

/* Three independent inputs share each quantized Gate/Up load in a distinct
 * kernel. The old shared2 diagnostic is deliberately not used here. This
 * provider requires scalar-GPU byte parity and real checkpoint validation
 * before its caller can enable it in production. */
static int shared_gu_shared3_encode(myds_metal_buffer_t *const out[3],
                                         const myds_metal_buffer_t *gate,
                                         const myds_metal_buffer_t *up,
                                         myds_metal_buffer_t *const input[3],
                                         uint32_t out_dim, uint32_t in_dim,
                                         ggml_type gate_type, ggml_type up_type,
                                         float clamp) {
    if (!out || !input || !gate || !up || !out_dim || !in_dim ||
        out_dim > INT32_MAX || in_dim > INT32_MAX ||
        !isfinite(clamp) || clamp < 0 || gate_type != up_type ||
        (gate_type != GGML_TYPE_Q4_K && gate_type != GGML_TYPE_Q8_0)) return 0;
    const uint32_t block = gate_type == GGML_TYPE_Q4_K ? 256u : 32u;
    const uint32_t bytes = gate_type == GGML_TYPE_Q4_K ? 144u : 34u;
    if (in_dim % block) return 0;
    const uint64_t row_bytes = (uint64_t)(in_dim / block) * bytes;
    if (row_bytes > UINT32_MAX ||
        myds_metal_buffer_size(gate) / row_bytes < out_dim ||
        myds_metal_buffer_size(up) / row_bytes < out_dim ||
        !myds_metal_buffer_data((myds_metal_buffer_t *)gate) ||
        !myds_metal_buffer_data((myds_metal_buffer_t *)up) ||
        myds_metal_buffer_offset(gate) % sizeof(uint16_t) ||
        myds_metal_buffer_offset(up) % sizeof(uint16_t)) return 0;
    const uint64_t output_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t input_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t weight_bytes = row_bytes * out_dim;
    /* Preflight every row before encoding anything: output views may be
     * separately allocated or strided, but must not overlap any reader. */
    for (uint32_t t = 0; t < 3; ++t) {
        if (!sized(out[t], out_dim) || !sized(input[t], in_dim) ||
            !myds_metal_buffer_data(out[t]) || !myds_metal_buffer_data(input[t]) ||
            myds_metal_buffer_offset(out[t]) % 16u ||
            myds_metal_buffer_offset(input[t]) % sizeof(float)) return 0;
    }
    for (uint32_t t = 0; t < 3; ++t) {
        if (shared_pair_overlap(out[t], output_bytes, gate, weight_bytes) ||
            shared_pair_overlap(out[t], output_bytes, up, weight_bytes)) return 0;
        for (uint32_t s = 0; s < 3; ++s) {
            if (s < t && shared_pair_overlap(out[t], output_bytes, out[s], output_bytes)) return 0;
            if (shared_pair_overlap(out[t], output_bytes, input[s], input_bytes)) return 0;
        }
    }
    const struct myds_v41_shared_swiglu_args args = {
        .in_dim = in_dim, .out_dim = out_dim,
        .row_bytes = (uint32_t)row_bytes, .clamp = clamp
    };
    const myds_metal_buffer_t *buffers[] = {
        gate, up, input[0], input[1], input[2], out[0], out[1], out[2]
    };
    if (gate_type == GGML_TYPE_Q4_K)
        return dispatch("kernel_myds_v41_shared_gate_up_swiglu_q4_shared3",
                        &args, sizeof(args), buffers, 8,
                        (out_dim + 7u) / 8u, 128u, 0);
    /* Use the same NR0 setting, NSG=4 and cross-SIMD reduction as scalar. */
    static int nr0_cached = 0;
    if (!nr0_cached) {
        const char *v = getenv("MYDS_SHARED_NR0");
        nr0_cached = (v && strcmp(v, "2") == 0) ? 2 : 4;
    }
    const uint32_t nr0 = (uint32_t)nr0_cached;
    const char *kernel = nr0 == 4u ? "kernel_myds_v41_shared_gate_up_swiglu_q8_shared3_r4"
                                  : "kernel_myds_v41_shared_gate_up_swiglu_q8_shared3_r2";
    return dispatch(kernel, &args, sizeof(args), buffers, 8,
                    (out_dim + nr0 - 1u) / nr0, 128u,
                    3u * 2u * nr0 * 32u * sizeof(float));
}

static _Atomic uint64_t shared_gu_shared3_requests;
static _Atomic uint64_t shared_gu_shared3_encoded;
static _Atomic uint64_t shared_gu_shared3_failures;

void myds_v41_shared_gu_shared3_counters(myds_v41_shared_gu_shared3_counters_t *out,
                                       bool reset) {
#define GU_SHARED3_COUNT(field) (reset ? atomic_exchange_explicit(&shared_gu_shared3_##field, 0, memory_order_relaxed) \
                                     : atomic_load_explicit(&shared_gu_shared3_##field, memory_order_relaxed))
    const myds_v41_shared_gu_shared3_counters_t value = {
        .requests = GU_SHARED3_COUNT(requests),
        .encoded = GU_SHARED3_COUNT(encoded),
        .failures = GU_SHARED3_COUNT(failures)
    };
#undef GU_SHARED3_COUNT
    if (out) *out = value;
}

int myds_v41_shared_gate_up_swiglu_shared3(myds_metal_buffer_t *const out[3],
                                         const myds_metal_buffer_t *gate,
                                         const myds_metal_buffer_t *up,
                                         myds_metal_buffer_t *const input[3],
                                         uint32_t out_dim, uint32_t in_dim,
                                         ggml_type gate_type, ggml_type up_type,
                                         float clamp) {
    atomic_fetch_add_explicit(&shared_gu_shared3_requests, 1, memory_order_relaxed);
    const int ok = shared_gu_shared3_encode(out, gate, up, input, out_dim, in_dim,
                                           gate_type, up_type, clamp);
    if (ok) atomic_fetch_add_explicit(&shared_gu_shared3_encoded, 1, memory_order_relaxed);
    else atomic_fetch_add_explicit(&shared_gu_shared3_failures, 1, memory_order_relaxed);
    return ok;
}

/* Unaccepted diagnostic candidate; production uses the scalar-PSO pair. */
int myds_v41_shared_gate_up_swiglu_batch2(myds_metal_buffer_t *out,
                                         const myds_metal_buffer_t *gate,
                                         const myds_metal_buffer_t *up,
                                         const myds_metal_buffer_t *input,
                                         uint32_t out_dim, uint32_t in_dim,
                                         ggml_type gate_type, ggml_type up_type,
                                         float clamp) {
    if (!out || !gate || !up || !input || !out_dim || !in_dim ||
        out_dim > INT32_MAX || in_dim > INT32_MAX ||
        !isfinite(clamp) || clamp < 0 ||
        myds_metal_buffer_offset(out) % 16u != 0) return 0;

    if (gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K) {
        if (in_dim % 256u) return 0;
        const uint32_t row_bytes = (in_dim / 256u) * 144u;
        if (myds_metal_buffer_size(gate) / row_bytes < out_dim ||
            myds_metal_buffer_size(up) / row_bytes < out_dim ||
            myds_metal_buffer_size(input) / 4u < 2 * (uint64_t)in_dim ||
            myds_metal_buffer_size(out) / 4u < 2 * (uint64_t)out_dim) return 0;

        struct myds_v41_shared_swiglu_args args = {
            .in_dim = in_dim,
            .out_dim = out_dim,
            .row_bytes = row_bytes,
            .clamp = clamp
        };

        const char *kernel_name = "kernel_myds_v41_shared_gate_up_swiglu_q4_shared2";
        const myds_metal_buffer_t *buffers[] = { gate, up, input, out };
        const uint32_t groups = (out_dim + 8u - 1u) / 8u; // NSG=4 * NR0=2 = 8 rows per threadgroup
        const uint32_t threads = 128u; // 32 * NSG (NSG=4)
        const size_t shmem_bytes = 0;

        return dispatch(kernel_name, &args, sizeof(args), buffers, 4,
                        groups, threads, shmem_bytes);
    }
    return 0;
}

int myds_v41_moe_eval(myds_context_t *ctx, uint32_t layer, uint32_t pos,
                      const myds_metal_buffer_t *ffn_norm,
                      const myds_metal_buffer_t *router_logits,
                      const myds_metal_buffer_t *shared_out) {
    if (!ctx || !ctx->model || layer >= ctx->model->n_layers) return 0;
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim=m->dim, width=m->moe_expert_dim, k=m->n_activated_experts;
    if (m->n_routed_experts!=384 || k!=6 || m->n_shared_experts!=1 ||
        !dim || !width || dim%256 || width%256 ||
        l->ffn_gate_exps_type!=GGML_TYPE_IQ2_XXS ||
        l->ffn_up_exps_type!=GGML_TYPE_IQ2_XXS ||
        l->ffn_down_exps_type!=GGML_TYPE_Q2_K) return 0;

    if (!myds_trace_f32("route_logits",layer,pos,router_logits,m->n_routed_experts) ||
        !myds_v41_route(ctx->selected_ids,ctx->selected_weights,ctx->moe_gate,ctx->moe_up,
                        router_logits,l->router_bias,m->n_routed_experts,k,m->routed_scaling_factor) ||
        !myds_trace_f32("route_probs",layer,pos,ctx->moe_gate,m->n_routed_experts) ||
        !myds_trace_f32("route_weights",layer,pos,ctx->selected_weights,k)) return 0;
    if (myds_trace_active()) {
        const myds_metal_buffer_t *id_buffers[] = {ctx->selected_ids,ctx->moe_up};
        if (!dispatch("kernel_myds_v41_ids_f32",&k,sizeof(k),id_buffers,2,1,32,0) ||
            !myds_trace_f32("route_ids",layer,pos,ctx->moe_up,k)) return 0;
    }
    if (!myds_trace_f32("shared_out",layer,pos,shared_out,dim)) return 0;
    myds_metal_barrier();
    if (!myds_metal_routed_moe(ctx->moe_out,ffn_norm,l->ffn_gate_exps,
        l->ffn_up_exps,l->ffn_down_exps,ctx->selected_ids,ctx->selected_weights,
        ctx->moe_gate,ctx->moe_up,ctx->moe_mid,ctx->moe_down,dim,width,
        m->n_routed_experts,k,l->ffn_gate_exps_type,l->ffn_up_exps_type,
        l->ffn_down_exps_type,m->swiglu_clamp_exp)) return 0;
    myds_metal_barrier();
    if (!myds_trace_f32("routed_out",layer,pos,ctx->moe_out,dim) ||
        !myds_metal_v41_add_bf16(ctx->ffn_out,ctx->moe_out,shared_out,dim)) return 0;
    return myds_trace_f32("ffnblock",layer,pos,ctx->ffn_out,dim);
}

static inline bool is_moe_concurrent_enabled(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        const char *v = getenv("MYDS_MOE_CONCURRENT");
        cached = (v && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

static inline bool is_moe_fuse_expand_enabled(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        const char *v = getenv("MYDS_MOE_FUSE_EXPAND");
        cached = (v && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

static int moe_fused_scoped(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
                       myds_metal_buffer_t *residual,
                       const myds_metal_buffer_t *after_attn,
                       const myds_metal_buffer_t *ffn_split, bool ordinary_decode_allowed) {
    (void)token; // V4.1 text router is score-based, without token-hash selection.
    if (!ctx || !ctx->model || layer >= ctx->model->n_layers) return 0;
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim = m->dim, width = m->moe_expert_dim, k = m->n_activated_experts;
    const bool trace_on = myds_trace_active();

    if (m->n_routed_experts != 384 || k != 6 || m->n_shared_experts != 1 ||
        !dim || !width || dim % 256 || width % 256 ||
        l->ffn_gate_exps_type != GGML_TYPE_IQ2_XXS ||
        l->ffn_up_exps_type != GGML_TYPE_IQ2_XXS ||
        l->ffn_down_exps_type != GGML_TYPE_Q2_K) return 0;

    const bool can_fuse = (residual && after_attn && ffn_split && is_moe_fuse_expand_enabled());

    /* 状态与跟踪门禁：若 trace 开启或显式关闭并发，退回原有严格串行路径 */
    if (trace_on || !is_moe_concurrent_enabled()) {
        const int prev_auto_barrier = myds_metal_get_auto_barrier();
        const int prev_concurrent = myds_metal_get_concurrent();

        if (prev_concurrent && !prev_auto_barrier) {
            myds_metal_barrier();
        }
        myds_metal_set_concurrent(0);
        myds_metal_set_auto_barrier(1);

        #define SERIAL_MOE_RETURN(val) do { \
            myds_metal_set_auto_barrier(prev_auto_barrier); \
            myds_metal_set_concurrent(prev_concurrent); \
            if (prev_concurrent && !prev_auto_barrier) { \
                myds_metal_barrier(); \
            } \
            return (val); \
        } while (0)

        /* 1. 优先派发 router_logits GEMV 与路由打分排序 */
        if (!myds_metal_v41_gemv(ctx->router_logits, l->router_gate, ctx->ffn_norm_out,
                                 m->n_routed_experts, dim, l->router_gate_type)) {
            SERIAL_MOE_RETURN(0);
        }
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(ctx->router_logits);
        if (trace_on && !myds_trace_f32("route_logits", layer, pos, ctx->router_logits, m->n_routed_experts)) {
            SERIAL_MOE_RETURN(0);
        }

        if (!myds_v41_route(ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up,
                            ctx->router_logits, l->router_bias, m->n_routed_experts, k, m->routed_scaling_factor)) {
            SERIAL_MOE_RETURN(0);
        }
        if (trace_on) {
            if (!myds_trace_f32("route_probs", layer, pos, ctx->moe_gate, m->n_routed_experts) ||
                !myds_trace_f32("route_weights", layer, pos, ctx->selected_weights, k)) {
                SERIAL_MOE_RETURN(0);
            }
            const myds_metal_buffer_t *id_buffers[] = {ctx->selected_ids, ctx->moe_up};
            if (!dispatch("kernel_myds_v41_ids_f32", &k, sizeof(k), id_buffers, 2, 1, 32, 0) ||
                !myds_trace_f32("route_ids", layer, pos, ctx->moe_up, k)) {
                SERIAL_MOE_RETURN(0);
            }
        }

        /* 2. 紧接着派发 shared expert (gate, up, swiglu, down) */
        int sh_swiglu_ok = 0;
        if (!trace_on && is_shared_fuse_enabled()) {
            sh_swiglu_ok = myds_v41_shared_gate_up_swiglu(
                ctx->shared_swiglu_out, l->shared_gate, l->shared_up, ctx->ffn_norm_out,
                width, dim, l->shared_gate_type, l->shared_up_type, m->swiglu_clamp_exp);
        }
        if (!sh_swiglu_ok) {
            if (!myds_v41_decode_gemv_bf16(ctx->shared_gate_out, l->shared_gate, ctx->ffn_norm_out,
                                 width, dim, l->shared_gate_type, trace_on) ||
                !myds_v41_decode_gemv_bf16(ctx->shared_up_out, l->shared_up, ctx->ffn_norm_out,
                                 width, dim, l->shared_up_type, trace_on)) {
                SERIAL_MOE_RETURN(0);
            }
            if (!myds_metal_get_auto_barrier()) {
                const myds_metal_buffer_t *shared_gu_deps[] = { ctx->shared_gate_out, ctx->shared_up_out };
                myds_metal_barrier_buffers(shared_gu_deps, 2);
            }
            if (!myds_v41_shared_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out,
                                       ctx->shared_up_out, width, m->swiglu_clamp_exp)) {
                SERIAL_MOE_RETURN(0);
            }
        }
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(ctx->shared_swiglu_out);
        if (!myds_v41_decode_gemv_bf16(ctx->shared_out, l->shared_down, ctx->shared_swiglu_out,
                             dim, width, l->shared_down_type, trace_on)) {
            SERIAL_MOE_RETURN(0);
        }
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(ctx->shared_out);
        if (trace_on && !myds_trace_f32("shared_out", layer, pos, ctx->shared_out, dim)) {
            SERIAL_MOE_RETURN(0);
        }

        /* 3. 此时路由选择与 shared expert 计算流已就绪，启动 routed MoE */
        myds_metal_barrier();
        if (!myds_metal_routed_moe_scoped(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
            l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
            ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
            m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
            l->ffn_down_exps_type, m->swiglu_clamp_exp, ordinary_decode_allowed && !trace_on)) {
            SERIAL_MOE_RETURN(0);
        }
        myds_metal_barrier();
        if (trace_on && !myds_trace_f32("routed_out", layer, pos, ctx->moe_out, dim)) {
            SERIAL_MOE_RETURN(0);
        }

        /* 4. 融合 shared_out 与 routed_moe */
        if (can_fuse) {
            if (!myds_metal_v41_moe_add_hc_expand(residual, ctx->ffn_out, ctx->moe_out, ctx->shared_out, after_attn, ffn_split, dim)) {
                SERIAL_MOE_RETURN(0);
            }
            if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(residual);
        } else {
            if (!myds_metal_v41_add_bf16(ctx->ffn_out, ctx->moe_out, ctx->shared_out, dim)) {
                SERIAL_MOE_RETURN(0);
            }
            if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(ctx->ffn_out);
            if (residual && after_attn && ffn_split) {
                if (!myds_metal_v41_hc_expand(residual, ctx->ffn_out, after_attn, ffn_split, dim)) {
                    SERIAL_MOE_RETURN(0);
                }
                if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(residual);
            }
        }
        bool final_ok = !trace_on || myds_trace_f32("ffnblock", layer, pos, ctx->ffn_out, dim);
        SERIAL_MOE_RETURN(final_ok ? 1 : 0);
        #undef SERIAL_MOE_RETURN
    }

    /* === S1 并发流水调度分支 === */
    const int prev_auto_barrier = myds_metal_get_auto_barrier();
    const int prev_concurrent = myds_metal_get_concurrent();
    myds_metal_set_auto_barrier(0);
    myds_metal_set_concurrent(1);

    /* 1. 优先并发派发 Router GEMV 流（小算子，立即在 GPU Cycle 0 启动） */
    if (!myds_metal_v41_gemv(ctx->router_logits, l->router_gate, ctx->ffn_norm_out,
                             m->n_routed_experts, dim, l->router_gate_type)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 2. 并发派发 Shared Expert Gate/Up/SwiGLU 流（与 Router GEMV 在 GPU 上并发） */
    int sh_fused_ok = 0;
    if (is_shared_fuse_enabled()) {
        sh_fused_ok = myds_v41_shared_gate_up_swiglu(
            ctx->shared_swiglu_out, l->shared_gate, l->shared_up, ctx->ffn_norm_out,
            width, dim, l->shared_gate_type, l->shared_up_type, m->swiglu_clamp_exp);
    }
    if (!sh_fused_ok) {
        if (!myds_v41_decode_gemv_bf16(ctx->shared_gate_out, l->shared_gate, ctx->ffn_norm_out,
                                      width, dim, l->shared_gate_type, false) ||
            !myds_v41_decode_gemv_bf16(ctx->shared_up_out, l->shared_up, ctx->ffn_norm_out,
                                      width, dim, l->shared_up_type, false)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
        const myds_metal_buffer_t *shared_gu_deps[] = { ctx->shared_gate_out, ctx->shared_up_out };
        myds_metal_barrier_buffers(shared_gu_deps, 2);
        if (!myds_v41_shared_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out,
                                   ctx->shared_up_out, width, m->swiglu_clamp_exp)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
    }

    /* 3. 提前入队 Shared Down GEMV（只依赖 shared_swiglu_out，在 routing 屏障前提交入流） */
    myds_metal_barrier_buffer(ctx->shared_swiglu_out);
    if (!myds_v41_decode_gemv_bf16(ctx->shared_out, l->shared_down, ctx->shared_swiglu_out,
                                  dim, width, l->shared_down_type, false)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 4. 仅同步 router_logits，派发路由打分与 Top-K 选择流水线 */
    myds_metal_barrier_buffer(ctx->router_logits);
    if (!myds_v41_route(ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up,
                        ctx->router_logits, l->router_bias, m->n_routed_experts, k, m->routed_scaling_factor)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 5. 仅同步路由输出与选择结果，启动 Routed MoE（与 Shared Down GEMV 在 GPU 上并发） */
    const myds_metal_buffer_t *moe_deps[] = {
        ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up
    };
    myds_metal_barrier_buffers(moe_deps, 4);
    if (!myds_metal_routed_moe_scoped(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
        l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
        m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
        l->ffn_down_exps_type, m->swiglu_clamp_exp, ordinary_decode_allowed && !trace_on)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 6. 等待 Routed MoE (moe_out) 与 Shared Expert (shared_out) 两路汇聚 */
    const myds_metal_buffer_t *final_deps[] = { ctx->moe_out, ctx->shared_out };
    myds_metal_barrier_buffers(final_deps, 2);

    if (can_fuse) {
        /* S10 深度算子融合：直接将 moe_out + shared_out 与 HC expand 四通道广播累加融合为单次派发 */
        if (!myds_metal_v41_moe_add_hc_expand(residual, ctx->ffn_out, ctx->moe_out, ctx->shared_out, after_attn, ffn_split, dim)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
        myds_metal_barrier_buffer(residual);
    } else {
        if (!myds_metal_v41_add_bf16(ctx->ffn_out, ctx->moe_out, ctx->shared_out, dim)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
        myds_metal_barrier_buffer(ctx->ffn_out);
        if (residual && after_attn && ffn_split) {
            if (!myds_metal_v41_hc_expand(residual, ctx->ffn_out, after_attn, ffn_split, dim)) {
                myds_metal_set_auto_barrier(prev_auto_barrier);
                myds_metal_set_concurrent(prev_concurrent);
                return 0;
            }
            myds_metal_barrier_buffer(residual);
        }
    }
    myds_metal_set_auto_barrier(prev_auto_barrier);
    myds_metal_set_concurrent(prev_concurrent);
    return 1;
}

int myds_v41_moe_fused(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
                       myds_metal_buffer_t *residual,
                       const myds_metal_buffer_t *after_attn,
                       const myds_metal_buffer_t *ffn_split) {
    return moe_fused_scoped(ctx, layer, token, pos, residual, after_attn, ffn_split, false);
}

int myds_v41_moe(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos) {
    return myds_v41_moe_fused(ctx, layer, token, pos, NULL, NULL, NULL);
}

static inline bool is_moe_pipeline_enabled(void) {
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        const char *v = getenv("MYDS_MOE_PIPELINE");
        cached = (v && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

int myds_v41_moe_pipeline_scoped(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
                          myds_metal_buffer_t *residual,
                          const myds_metal_buffer_t *after_attn,
                          myds_metal_buffer_t *ffn_split,
                          const myds_metal_buffer_t *ffn_fn,
                          const myds_metal_buffer_t *ffn_scale,
                          const myds_metal_buffer_t *ffn_base,
                          myds_metal_buffer_t *flat_norm,
                          myds_metal_buffer_t *mix,
                          float hc_eps, bool ordinary_decode_allowed) {
    (void)token;
    if (!ctx || !ctx->model || layer >= ctx->model->n_layers) return 0;
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim = m->dim, width = m->moe_expert_dim, k = m->n_activated_experts;
    const bool trace_on = myds_trace_active();

    if (trace_on || !is_moe_concurrent_enabled() || !is_moe_pipeline_enabled() ||
        !ffn_split || !after_attn || !ffn_fn || !ffn_scale || !ffn_base || !flat_norm || !mix) {
        /* 回退串行调度：先做 hc_mix，再调 myds_v41_moe_fused */
        if (!myds_metal_v41_hc_mix(ffn_split, after_attn, ffn_fn, GGML_TYPE_F16,
                                   ffn_scale, ffn_base, flat_norm, mix, dim, m->norm_eps, hc_eps)) return 0;
        if (trace_on) {
            if (!myds_trace_f32("hc_ffn_mix", layer, pos, mix, 24) ||
                !myds_trace_f32("hc_ffn_split", layer, pos, ffn_split, 24)) return 0;
        }
        return moe_fused_scoped(ctx, layer, token, pos, residual, after_attn, ffn_split, ordinary_decode_allowed && !trace_on);
    }

    if (m->n_routed_experts != 384 || k != 6 || m->n_shared_experts != 1 ||
        !dim || !width || dim % 256 || width % 256 ||
        l->ffn_gate_exps_type != GGML_TYPE_IQ2_XXS ||
        l->ffn_up_exps_type != GGML_TYPE_IQ2_XXS ||
        l->ffn_down_exps_type != GGML_TYPE_Q2_K) return 0;

    /* === S11 跨阶段全并发流水线 === */
    const int prev_auto_barrier = myds_metal_get_auto_barrier();
    const int prev_concurrent = myds_metal_get_concurrent();
    myds_metal_set_auto_barrier(0);
    myds_metal_set_concurrent(1);

    /* 1. 并发派发 HC Mix 流 (RMSNorm 20480 + GEMV 24x20480 + Sinkhorn)（只读 after_attn，写 ffn_split） */
    if (!myds_metal_v41_hc_mix(ffn_split, after_attn, ffn_fn, GGML_TYPE_F16,
                               ffn_scale, ffn_base, flat_norm, mix, dim, m->norm_eps, hc_eps)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 2. 优先并发派发 Router GEMV 流（小算子，立即在 GPU Cycle 0 启动） */
    if (!myds_metal_v41_gemv(ctx->router_logits, l->router_gate, ctx->ffn_norm_out,
                             m->n_routed_experts, dim, l->router_gate_type)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 3. 并发派发 Shared Expert Gate/Up/SwiGLU 流（与 Router GEMV 在 GPU 上并发） */
    int sh_fused_ok = 0;
    if (is_shared_fuse_enabled()) {
        sh_fused_ok = myds_v41_shared_gate_up_swiglu(
            ctx->shared_swiglu_out, l->shared_gate, l->shared_up, ctx->ffn_norm_out,
            width, dim, l->shared_gate_type, l->shared_up_type, m->swiglu_clamp_exp);
    }
    if (!sh_fused_ok) {
        if (!myds_v41_decode_gemv_bf16(ctx->shared_gate_out, l->shared_gate, ctx->ffn_norm_out,
                                      width, dim, l->shared_gate_type, false) ||
            !myds_v41_decode_gemv_bf16(ctx->shared_up_out, l->shared_up, ctx->ffn_norm_out,
                                      width, dim, l->shared_up_type, false)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
        const myds_metal_buffer_t *shared_gu_deps[] = { ctx->shared_gate_out, ctx->shared_up_out };
        myds_metal_barrier_buffers(shared_gu_deps, 2);
        if (!myds_v41_shared_swiglu(ctx->shared_swiglu_out, ctx->shared_gate_out,
                                   ctx->shared_up_out, width, m->swiglu_clamp_exp)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
    }

    /* 4. 提前入队 Shared Down GEMV（只依赖 shared_swiglu_out，在 routing 屏障前提交入流） */
    myds_metal_barrier_buffer(ctx->shared_swiglu_out);
    if (!myds_v41_decode_gemv_bf16(ctx->shared_out, l->shared_down, ctx->shared_swiglu_out,
                                  dim, width, l->shared_down_type, false)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 5. 仅同步 router_logits，派发路由打分与 Top-K 选择流水线 */
    myds_metal_barrier_buffer(ctx->router_logits);
    if (!myds_v41_route(ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up,
                        ctx->router_logits, l->router_bias, m->n_routed_experts, k, m->routed_scaling_factor)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 6. 仅同步路由输出与选择结果，启动 Routed MoE（与 Shared Down GEMV 在 GPU 上并发） */
    const myds_metal_buffer_t *moe_deps[] = {
        ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up
    };
    myds_metal_barrier_buffers(moe_deps, 4);
    if (!myds_metal_routed_moe_scoped(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
        l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
        m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
        l->ffn_down_exps_type, m->swiglu_clamp_exp, ordinary_decode_allowed && !trace_on)) {
        myds_metal_set_auto_barrier(prev_auto_barrier);
        myds_metal_set_concurrent(prev_concurrent);
        return 0;
    }

    /* 7. 等待 Routed MoE (moe_out), Shared Expert (shared_out), 和 HC Mix (ffn_split) 三路全部汇聚 */
    const myds_metal_buffer_t *final_deps[] = { ctx->moe_out, ctx->shared_out, ffn_split };
    myds_metal_barrier_buffers(final_deps, 3);

    const bool can_fuse = (residual && after_attn && ffn_split && is_moe_fuse_expand_enabled());
    if (can_fuse) {
        /* S10 深度算子融合：直接将 moe_out + shared_out 与 HC expand 四通道广播累加融合为单次派发 */
        if (!myds_metal_v41_moe_add_hc_expand(residual, ctx->ffn_out, ctx->moe_out, ctx->shared_out, after_attn, ffn_split, dim)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
        myds_metal_barrier_buffer(residual);
    } else {
        if (!myds_metal_v41_add_bf16(ctx->ffn_out, ctx->moe_out, ctx->shared_out, dim)) {
            myds_metal_set_auto_barrier(prev_auto_barrier);
            myds_metal_set_concurrent(prev_concurrent);
            return 0;
        }
        myds_metal_barrier_buffer(ctx->ffn_out);
        if (residual && after_attn && ffn_split) {
            if (!myds_metal_v41_hc_expand(residual, ctx->ffn_out, after_attn, ffn_split, dim)) {
                myds_metal_set_auto_barrier(prev_auto_barrier);
                myds_metal_set_concurrent(prev_concurrent);
                return 0;
            }
            myds_metal_barrier_buffer(residual);
        }
    }
    myds_metal_set_auto_barrier(prev_auto_barrier);
    myds_metal_set_concurrent(prev_concurrent);
    return 1;
}

int myds_v41_moe_pipeline(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
                          myds_metal_buffer_t *residual,
                          const myds_metal_buffer_t *after_attn,
                          myds_metal_buffer_t *ffn_split,
                          const myds_metal_buffer_t *ffn_fn,
                          const myds_metal_buffer_t *ffn_scale,
                          const myds_metal_buffer_t *ffn_base,
                          myds_metal_buffer_t *flat_norm,
                          myds_metal_buffer_t *mix, float hc_eps) {
    return myds_v41_moe_pipeline_scoped(ctx, layer, token, pos, residual, after_attn,
        ffn_split, ffn_fn, ffn_scale, ffn_base, flat_norm, mix, hc_eps, false);
}

int myds_dspark_route_fused(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
                            const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
                            uint32_t top_k, float scale, uint32_t rows) {
    if (!selected || !weights || !logits || rows == 0 || top_k == 0 || top_k > 32 ||
        !isfinite(scale) || scale <= 0.0f) return 0;

    const uint32_t experts = 128;
    const uint64_t total = (uint64_t)experts * (uint64_t)rows;
    /* Shader row indices and count are uint32_t. Validate all buffer views
     * before dispatch; a short view must fail without issuing GPU work. */
    if (total > UINT32_MAX || !sized(logits, (size_t)total) ||
        !sized(selected, (size_t)top_k * rows) ||
        !sized(weights, (size_t)top_k * rows) ||
        (bias && !sized(bias, experts))) return 0;
    struct route_args args = {
        .count = (uint32_t)total,
        .top_k = top_k,
        .op = (bias != NULL) ? 1u : 0u,
        .scale = scale,
        .rows = rows,
        .row = experts,
        .row4 = experts / 4u
    };

    const myds_metal_buffer_t *fused_bufs[] = {logits, bias ? bias : logits, selected, weights};
    /* 共享内存: 128 ints + 128 floats + 128 floats + 32 floats = 1664 字节 */
    const size_t shmem_bytes = 128 * sizeof(int32_t) + 128 * sizeof(float) +
                               128 * sizeof(float) + 32 * sizeof(float);

    if (dispatch("kernel_myds_dspark_route_fused_128", &args, sizeof(args), fused_bufs, 4,
                 rows, 128, shmem_bytes)) {
        const myds_metal_buffer_t *out_bufs[] = {selected, weights};
        myds_metal_barrier_buffers(out_bufs, 2);
        return 1;
    }
    return 0;
}

/* ABI shared with metal/moe.metal's ds4_metal_glm_routed_moe_args. Those
 * kernels implement Q2_K addressing and a dynamic expert count; the target's
 * IQ2_XXS and sum6 kernels cannot execute DSpark's Q2_K/top-3 weights. */
struct dspark_moe_args {
    uint32_t in_dim, mid_dim, out_dim, n_total_expert, n_expert_used;
    uint32_t n_tokens, mid_token_stride, down_type;
    float swiglu_clamp;
    int32_t tp_rank, tp_world, tp_expert_base;
    uint64_t gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes;
    uint64_t down_expert_bytes, down_row_bytes;
};
_Static_assert(sizeof(struct dspark_moe_args) == 96, "DSpark MoE shader ABI");

int myds_dspark_moe(myds_metal_buffer_t *moe_out,
    const myds_metal_buffer_t *x_in,
    const myds_metal_buffer_t *gate_exps, const myds_metal_buffer_t *up_exps,
    const myds_metal_buffer_t *down_exps,
    const myds_metal_buffer_t *selected_ids, const myds_metal_buffer_t *selected_weights,
    myds_metal_buffer_t *scratch_gate, myds_metal_buffer_t *scratch_up,
    myds_metal_buffer_t *scratch_mid, myds_metal_buffer_t *scratch_down,
    uint32_t dim, uint32_t expert_dim, uint32_t n_routed_experts,
    uint32_t n_activated_experts, ggml_type gate_type, ggml_type up_type,
    ggml_type down_type, float clamp_val) {
    (void)scratch_gate; (void)scratch_up; (void)scratch_down;
    if (!dim || !expert_dim || dim % 256u || expert_dim % 256u ||
        dim > INT32_MAX || expert_dim > INT32_MAX / 3u ||
        n_routed_experts < 3u || n_routed_experts > INT32_MAX ||
        n_activated_experts != 3u || gate_type != GGML_TYPE_Q2_K ||
        up_type != GGML_TYPE_Q2_K || down_type != GGML_TYPE_Q2_K ||
        !isfinite(clamp_val) || clamp_val < 0.0f) return 0;

    const uint64_t gate_row = (uint64_t)(dim / 256u) * 84u;
    const uint64_t down_row = (uint64_t)(expert_dim / 256u) * 84u;
    const uint64_t gate_expert = gate_row * expert_dim;
    const uint64_t down_expert = down_row * dim;
    if (!gate_exps || !up_exps || !down_exps ||
        gate_expert > SIZE_MAX / n_routed_experts ||
        down_expert > SIZE_MAX / n_routed_experts ||
        myds_metal_buffer_size(gate_exps) < gate_expert * n_routed_experts ||
        myds_metal_buffer_size(up_exps) < gate_expert * n_routed_experts ||
        myds_metal_buffer_size(down_exps) < down_expert * n_routed_experts ||
        !sized(moe_out, dim) || !sized(x_in, dim) || !sized(selected_ids, 3u) ||
        !sized(selected_weights, 3u) || !sized(scratch_mid, (size_t)3u * expert_dim)) return 0;
    /* Both PSOs must be available before encoding either half of this graph. */
    const char *pair_kernel = "kernel_glm_q2_K_pair_swiglu_f32";
    const char *down_kernel = "kernel_glm_q2_K_down_f32";
    if (myds_metal_kernel_max_threads(pair_kernel) < 64u ||
        myds_metal_kernel_max_threads(down_kernel) < 64u) return 0;

    const struct dspark_moe_args args = {
        .in_dim = dim, .mid_dim = expert_dim, .out_dim = dim,
        .n_total_expert = n_routed_experts, .n_expert_used = 3u,
        .n_tokens = 1u, .mid_token_stride = 3u * expert_dim,
        .down_type = GGML_TYPE_Q2_K, .swiglu_clamp = clamp_val,
        .tp_rank = 0, .tp_world = 1, .tp_expert_base = 0,
        .gate_expert_bytes = gate_expert, .gate_row_bytes = gate_row,
        .up_expert_bytes = gate_expert, .up_row_bytes = gate_row,
        .down_expert_bytes = down_expert, .down_row_bytes = down_row,
    };
    /* GLM Q2 kernels use two SIMD groups with two output rows each. This
     * geometry is independent of the target's MYDS_MOE_* NSG settings. */
    const uint32_t threads[3] = {32u, 2u, 1u};
    const uint32_t pair_groups[3] = {(expert_dim + 3u) / 4u, 3u, 1u};
    const myds_metal_buffer_t *pair_buffers[] = {
        gate_exps, up_exps, x_in, selected_ids, selected_weights, scratch_mid,
    };
    if (!myds_metal_dispatch_kernel(pair_kernel, &args, sizeof(args), pair_buffers,
                                    6u, pair_groups, threads, 16u)) return 0;
    myds_metal_barrier_buffer(scratch_mid);
    const uint32_t down_groups[3] = {(dim + 3u) / 4u, 1u, 1u};
    const myds_metal_buffer_t *down_buffers[] = {
        down_exps, selected_ids, scratch_mid, moe_out,
    };
    if (!myds_metal_dispatch_kernel(down_kernel, &args, sizeof(args), down_buffers,
                                    4u, down_groups, threads, 16u)) return 0;
    myds_metal_barrier_buffer(moe_out);
    return 1;
}



