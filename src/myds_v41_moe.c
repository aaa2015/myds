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
static _Atomic uint64_t hc_overlap_requests, hc_overlap_encoded, hc_overlap_failures;
static _Atomic int hc_overlap_mode;
int myds_v41_moe_hc_overlap_parse_mode(void) {
    const char *v = getenv("MYDS_MOE_HC_OVERLAP");
    const int mode=(!v || !strcmp(v,"0")) ? 0 : !strcmp(v,"1") ? 1 : -1;
    if (mode<0) MYDS_LOG_E("MYDS_MOE_HC_OVERLAP must be exactly 0 or 1");
    atomic_store_explicit(&hc_overlap_mode,mode,memory_order_relaxed);
    return mode;
}
int myds_v41_moe_hc_overlap_enabled(void) { return atomic_load_explicit(&hc_overlap_mode,memory_order_relaxed); }
void myds_v41_moe_hc_overlap_reset_mode(void) { atomic_store_explicit(&hc_overlap_mode,0,memory_order_relaxed); }
void myds_v41_moe_hc_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
#define HC_OVERLAP_COUNT(field) (reset ? atomic_exchange_explicit(&hc_overlap_##field, 0, memory_order_relaxed) \
                                     : atomic_load_explicit(&hc_overlap_##field, memory_order_relaxed))
    const myds_metal_decode_candidate_counters_t value = {
        .requests=HC_OVERLAP_COUNT(requests), .encoded=HC_OVERLAP_COUNT(encoded),
        .failures=HC_OVERLAP_COUNT(failures)};
    if (out) *out=value;
#undef HC_OVERLAP_COUNT
}
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

static int route_batch_impl(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probs, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t experts, uint32_t top_k, float scale, uint32_t rows, bool emit_final_barrier) {
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
        if (emit_final_barrier) {
            const myds_metal_buffer_t *out_bufs[] = {selected, weights};
            myds_metal_barrier_buffers(out_bufs, 2);
        }
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
            if (emit_final_barrier) {
                const myds_metal_buffer_t *out_bufs[] = {selected, weights};
                myds_metal_barrier_buffers(out_bufs, 2);
            }
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
            if (emit_final_barrier) {
                const myds_metal_buffer_t *out_bufs[] = {selected, weights};
                myds_metal_barrier_buffers(out_bufs, 2);
            }
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
    if (emit_final_barrier) {
        const myds_metal_buffer_t *out_bufs[] = {selected, weights};
        myds_metal_barrier_buffers(out_bufs, 2);
    }
    return ok;
}

int myds_v41_route_batch(myds_metal_buffer_t *selected, myds_metal_buffer_t *weights,
    myds_metal_buffer_t *probs, myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t experts, uint32_t top_k, float scale, uint32_t rows) {
    return route_batch_impl(selected, weights, probs, scores, logits, bias,
                            experts, top_k, scale, rows, true);
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

    const bool overlap_down = ordinary_decode_allowed && !trace_on && myds_metal_moe_down_overlap_enabled();
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
    if (!overlap_down && !myds_v41_decode_gemv_bf16(ctx->shared_out, l->shared_down, ctx->shared_swiglu_out,
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

    /* 5. 仅同步路由输出与选择结果，启动 Routed MoE（可选 down pair 在 helper 的 MID join 后编码） */
    const myds_metal_buffer_t *moe_deps[] = {
        ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up
    };
    myds_metal_barrier_buffers(moe_deps, 4);
    const int routed_ok = overlap_down
        ? myds_metal_routed_moe_down_overlap_scoped(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
        l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
        m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
        l->ffn_down_exps_type, m->swiglu_clamp_exp, ordinary_decode_allowed && !trace_on, ctx->shared_out, l->shared_down, ctx->shared_swiglu_out, l->shared_down_type)
        : myds_metal_routed_moe_scoped(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
        l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
        m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
        l->ffn_down_exps_type, m->swiglu_clamp_exp, ordinary_decode_allowed && !trace_on);
    if (!routed_ok) {
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

/* A separate ordinary101 schedule. It changes enqueue order and joins only;
 * every HC/router/shared/routed/expand operation retains its production PSO,
 * ABI, thread geometry and arithmetic. The caller has already joined the
 * after_attn and FFN-normalized inputs and previous scratch consumers. */
static bool hc_scratch_independent(myds_context_t *ctx, uint32_t layer,
    myds_metal_buffer_t *residual, const myds_metal_buffer_t *after_attn,
    myds_metal_buffer_t *ffn_split, const myds_metal_buffer_t *ffn_fn,
    const myds_metal_buffer_t *ffn_scale, const myds_metal_buffer_t *ffn_base,
    myds_metal_buffer_t *flat_norm, myds_metal_buffer_t *mix) {
    const myds_layer_t *l=&ctx->model->layers[layer];
    const myds_metal_buffer_t *write[]={flat_norm,mix,ffn_split};
    const size_t write_bytes[]={81920u,96u,96u};
    const myds_metal_buffer_t *other[]={residual,after_attn,ffn_fn,ffn_scale,ffn_base,
        ctx->ffn_norm_out,ctx->router_logits,l->router_gate,l->router_bias,
        ctx->shared_swiglu_out,l->shared_gate,l->shared_up,l->shared_down,ctx->shared_out,
        l->ffn_gate_exps,l->ffn_up_exps,l->ffn_down_exps,ctx->selected_ids,ctx->selected_weights,
        ctx->moe_gate,ctx->moe_up,ctx->moe_mid,ctx->moe_down,ctx->moe_out,ctx->ffn_out};
    for (unsigned i=0;i<3u;++i) {
        const uintptr_t a=(uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)write[i]);
        if (!a || write_bytes[i]>myds_metal_buffer_size(write[i]) || write_bytes[i]>UINTPTR_MAX-a) return false;
        for (unsigned j=0;j<sizeof(other)/sizeof(*other);++j) {
            const uintptr_t b=(uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)other[j]);
            const size_t bytes=myds_metal_buffer_size(other[j]);
            if (!b || !bytes || bytes>UINTPTR_MAX-b ||
                (a<=b ? b-a<write_bytes[i] : a-b<bytes)) return false;
        }
    }
    return true;
}

static int moe_hc_overlap_101(myds_context_t *ctx, uint32_t layer,
    myds_metal_buffer_t *residual, const myds_metal_buffer_t *after_attn,
    myds_metal_buffer_t *ffn_split, const myds_metal_buffer_t *ffn_fn,
    const myds_metal_buffer_t *ffn_scale, const myds_metal_buffer_t *ffn_base,
    myds_metal_buffer_t *flat_norm, myds_metal_buffer_t *mix, float hc_eps) {
    myds_model_t *m=ctx->model;
    myds_layer_t *l=&m->layers[layer];
    const uint32_t dim=m->dim, width=m->moe_expert_dim, k=m->n_activated_experts;
    const int previous_auto=myds_metal_get_auto_barrier();
    const int previous_concurrent=myds_metal_get_concurrent();
    myds_metal_set_auto_barrier(0);
    myds_metal_set_concurrent(1);
#define HC_STAGE_CHECK(call) do { if (!(call)) goto failed; } while (0)
    HC_STAGE_CHECK(hc_scratch_independent(ctx,layer,residual,after_attn,ffn_split,
        ffn_fn,ffn_scale,ffn_base,flat_norm,mix));
    HC_STAGE_CHECK(myds_metal_v41_hc_mix_stages_preflight(ffn_split,after_attn,ffn_fn,
        ffn_scale,ffn_base,flat_norm,mix,dim,m->norm_eps,hc_eps));

    /* Cohort1: the normalization is independent of Router/SharedGU. */
    HC_STAGE_CHECK(myds_metal_v41_hc_mix_norm_stage(flat_norm,after_attn,dim,m->norm_eps));
    HC_STAGE_CHECK(myds_metal_v41_gemv(ctx->router_logits,l->router_gate,ctx->ffn_norm_out,
        m->n_routed_experts,dim,l->router_gate_type));
    HC_STAGE_CHECK(myds_metal_shared_q4_v41_nb_enabled()
        ? myds_metal_shared_q4_v41_nb_gate_up(ctx->shared_swiglu_out,l->shared_gate,l->shared_up,
            ctx->ffn_norm_out,width,dim,m->swiglu_clamp_exp)
        : myds_v41_shared_gate_up_swiglu(ctx->shared_swiglu_out,l->shared_gate,
            l->shared_up,ctx->ffn_norm_out,width,dim,l->shared_gate_type,l->shared_up_type,m->swiglu_clamp_exp));
    const myds_metal_buffer_t *cohort1[]={flat_norm,ctx->router_logits,ctx->shared_swiglu_out};
    myds_metal_barrier_buffers(cohort1,3);

    /* Cohort2: original HC projection alongside the production router. Keep
     * the router's existing internal output join, plus this full union join. */
    HC_STAGE_CHECK(myds_metal_v41_hc_mix_project_stage(mix,ffn_fn,flat_norm,dim));
    HC_STAGE_CHECK(myds_v41_route(ctx->selected_ids,ctx->selected_weights,ctx->moe_gate,
        ctx->moe_up,ctx->router_logits,l->router_bias,m->n_routed_experts,k,m->routed_scaling_factor));
    const myds_metal_buffer_t *cohort2[]={mix,ctx->selected_ids,ctx->selected_weights,ctx->moe_gate,ctx->moe_up};
    myds_metal_barrier_buffers(cohort2,5);

    /* Cohort3: the unchanged one-thread Sinkhorn dispatch is independent of
     * IQ2 GU, whose original MID join and paired down scheduling stay inside
     * the production helper. Encoded means the actual stage3 was accepted. */
    HC_STAGE_CHECK(myds_metal_v41_hc_mix_split_stage(ffn_split,mix,ffn_scale,ffn_base,dim,hc_eps));
    atomic_fetch_add_explicit(&hc_overlap_encoded,1,memory_order_relaxed);
    int (*down_encode)(myds_metal_buffer_t *,const myds_metal_buffer_t *,
        const myds_metal_buffer_t *,const myds_metal_buffer_t *,const myds_metal_buffer_t *,
        const myds_metal_buffer_t *,const myds_metal_buffer_t *,myds_metal_buffer_t *,
        myds_metal_buffer_t *,myds_metal_buffer_t *,myds_metal_buffer_t *,uint32_t,uint32_t,
        uint32_t,uint32_t,ggml_type,ggml_type,ggml_type,float,bool,myds_metal_buffer_t *,
        const myds_metal_buffer_t *,const myds_metal_buffer_t *,ggml_type) =
        myds_metal_shared_q4_v41_nb_enabled() ? myds_metal_routed_moe_down_overlap_nbhc_scoped :
            myds_metal_routed_moe_down_overlap_scoped;
    HC_STAGE_CHECK(down_encode(ctx->moe_out,ctx->ffn_norm_out,
        l->ffn_gate_exps,l->ffn_up_exps,l->ffn_down_exps,ctx->selected_ids,ctx->selected_weights,
        ctx->moe_gate,ctx->moe_up,ctx->moe_mid,ctx->moe_down,dim,width,m->n_routed_experts,k,
        l->ffn_gate_exps_type,l->ffn_up_exps_type,l->ffn_down_exps_type,m->swiglu_clamp_exp,true,
        ctx->shared_out,l->shared_down,ctx->shared_swiglu_out,l->shared_down_type));
    const myds_metal_buffer_t *final_deps[]={ctx->moe_out,ctx->shared_out,ffn_split};
    myds_metal_barrier_buffers(final_deps,3);
    HC_STAGE_CHECK(myds_metal_v41_moe_add_hc_expand(residual,ctx->ffn_out,ctx->moe_out,
        ctx->shared_out,after_attn,ffn_split,dim));
    myds_metal_barrier_buffer(residual);
    myds_metal_set_auto_barrier(previous_auto);
    myds_metal_set_concurrent(previous_concurrent);
#undef HC_STAGE_CHECK
    return 1;
failed:
    atomic_fetch_add_explicit(&hc_overlap_failures,1,memory_order_relaxed);
    myds_metal_set_auto_barrier(previous_auto);
    myds_metal_set_concurrent(previous_concurrent);
    MYDS_LOG_E("MoE HC overlap selected stage failed; refusing fallback");
    return 0;
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
    const int hc_overlap=myds_v41_moe_hc_overlap_enabled();
    if (hc_overlap < 0) return 0;
    if (!ctx || !ctx->model || layer >= ctx->model->n_layers) return 0;
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim = m->dim, width = m->moe_expert_dim, k = m->n_activated_experts;
    const bool trace_on = myds_trace_active();

    const bool hc_overlap_eligible=hc_overlap && ordinary_decode_allowed && !trace_on &&
        myds_metal_batch_active() && is_moe_concurrent_enabled() && is_moe_pipeline_enabled() &&
        is_shared_fuse_enabled() && is_moe_fuse_expand_enabled() &&
        myds_metal_moe_down_overlap_enabled() && myds_metal_moe_q2_v41_enabled() &&
        !myds_metal_moe_iq2_v41_enabled() && myds_metal_dense_q4_v41_enabled() &&
        dim==5120u && width==2304u && m->n_routed_experts==384u && k==6u && m->n_shared_experts==1u &&
        l->ffn_gate_exps_type==GGML_TYPE_IQ2_XXS && l->ffn_up_exps_type==GGML_TYPE_IQ2_XXS &&
        l->ffn_down_exps_type==GGML_TYPE_Q2_K && l->shared_gate_type==GGML_TYPE_Q4_K &&
        l->shared_up_type==GGML_TYPE_Q4_K && l->shared_down_type==GGML_TYPE_Q4_K &&
        residual && after_attn && ffn_split && ffn_fn && ffn_scale && ffn_base && flat_norm && mix;
    if (hc_overlap && ordinary_decode_allowed && !trace_on) {
        atomic_fetch_add_explicit(&hc_overlap_requests,1,memory_order_relaxed);
        if (!hc_overlap_eligible) {
            atomic_fetch_add_explicit(&hc_overlap_failures,1,memory_order_relaxed);
            MYDS_LOG_E("MoE HC overlap ordinary scope conflicts with fixed101 schedule; refusing fallback");
            return 0;
        }
        return moe_hc_overlap_101(ctx,layer,residual,after_attn,ffn_split,ffn_fn,ffn_scale,
            ffn_base,flat_norm,mix,hc_eps);
    }

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

    const bool overlap_down = ordinary_decode_allowed && !trace_on && myds_metal_moe_down_overlap_enabled();
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
    if (!overlap_down && !myds_v41_decode_gemv_bf16(ctx->shared_out, l->shared_down, ctx->shared_swiglu_out,
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

    /* 6. 仅同步路由输出与选择结果，启动 Routed MoE（可选 down pair 在 helper 的 MID join 后编码） */
    const myds_metal_buffer_t *moe_deps[] = {
        ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up
    };
    myds_metal_barrier_buffers(moe_deps, 4);
    const int routed_ok = overlap_down
        ? myds_metal_routed_moe_down_overlap_scoped(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
        l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
        m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
        l->ffn_down_exps_type, m->swiglu_clamp_exp, ordinary_decode_allowed && !trace_on, ctx->shared_out, l->shared_down, ctx->shared_swiglu_out, l->shared_down_type)
        : myds_metal_routed_moe_scoped(ctx->moe_out, ctx->ffn_norm_out, l->ffn_gate_exps,
        l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width,
        m->n_routed_experts, k, l->ffn_gate_exps_type, l->ffn_up_exps_type,
        l->ffn_down_exps_type, m->swiglu_clamp_exp, ordinary_decode_allowed && !trace_on);
    if (!routed_ok) {
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

static _Atomic uint64_t ffn_hc_overlap_requests, ffn_hc_overlap_encoded, ffn_hc_overlap_failures;
static _Atomic int ffn_hc_overlap_mode;

int myds_v41_ffn_hc_overlap_parse_mode(void) {
    const char *v = getenv("MYDS_FFN_HC_OVERLAP");
    const int mode = (!v || !strcmp(v, "0")) ? 0 : !strcmp(v, "1") ? 1 : -1;
    if (mode < 0) MYDS_LOG_E("MYDS_FFN_HC_OVERLAP must be exactly 0 or 1");
    atomic_store_explicit(&ffn_hc_overlap_mode, mode, memory_order_relaxed);
    return mode;
}

int myds_v41_ffn_hc_overlap_enabled(void) {
    return atomic_load_explicit(&ffn_hc_overlap_mode, memory_order_relaxed);
}

void myds_v41_ffn_hc_overlap_reset_mode(void) {
    atomic_store_explicit(&ffn_hc_overlap_mode, 0, memory_order_relaxed);
}

void myds_v41_ffn_hc_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
#define FFN_HC_OVERLAP_COUNT(field) \
    (reset ? atomic_exchange_explicit(&ffn_hc_overlap_##field, 0, memory_order_relaxed) \
           : atomic_load_explicit(&ffn_hc_overlap_##field, memory_order_relaxed))
    const myds_metal_decode_candidate_counters_t value = {
        .requests = FFN_HC_OVERLAP_COUNT(requests),
        .encoded = FFN_HC_OVERLAP_COUNT(encoded),
        .failures = FFN_HC_OVERLAP_COUNT(failures)};
    if (out) *out = value;
#undef FFN_HC_OVERLAP_COUNT
}

static bool ffn_hc_scratch_independent(myds_context_t *ctx, uint32_t layer,
    myds_metal_buffer_t *residual, const myds_metal_buffer_t *after_attn,
    const myds_metal_buffer_t *attn_split, myds_metal_buffer_t *ffn_split,
    const myds_metal_buffer_t *ffn_fn, const myds_metal_buffer_t *ffn_scale,
    const myds_metal_buffer_t *ffn_base, myds_metal_buffer_t *flat_norm,
    myds_metal_buffer_t *mix) {
    const myds_layer_t *l = &ctx->model->layers[layer];
    const myds_metal_buffer_t *write[] = {flat_norm, mix, ffn_split, ctx->ffn_norm_out, ctx->x};
    const size_t write_bytes[] = {81920u, 96u, 96u, 20480u, 20480u};
    const myds_metal_buffer_t *other[] = {residual, after_attn, attn_split, l->ffn_norm,
        ffn_fn, ffn_scale, ffn_base, ctx->router_logits, l->router_gate, l->router_bias,
        ctx->shared_swiglu_out, l->shared_gate, l->shared_up, l->shared_down, ctx->shared_out,
        l->ffn_gate_exps, l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, ctx->moe_out, ctx->ffn_out};
    for (unsigned i = 0; i < 5u; ++i) {
        const uintptr_t a = (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)write[i]);
        if (!a || write_bytes[i] > myds_metal_buffer_size(write[i]) || write_bytes[i] > UINTPTR_MAX - a) return false;
        for (unsigned j = 0; j < i; ++j) {
            const uintptr_t b = (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)write[j]);
            if (a <= b ? b - a < write_bytes[i] : a - b < write_bytes[j]) return false;
        }
        for (unsigned j = 0; j < sizeof(other) / sizeof(*other); ++j) {
            const uintptr_t b = (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)other[j]);
            const size_t bytes = myds_metal_buffer_size(other[j]);
            if (!b || !bytes || bytes > UINTPTR_MAX - b ||
                (a <= b ? b - a < write_bytes[i] : a - b < bytes)) return false;
        }
    }
    return true;
}

int myds_v41_moe_ffn_hc_overlap_scoped(myds_context_t *ctx, uint32_t layer, uint32_t token, uint32_t pos,
    myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *after_attn,
    const myds_metal_buffer_t *attn_split,
    myds_metal_buffer_t *ffn_split,
    const myds_metal_buffer_t *ffn_fn,
    const myds_metal_buffer_t *ffn_scale,
    const myds_metal_buffer_t *ffn_base,
    myds_metal_buffer_t *flat_norm,
    myds_metal_buffer_t *mix,
    float hc_eps, bool ordinary_decode_allowed) {
    (void)token;
    (void)pos;
    const int ffn_hc_mode = myds_v41_ffn_hc_overlap_enabled();
    const int hc_overlap = myds_v41_moe_hc_overlap_enabled();
    if (ffn_hc_mode != 1 || hc_overlap < 0) return 0;

    const bool trace_on = myds_trace_active();
    if (!ordinary_decode_allowed || trace_on) return 0;

    atomic_fetch_add_explicit(&ffn_hc_overlap_requests, 1, memory_order_relaxed);
    if (hc_overlap) atomic_fetch_add_explicit(&hc_overlap_requests, 1, memory_order_relaxed);

    if (!ctx || !ctx->model || !ctx->model->layers || layer >= ctx->model->n_layers) {
        atomic_fetch_add_explicit(&ffn_hc_overlap_failures, 1, memory_order_relaxed);
        if (hc_overlap) atomic_fetch_add_explicit(&hc_overlap_failures, 1, memory_order_relaxed);
        return 0;
    }
    myds_model_t *m = ctx->model;
    myds_layer_t *l = &m->layers[layer];
    const uint32_t dim = m->dim, width = m->moe_expert_dim, k = m->n_activated_experts;

    const char *fuse_hc_norm_env = getenv("MYDS_FUSE_HC_SUM_RMSNORM");
    const bool fuse_hc_norm_on = !fuse_hc_norm_env || atoi(fuse_hc_norm_env) != 0;
    const bool eligible = myds_metal_batch_active() && is_moe_concurrent_enabled() &&
        is_moe_pipeline_enabled() && is_shared_fuse_enabled() && is_moe_fuse_expand_enabled() &&
        fuse_hc_norm_on &&
        myds_metal_moe_down_overlap_enabled() && myds_metal_moe_q2_v41_enabled() &&
        !myds_metal_moe_iq2_v41_enabled() && myds_metal_dense_q4_v41_enabled() &&
        dim == 5120u && width == 2304u && m->n_routed_experts == 384u && k == 6u && m->n_shared_experts == 1u &&
        isfinite(m->norm_eps) && m->norm_eps >= 0.0f && isfinite(hc_eps) && hc_eps >= 0.0f &&
        l->ffn_gate_exps_type == GGML_TYPE_IQ2_XXS && l->ffn_up_exps_type == GGML_TYPE_IQ2_XXS &&
        l->ffn_down_exps_type == GGML_TYPE_Q2_K && l->shared_gate_type == GGML_TYPE_Q4_K &&
        l->shared_up_type == GGML_TYPE_Q4_K && l->shared_down_type == GGML_TYPE_Q4_K &&
        residual && after_attn && attn_split && ffn_split && ffn_fn && ffn_scale && ffn_base && flat_norm && mix;
    if (!eligible) {
        atomic_fetch_add_explicit(&ffn_hc_overlap_failures, 1, memory_order_relaxed);
        if (hc_overlap) atomic_fetch_add_explicit(&hc_overlap_failures, 1, memory_order_relaxed);
        MYDS_LOG_E("FFN HC overlap ordinary scope preconditions failed; refusing fallback");
        return 0;
    }

    const int previous_auto = myds_metal_get_auto_barrier();
    const int previous_concurrent = myds_metal_get_concurrent();
    myds_metal_set_auto_barrier(0);
    myds_metal_set_concurrent(1);
#define FFN_HC_CHECK(call) do { if (!(call)) goto failed; } while (0)
    FFN_HC_CHECK(ffn_hc_scratch_independent(ctx, layer, residual, after_attn, attn_split,
        ffn_split, ffn_fn, ffn_scale, ffn_base, flat_norm, mix));
    FFN_HC_CHECK(myds_metal_v41_hc_mix_stages_preflight(ffn_split, after_attn, ffn_fn,
        ffn_scale, ffn_base, flat_norm, mix, dim, m->norm_eps, hc_eps));

    /* Cohort 0: FFN input HC sum+RMSNorm and HC mix norm stage both read only
     * after_attn (and attn_split / ffn_norm) and write disjoint buffers. */
    FFN_HC_CHECK(myds_metal_v41_hc_sum_rmsnorm(ctx->ffn_norm_out, ctx->x, after_attn,
        attn_split, l->ffn_norm, dim, m->norm_eps, true));
    FFN_HC_CHECK(myds_metal_v41_hc_mix_norm_stage(flat_norm, after_attn, dim, m->norm_eps));
    const myds_metal_buffer_t *cohort0[] = { ctx->ffn_norm_out, flat_norm };
    myds_metal_barrier_buffers(cohort0, 2);

    /* Cohort 1: HC mix projection (983 KB F16 weights) and Router GEMV (3.9 MB F32
     * weights) run concurrently so router_logits is ready for Cohort 2 as early as possible. */
    FFN_HC_CHECK(myds_metal_v41_hc_mix_project_stage(mix, ffn_fn, flat_norm, dim));
    FFN_HC_CHECK(myds_metal_v41_gemv(ctx->router_logits, l->router_gate, ctx->ffn_norm_out,
        m->n_routed_experts, dim, l->router_gate_type));
    const myds_metal_buffer_t *cohort1[] = { mix, ctx->router_logits };
    myds_metal_barrier_buffers(cohort1, 2);

    /* Cohort 2: 1-threadgroup Sinkhorn split stage, 1-threadgroup router top-6
     * selection, and 288-threadgroup Shared Expert Gate+Up SwiGLU all run
     * concurrently, followed by a single unified barrier. */
    FFN_HC_CHECK(myds_metal_v41_hc_mix_split_stage(ffn_split, mix, ffn_scale, ffn_base, dim, hc_eps));
    if (hc_overlap) atomic_fetch_add_explicit(&hc_overlap_encoded, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&ffn_hc_overlap_encoded, 1, memory_order_relaxed);
    FFN_HC_CHECK(route_batch_impl(ctx->selected_ids, ctx->selected_weights, ctx->moe_gate,
        ctx->moe_up, ctx->router_logits, l->router_bias, m->n_routed_experts, k,
        m->routed_scaling_factor, 1, false));
    FFN_HC_CHECK(myds_metal_shared_q4_v41_nb_enabled()
        ? myds_metal_shared_q4_v41_nb_gate_up(ctx->shared_swiglu_out, l->shared_gate, l->shared_up,
            ctx->ffn_norm_out, width, dim, m->swiglu_clamp_exp)
        : myds_v41_shared_gate_up_swiglu(ctx->shared_swiglu_out, l->shared_gate,
            l->shared_up, ctx->ffn_norm_out, width, dim, l->shared_gate_type, l->shared_up_type, m->swiglu_clamp_exp));
    const myds_metal_buffer_t *cohort2[] = {
        ctx->selected_ids, ctx->selected_weights, ctx->moe_gate, ctx->moe_up, ctx->shared_swiglu_out
    };
    myds_metal_barrier_buffers(cohort2, 5);

    /* Cohort 3: Routed MoE Gate+Up + paired Down + Shared Down. */
    int (*down_encode)(myds_metal_buffer_t *, const myds_metal_buffer_t *,
        const myds_metal_buffer_t *, const myds_metal_buffer_t *, const myds_metal_buffer_t *,
        const myds_metal_buffer_t *, const myds_metal_buffer_t *, myds_metal_buffer_t *,
        myds_metal_buffer_t *, myds_metal_buffer_t *, myds_metal_buffer_t *, uint32_t, uint32_t,
        uint32_t, uint32_t, ggml_type, ggml_type, ggml_type, float, bool, myds_metal_buffer_t *,
        const myds_metal_buffer_t *, const myds_metal_buffer_t *, ggml_type) =
        myds_metal_shared_q4_v41_nb_enabled() ? myds_metal_routed_moe_down_overlap_nbhc_scoped :
            myds_metal_routed_moe_down_overlap_scoped;
    FFN_HC_CHECK(down_encode(ctx->moe_out, ctx->ffn_norm_out,
        l->ffn_gate_exps, l->ffn_up_exps, l->ffn_down_exps, ctx->selected_ids, ctx->selected_weights,
        ctx->moe_gate, ctx->moe_up, ctx->moe_mid, ctx->moe_down, dim, width, m->n_routed_experts, k,
        l->ffn_gate_exps_type, l->ffn_up_exps_type, l->ffn_down_exps_type, m->swiglu_clamp_exp, true,
        ctx->shared_out, l->shared_down, ctx->shared_swiglu_out, l->shared_down_type));
    const myds_metal_buffer_t *final_deps[] = { ctx->moe_out, ctx->shared_out, ffn_split };
    myds_metal_barrier_buffers(final_deps, 3);
    FFN_HC_CHECK(myds_metal_v41_moe_add_hc_expand(residual, ctx->ffn_out, ctx->moe_out,
        ctx->shared_out, after_attn, ffn_split, dim));
    myds_metal_barrier_buffer(residual);
    myds_metal_set_auto_barrier(previous_auto);
    myds_metal_set_concurrent(previous_concurrent);
#undef FFN_HC_CHECK
    return 1;
failed:
    atomic_fetch_add_explicit(&ffn_hc_overlap_failures, 1, memory_order_relaxed);
    if (hc_overlap) atomic_fetch_add_explicit(&hc_overlap_failures, 1, memory_order_relaxed);
    myds_metal_set_auto_barrier(previous_auto);
    myds_metal_set_concurrent(previous_concurrent);
    MYDS_LOG_E("FFN HC overlap selected stage failed; refusing fallback");
    return 0;
}

static _Atomic uint64_t attn_hc_overlap_requests, attn_hc_overlap_encoded, attn_hc_overlap_failures;
static _Atomic int attn_hc_overlap_mode;

int myds_v41_attn_hc_overlap_parse_mode(void) {
    const char *v = getenv("MYDS_ATTN_HC_OVERLAP");
    const int mode = (!v || !strcmp(v, "0")) ? 0 : !strcmp(v, "1") ? 1 : -1;
    if (mode < 0) MYDS_LOG_E("MYDS_ATTN_HC_OVERLAP must be exactly 0 or 1");
    atomic_store_explicit(&attn_hc_overlap_mode, mode, memory_order_relaxed);
    return mode;
}

int myds_v41_attn_hc_overlap_enabled(void) {
    return atomic_load_explicit(&attn_hc_overlap_mode, memory_order_relaxed);
}

void myds_v41_attn_hc_overlap_reset_mode(void) {
    atomic_store_explicit(&attn_hc_overlap_mode, 0, memory_order_relaxed);
}

void myds_v41_attn_hc_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
#define ATTN_HC_OVERLAP_COUNT(field) \
    (reset ? atomic_exchange_explicit(&attn_hc_overlap_##field, 0, memory_order_relaxed) \
           : atomic_load_explicit(&attn_hc_overlap_##field, memory_order_relaxed))
    const myds_metal_decode_candidate_counters_t value = {
        .requests = ATTN_HC_OVERLAP_COUNT(requests),
        .encoded = ATTN_HC_OVERLAP_COUNT(encoded),
        .failures = ATTN_HC_OVERLAP_COUNT(failures)};
    if (out) *out = value;
#undef ATTN_HC_OVERLAP_COUNT
}

static _Atomic uint64_t indexer_decode_overlap_requests, indexer_decode_overlap_encoded, indexer_decode_overlap_failures;
static _Atomic int indexer_decode_overlap_mode;

int myds_v41_indexer_decode_overlap_parse_mode(void) {
    const char *v = getenv("MYDS_INDEXER_DECODE_OVERLAP");
    const int mode = (!v || !strcmp(v, "0")) ? 0 : !strcmp(v, "1") ? 1 : -1;
    if (mode < 0) MYDS_LOG_E("MYDS_INDEXER_DECODE_OVERLAP must be exactly 0 or 1");
    atomic_store_explicit(&indexer_decode_overlap_mode, mode, memory_order_relaxed);
    return mode;
}

int myds_v41_indexer_decode_overlap_enabled(void) {
    return atomic_load_explicit(&indexer_decode_overlap_mode, memory_order_relaxed);
}

void myds_v41_indexer_decode_overlap_reset_mode(void) {
    atomic_store_explicit(&indexer_decode_overlap_mode, 0, memory_order_relaxed);
}

void myds_v41_indexer_decode_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
#define INDEXER_DECODE_OVERLAP_COUNT(field) \
    (reset ? atomic_exchange_explicit(&indexer_decode_overlap_##field, 0, memory_order_relaxed) \
           : atomic_load_explicit(&indexer_decode_overlap_##field, memory_order_relaxed))
    const myds_metal_decode_candidate_counters_t value = {
        .requests = INDEXER_DECODE_OVERLAP_COUNT(requests),
        .encoded = INDEXER_DECODE_OVERLAP_COUNT(encoded),
        .failures = INDEXER_DECODE_OVERLAP_COUNT(failures)};
    if (out) *out = value;
#undef INDEXER_DECODE_OVERLAP_COUNT
}

static uint32_t decode_ratio_for(uint32_t l) { return l < 2u ? 0u : (l < 20u ? 2u : 1u); }
static bool decode_kv_source(uint32_t l) { return l == 2u || l == 8u || l == 14u || l == 20u; }
static bool decode_index_source(uint32_t l) {
    return decode_kv_source(l) || (l >= 24u && l < 40u && l % 4u == 0u);
}

static bool env_default_on(const char *name) {
    const char *v = getenv(name);
    return !v || atoi(v) != 0;
}

static bool env_not_zero_literal(const char *name) {
    const char *v = getenv(name);
    return !(v && strcmp(v, "0") == 0);
}

static bool env_is_one_literal(const char *name) {
    const char *v = getenv(name);
    return v && strcmp(v, "1") == 0;
}

static bool env_block_budget_default(void) {
    const char *v = getenv("MYDS_BLOCK_BUDGET");
    return !v || atoi(v) <= 0;
}

typedef struct {
    const myds_metal_buffer_t *buf;
    size_t min_bytes;
} decode_buf_span_t;

static bool decode_spans_disjoint(const decode_buf_span_t *spans, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!spans[i].buf || !spans[i].min_bytes) return false;
        const uintptr_t a = (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)spans[i].buf);
        const size_t sa = myds_metal_buffer_size(spans[i].buf);
        if (!a || spans[i].min_bytes > sa || sa > UINTPTR_MAX - a) return false;
        for (size_t j = 0; j < i; ++j) {
            const uintptr_t b = (uintptr_t)myds_metal_buffer_data((myds_metal_buffer_t *)spans[j].buf);
            const size_t sb = myds_metal_buffer_size(spans[j].buf);
            if (a <= b ? (b - a < sa) : (a - b < sb)) return false;
        }
    }
    return true;
}

static bool decode_layer_buffers_valid(const myds_context_t *ctx,
    const myds_v41_decode_attn_args_t *args, bool check_indexer, bool check_kv_reuse) {
    const myds_layer_t *base = &ctx->model->layers[args->layer];
    const uint32_t il = args->layer;
    const uint32_t ratio = decode_ratio_for(il);
    const uint32_t count = ratio ? (args->pos + 1u) / ratio : 0u;
    const bool is_kv = decode_kv_source(il);
    const bool is_idx = decode_index_source(il);
    const bool kv_commit = is_kv && ratio != 0u && ((args->pos + 1u) % ratio == 0u);
    const uint32_t blocks = (count + 7u) / 8u;

    decode_buf_span_t spans[48];
    size_t n = 0;
#define ADD_SPAN(b, bytes) do { spans[n++] = (decode_buf_span_t){(b), (bytes)}; } while (0)
    ADD_SPAN(args->flat_norm, 81920u);
    ADD_SPAN(args->mix, 96u);
    ADD_SPAN(args->attn_split, 96u);
    ADD_SPAN(args->residual, 81920u);
    ADD_SPAN(args->hc_in_pre, il == 0u ? 16u : 96u);
    ADD_SPAN(args->attn_fn, 1u);
    ADD_SPAN(args->attn_scale, 12u);
    ADD_SPAN(args->attn_base, 96u);
    ADD_SPAN(args->window, 262144u);
    ADD_SPAN(ctx->x, 20480u);
    ADD_SPAN(ctx->x_norm, 20480u);
    ADD_SPAN(ctx->qr, 5120u);
    ADD_SPAN(ctx->qr_norm, 5120u);
    ADD_SPAN(ctx->q, 131072u);
    ADD_SPAN(ctx->kv_act, 2048u);
    ADD_SPAN(base->attn_norm, 20480u);
    ADD_SPAN(base->attn_q_a, 1u);
    ADD_SPAN(base->attn_q_a_norm, 5120u);
    ADD_SPAN(base->attn_q_b, 1u);
    ADD_SPAN(base->attn_kv, 1u);
    ADD_SPAN(base->attn_kv_a_norm, 2048u);

    if (check_kv_reuse) {
        if (!args->gpu_scratch) return false;
        ADD_SPAN(myds_v41_gpu_scratch_kv_f16(args->gpu_scratch), 640u * 512u * 2u);
        ADD_SPAN(myds_v41_gpu_scratch_pad(args->gpu_scratch), 2u * 32u * 512u * 2u + 32u * 2u);
        ADD_SPAN(myds_v41_gpu_scratch_partial(args->gpu_scratch), 64u * 514u * 32u * 4u);
    }

    if (check_indexer && (is_kv || is_idx)) {
        if (is_kv) {
            ADD_SPAN(args->compressor_kv, 1u);
            ADD_SPAN(args->pool_kv, 2048u);
            ADD_SPAN(args->latent, 2048u);
            if (ratio == 2u) {
                ADD_SPAN(args->compressor_gate, 1u);
                ADD_SPAN(args->pool_score, 2048u);
                ADD_SPAN(args->previous_kv, 2048u);
                ADD_SPAN(args->previous_score, 2048u);
            }
            if (kv_commit) {
                ADD_SPAN(args->compressor_norm, 2048u);
                ADD_SPAN(args->index_k_weight, 131072u);
                ADD_SPAN(args->index_k_norm, 512u);
                ADD_SPAN(args->index_k_buf, 512u);
            }
        }
        if (kv_commit || (is_idx && count > 0u)) {
            const size_t rows = count > 0u ? (size_t)count : 1u;
            ADD_SPAN(args->compressed_kv, rows * 1024u);
            ADD_SPAN(args->index_cache, rows * 256u);
        }
        if (is_idx && count > 0u) {
            if (!args->gpu_scratch) return false;
            ADD_SPAN(args->index_q_weight, 10485760u);
            ADD_SPAN(args->index_proj_weight, 327680u);
            ADD_SPAN(args->index_q_buf, 16384u);
            ADD_SPAN(args->index_weights, 128u);
            ADD_SPAN(args->index_scores, (size_t)count * 4u);
            ADD_SPAN(args->selected, 2048u);
            if (il >= 20u) ADD_SPAN(args->block_mask, (size_t)blocks * 4u);
            if (il == 20u) {
                ADD_SPAN(args->block_scores, (size_t)blocks * 4u);
                ADD_SPAN(args->block_selected, 8192u);
            }
        }
    }
#undef ADD_SPAN
    return decode_spans_disjoint(spans, n);
}

static int ref_project_f16(myds_metal_buffer_t *out, const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *in, uint32_t rows, uint32_t width, bool round, bool ordinary_allowed) {
    if (ordinary_allowed && round && myds_metal_f16_gemv_bf16_fuse_enabled() == 1) {
        if (myds_metal_get_auto_barrier()) myds_metal_barrier();
        else myds_metal_barrier_buffer(in);
        if (!myds_metal_v41_gemv_f16_bf16(out, w, in, rows, width)) return 0;
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(out);
        return 1;
    }
    if (myds_metal_get_auto_barrier()) myds_metal_barrier();
    else myds_metal_barrier_buffer(in);
    if (!myds_metal_v41_gemv(out, w, in, rows, width, GGML_TYPE_F16)) return 0;
    if (round) {
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(out);
        if (!myds_metal_v41_bf16(out, rows)) return 0;
    }
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(out);
    return 1;
}

static int ref_publish_compressor_decode(const myds_context_t *ctx,
    const myds_v41_decode_attn_args_t *args, bool ordinary_allowed) {
    const uint32_t il = args->layer;
    const uint32_t ratio = decode_ratio_for(il);
    if (!ratio || !decode_kv_source(il)) return 1;
    const uint32_t count = (args->pos + 1u) / ratio;
    const float eps = ctx->model->norm_eps;
#define CHECK(x) do { if (!(x)) return 0; } while (0)
    CHECK(ref_project_f16(args->pool_kv, args->compressor_kv, ctx->x_norm, 512u, 5120u, ratio == 1u, ordinary_allowed));
    if (ratio == 2u) {
        CHECK(ref_project_f16(args->pool_score, args->compressor_gate, ctx->x_norm, 512u, 5120u, false, false));
        if (!myds_metal_get_auto_barrier()) {
            const myds_metal_buffer_t *pool_deps[] = { args->pool_kv, args->pool_score };
            myds_metal_barrier_buffers(pool_deps, 2);
        }
        CHECK(myds_v41_gpu_pool2(args->latent, args->pool_kv, args->pool_score,
            args->previous_kv, args->previous_score, 512u, args->pos));
    } else {
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->pool_kv);
        CHECK(myds_metal_buffer_copy(args->latent, 0, args->pool_kv, 0, 512u * 4u));
    }
    if ((args->pos + 1u) % ratio == 0u) {
        const uint32_t start = args->pos + 1u - ratio;
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->latent);
        CHECK(myds_metal_v41_rmsnorm(args->latent, args->latent, args->compressor_norm, 512u, eps, true));
        CHECK(ref_project_f16(args->index_k_buf, args->index_k_weight, args->latent, 128u, 512u, true, ordinary_allowed));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_k_buf);
        CHECK(myds_metal_v41_rmsnorm(args->index_k_buf, args->index_k_buf, args->index_k_norm, 128u, eps, true));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_k_buf);
        if (ordinary_allowed && myds_metal_indexer_rope_quant_fuse_enabled() == 1) {
            CHECK(myds_v41_gpu_indexer_rope_quant(args->index_k_buf, 1u, start));
        } else {
            CHECK(myds_v41_gpu_rope(args->index_k_buf, 128u, 1u, start, true, false));
            if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_k_buf);
            CHECK(myds_v41_gpu_quant(args->index_k_buf, 128u, 1u, 2u));
        }
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_k_buf);
        CHECK(myds_v41_gpu_pack_bf16(args->index_cache, (size_t)(count - 1u) * 128u, args->index_k_buf, 0, 128u));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->latent);
        CHECK(myds_v41_gpu_rope(args->latent, 512u, 1u, start, true, false));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->latent);
        CHECK(myds_v41_gpu_quant(args->latent, 512u, 1u, 3u));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->latent);
        CHECK(myds_v41_gpu_pack_bf16(args->compressed_kv, (size_t)(count - 1u) * 512u, args->latent, 0, 512u));
        if (!myds_metal_get_auto_barrier()) {
            const myds_metal_buffer_t *comp_out[] = { args->index_cache, args->compressed_kv };
            myds_metal_barrier_buffers(comp_out, 2);
        }
    }
#undef CHECK
    return 1;
}

static int ref_publish_index_weights_decode(const myds_context_t *ctx,
    const myds_v41_decode_attn_args_t *args, bool ordinary_allowed) {
    const uint32_t il = args->layer;
    const uint32_t ratio = decode_ratio_for(il);
    if (!ratio) return 1;
    const uint32_t count = (args->pos + 1u) / ratio;
    if (count && decode_index_source(il)) {
        return ref_project_f16(args->index_weights, args->index_proj_weight, ctx->x_norm,
                               32u, 5120u, true, ordinary_allowed);
    }
    return 1;
}

static int ref_publish_indexer_one(const myds_context_t *ctx,
    const myds_v41_decode_attn_args_t *args, bool ordinary_allowed) {
    const uint32_t il = args->layer;
    const uint32_t ratio = decode_ratio_for(il);
    if (!ratio) return 1;
    const uint32_t count = (args->pos + 1u) / ratio;
    if (!count || !decode_index_source(il)) return 1;
#define CHECK(x) do { if (!(x)) return 0; } while (0)
    CHECK(ref_project_f16(args->index_q_buf, args->index_q_weight, ctx->qr_norm, 32u * 128u, 1280u, true, ordinary_allowed));
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_q_buf);
    if (ordinary_allowed && myds_metal_indexer_rope_quant_fuse_enabled() == 1) {
        CHECK(myds_v41_gpu_indexer_rope_quant(args->index_q_buf, 32u, args->pos));
    } else {
        CHECK(myds_v41_gpu_rope(args->index_q_buf, 128u, 32u, args->pos, true, false));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_q_buf);
        CHECK(myds_v41_gpu_quant(args->index_q_buf, 128u, 32u, 2u));
    }
    const myds_metal_buffer_t *b_mask = (il > 20u) ? args->block_mask : NULL;
    if (!myds_metal_get_auto_barrier()) {
        const myds_metal_buffer_t *score_deps[] = { args->index_q_buf, args->index_weights, args->index_cache };
        myds_metal_barrier_buffers(score_deps, 3);
    }
    if (ordinary_allowed && myds_v41_gpu_indexer_head_slots_enabled() == 1) {
        CHECK(myds_v41_gpu_indexscores_head_slots(args->index_scores, args->index_q_buf,
            args->index_weights, args->index_cache, b_mask, count, 32u, true));
    } else {
        CHECK(myds_v41_gpu_indexscores_masked_bf16(args->index_scores, args->index_q_buf,
            args->index_weights, args->index_cache, b_mask, count, 32u));
    }
    if (il == 20u) {
        const uint32_t blocks = (count + 7u) / 8u;
        const uint32_t top = blocks < 2048u ? blocks : 2048u;
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_scores);
        CHECK(myds_v41_gpu_candidates(args->block_scores, args->index_scores, count, args->pos, ratio));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->block_scores);
        CHECK(myds_v41_gpu_topk(args->block_selected, args->block_scores, blocks, top, args->gpu_scratch));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->block_selected);
        CHECK(myds_v41_gpu_topkmask(args->block_mask, args->block_selected, blocks, top));
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->block_mask);
    } else if (il > 20u) {
        if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_scores);
        CHECK(myds_v41_gpu_filter(args->index_scores, b_mask, count, args->pos, ratio));
    }
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->index_scores);
    const uint32_t top_k = count < 512u ? count : 512u;
    if (ordinary_allowed && myds_v41_gpu_topk_threshold_enabled() == 1) {
        CHECK(myds_v41_gpu_topk_threshold(args->selected, args->index_scores, count, top_k, args->gpu_scratch));
    } else {
        CHECK(myds_v41_gpu_topk(args->selected, args->index_scores, count, top_k, args->gpu_scratch));
    }
    if (!myds_metal_get_auto_barrier()) myds_metal_barrier_buffer(args->selected);
#undef CHECK
    return 1;
}

int myds_v41_decode_attn_indexer_scoped(myds_context_t *ctx,
    const myds_v41_decode_attn_args_t *args, bool ordinary_decode_allowed) {
    const int attn_hc_mode = myds_v41_attn_hc_overlap_enabled();
    const int idx_overlap_mode = myds_v41_indexer_decode_overlap_enabled();
    const int kv_reuse_mode = myds_v41_attn_kv_reuse_enabled();
    if (attn_hc_mode < 0 || idx_overlap_mode < 0 || kv_reuse_mode < 0) return 0;

    const bool trace_on = myds_trace_active();
    const bool in_idx_scope = !args || args->layer >= 40u || decode_index_source(args->layer);
    const bool want_attn_hc = (attn_hc_mode == 1) && ordinary_decode_allowed && !trace_on;
    const bool want_idx_overlap = (idx_overlap_mode == 1) && ordinary_decode_allowed && !trace_on && in_idx_scope;
    const bool want_kv_reuse = (kv_reuse_mode == 1) && ordinary_decode_allowed && !trace_on;

    if (!want_kv_reuse) {
        myds_v41_gpu_attn_kv_reuse_reset(args ? args->gpu_scratch : NULL);
    }

    if (want_attn_hc) atomic_fetch_add_explicit(&attn_hc_overlap_requests, 1, memory_order_relaxed);
    if (want_idx_overlap) atomic_fetch_add_explicit(&indexer_decode_overlap_requests, 1, memory_order_relaxed);

    const bool struct_ok = ctx && ctx->model && ctx->model->layers && args &&
        args->layer < ctx->model->n_layers && ctx->model->n_layers == 40u &&
        ctx->model->dim == 5120u && ctx->model->n_heads == 64u && ctx->model->head_dim == 512u &&
        isfinite(ctx->model->norm_eps) && ctx->model->norm_eps >= 0.0f &&
        isfinite(args->hc_eps) && args->hc_eps >= 0.0f &&
        args->pos < args->capacity && args->pos < ctx->cfg.max_seq_len;

    if (!struct_ok) {
        if (want_attn_hc) atomic_fetch_add_explicit(&attn_hc_overlap_failures, 1, memory_order_relaxed);
        if (want_idx_overlap) atomic_fetch_add_explicit(&indexer_decode_overlap_failures, 1, memory_order_relaxed);
        if (want_kv_reuse) {
            myds_v41_gpu_attn_kv_reuse_reset(args ? args->gpu_scratch : NULL);
            myds_v41_attn_kv_reuse_record_prestage_failure();
        }
        return 0;
    }

    const myds_model_t *m = ctx->model;
    const myds_layer_t *base = &m->layers[args->layer];
    const uint32_t il = args->layer;
    const uint32_t pos = args->pos;
    const uint32_t ratio = decode_ratio_for(il);
    const bool compressed = ratio != 0u;
    const uint32_t count = compressed ? (pos + 1u) / ratio : 0u;
    const bool is_kv = decode_kv_source(il);
    const bool is_idx = decode_index_source(il);
    const bool kv_commit = is_kv && compressed && ((pos + 1u) % ratio == 0u);

    const bool base_pipeline_ok = myds_metal_batch_active() &&
        env_default_on("MYDS_QKV_PIPELINE") &&
        env_default_on("MYDS_ATTN_OUTPUT_PIPELINE") &&
        env_default_on("MYDS_DECODE_CONCURRENT") &&
        env_default_on("MYDS_ATTN_INDEXER_PIPELINE") &&
        env_default_on("MYDS_ATTN_INDEXER_EARLY") &&
        env_default_on("MYDS_FUSE_HC_SUM_RMSNORM") &&
        env_not_zero_literal("MYDS_QKV_FUSE_BF16") &&
        env_not_zero_literal("MYDS_QKV_FUSE_POST") &&
        !env_is_one_literal("MYDS_QKV_CONCURRENT") &&
        myds_metal_dense_q4_v41_enabled() == 1 &&
        (base->attn_q_a_type == GGML_TYPE_Q8_0 || base->attn_q_a_type == GGML_TYPE_Q4_K) &&
        base->attn_q_b_type == GGML_TYPE_Q4_K &&
        (base->attn_kv_type == GGML_TYPE_F16 || base->attn_kv_type == GGML_TYPE_Q8_0 ||
         base->attn_kv_type == GGML_TYPE_Q4_K);

    const bool idx_pipeline_ok = base_pipeline_ok &&
        myds_metal_f16_gemv_bf16_fuse_enabled() == 1 &&
        myds_metal_indexer_rope_quant_fuse_enabled() == 1 &&
        myds_v41_gpu_indexer_head_slots_enabled() == 1 &&
        myds_v41_gpu_topk_threshold_enabled() == 0 &&
        !env_is_one_literal("MYDS_INDEXER_COMPACT_SCORER") &&
        env_block_budget_default();

    const bool kv_reuse_pipeline_ok = base_pipeline_ok &&
        args->gpu_scratch &&
        env_default_on("MYDS_ATTN_DIRECT_PACK") &&
        myds_v41_gpu_attention_direct_pack_fuse_enabled() == 0 &&
        myds_v41_attention_rope_fuse_enabled() == 1 &&
        myds_v41_gpu_attention_packed_register_enabled() == 1;

    if (!want_attn_hc && !want_idx_overlap && !want_kv_reuse) {
        if (!decode_layer_buffers_valid(ctx, args, true, false)) return 0;
        const int prev_auto = myds_metal_get_auto_barrier();
        const int prev_conc = myds_metal_get_concurrent();
        myds_metal_set_auto_barrier(0);
        myds_metal_set_concurrent(1);
#define REF_CHECK(call) do { if (!(call)) { \
    myds_metal_set_auto_barrier(prev_auto); myds_metal_set_concurrent(prev_conc); return 0; } } while (0)
        REF_CHECK(myds_metal_v41_hc_sum_rmsnorm(ctx->x_norm, ctx->x, args->residual,
            args->hc_in_pre, base->attn_norm, 5120u, m->norm_eps, true));
        myds_metal_barrier_buffer(ctx->x_norm);
        REF_CHECK(myds_metal_v41_hc_mix(args->attn_split, args->residual, args->attn_fn, GGML_TYPE_F16,
            args->attn_scale, args->attn_base, args->flat_norm, args->mix, 5120u, m->norm_eps, args->hc_eps));
        REF_CHECK(myds_metal_v41_gemv_bf16(ctx->qr, base->attn_q_a, ctx->x_norm, 1280u, 5120u, base->attn_q_a_type));
        REF_CHECK(myds_metal_v41_gemv_bf16(ctx->kv_act, base->attn_kv, ctx->x_norm, 512u, 5120u, base->attn_kv_type));
        REF_CHECK(ref_publish_compressor_decode(ctx, args, ordinary_decode_allowed && !trace_on));
        REF_CHECK(ref_publish_index_weights_decode(ctx, args, ordinary_decode_allowed && !trace_on));
        const myds_metal_buffer_t *qkv_out[] = { ctx->qr, ctx->kv_act };
        myds_metal_barrier_buffers(qkv_out, 2);
        REF_CHECK(myds_metal_v41_rmsnorm(ctx->qr_norm, ctx->qr, base->attn_q_a_norm, 1280u, m->norm_eps, true));
        REF_CHECK(myds_metal_v41_rmsnorm(ctx->kv_act, ctx->kv_act, base->attn_kv_a_norm, 512u, m->norm_eps, true));
        const myds_metal_buffer_t *norm_out[] = { ctx->qr_norm, ctx->kv_act };
        myds_metal_barrier_buffers(norm_out, 2);
        if (ordinary_decode_allowed && !trace_on && myds_metal_dense_q4_v41_enabled() == 1) {
            REF_CHECK(myds_metal_dense_q4_v41(ctx->q, base->attn_q_b, ctx->qr_norm, 64u * 512u, 1280u, true));
        } else {
            REF_CHECK(myds_metal_v41_gemv_bf16(ctx->q, base->attn_q_b, ctx->qr_norm, 64u * 512u, 1280u, base->attn_q_b_type));
        }
        REF_CHECK(ref_publish_indexer_one(ctx, args, ordinary_decode_allowed && !trace_on));
        myds_metal_barrier_buffer(ctx->q);
        REF_CHECK(myds_v41_gpu_qkv_finish(ctx->q, ctx->kv_act, args->window, pos, compressed));
        const myds_metal_buffer_t *attn_in_deps[] = {
            ctx->q, args->window, count ? args->selected : NULL, count ? args->compressed_kv : NULL
        };
        myds_metal_barrier_buffers(attn_in_deps, 4);
#undef REF_CHECK
        myds_metal_set_auto_barrier(prev_auto);
        myds_metal_set_concurrent(prev_conc);
        return 1;
    }

    if ((want_attn_hc && !base_pipeline_ok) || (want_idx_overlap && !idx_pipeline_ok) ||
        (want_kv_reuse && !kv_reuse_pipeline_ok) ||
        !decode_layer_buffers_valid(ctx, args, true, want_kv_reuse) ||
        (want_kv_reuse && !myds_v41_gpu_attn_kv_reuse_prepare_layer(args->gpu_scratch, il, count, is_idx))) {
        if (want_attn_hc) atomic_fetch_add_explicit(&attn_hc_overlap_failures, 1, memory_order_relaxed);
        if (want_idx_overlap) atomic_fetch_add_explicit(&indexer_decode_overlap_failures, 1, memory_order_relaxed);
        if (want_kv_reuse) {
            myds_v41_gpu_attn_kv_reuse_reset(args->gpu_scratch);
            myds_v41_attn_kv_reuse_record_prestage_failure();
        }
        MYDS_LOG_E("Decode attention/indexer overlap preconditions failed; refusing fallback");
        return 0;
    }

    const int previous_auto = myds_metal_get_auto_barrier();
    const int previous_concurrent = myds_metal_get_concurrent();
    myds_metal_set_auto_barrier(0);
    myds_metal_set_concurrent(1);

#define STAGE_CHECK(call) do { if (!(call)) goto failed; } while (0)
    if (want_attn_hc) {
        STAGE_CHECK(myds_metal_v41_hc_mix_stages_preflight(args->attn_split, args->residual,
            args->attn_fn, args->attn_scale, args->attn_base, args->flat_norm, args->mix,
            5120u, m->norm_eps, args->hc_eps));
    }

    /* Cohort 1: fused HC sum+RMSNorm for x_norm, plus HC mix norm stage when enabled. */
    STAGE_CHECK(myds_metal_v41_hc_sum_rmsnorm(ctx->x_norm, ctx->x, args->residual,
        args->hc_in_pre, base->attn_norm, 5120u, m->norm_eps, true));
    if (want_attn_hc) {
        STAGE_CHECK(myds_metal_v41_hc_mix_norm_stage(args->flat_norm, args->residual, 5120u, m->norm_eps));
        const myds_metal_buffer_t *cohort1[] = { ctx->x_norm, args->flat_norm };
        myds_metal_barrier_buffers(cohort1, 2);
    } else {
        myds_metal_barrier_buffer(ctx->x_norm);
    }

    if (!want_idx_overlap) {
        /* Reference compressor/indexer schedule (with optional MYDS_ATTN_HC_OVERLAP and MYDS_ATTN_KV_REUSE). */
        if (want_attn_hc) {
            STAGE_CHECK(myds_metal_v41_hc_mix_project_stage(args->mix, args->attn_fn, args->flat_norm, 5120u));
        } else {
            STAGE_CHECK(myds_metal_v41_hc_mix(args->attn_split, args->residual, args->attn_fn, GGML_TYPE_F16,
                args->attn_scale, args->attn_base, args->flat_norm, args->mix, 5120u, m->norm_eps, args->hc_eps));
        }
        STAGE_CHECK(myds_metal_v41_gemv_bf16(ctx->qr, base->attn_q_a, ctx->x_norm, 1280u, 5120u, base->attn_q_a_type));
        STAGE_CHECK(myds_metal_v41_gemv_bf16(ctx->kv_act, base->attn_kv, ctx->x_norm, 512u, 5120u, base->attn_kv_type));
        STAGE_CHECK(ref_publish_compressor_decode(ctx, args, true));
        STAGE_CHECK(ref_publish_index_weights_decode(ctx, args, true));
        const myds_metal_buffer_t *cohort2[] = { ctx->qr, ctx->kv_act, want_attn_hc ? args->mix : NULL };
        myds_metal_barrier_buffers(cohort2, 3);

        const bool defer_hc_split = want_attn_hc && want_kv_reuse && !is_idx;
        if (want_attn_hc && !defer_hc_split) {
            STAGE_CHECK(myds_metal_v41_hc_mix_split_stage(args->attn_split, args->mix,
                args->attn_scale, args->attn_base, 5120u, args->hc_eps));
            atomic_fetch_add_explicit(&attn_hc_overlap_encoded, 1, memory_order_relaxed);
        }
        STAGE_CHECK(myds_metal_v41_rmsnorm(ctx->qr_norm, ctx->qr, base->attn_q_a_norm, 1280u, m->norm_eps, true));
        STAGE_CHECK(myds_metal_v41_rmsnorm(ctx->kv_act, ctx->kv_act, base->attn_kv_a_norm, 512u, m->norm_eps, true));
        const myds_metal_buffer_t *cohort3[] = { ctx->qr_norm, ctx->kv_act };
        myds_metal_barrier_buffers(cohort3, 2);

        if (want_kv_reuse && !is_idx) {
            myds_metal_buffer_t *partial_scratch = myds_v41_gpu_scratch_partial(args->gpu_scratch);
            const myds_metal_buffer_t *kv_f16_scratch = myds_v41_gpu_scratch_kv_f16(args->gpu_scratch);
            const myds_metal_buffer_t *pad_scratch = myds_v41_gpu_scratch_pad(args->gpu_scratch);
            if (defer_hc_split) {
                STAGE_CHECK(myds_metal_v41_hc_mix_split_stage(args->attn_split, args->mix,
                    args->attn_scale, args->attn_base, 5120u, args->hc_eps));
                atomic_fetch_add_explicit(&attn_hc_overlap_encoded, 1, memory_order_relaxed);
            }
            STAGE_CHECK(myds_v41_gpu_qkv_finish(partial_scratch, ctx->kv_act, args->window, pos, compressed));
            STAGE_CHECK(myds_metal_dense_q4_v41(ctx->q, base->attn_q_b, ctx->qr_norm, 64u * 512u, 1280u, true));
            const myds_metal_buffer_t *cohort4[] = {
                ctx->q, args->window, partial_scratch, kv_f16_scratch,
                defer_hc_split ? args->attn_split : NULL
            };
            myds_metal_barrier_buffers(cohort4, 5);
            STAGE_CHECK(myds_v41_gpu_rope(ctx->q, 512u, 64u, pos, compressed, false));
            STAGE_CHECK(myds_v41_gpu_attn_kv_reuse_prestage_window(args->gpu_scratch, args->window, pos));
            const myds_metal_buffer_t *attn_in_deps[] = {
                ctx->q, kv_f16_scratch, pad_scratch
            };
            myds_metal_barrier_buffers(attn_in_deps, 3);
            myds_v41_gpu_attn_kv_reuse_mark_fenced(args->gpu_scratch);
            myds_metal_set_auto_barrier(previous_auto);
            myds_metal_set_concurrent(previous_concurrent);
            return 1;
        }
        STAGE_CHECK(myds_metal_dense_q4_v41(ctx->q, base->attn_q_b, ctx->qr_norm, 64u * 512u, 1280u, true));
        STAGE_CHECK(ref_publish_indexer_one(ctx, args, true));
        myds_metal_barrier_buffer(ctx->q);
        STAGE_CHECK(myds_v41_gpu_qkv_finish(ctx->q, ctx->kv_act, args->window, pos, compressed));
        const bool fence_comp = count > 0u && (!want_kv_reuse || is_idx);
        const myds_metal_buffer_t *attn_in_deps[] = {
            ctx->q, args->window,
            fence_comp ? args->selected : NULL,
            fence_comp ? args->compressed_kv : NULL,
            want_kv_reuse ? myds_v41_gpu_scratch_kv_f16(args->gpu_scratch) : NULL,
            want_kv_reuse ? myds_v41_gpu_scratch_pad(args->gpu_scratch) : NULL
        };
        myds_metal_barrier_buffers(attn_in_deps, want_kv_reuse ? 6u : 4u);
        if (want_kv_reuse) myds_v41_gpu_attn_kv_reuse_mark_fenced(args->gpu_scratch);
        myds_metal_set_auto_barrier(previous_auto);
        myds_metal_set_concurrent(previous_concurrent);
        return 1;
    }

    /* Overlapped compressor + indexer + QKV cohort schedule on index_source layers. */
    const bool direct_latent_r1 = want_kv_reuse && is_kv && ratio == 1u;
    const bool direct_prev_r2 = want_kv_reuse && is_kv && ratio == 2u && !kv_commit;
    const bool defer_idx_qb = want_kv_reuse && count > 0u;

    /* Cohort 2: all x_norm (and flat_norm) consumers dispatched without intermediate barriers. */
    if (want_attn_hc) {
        STAGE_CHECK(myds_metal_v41_hc_mix_project_stage(args->mix, args->attn_fn, args->flat_norm, 5120u));
    } else {
        STAGE_CHECK(myds_metal_v41_hc_mix(args->attn_split, args->residual, args->attn_fn, GGML_TYPE_F16,
            args->attn_scale, args->attn_base, args->flat_norm, args->mix, 5120u, m->norm_eps, args->hc_eps));
    }
    if (defer_idx_qb) {
        STAGE_CHECK(myds_metal_v41_gemv_f16_bf16(args->index_weights, args->index_proj_weight, ctx->x_norm, 32u, 5120u));
    }
    STAGE_CHECK(myds_metal_v41_gemv_bf16(ctx->qr, base->attn_q_a, ctx->x_norm, 1280u, 5120u, base->attn_q_a_type));
    STAGE_CHECK(myds_metal_v41_gemv_bf16(ctx->kv_act, base->attn_kv, ctx->x_norm, 512u, 5120u, base->attn_kv_type));
    if (is_kv) {
        if (ratio == 1u) {
            myds_metal_buffer_t *kv_out = direct_latent_r1 ? args->latent : args->pool_kv;
            STAGE_CHECK(myds_metal_v41_gemv_f16_bf16(kv_out, args->compressor_kv, ctx->x_norm, 512u, 5120u));
        } else {
            myds_metal_buffer_t *kv_out = direct_prev_r2 ? args->previous_kv : args->pool_kv;
            myds_metal_buffer_t *score_out = direct_prev_r2 ? args->previous_score : args->pool_score;
            STAGE_CHECK(myds_metal_v41_gemv(kv_out, args->compressor_kv, ctx->x_norm, 512u, 5120u, GGML_TYPE_F16));
            STAGE_CHECK(myds_metal_v41_gemv(score_out, args->compressor_gate, ctx->x_norm, 512u, 5120u, GGML_TYPE_F16));
        }
    }
    if (count > 0u && !defer_idx_qb) {
        STAGE_CHECK(myds_metal_v41_gemv_f16_bf16(args->index_weights, args->index_proj_weight, ctx->x_norm, 32u, 5120u));
    }
    const myds_metal_buffer_t *cohort2[] = {
        ctx->qr, ctx->kv_act,
        want_attn_hc ? args->mix : NULL,
        is_kv ? (direct_latent_r1 ? args->latent : (direct_prev_r2 ? args->previous_kv : args->pool_kv)) : NULL,
        (is_kv && ratio == 2u) ? (direct_prev_r2 ? args->previous_score : args->pool_score) : NULL,
        count ? args->index_weights : NULL
    };
    myds_metal_barrier_buffers(cohort2, 6);

    /* Cohort 3: qr_norm + kv_act_norm + pool2/copy/r1-norm (plus HC split stage when !defer_idx_qb). */
    if (want_attn_hc && !defer_idx_qb) {
        STAGE_CHECK(myds_metal_v41_hc_mix_split_stage(args->attn_split, args->mix,
            args->attn_scale, args->attn_base, 5120u, args->hc_eps));
        atomic_fetch_add_explicit(&attn_hc_overlap_encoded, 1, memory_order_relaxed);
    }
    STAGE_CHECK(myds_metal_v41_rmsnorm(ctx->qr_norm, ctx->qr, base->attn_q_a_norm, 1280u, m->norm_eps, true));
    STAGE_CHECK(myds_metal_v41_rmsnorm(ctx->kv_act, ctx->kv_act, base->attn_kv_a_norm, 512u, m->norm_eps, true));
    if (is_kv) {
        if (ratio == 2u) {
            if (!direct_prev_r2) {
                STAGE_CHECK(myds_v41_gpu_pool2(args->latent, args->pool_kv, args->pool_score,
                    args->previous_kv, args->previous_score, 512u, pos));
            }
        } else if (direct_latent_r1) {
            STAGE_CHECK(myds_metal_v41_rmsnorm(args->latent, args->latent, args->compressor_norm, 512u, m->norm_eps, true));
        } else {
            STAGE_CHECK(myds_metal_buffer_copy(args->latent, 0, args->pool_kv, 0, 512u * 4u));
        }
    }
    const myds_metal_buffer_t *cohort3[] = {
        ctx->qr_norm, ctx->kv_act, kv_commit ? args->latent : NULL
    };
    myds_metal_barrier_buffers(cohort3, 3);

    /* Cohort 4: index_q GEMV + compressor_norm/r1-index_k + window update (plus Q_B GEMV when !defer_idx_qb). */
    if (!defer_idx_qb) {
        STAGE_CHECK(myds_metal_dense_q4_v41(ctx->q, base->attn_q_b, ctx->qr_norm, 64u * 512u, 1280u, true));
    }
    if (defer_idx_qb && kv_commit) {
        if (direct_latent_r1) {
            STAGE_CHECK(myds_metal_v41_gemv_f16_bf16(args->index_k_buf, args->index_k_weight, args->latent, 128u, 512u));
        } else {
            STAGE_CHECK(myds_metal_v41_rmsnorm(args->latent, args->latent, args->compressor_norm, 512u, m->norm_eps, true));
        }
    }
    if (want_kv_reuse) {
        myds_metal_buffer_t *partial_scratch = myds_v41_gpu_scratch_partial(args->gpu_scratch);
        STAGE_CHECK(myds_v41_gpu_qkv_finish(partial_scratch, ctx->kv_act, args->window, pos, compressed));
    }
    if (count > 0u) {
        STAGE_CHECK(myds_metal_v41_gemv_f16_bf16(args->index_q_buf, args->index_q_weight, ctx->qr_norm, 32u * 128u, 1280u));
    }
    if (!defer_idx_qb && kv_commit) {
        if (direct_latent_r1) {
            STAGE_CHECK(myds_metal_v41_gemv_f16_bf16(args->index_k_buf, args->index_k_weight, args->latent, 128u, 512u));
        } else {
            STAGE_CHECK(myds_metal_v41_rmsnorm(args->latent, args->latent, args->compressor_norm, 512u, m->norm_eps, true));
        }
    }
    if (want_kv_reuse) {
        const myds_metal_buffer_t *cohort4[] = {
            defer_idx_qb ? NULL : ctx->q, args->window,
            myds_v41_gpu_scratch_partial(args->gpu_scratch),
            myds_v41_gpu_scratch_kv_f16(args->gpu_scratch),
            count ? args->index_q_buf : NULL,
            kv_commit ? (direct_latent_r1 ? args->index_k_buf : args->latent) : NULL
        };
        myds_metal_barrier_buffers(cohort4, 6);
    } else {
        const myds_metal_buffer_t *cohort4[] = {
            ctx->q, count ? args->index_q_buf : NULL, kv_commit ? args->latent : NULL
        };
        myds_metal_barrier_buffers(cohort4, 3);
    }

    /* Cohort 5: prestage_window + indexer_rope_quant(index_q) + index_k/latent commit (plus q rope when !defer_idx_qb). */
    if (want_kv_reuse) {
        if (!defer_idx_qb) {
            STAGE_CHECK(myds_v41_gpu_rope(ctx->q, 512u, 64u, pos, compressed, false));
        }
        STAGE_CHECK(myds_v41_gpu_attn_kv_reuse_prestage_window(args->gpu_scratch, args->window, pos));
    } else {
        STAGE_CHECK(myds_v41_gpu_qkv_finish(ctx->q, ctx->kv_act, args->window, pos, compressed));
    }
    if (count > 0u) {
        STAGE_CHECK(myds_v41_gpu_indexer_rope_quant(args->index_q_buf, 32u, pos));
    }
    if (kv_commit) {
        const uint32_t start = pos + 1u - ratio;
        const myds_metal_buffer_t *commit_deps[] = { args->index_k_buf, args->latent };
        if (!direct_latent_r1) {
            STAGE_CHECK(myds_metal_v41_gemv_f16_bf16(args->index_k_buf, args->index_k_weight, args->latent, 128u, 512u));
            if (defer_idx_qb) {
                STAGE_CHECK(myds_metal_dense_q4_v41_qb_slice(ctx->q, base->attn_q_b, ctx->qr_norm,
                    0u, 4u * 512u, true));
            }
            myds_metal_barrier_buffers(commit_deps, 2);
        }
        STAGE_CHECK(myds_metal_v41_rmsnorm(args->index_k_buf, args->index_k_buf, args->index_k_norm, 128u, m->norm_eps, true));
        STAGE_CHECK(myds_v41_gpu_rope(args->latent, 512u, 1u, start, true, false));
        if (defer_idx_qb && direct_latent_r1) {
            STAGE_CHECK(myds_metal_dense_q4_v41_qb_slice(ctx->q, base->attn_q_b, ctx->qr_norm,
                0u, 4u * 512u, true));
        }
        myds_metal_barrier_buffers(commit_deps, 2);
        STAGE_CHECK(myds_v41_gpu_indexer_rope_quant(args->index_k_buf, 1u, start));
        STAGE_CHECK(myds_v41_gpu_quant(args->latent, 512u, 1u, 3u));
        if (defer_idx_qb) {
            STAGE_CHECK(myds_metal_dense_q4_v41_qb_slice(ctx->q, base->attn_q_b, ctx->qr_norm,
                4u * 512u, 6u * 512u, false));
        }
        myds_metal_barrier_buffers(commit_deps, 2);
        STAGE_CHECK(myds_v41_gpu_pack_bf16(args->index_cache, (size_t)(count - 1u) * 128u, args->index_k_buf, 0, 128u));
        STAGE_CHECK(myds_v41_gpu_pack_bf16(args->compressed_kv, (size_t)(count - 1u) * 512u, args->latent, 0, 512u));
        if (defer_idx_qb) {
            STAGE_CHECK(myds_metal_dense_q4_v41_qb_slice(ctx->q, base->attn_q_b, ctx->qr_norm,
                10u * 512u, 6u * 512u, false));
        }
    } else if (defer_idx_qb) {
        STAGE_CHECK(myds_metal_dense_q4_v41_qb_slice(ctx->q, base->attn_q_b, ctx->qr_norm,
            0u, 4u * 512u, true));
    }

    if (count > 0u) {
        const uint32_t qb_c5_heads = kv_commit ? 16u : 4u;
        const uint32_t qb_c6_heads = 16u;
        const uint32_t blocks = (count + 7u) / 8u;
        const bool need_block_mask = !want_kv_reuse || blocks > 2048u;
        const myds_metal_buffer_t *b_mask = (il > 20u && need_block_mask) ? args->block_mask : NULL;
        const myds_metal_buffer_t *score_deps[] = {
            args->index_q_buf, args->index_weights, args->index_cache,
            kv_commit ? args->compressed_kv : NULL, b_mask,
            defer_idx_qb ? NULL : ctx->q, args->window,
            want_kv_reuse ? myds_v41_gpu_scratch_kv_f16(args->gpu_scratch) : NULL,
            want_kv_reuse ? myds_v41_gpu_scratch_pad(args->gpu_scratch) : NULL
        };
        myds_metal_barrier_buffers(score_deps, want_kv_reuse ? 9u : 7u);
        if (want_kv_reuse) myds_v41_gpu_attn_kv_reuse_mark_fenced(args->gpu_scratch);
        STAGE_CHECK(myds_v41_gpu_indexscores_head_slots(args->index_scores, args->index_q_buf,
            args->index_weights, args->index_cache, b_mask, count, 32u, true));
        if (defer_idx_qb) {
            STAGE_CHECK(myds_metal_dense_q4_v41_qb_slice(ctx->q, base->attn_q_b, ctx->qr_norm,
                qb_c5_heads * 512u, qb_c6_heads * 512u, false));
        }
        myds_metal_barrier_buffer(args->index_scores);
        if (il == 20u && need_block_mask) {
            const uint32_t top_blocks = blocks < 2048u ? blocks : 2048u;
            STAGE_CHECK(myds_v41_gpu_candidates(args->block_scores, args->index_scores, count, pos, ratio));
            myds_metal_barrier_buffer(args->block_scores);
            STAGE_CHECK(myds_v41_gpu_topk(args->block_selected, args->block_scores, blocks, top_blocks, args->gpu_scratch));
            STAGE_CHECK(myds_v41_gpu_topkmask(args->block_mask, args->block_selected, blocks, top_blocks));
        }
        if (defer_idx_qb) {
            if (want_attn_hc) {
                STAGE_CHECK(myds_metal_v41_hc_mix_split_stage(args->attn_split, args->mix,
                    args->attn_scale, args->attn_base, 5120u, args->hc_eps));
                atomic_fetch_add_explicit(&attn_hc_overlap_encoded, 1, memory_order_relaxed);
            }
            myds_v41_gpu_attn_kv_reuse_defer_q_rope(args->gpu_scratch, ctx->q, compressed);
        }
        const uint32_t top_k = count < 512u ? count : 512u;
        STAGE_CHECK(myds_v41_gpu_topk(args->selected, args->index_scores, count, top_k, args->gpu_scratch));
        if (defer_idx_qb) {
            const uint32_t qb_c7_offset = (qb_c5_heads + qb_c6_heads) * 512u;
            STAGE_CHECK(myds_metal_dense_q4_v41_qb_slice(ctx->q, base->attn_q_b, ctx->qr_norm,
                qb_c7_offset, 64u * 512u - qb_c7_offset, false));
            const myds_metal_buffer_t *topk_q_deps[] = {
                args->selected, ctx->q, want_attn_hc ? args->attn_split : NULL
            };
            myds_metal_barrier_buffers(topk_q_deps, 3);
        }
    } else {
        const myds_metal_buffer_t *attn_in_deps[] = {
            ctx->q, args->window,
            want_kv_reuse ? myds_v41_gpu_scratch_kv_f16(args->gpu_scratch) : NULL,
            want_kv_reuse ? myds_v41_gpu_scratch_pad(args->gpu_scratch) : NULL
        };
        myds_metal_barrier_buffers(attn_in_deps, want_kv_reuse ? 4u : 2u);
        if (want_kv_reuse) myds_v41_gpu_attn_kv_reuse_mark_fenced(args->gpu_scratch);
    }

    atomic_fetch_add_explicit(&indexer_decode_overlap_encoded, 1, memory_order_relaxed);
    myds_metal_set_auto_barrier(previous_auto);
    myds_metal_set_concurrent(previous_concurrent);
#undef STAGE_CHECK
    return 1;

failed:
    if (want_attn_hc) atomic_fetch_add_explicit(&attn_hc_overlap_failures, 1, memory_order_relaxed);
    if (want_idx_overlap) atomic_fetch_add_explicit(&indexer_decode_overlap_failures, 1, memory_order_relaxed);
    if (want_kv_reuse) {
        myds_v41_gpu_attn_kv_reuse_reset(args->gpu_scratch);
        myds_v41_attn_kv_reuse_record_prestage_failure();
    }
    myds_metal_set_auto_barrier(previous_auto);
    myds_metal_set_concurrent(previous_concurrent);
    MYDS_LOG_E("Decode attention/indexer overlap stage failed; refusing fallback");
    return 0;
}




