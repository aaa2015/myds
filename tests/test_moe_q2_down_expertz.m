/* Standalone Component Unit Test and Benchmark for D3 MoE Down Expert-in-grid.z.
 * Compares:
 *   - Baseline: kernel_mul_mv_id_q2_K_sum6_v41_f32 (single dispatch, 82K threads, in-kernel 6-expert loop)
 *   - Candidate A (0-ULP): kernel_mul_mv_id_q2_K_sum6_expertz_lane_f32 + kernel_mul_mv_id_q2_K_sum6_merge_lane_f32
 *                          (Pass 1: 491K threads, grid=(1280,1,6), writes 3.93 MB lane partials;
 *                           Pass 2: 82K threads, grid=(1280,1,1), accumulates 6 slots per lane before simd_sum)
 *   - Candidate B (Compact): kernel_mul_mv_id_q2_K_sum6_expertz_direct_f32 + kernel_mul_mv_id_q2_K_sum6_merge_expert_f32
 *                          (Pass 1: 491K threads, grid=(1280,1,6), writes 123 KB expert sums;
 *                           Pass 2: 5.1K threads, grid=(160,1,1), sums 6 floats per row)
 *
 * Verifies exact 0-ULP bit-exact equivalence and measures hardware execution latency & bandwidth.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(v) do { if (!(v)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #v); exit(1); } } while (0)

static const char *SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

#define QK_K 256

struct block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

struct ds4_metal_args_mul_mv_id {
    int32_t nei0, nei1; uint64_t nbi1; int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13; uint64_t nb10, nb11, nb12; int32_t ne0, ne1; uint64_t nb1;
    int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base;
};

// =========================================================================
// 1. Production Reference Baseline
// =========================================================================
kernel void kernel_mul_mv_id_q2_K_sum6_v41_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * add_in,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 2;
    const int nb = 9;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * 24u);
    device const char *token_src1 = src1 + (uint64_t)token * 55296u;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q2_K * x = (device const block_q2_K *)(src0s + (uint)expert*3870720u + (uint)first_row*756u);
        device const float * y = (device const float *)(token_src1 + expert_slot*9216u);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                    acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                    acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                    acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                    acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                    acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                    acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                    acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                }
                float dall = dh[0];
                float dmin = dh[1] * (1.f/16.f);
                sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                     (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                     (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                     (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                             dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                     sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                qs += 756u/2;
                sc += 756u;
                dh += 756u/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * 20480u);
    for (int row = 0; row < nr0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            float outv = sum_all;
            if (args.tp_addend) {
                outv += ((device const float *) add_in)[first_row + row];
            }
            dst_f32[first_row + row] = outv;
        }
    }
}

// =========================================================================
// 2. Candidate A: Expert in grid.z with Lane-Level Partials (Strict 0-ULP)
// =========================================================================
kernel void kernel_mul_mv_id_q2_K_sum6_expertz_lane_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       float * parts_lane, // [slot][row][lane] -> 6 * 5120 * 32 floats = 3.93 MB
        device const char * ids,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 2;
    const int nb = 9;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    const int expert_slot = tgpig.z; // 0..5
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * 24u);
    device const char *token_src1 = src1 + (uint64_t)token * 55296u;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    const int32_t expert = token_ids[expert_slot];
    device const block_q2_K * x = (device const block_q2_K *)(src0s + (uint)expert*3870720u + (uint)first_row*756u);
    device const float * y = (device const float *)(token_src1 + expert_slot*9216u);
    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    float sumf[nr0] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[32];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * (1.f/16.f);
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += 756u/2;
            sc += 756u;
            dh += 756u/2;
        }

        y4 += 4 * QK_K;
    }

    // Write lane partial sum
    for (short row = 0; row < nr0; row++) {
        const int r = first_row + row;
        if (r < 5120) {
            const uint idx = ((uint)expert_slot * 5120u + (uint)r) * 32u + (uint)tiisg;
            parts_lane[idx] = sumf[row];
        }
    }
}

kernel void kernel_mul_mv_id_q2_K_sum6_merge_lane_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const float * parts_lane,
        device       char * dst,
        device const char * add_in,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 2;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device float * dst_f32 = (device float *)(dst + (uint64_t)token * 20480u);

    for (short row = 0; row < nr0; row++) {
        const int r = first_row + row;
        if (r < 5120) {
            float sum_lane = 0.f;
            #pragma unroll
            for (int s = 0; s < 6; s++) {
                const uint idx = ((uint)s * 5120u + (uint)r) * 32u + (uint)tiisg;
                sum_lane += parts_lane[idx];
            }
            const float sum_all = simd_sum(sum_lane);
            if (tiisg == 0) {
                float outv = sum_all;
                if (args.tp_addend) {
                    outv += ((device const float *) add_in)[r];
                }
                dst_f32[r] = outv;
            }
        }
    }
}

// =========================================================================
// 3. Candidate B: Expert in grid.z with Direct Threadgroup Reduction (123 KB)
// =========================================================================
kernel void kernel_mul_mv_id_q2_K_sum6_expertz_direct_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       float * parts_expert, // [slot][row] -> 6 * 5120 floats = 122.88 KB
        device const char * ids,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = 2;
    const int nb = 9;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    const int expert_slot = tgpig.z;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * 24u);
    device const char *token_src1 = src1 + (uint64_t)token * 55296u;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    const int32_t expert = token_ids[expert_slot];
    device const block_q2_K * x = (device const block_q2_K *)(src0s + (uint)expert*3870720u + (uint)first_row*756u);
    device const float * y = (device const float *)(token_src1 + expert_slot*9216u);
    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    float sumf[nr0] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[32];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * (1.f/16.f);
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += 756u/2;
            sc += 756u;
            dh += 756u/2;
        }

        y4 += 4 * QK_K;
    }

    // Direct reduction inside threadgroup
    for (short row = 0; row < nr0; row++) {
        const int r = first_row + row;
        if (r < 5120) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                parts_expert[(uint)expert_slot * 5120u + (uint)r] = sum_all;
            }
        }
    }
}

kernel void kernel_mul_mv_id_q2_K_sum6_merge_expert_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const float * parts_expert,
        device       char * dst,
        device const char * add_in,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    const uint r = tgpig.x * 32u + (uint)tiitg;
    if (r < 5120u) {
        device float * dst_f32 = (device float *)(dst + (uint64_t)tgpig.y * 20480u);
        float sum = 0.f;
        #pragma unroll
        for (int s = 0; s < 6; s++) {
            sum += parts_expert[(uint)s * 5120u + r];
        }
        if (args.tp_addend) {
            sum += ((device const float *) add_in)[r];
        }
        dst_f32[r] = sum;
    }
}
)";

static uint64_t rng_state = 123456789012345ULL;
static uint64_t xorshift64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static void fill_q2_data(void *ptr, size_t bytes) {
    uint8_t *p = (uint8_t *)ptr;
    REQUIRE(bytes % 84 == 0);
    size_t blocks = bytes / 84;
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t *q = p + b * 84;
        for (size_t j = 0; j < 80; j += 8) {
            uint64_t v = xorshift64();
            memcpy(q + j, &v, 8);
        }
        uint16_t d = 0x2000u + (xorshift64() % 0x0800u);
        uint16_t m = 0x1800u + (xorshift64() % 0x0800u);
        memcpy(q + 80, &d, 2);
        memcpy(q + 82, &m, 2);
    }
}

static void fill_float_data(float *p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        p[i] = ((float)(xorshift64() % 2000001) - 1000000.f) / 1000000.f;
    }
}

static uint32_t float_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static uint32_t ulp_diff(float a, float b) {
    if (isnan(a) || isnan(b)) return 0xFFFFFFFFu;
    if (a == b) return 0;
    uint32_t ua = float_to_bits(a);
    uint32_t ub = float_to_bits(b);
    int32_t ia = (int32_t)ua;
    int32_t ib = (int32_t)ub;
    if (ia < 0) ia = 0x80000000 - ia;
    if (ib < 0) ib = 0x80000000 - ib;
    return (uint32_t)abs(ia - ib);
}

struct ds4_metal_args_mul_mv_id {
    int32_t nei0, nei1; uint64_t nbi1; int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13; uint64_t nb10, nb11, nb12; int32_t ne0, ne1; uint64_t nb1;
    int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base;
};

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    @autoreleasepool {
        printf("=== D3 MoE Down Expert-in-grid.z Unit Test & Microbenchmark ===\n");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        REQUIRE(device);
        printf("GPU: %s\n", [device.name UTF8String]);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        REQUIRE(queue);

        // Compile Metal shaders
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.fastMathEnabled = YES;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:SHADER_SOURCE]
                                                      options:opts error:&err];
        if (!library) {
            fprintf(stderr, "Shader compile error: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> pso_base = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_v41_f32"] error:&err];
        REQUIRE(pso_base);
        id<MTLComputePipelineState> pso_candA_pass1 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_expertz_lane_f32"] error:&err];
        REQUIRE(pso_candA_pass1);
        id<MTLComputePipelineState> pso_candA_pass2 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_merge_lane_f32"] error:&err];
        REQUIRE(pso_candA_pass2);
        id<MTLComputePipelineState> pso_candB_pass1 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_expertz_direct_f32"] error:&err];
        REQUIRE(pso_candB_pass1);
        id<MTLComputePipelineState> pso_candB_pass2 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_merge_expert_f32"] error:&err];
        REQUIRE(pso_candB_pass2);

        // Constants
        const uint32_t DIM = 5120;
        const uint32_t WIDTH = 2304;
        const uint32_t ACTIVE = 6;
        const uint32_t EXPERTS = 384;
        const size_t expert_bytes = 3870720;
        const size_t all_weights_bytes = (size_t)EXPERTS * expert_bytes; // 384 * 3.87 MB = 1.48 GB
        const size_t mid_bytes = (size_t)ACTIVE * WIDTH * sizeof(float); // 6 * 2304 * 4 = 55.296 KB
        const size_t dst_bytes = (size_t)DIM * sizeof(float); // 5120 * 4 = 20.48 KB
        const size_t parts_lane_bytes = (size_t)ACTIVE * DIM * 32 * sizeof(float); // 3.93 MB
        const size_t parts_expert_bytes = (size_t)ACTIVE * DIM * sizeof(float); // 122.88 KB

        printf("Allocating test buffers: weights (active 6 = %.2f MB), mid (%.1f KB), parts_lane (%.2f MB), parts_expert (%.1f KB)...\n",
               (double)(ACTIVE * expert_bytes) / 1e6, (double)mid_bytes / 1e3,
               (double)parts_lane_bytes / 1e6, (double)parts_expert_bytes / 1e3);

        // Allocate buffers
        // We allocate 16 expert weight blocks to fit test IDs
        id<MTLBuffer> buf_weights = [device newBufferWithLength:16 * expert_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mid = [device newBufferWithLength:mid_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ids = [device newBufferWithLength:ACTIVE * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_dst_base = [device newBufferWithLength:dst_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_dst_candA = [device newBufferWithLength:dst_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_dst_candB = [device newBufferWithLength:dst_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_parts_lane = [device newBufferWithLength:parts_lane_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_parts_expert = [device newBufferWithLength:parts_expert_bytes options:MTLResourceStorageModeShared];

        fill_q2_data(buf_weights.contents, 16 * expert_bytes);
        fill_float_data((float *)buf_mid.contents, ACTIVE * WIDTH);

        int32_t *ids = (int32_t *)buf_ids.contents;
        ids[0] = 0; ids[1] = 2; ids[2] = 5; ids[3] = 7; ids[4] = 9; ids[5] = 11;

        struct ds4_metal_args_mul_mv_id args = {
            .nei0 = 6, .nei1 = 1, .nbi1 = 24, .ne00 = (int32_t)WIDTH, .ne01 = (int32_t)DIM, .ne02 = 384,
            .nb00 = 84, .nb01 = 756, .nb02 = (uint64_t)expert_bytes,
            .ne10 = (int32_t)WIDTH, .ne11 = 6, .ne12 = 1, .ne13 = 1,
            .nb10 = 4, .nb11 = 9216, .nb12 = 55296,
            .ne0 = (int32_t)DIM, .ne1 = 1, .nb1 = 20480,
            .nr0 = 2, .tp_rank = 0, .tp_world = 1, .tp_addend = 0, .tp_expert_base = 0
        };

        // --- 0-ULP Parity Testing across 10 random seeds ---
        printf("\n--- Running 0-ULP Parity Suite across 10 random seeds ---\n");
        int all_candA_0ulp = 1;
        uint32_t max_candB_ulp = 0;

        for (int seed = 1; seed <= 10; seed++) {
            rng_state = 10007ULL * seed + 42ULL;
            fill_q2_data(buf_weights.contents, 16 * expert_bytes);
            fill_float_data((float *)buf_mid.contents, ACTIVE * WIDTH);
            memset(buf_dst_base.contents, 0, dst_bytes);
            memset(buf_dst_candA.contents, 0, dst_bytes);
            memset(buf_dst_candB.contents, 0, dst_bytes);

            // 1. Run Baseline
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_base];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_dst_base offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc setBuffer:buf_dst_base offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // 2. Run Candidate A (Lane partials)
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                // Pass 1: expertz
                [enc setComputePipelineState:pso_candA_pass1];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_parts_lane offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 6) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Pass 2: merge
                [enc setComputePipelineState:pso_candA_pass2];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_parts_lane offset:0 atIndex:1];
                [enc setBuffer:buf_dst_candA offset:0 atIndex:2];
                [enc setBuffer:buf_dst_candA offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // 3. Run Candidate B (Expert sums)
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                // Pass 1: expertz direct
                [enc setComputePipelineState:pso_candB_pass1];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_parts_expert offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 6) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Pass 2: merge expert
                [enc setComputePipelineState:pso_candB_pass2];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_parts_expert offset:0 atIndex:1];
                [enc setBuffer:buf_dst_candB offset:0 atIndex:2];
                [enc setBuffer:buf_dst_candB offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(160, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // Check parity
            int matchA = (memcmp(buf_dst_base.contents, buf_dst_candA.contents, dst_bytes) == 0);
            if (!matchA) all_candA_0ulp = 0;

            const float *fb = (const float *)buf_dst_base.contents;
            const float *fa = (const float *)buf_dst_candA.contents;
            const float *fc = (const float *)buf_dst_candB.contents;

            uint32_t seed_max_b_ulp = 0;
            for (uint32_t r = 0; r < DIM; ++r) {
                uint32_t u = ulp_diff(fb[r], fc[r]);
                if (u > seed_max_b_ulp) seed_max_b_ulp = u;
            }
            if (seed_max_b_ulp > max_candB_ulp) max_candB_ulp = seed_max_b_ulp;

            printf("  Seed %2d: Cand A (0-ULP): %s | Cand B (compact): max_ulp=%u\n",
                   seed, matchA ? "PASS 0-ULP (memcmp == 0)" : "FAIL", seed_max_b_ulp);
            if (!matchA) {
                for (uint32_t r = 0; r < DIM; ++r) {
                    if (fb[r] != fa[r]) {
                        printf("    First mismatch at row %u: base=%f (0x%08x) vs candA=%f (0x%08x) diff_ulp=%u\n",
                               r, fb[r], float_to_bits(fb[r]), fa[r], float_to_bits(fa[r]), ulp_diff(fb[r], fa[r]));
                        break;
                    }
                }
            }
        }

        if (all_candA_0ulp) {
            printf(">>> CANDIDATE A (Lane partials): 100%% 0-ULP BIT-EXACT PASS! <<<\n");
        } else {
            printf(">>> CANDIDATE A: FAILED 0-ULP! <<<\n");
        }
        printf(">>> CANDIDATE B (Compact): Max ULP across all seeds = %u ULPs <<<\n", max_candB_ulp);

        // --- Microbenchmarking ---
        printf("\n--- Microbenchmarking: 2000 runs per arm ---\n");
        const int ITERS = 2000;

        // Warmup
        for (int i = 0; i < 200; ++i) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_base];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_dst_base offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc setBuffer:buf_dst_base offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);

        // 1. Measure Baseline
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; ++i) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_base];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_dst_base offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc setBuffer:buf_dst_base offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        uint64_t dt_base = mach_absolute_time() - t0;
        double us_base = (double)dt_base * tb.numer / tb.denom / 1000.0 / ITERS;

        // 2. Measure Candidate A (Lane partials)
        t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; ++i) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_candA_pass1];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_parts_lane offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 6) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                [enc setComputePipelineState:pso_candA_pass2];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_parts_lane offset:0 atIndex:1];
                [enc setBuffer:buf_dst_candA offset:0 atIndex:2];
                [enc setBuffer:buf_dst_candA offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        uint64_t dt_candA = mach_absolute_time() - t0;
        double us_candA = (double)dt_candA * tb.numer / tb.denom / 1000.0 / ITERS;

        // 3. Measure Candidate B (Expert sums)
        t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; ++i) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_candB_pass1];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_parts_expert offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 6) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                [enc setComputePipelineState:pso_candB_pass2];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_parts_expert offset:0 atIndex:1];
                [enc setBuffer:buf_dst_candB offset:0 atIndex:2];
                [enc setBuffer:buf_dst_candB offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(160, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        uint64_t dt_candB = mach_absolute_time() - t0;
        double us_candB = (double)dt_candB * tb.numer / tb.denom / 1000.0 / ITERS;

        double weight_mb = (double)(ACTIVE * expert_bytes) / 1e6;
        double bw_base = weight_mb / (us_base / 1e6) / 1000.0;
        double bw_candA = weight_mb / (us_candA / 1e6) / 1000.0;
        double bw_candB = weight_mb / (us_candB / 1e6) / 1000.0;

        printf("\n=======================================================\n");
        printf("MICROBENCHMARK RESULTS (%d iterations synchronous):\n", ITERS);
        printf("  Baseline (1 pass, 82K thd)     : %8.3f us/call | Bandwidth: %6.1f GB/s\n", us_base, bw_base);
        printf("  Candidate A (0-ULP 3.93MB part): %8.3f us/call | Bandwidth: %6.1f GB/s (Delta: %+.1f%%, %+.2f us)\n",
               us_candA, bw_candA, (us_candA - us_base) / us_base * 100.0, us_candA - us_base);
        printf("  Candidate B (Compact 123KB sum): %8.3f us/call | Bandwidth: %6.1f GB/s (Delta: %+.1f%%, %+.2f us)\n",
               us_candB, bw_candB, (us_candB - us_base) / us_base * 100.0, us_candB - us_base);
        printf("-------------------------------------------------------\n");
        printf("  Single token delta (40 layers):\n");
        printf("    Candidate A: %+.3f ms/token\n", (us_candA - us_base) * 40.0 / 1000.0);
        printf("    Candidate B: %+.3f ms/token\n", (us_candB - us_base) * 40.0 / 1000.0);
        printf("=======================================================\n");

        // --- Pipelined Batch Benchmark (40 dispatches per command buffer) ---
        printf("\n--- Pipelined Batch Benchmark: 100 runs of 40 dispatches per command buffer ---\n");
        const int BATCH_RUNS = 100;
        const int DISPATCHES_PER_CB = 40;

        // Baseline pipelined
        t0 = mach_absolute_time();
        for (int r = 0; r < BATCH_RUNS; ++r) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_base];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:buf_weights offset:0 atIndex:1];
                [enc setBuffer:buf_mid offset:0 atIndex:2];
                [enc setBuffer:buf_dst_base offset:0 atIndex:3];
                [enc setBuffer:buf_ids offset:0 atIndex:4];
                [enc setBuffer:buf_dst_base offset:0 atIndex:5];
                for (int d = 0; d < DISPATCHES_PER_CB; ++d) {
                    [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        uint64_t dt_pipe_base = mach_absolute_time() - t0;
        double us_pipe_base = (double)dt_pipe_base * tb.numer / tb.denom / 1000.0 / (BATCH_RUNS * DISPATCHES_PER_CB);

        // Candidate A pipelined
        t0 = mach_absolute_time();
        for (int r = 0; r < BATCH_RUNS; ++r) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int d = 0; d < DISPATCHES_PER_CB; ++d) {
                    [enc setComputePipelineState:pso_candA_pass1];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:buf_weights offset:0 atIndex:1];
                    [enc setBuffer:buf_mid offset:0 atIndex:2];
                    [enc setBuffer:buf_parts_lane offset:0 atIndex:3];
                    [enc setBuffer:buf_ids offset:0 atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 6) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:pso_candA_pass2];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:buf_parts_lane offset:0 atIndex:1];
                    [enc setBuffer:buf_dst_candA offset:0 atIndex:2];
                    [enc setBuffer:buf_dst_candA offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        uint64_t dt_pipe_candA = mach_absolute_time() - t0;
        double us_pipe_candA = (double)dt_pipe_candA * tb.numer / tb.denom / 1000.0 / (BATCH_RUNS * DISPATCHES_PER_CB);

        // Candidate B pipelined
        t0 = mach_absolute_time();
        for (int r = 0; r < BATCH_RUNS; ++r) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int d = 0; d < DISPATCHES_PER_CB; ++d) {
                    [enc setComputePipelineState:pso_candB_pass1];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:buf_weights offset:0 atIndex:1];
                    [enc setBuffer:buf_mid offset:0 atIndex:2];
                    [enc setBuffer:buf_parts_expert offset:0 atIndex:3];
                    [enc setBuffer:buf_ids offset:0 atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1280, 1, 6) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:pso_candB_pass2];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:buf_parts_expert offset:0 atIndex:1];
                    [enc setBuffer:buf_dst_candB offset:0 atIndex:2];
                    [enc setBuffer:buf_dst_candB offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(160, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        uint64_t dt_pipe_candB = mach_absolute_time() - t0;
        double us_pipe_candB = (double)dt_pipe_candB * tb.numer / tb.denom / 1000.0 / (BATCH_RUNS * DISPATCHES_PER_CB);

        double bw_pipe_base = weight_mb / (us_pipe_base / 1e6) / 1000.0;
        double bw_pipe_candA = weight_mb / (us_pipe_candA / 1e6) / 1000.0;
        double bw_pipe_candB = weight_mb / (us_pipe_candB / 1e6) / 1000.0;

        printf("\n=======================================================\n");
        printf("PIPELINED BATCH RESULTS (4000 total dispatches):\n");
        printf("  Baseline (1 pass, 82K thd)     : %8.3f us/call | Bandwidth: %6.1f GB/s\n", us_pipe_base, bw_pipe_base);
        printf("  Candidate A (0-ULP 3.93MB part): %8.3f us/call | Bandwidth: %6.1f GB/s (Delta: %+.1f%%, %+.2f us)\n",
               us_pipe_candA, bw_pipe_candA, (us_pipe_candA - us_pipe_base) / us_pipe_base * 100.0, us_pipe_candA - us_pipe_base);
        printf("  Candidate B (Compact 123KB sum): %8.3f us/call | Bandwidth: %6.1f GB/s (Delta: %+.1f%%, %+.2f us)\n",
               us_pipe_candB, bw_pipe_candB, (us_pipe_candB - us_pipe_base) / us_pipe_base * 100.0, us_pipe_candB - us_pipe_base);
        printf("-------------------------------------------------------\n");
        printf("  Single token delta (40 layers):\n");
        printf("    Candidate A: %+.3f ms/token\n", (us_pipe_candA - us_pipe_base) * 40.0 / 1000.0);
        printf("    Candidate B: %+.3f ms/token\n", (us_pipe_candB - us_pipe_base) * 40.0 / 1000.0);
        printf("=======================================================\n");
    }
    return 0;
}
