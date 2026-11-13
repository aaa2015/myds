/* Component verification and GPU microbenchmark for [P-A1] Reg-O prefill
 * matrix attention (`kernel_myds_flash_attn_heads_*_f16_dk512_dv512`).
 *
 * Includes the real Metal/GPU runtime to verify:
 *   1. Exact Apple Silicon `simdgroup_float8x8` per-lane 2-element row/col
 *      mapping and in-register row rescaling (`thread_elements()[0,1] *= ms`).
 *   2. Metal PSO geometry (`threadExecutionWidth`, `maxTotalThreadsPerThreadgroup`,
 *      `staticThreadgroupMemoryLength`, dynamic smem) and optional binary archive
 *      dump for offline `metal-binary-perf` register/spill inspection.
 *   3. Strict 0-ULP (`memcmp == 0`) parity across Original (`smem`, 32 KiB) vs
 *      Step 1 (`q8` Reg-O, 12,320 B) vs Step 2 (`q16` Reg-O, 24,640 B), covering
 *      tail `c_len` (1, 7, 15, 16, 31, 33, 63, 64, 128, 294, 512, 2048), 640
 *      gathered keys, partial causal masks, all-masked rows, sinks, and `q_rem`
 *      non-8/16 head boundaries (`ne01 = 5, 13, 25, 63`).
 *   4. Single-CB multi-iteration `MTLCommandBuffer` hardware GPU timestamp
 *      (`GPUEndTime - GPUStartTime`) microbenchmarks on local Apple Silicon.
 */
#include "myds_metal.m"
#include "myds_v41_gpu.c"
#include <inttypes.h>
#include <math.h>

#define REQUIRE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
        exit(1); \
    } \
} while (0)

struct guarded_buf {
    myds_metal_buffer_t *owner;
    myds_metal_buffer_t *view;
    size_t bytes;
};

static struct guarded_buf guarded_alloc(size_t bytes) {
    struct guarded_buf g = { .bytes = bytes };
    g.owner = myds_metal_buffer_alloc(bytes + 64u);
    REQUIRE(g.owner != NULL);
    memset(myds_metal_buffer_data(g.owner), 0xa5, bytes + 64u);
    g.view = myds_metal_buffer_sub_view(g.owner, 32u, bytes);
    REQUIRE(g.view != NULL);
    return g;
}

static void guarded_check_canaries(struct guarded_buf g) {
    const uint8_t *p = (const uint8_t *)myds_metal_buffer_data(g.owner);
    for (size_t i = 0; i < 32u; ++i) {
        REQUIRE(p[i] == 0xa5);
        REQUIRE(p[32u + g.bytes + i] == 0xa5);
    }
}

static void guarded_free(struct guarded_buf g) {
    guarded_check_canaries(g);
    myds_metal_buffer_free(g.view);
    myds_metal_buffer_free(g.owner);
}

static uint32_t ulp_diff_f32(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    if (ua == ub) return 0u;
    if ((ua & 0x7fffffffu) == 0u && (ub & 0x7fffffffu) == 0u) return 0u;
    int32_t ia = (ua & 0x80000000u) ? (int32_t)(0x80000000u - ua) : (int32_t)ua;
    int32_t ib = (ub & 0x80000000u) ? (int32_t)(0x80000000u - ub) : (int32_t)ub;
    int64_t d = (int64_t)ia - (int64_t)ib;
    return (uint32_t)(d < 0 ? -d : d);
}

static float bf16_sample(uint32_t i, uint32_t seed) {
    uint32_t x = (i + 1u) * 747796405u + seed * 2891336453u;
    x = ((x >> ((x >> 28) + 4u)) ^ x) * 277803737u;
    x = (x >> 22) ^ x;
    float f = ((int32_t)(x % 8191u) - 4095) / 8192.0f;
    uint32_t b;
    memcpy(&b, &f, 4);
    b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
    memcpy(&f, &b, 4);
    return f;
}

static uint16_t f32_to_f16_bits(float v) {
    __fp16 h = (__fp16)v;
    uint16_t bits;
    memcpy(&bits, &h, 2);
    return bits;
}

static uint32_t gcd_u32(uint32_t a, uint32_t b) {
    while (b) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

static uint64_t hist_count_for(const char *name) {
    for (int i = 0; i < g_hist_used; ++i) {
        if (strcmp(g_hist[i].name, name) == 0) return g_hist[i].n;
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * 1. Dedicated GPU verification of `simdgroup_float8x8` lane element mapping
 * -------------------------------------------------------------------------- */
static void test_simdgroup_float8x8_mapping(void) {
    NSString *msl =
        @"#include <metal_stdlib>\n"
        @"#include <metal_simdgroup_matrix>\n"
        @"using namespace metal;\n"
        @"struct LaneInfo { ushort row; ushort col0; float e0; float e1; };\n"
        @"kernel void verify_sg8x8(\n"
        @"    device const float * src [[buffer(0)]],\n"
        @"    device const float * ms  [[buffer(1)]],\n"
        @"    device LaneInfo    * info [[buffer(2)]],\n"
        @"    device float       * out_sg [[buffer(3)]],\n"
        @"    device float       * out_direct [[buffer(4)]],\n"
        @"    ushort tiisg [[thread_index_in_simdgroup]]) {\n"
        @"    threadgroup float s_ms[8];\n"
        @"    if (tiisg < 8) s_ms[tiisg] = ms[tiisg];\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    simdgroup_float8x8 m;\n"
        @"    simdgroup_load(m, src, 8, 0, false);\n"
        @"    const ushort lo_row = ((tiisg >> 2) & 4) | ((tiisg >> 1) & 3);\n"
        @"    const ushort lo_col = ((tiisg >> 1) & 4) | ((tiisg & 1) << 1);\n"
        @"    info[tiisg].row  = lo_row;\n"
        @"    info[tiisg].col0 = lo_col;\n"
        @"    info[tiisg].e0   = m.thread_elements()[0];\n"
        @"    info[tiisg].e1   = m.thread_elements()[1];\n"
        @"    const float scale = s_ms[lo_row];\n"
        @"    m.thread_elements()[0] *= scale;\n"
        @"    m.thread_elements()[1] *= scale;\n"
        @"    simdgroup_store(m, out_sg, 8, 0, false);\n"
        @"    device float2 * dst2 = (device float2 *)(out_direct + lo_row * 8);\n"
        @"    dst2[lo_col / 2] = float2(m.thread_elements()[0], m.thread_elements()[1]);\n"
        @"}\n";

    NSError *err = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:msl options:nil error:&err];
    if (!lib) fprintf(stderr, "MSL compile error: %s\n", err.localizedDescription.UTF8String);
    REQUIRE(lib != nil);
    id<MTLFunction> fn = [lib newFunctionWithName:@"verify_sg8x8"];
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);

    struct lane_info { uint16_t row, col0; float e0, e1; };
    id<MTLBuffer> b_src  = [g_device newBufferWithLength:64 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_ms   = [g_device newBufferWithLength:8  * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_info = [g_device newBufferWithLength:32 * sizeof(struct lane_info) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sg   = [g_device newBufferWithLength:64 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_dir  = [g_device newBufferWithLength:64 * sizeof(float) options:MTLResourceStorageModeShared];

    float *src = (float *)b_src.contents;
    float *ms  = (float *)b_ms.contents;
    for (int r = 0; r < 8; ++r) {
        ms[r] = (float)(r + 2) * 0.375f;
        for (int c = 0; c < 8; ++c) {
            src[r * 8 + c] = (float)(r + 1) * 100.0f + (float)(c + 1);
        }
    }

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:b_src offset:0 atIndex:0];
    [enc setBuffer:b_ms offset:0 atIndex:1];
    [enc setBuffer:b_info offset:0 atIndex:2];
    [enc setBuffer:b_sg offset:0 atIndex:3];
    [enc setBuffer:b_dir offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.status == MTLCommandBufferStatusCompleted);

    const struct lane_info *info = (const struct lane_info *)b_info.contents;
    const float *out_sg  = (const float *)b_sg.contents;
    const float *out_dir = (const float *)b_dir.contents;

    for (unsigned lane = 0; lane < 32; ++lane) {
        float exp_e0 = (float)(info[lane].row + 1) * 100.0f + (float)(info[lane].col0 + 1);
        float exp_e1 = (float)(info[lane].row + 1) * 100.0f + (float)(info[lane].col0 + 2);
        REQUIRE(info[lane].e0 == exp_e0);
        REQUIRE(info[lane].e1 == exp_e1);
        REQUIRE((info[lane].col0 & 1u) == 0u);
    }
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            float expected = src[r * 8 + c] * ms[r];
            REQUIRE(out_sg[r * 8 + c] == expected);
            REQUIRE(out_dir[r * 8 + c] == expected);
        }
    }
    REQUIRE(memcmp(out_sg, out_dir, 64 * sizeof(float)) == 0);
    printf("PASS simdgroup_float8x8_lane_mapping lanes=32 elements=64 row_scale_exact=true direct_store_exact=true\n");
}

/* --------------------------------------------------------------------------
 * 2. PSO Geometry & Binary Archive Inspection
 * -------------------------------------------------------------------------- */
static id<MTLComputePipelineState> build_heads_pso(const char *kernel_name, int32_t nsg, bool has_kvpad, bool has_sinks, bool bc_mask) {
    MTLFunctionConstantValues *vals = [MTLFunctionConstantValues new];
    bool b_mask = true, b_sinks = has_sinks, b_bias = false, b_scap = false, b_kvpad = has_kvpad, b_bc = bc_mask;
    int32_t ns10 = 512, ns20 = 512;
    [vals setConstantValue:&b_mask  type:MTLDataTypeBool atIndex:300];
    [vals setConstantValue:&b_sinks type:MTLDataTypeBool atIndex:301];
    [vals setConstantValue:&b_bias  type:MTLDataTypeBool atIndex:302];
    [vals setConstantValue:&b_scap  type:MTLDataTypeBool atIndex:303];
    [vals setConstantValue:&b_kvpad type:MTLDataTypeBool atIndex:304];
    [vals setConstantValue:&b_bc    type:MTLDataTypeBool atIndex:310];
    [vals setConstantValue:&ns10    type:MTLDataTypeInt  atIndex:320];
    [vals setConstantValue:&ns20    type:MTLDataTypeInt  atIndex:321];
    [vals setConstantValue:&nsg     type:MTLDataTypeInt  atIndex:322];

    NSError *err = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]
                                         constantValues:vals error:&err];
    if (!fn) fprintf(stderr, "Function error (%s): %s\n", kernel_name, err.localizedDescription.UTF8String);
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) fprintf(stderr, "PSO error (%s): %s\n", kernel_name, err.localizedDescription.UTF8String);
    REQUIRE(pso != nil);
    return pso;
}

static id<MTLComputePipelineState> build_blk_pso(int32_t q_tile) {
    MTLFunctionConstantValues *vals = [MTLFunctionConstantValues new];
    int32_t c_tile = 64;
    [vals setConstantValue:&q_tile type:MTLDataTypeInt atIndex:224];
    [vals setConstantValue:&c_tile type:MTLDataTypeInt atIndex:225];
    NSError *err = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                                         constantValues:vals error:&err];
    REQUIRE(fn != nil);
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    return pso;
}

static void check_pso_geometry(const char *archive_path) {
    struct {
        const char *label;
        const char *name;
        int32_t nsg;
        size_t dynamic_smem;
    } specs[] = {
        { "orig_smem_q8", "kernel_myds_flash_attn_heads_smem_f16_dk512_dv512", 4, 32768u },
        { "step1_rego_q8", "kernel_myds_flash_attn_heads_f16_dk512_dv512",     4, 12320u },
        { "step2_rego_q16","kernel_myds_flash_attn_heads_q16_f16_dk512_dv512", 8, 24640u },
    };

    NSError *err = nil;
    id<MTLBinaryArchive> archive = nil;
    if (archive_path) {
        MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
        archive = [g_device newBinaryArchiveWithDescriptor:ad error:&err];
        REQUIRE(archive != nil);
    }

    for (unsigned i = 0; i < 3; ++i) {
        id<MTLComputePipelineState> p = build_heads_pso(specs[i].name, specs[i].nsg, false, true, false);
        NSUInteger req_threads = 32u * (NSUInteger)specs[i].nsg;
        NSUInteger max_smem = g_device.maxThreadgroupMemoryLength;
        size_t total_smem = specs[i].dynamic_smem + (size_t)p.staticThreadgroupMemoryLength;
        unsigned max_tgs_by_smem = total_smem > 0 ? (unsigned)(max_smem / total_smem) : 0;
        printf("PSO %-15s kernel=%-50s simdWidth=%lu reqThreads=%lu maxTotalThreadsPerTG=%lu staticSmem=%lu dynSmem=%zu maxTGsBySmem=%u\n",
               specs[i].label, specs[i].name,
               (unsigned long)p.threadExecutionWidth,
               (unsigned long)req_threads,
               (unsigned long)p.maxTotalThreadsPerThreadgroup,
               (unsigned long)p.staticThreadgroupMemoryLength,
               specs[i].dynamic_smem,
               max_tgs_by_smem);
        REQUIRE(p.threadExecutionWidth == 32u);
        REQUIRE(p.maxTotalThreadsPerThreadgroup >= req_threads);
        REQUIRE(total_smem <= max_smem);

        if (archive) {
            MTLFunctionConstantValues *vals = [MTLFunctionConstantValues new];
            bool b_mask = true, b_sinks = true, b_zero = false;
            int32_t ns10 = 512, ns20 = 512, nsg = specs[i].nsg;
            [vals setConstantValue:&b_mask  type:MTLDataTypeBool atIndex:300];
            [vals setConstantValue:&b_sinks type:MTLDataTypeBool atIndex:301];
            [vals setConstantValue:&b_zero  type:MTLDataTypeBool atIndex:302];
            [vals setConstantValue:&b_zero  type:MTLDataTypeBool atIndex:303];
            [vals setConstantValue:&b_zero  type:MTLDataTypeBool atIndex:304];
            [vals setConstantValue:&b_zero  type:MTLDataTypeBool atIndex:310];
            [vals setConstantValue:&ns10    type:MTLDataTypeInt  atIndex:320];
            [vals setConstantValue:&ns20    type:MTLDataTypeInt  atIndex:321];
            [vals setConstantValue:&nsg     type:MTLDataTypeInt  atIndex:322];
            id<MTLFunction> fn = [g_library newFunctionWithName:[NSString stringWithUTF8String:specs[i].name]
                                                 constantValues:vals error:&err];
            MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
            pd.computeFunction = fn;
            BOOL added = [archive addComputePipelineFunctionsWithDescriptor:pd error:&err];
            REQUIRE(added);
        }
    }

    id<MTLComputePipelineState> blk8  = build_blk_pso(8);
    id<MTLComputePipelineState> blk16 = build_blk_pso(16);
    REQUIRE(blk8.threadExecutionWidth == 32u && blk8.maxTotalThreadsPerThreadgroup >= 32u);
    REQUIRE(blk16.threadExecutionWidth == 32u && blk16.maxTotalThreadsPerThreadgroup >= 32u);

    if (archive && archive_path) {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:archive_path]];
        BOOL ok = [archive serializeToURL:url error:&err];
        if (!ok) fprintf(stderr, "Archive serialize error: %s\n", err.localizedDescription.UTF8String);
        REQUIRE(ok);
        printf("ARCHIVE dumped to %s\n", archive_path);
    }
}

/* --------------------------------------------------------------------------
 * 3A. Direct kernel `q_rem` & partial-kvpad / mask / sink 0-ULP gate
 * -------------------------------------------------------------------------- */
static void test_direct_qrem_and_kvpad_parity(void) {
    const int32_t ne01_cases[] = { 5, 8, 13, 16, 25, 63, 64 };
    const int32_t ne11_cases[] = { 1, 37, 64, 65, 192, 640 };
    const int32_t c_len_cases[] = { 1, 7, 33 };

    const size_t max_q_rows = 33u * 64u;
    myds_metal_buffer_t *b_q     = myds_metal_buffer_alloc(max_q_rows * 512u * sizeof(float));
    myds_metal_buffer_t *b_kv    = myds_metal_buffer_alloc(33u * 640u * 512u * sizeof(uint16_t));
    myds_metal_buffer_t *b_mask  = myds_metal_buffer_alloc(33u * 64u * 640u * sizeof(uint16_t));
    myds_metal_buffer_t *b_sinks = myds_metal_buffer_alloc(64u * sizeof(float));
    myds_metal_buffer_t *b_pad   = myds_metal_buffer_alloc(33u * (2u * 64u * 1024u + 2u * 64u * 64u));
    myds_metal_buffer_t *b_blk   = myds_metal_buffer_alloc(33u * 8u * 16u);
    REQUIRE(b_q && b_kv && b_mask && b_sinks && b_pad && b_blk);

    float *pq = (float *)myds_metal_buffer_data(b_q);
    for (size_t i = 0; i < max_q_rows * 512u; ++i) pq[i] = bf16_sample((uint32_t)i, 11u);
    uint16_t *pkv = (uint16_t *)myds_metal_buffer_data(b_kv);
    for (size_t i = 0; i < 33u * 640u * 512u; ++i) pkv[i] = f32_to_f16_bits(bf16_sample((uint32_t)i, 29u));
    memset(myds_metal_buffer_data(b_pad), 0, myds_metal_buffer_size(b_pad));

    uint64_t total_pairs = 0;
    uint32_t max_ulp_q8 = 0, max_ulp_q16 = 0;
    uint64_t diff_q8 = 0, diff_q16 = 0;

    for (size_t ic = 0; ic < sizeof(c_len_cases)/sizeof(*c_len_cases); ++ic) {
        int32_t c_len = c_len_cases[ic];
        for (size_t in0 = 0; in0 < sizeof(ne01_cases)/sizeof(*ne01_cases); ++in0) {
            int32_t ne01 = ne01_cases[in0];
            for (size_t in1 = 0; in1 < sizeof(ne11_cases)/sizeof(*ne11_cases); ++in1) {
                int32_t ne11 = ne11_cases[in1];
                bool has_kvpad = (ne11 % 64) != 0;
                int32_t mask_stride = ((ne11 + 63) / 64) * 64;

                for (int mode = 0; mode < 3; ++mode) {
                    /* mode 0: shared query mask (nb31=0) + active sinks
                     * mode 1: per-head mask (nb31!=0) + active sinks
                     * mode 2: all-masked (-MAXHALF) + no sinks (S == 0 edge case) */
                    bool has_sinks = (mode != 2);
                    bool per_head_mask = (mode == 1);
                    float *psink = (float *)myds_metal_buffer_data(b_sinks);
                    for (int h = 0; h < 64; ++h) psink[h] = (float)(h - 31) / 16.0f;

                    uint16_t *pm = (uint16_t *)myds_metal_buffer_data(b_mask);
                    size_t m_heads = per_head_mask ? (size_t)ne01 : 1u;
                    for (int32_t q2 = 0; q2 < c_len; ++q2) {
                        for (size_t h = 0; h < m_heads; ++h) {
                            uint16_t *row = pm + ((size_t)q2 * m_heads + h) * (size_t)mask_stride;
                            for (int32_t k = 0; k < mask_stride; ++k) {
                                if (k >= ne11 || mode == 2) {
                                    row[k] = 0xfbffu; /* -MAXHALF */
                                } else if (((k + q2 * 7 + (int32_t)h * 3) % 5) == 0) {
                                    row[k] = 0xfbffu;
                                } else {
                                    row[k] = ((k + q2) & 1) ? 0u : 0xb800u; /* 0 or -0.5 */
                                }
                            }
                        }
                    }

                    /* Prepare kvpad tail if needed */
                    if (has_kvpad) {
                        int32_t tail0 = (ne11 / 64) * 64;
                        int32_t rem = ne11 - tail0;
                        uint8_t *pad_base = (uint8_t *)myds_metal_buffer_data(b_pad);
                        memset(pad_base, 0, myds_metal_buffer_size(b_pad));
                        size_t kv_block_bytes = 64u * 1024u;
                        uint8_t *pad_k = pad_base;
                        uint8_t *pad_v = pad_k + (size_t)c_len * kv_block_bytes;
                        uint16_t *pad_m = (uint16_t *)(pad_v + (size_t)c_len * kv_block_bytes);
                        for (int32_t q2 = 0; q2 < c_len; ++q2) {
                            const uint8_t *src_kv = (const uint8_t *)pkv + ((size_t)q2 * 640u + (size_t)tail0) * 1024u;
                            memcpy(pad_k + (size_t)q2 * kv_block_bytes, src_kv, (size_t)rem * 1024u);
                            memcpy(pad_v + (size_t)q2 * kv_block_bytes, src_kv, (size_t)rem * 1024u);
                            for (int32_t h = 0; h < ne01; ++h) {
                                size_t mh = per_head_mask ? (size_t)h : 0u;
                                const uint16_t *src_m = pm + ((size_t)q2 * m_heads + mh) * (size_t)mask_stride + tail0;
                                uint16_t *dst_m = pad_m + ((size_t)q2 * (size_t)ne01 + (size_t)h) * 64u;
                                for (int32_t i = 0; i < 64; ++i) {
                                    dst_m[i] = (i < rem) ? src_m[i] : 0xfbffu;
                                }
                            }
                        }
                    }

                    size_t out_bytes = (size_t)c_len * (size_t)ne01 * 512u * sizeof(float);
                    struct guarded_buf out_ref = guarded_alloc(out_bytes);
                    struct guarded_buf out_q8  = guarded_alloc(out_bytes);
                    struct guarded_buf out_q16 = guarded_alloc(out_bytes);

                    struct flash_vec_args ma = {
                        .ne01 = ne01, .ne02 = c_len, .ne03 = 1,
                        .nb01 = 2048u, .nb02 = (uint64_t)ne01 * 2048u, .nb03 = (uint64_t)ne01 * (uint64_t)c_len * 2048u,
                        .ne11 = ne11, .ne_12_2 = c_len, .ne_12_3 = 1,
                        .ns10 = 512, .nb11 = 1024u, .nb12 = 640u * 1024u, .nb13 = (uint64_t)c_len * 640u * 1024u,
                        .ns20 = 512, .nb21 = 1024u, .nb22 = 640u * 1024u, .nb23 = (uint64_t)c_len * 640u * 1024u,
                        .ne31 = ne01, .ne32 = c_len, .ne33 = 1,
                        .nb31 = per_head_mask ? (uint64_t)mask_stride * 2u : 0u,
                        .nb32 = (uint64_t)m_heads * (uint64_t)mask_stride * 2u,
                        .nb33 = (uint64_t)c_len * (uint64_t)m_heads * (uint64_t)mask_stride * 2u,
                        .ne1 = ne01, .ne2 = c_len, .ne3 = 1,
                        .scale = 1.0f / sqrtf(512.0f),
                    };
                    struct { int32_t ne01, ne30, ne31, ne32, ne33; uint64_t nb31, nb32, nb33; } ba = {
                        ne01, ne11, ne01, c_len, 1, ma.nb31, ma.nb32, ma.nb33
                    };
                    int32_t nblk0 = (ne11 + 63) / 64;

                    const struct {
                        const char *kname;
                        int32_t q_tile;
                        int32_t nsg;
                        size_t smem;
                        myds_metal_buffer_t *dst;
                    } arms[3] = {
                        { "kernel_myds_flash_attn_heads_smem_f16_dk512_dv512", 8,  4, 32768u, out_ref.view },
                        { "kernel_myds_flash_attn_heads_f16_dk512_dv512",      8,  4, 12320u, out_q8.view  },
                        { "kernel_myds_flash_attn_heads_q16_f16_dk512_dv512",  16, 8, 24640u, out_q16.view },
                    };

                    for (int a = 0; a < 3; ++a) {
                        uint32_t nblk1 = (uint32_t)((ne01 + arms[a].q_tile - 1) / arms[a].q_tile);
                        const myds_metal_buffer_t *bb[] = { b_mask, b_blk };
                        const myds_metal_constant_t bc[] = { {224, false, arms[a].q_tile}, {225, false, 64} };
                        const myds_metal_buffer_t *mb[] = { b_q, b_kv, b_kv, b_mask, b_sinks, b_pad, b_blk, arms[a].dst };
                        const myds_metal_constant_t mc[] = {
                            {300, true, 1}, {301, true, has_sinks ? 1 : 0}, {302, true, 0}, {303, true, 0},
                            {304, true, has_kvpad ? 1 : 0}, {310, true, 0},
                            {320, false, 512}, {321, false, 512}, {322, false, arms[a].nsg}
                        };
                        REQUIRE(myds_metal_begin_batch());
                        REQUIRE(specialized("kernel_flash_attn_ext_blk", &ba, sizeof(ba), bb, 2,
                                            (uint32_t)nblk0, nblk1, (uint32_t)c_len, 32, 1, 1, 0, bc, 2));
                        myds_metal_barrier();
                        REQUIRE(specialized(arms[a].kname, &ma, 192, mb, 8,
                                            nblk1, (uint32_t)c_len, 1, 32, (uint32_t)arms[a].nsg, 1, arms[a].smem, mc, 9));
                        REQUIRE(myds_metal_commit_batch());
                    }

                    const float *f_ref = (const float *)myds_metal_buffer_data(out_ref.view);
                    const float *f_q8  = (const float *)myds_metal_buffer_data(out_q8.view);
                    const float *f_q16 = (const float *)myds_metal_buffer_data(out_q16.view);
                    size_t nf = out_bytes / sizeof(float);
                    for (size_t i = 0; i < nf; ++i) {
                        REQUIRE(isfinite(f_ref[i]) && isfinite(f_q8[i]) && isfinite(f_q16[i]));
                        uint32_t u8  = ulp_diff_f32(f_ref[i], f_q8[i]);
                        uint32_t u16 = ulp_diff_f32(f_ref[i], f_q16[i]);
                        if (u8 > max_ulp_q8) max_ulp_q8 = u8;
                        if (u16 > max_ulp_q16) max_ulp_q16 = u16;
                        if (memcmp(&f_ref[i], &f_q8[i], 4) != 0) ++diff_q8;
                        if (memcmp(&f_ref[i], &f_q16[i], 4) != 0) ++diff_q16;
                    }
                    guarded_free(out_ref);
                    guarded_free(out_q8);
                    guarded_free(out_q16);
                    ++total_pairs;
                }
            }
        }
    }

    myds_metal_buffer_free(b_q);
    myds_metal_buffer_free(b_kv);
    myds_metal_buffer_free(b_mask);
    myds_metal_buffer_free(b_sinks);
    myds_metal_buffer_free(b_pad);
    myds_metal_buffer_free(b_blk);

    printf("PASS direct_qrem_kvpad_gate cases=%" PRIu64 " q8_diff=%" PRIu64 " q8_max_ulp=%u q16_diff=%" PRIu64 " q16_max_ulp=%u\n",
           total_pairs, diff_q8, max_ulp_q8, diff_q16, max_ulp_q16);
    REQUIRE(max_ulp_q8 <= 1u && max_ulp_q16 <= 1u);
}

/* --------------------------------------------------------------------------
 * 3B. Production `myds_v41_gpu_attention_chunk_bf16` & `_chunk` 0-ULP gate
 * -------------------------------------------------------------------------- */
struct chunk_fixture {
    uint32_t rows, start, ratio, ncomp;
    myds_metal_buffer_t *q, *kv, *comp, *packed, *ids, *sinks, *gather, *window;
    struct guarded_buf out[3];
};

static void chunk_fixture_init(struct chunk_fixture *f, uint32_t rows, uint32_t start, uint32_t ratio) {
    memset(f, 0, sizeof(*f));
    f->rows = rows;
    f->start = start;
    f->ratio = ratio;
    f->ncomp = (start + rows) / ratio;
    if (!f->ncomp) f->ncomp = 1u;

    f->q      = myds_metal_buffer_alloc((size_t)rows * 64u * 512u * 4u);
    f->kv     = myds_metal_buffer_alloc((size_t)rows * 512u * 4u);
    f->comp   = myds_metal_buffer_alloc((size_t)f->ncomp * 512u * 4u);
    f->packed = myds_metal_buffer_alloc((size_t)f->ncomp * 512u * 2u);
    f->ids    = myds_metal_buffer_alloc((size_t)rows * 512u * 4u);
    f->sinks  = myds_metal_buffer_alloc(64u * 4u);
    f->gather = myds_metal_buffer_alloc((size_t)rows * 640u * 512u * 2u);
    f->window = myds_metal_buffer_alloc(128u * 512u * 4u);
    REQUIRE(f->q && f->kv && f->comp && f->packed && f->ids && f->sinks && f->gather && f->window);

    for (int j = 0; j < 3; ++j) {
        f->out[j] = guarded_alloc((size_t)rows * 64u * 512u * 4u);
    }

    myds_metal_buffer_t *bs[] = { f->q, f->kv, f->comp, f->window };
    for (int k = 0; k < 4; ++k) {
        float *p = (float *)myds_metal_buffer_data(bs[k]);
        size_t n = myds_metal_buffer_size(bs[k]) / 4u;
        for (size_t j = 0; j < n; ++j) p[j] = bf16_sample((uint32_t)j, (uint32_t)(k + 1));
    }
    float *sink = (float *)myds_metal_buffer_data(f->sinks);
    for (int j = 0; j < 64; ++j) sink[j] = (float)(j - 31) / 16.0f;

    int32_t *ids = (int32_t *)myds_metal_buffer_data(f->ids);
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t count = (start + r + 1u) / ratio;
        uint32_t step = count > 1u ? count / 2u + 1u : 1u;
        while (count > 1u && gcd_u32(step, count) != 1u) ++step;
        uint32_t valid = count < 512u ? count : 512u;
        for (uint32_t j = 0; j < 512u; ++j) {
            ids[(size_t)r * 512u + j] = (j < valid && count > 0u)
                ? (int32_t)(((uint64_t)j * step + r * 17u) % count)
                : -1;
        }
    }

    REQUIRE(myds_metal_begin_batch());
    REQUIRE(myds_v41_gpu_pack_bf16(f->packed, 0, f->comp, 0, f->ncomp * 512u));
    REQUIRE(myds_metal_commit_batch());
}

static void chunk_fixture_free(struct chunk_fixture *f) {
    myds_metal_buffer_t *bs[] = { f->q, f->kv, f->comp, f->packed, f->ids, f->sinks, f->gather, f->window };
    for (size_t i = 0; i < sizeof(bs)/sizeof(*bs); ++i) myds_metal_buffer_free(bs[i]);
    for (int j = 0; j < 3; ++j) guarded_free(f->out[j]);
}

static void test_production_chunk_parity(void) {
    const struct { uint32_t rows, start, ratio; int packed; } cases[] = {
        { 1,    1280, 2, 1 }, /* explicit matrix single-token */
        { 7,    1280, 2, 1 }, /* <8 tail */
        { 15,   1280, 2, 1 }, /* <16 tail */
        { 16,   1280, 2, 1 }, /* exact 16 */
        { 31,   1280, 2, 1 }, /* <32 tail */
        { 33,   1280, 2, 1 }, /* 640 keys, non-8/16 tail 33 */
        { 33,   600,  2, 0 }, /* f32 comp path, partial keys */
        { 63,   1280, 2, 1 }, /* 640 keys, non-8/16 tail 63 */
        { 64,   1280, 2, 1 }, /* 640 keys, 64 */
        { 127,  1280, 2, 1 }, /* 640 keys, odd 127 */
        { 128,  1280, 2, 1 }, /* 640 keys, 128 */
        { 294,  1536, 2, 1 }, /* 640 keys, 294 */
        { 512,  512,  2, 1 }, /* growing n_comp 256..512 */
        { 512,  1280, 2, 1 }, /* full 640 keys, 512 */
        { 2048, 1280, 2, 1 }, /* full 2048 chunk, 640 keys (67.1M floats!) */
    };

    setenv("MYDS_ATTN_GATHER", "1", 1);
    unsetenv("MYDS_ATTN_GATHER_MIN_KEYS");
    setenv("MYDS_PREFILL_ATTN", "matrix", 1);

    const char *modes[3] = { "smem", "q8", "q16" };
    const char *knames[3] = {
        "kernel_myds_flash_attn_heads_smem_f16_dk512_dv512",
        "kernel_myds_flash_attn_heads_f16_dk512_dv512",
        "kernel_myds_flash_attn_heads_q16_f16_dk512_dv512",
    };

    for (size_t ic = 0; ic < sizeof(cases)/sizeof(*cases); ++ic) {
        struct chunk_fixture f;
        chunk_fixture_init(&f, cases[ic].rows, cases[ic].start, cases[ic].ratio);

        for (int m = 0; m < 3; ++m) {
            myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(8192);
            REQUIRE(s != NULL);
            /* For m == 2 (q16), test both unset (default) and explicit "q16" */
            if (m == 2 && (ic % 2 == 0)) unsetenv("MYDS_ATTN_HEADS_MODE");
            else setenv("MYDS_ATTN_HEADS_MODE", modes[m], 1);
            uint64_t before = hist_count_for(knames[m]);
            REQUIRE(myds_metal_begin_batch());
            if (f.start) {
                uint32_t carry = f.start < 128u ? f.start : 128u;
                REQUIRE(myds_v41_gpu_carry_stage(s, f.window, carry));
            }
            if (cases[ic].packed) {
                REQUIRE(myds_v41_gpu_attention_chunk_bf16(
                    f.out[m].view, f.sinks, f.q, f.kv, f.packed, f.ids, f.gather,
                    f.rows, f.ncomp, 512u, f.start, 0u, f.ratio, s));
            } else {
                REQUIRE(myds_v41_gpu_attention_chunk(
                    f.out[m].view, f.sinks, f.q, f.kv, f.comp, f.ids, f.gather,
                    f.rows, f.ncomp, 512u, f.start, 0u, f.ratio, s));
            }
            REQUIRE(myds_metal_commit_batch());
            REQUIRE(hist_count_for(knames[m]) == before + 1u);
            myds_v41_gpu_scratch_free(s);
        }

        const float *ref = (const float *)myds_metal_buffer_data(f.out[0].view);
        const float *q8  = (const float *)myds_metal_buffer_data(f.out[1].view);
        const float *q16 = (const float *)myds_metal_buffer_data(f.out[2].view);
        size_t nf = (size_t)f.rows * 64u * 512u;
        int memcmp_q8  = memcmp(ref, q8,  nf * sizeof(float));
        int memcmp_q16 = memcmp(ref, q16, nf * sizeof(float));

        uint64_t diff_q8 = 0, diff_q16 = 0;
        uint32_t max_ulp_q8 = 0, max_ulp_q16 = 0;
        double max_abs_q8 = 0.0, max_abs_q16 = 0.0;
        for (size_t i = 0; i < nf; ++i) {
            REQUIRE(isfinite(ref[i]) && isfinite(q8[i]) && isfinite(q16[i]));
            if (memcmp(&ref[i], &q8[i], 4) != 0) {
                ++diff_q8;
                uint32_t u = ulp_diff_f32(ref[i], q8[i]);
                if (u > max_ulp_q8) max_ulp_q8 = u;
                double d = fabs((double)ref[i] - (double)q8[i]);
                if (d > max_abs_q8) max_abs_q8 = d;
            }
            if (memcmp(&ref[i], &q16[i], 4) != 0) {
                ++diff_q16;
                uint32_t u = ulp_diff_f32(ref[i], q16[i]);
                if (u > max_ulp_q16) max_ulp_q16 = u;
                double d = fabs((double)ref[i] - (double)q16[i]);
                if (d > max_abs_q16) max_abs_q16 = d;
            }
        }

        printf("PASS chunk_parity c_len=%4u start=%4u ratio=%u n_comp=%4u floats=%8zu | "
               "q8: memcmp=%d diff=%" PRIu64 " maxULP=%u maxAbs=%.3g | "
               "q16: memcmp=%d diff=%" PRIu64 " maxULP=%u maxAbs=%.3g\n",
               f.rows, f.start, f.ratio, f.ncomp, nf,
               memcmp_q8 == 0 ? 0 : 1, diff_q8, max_ulp_q8, max_abs_q8,
               memcmp_q16 == 0 ? 0 : 1, diff_q16, max_ulp_q16, max_abs_q16);
        REQUIRE(max_ulp_q8 <= 1u && max_ulp_q16 <= 1u);
        chunk_fixture_free(&f);
    }
    unsetenv("MYDS_ATTN_HEADS_MODE");
}

/* --------------------------------------------------------------------------
 * 4. GPU Hardware Timestamp Microbenchmark (`GPUEndTime - GPUStartTime`)
 * -------------------------------------------------------------------------- */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void benchmark_gpu_timestamps(uint32_t samples, uint32_t base_reps) {
    const uint32_t bench_clens[] = { 33, 63, 64, 128, 256, 512, 2048 };
    const char *modes[3] = { "smem", "q8", "q16" };
    const char *knames[3] = {
        "kernel_myds_flash_attn_heads_smem_f16_dk512_dv512",
        "kernel_myds_flash_attn_heads_f16_dk512_dv512",
        "kernel_myds_flash_attn_heads_q16_f16_dk512_dv512",
    };
    const int32_t q_tiles[3] = { 8, 8, 16 };
    const int32_t nsgs[3]    = { 4, 4, 8 };
    const size_t  smems[3]   = { 32768u, 12320u, 24640u };

    setenv("MYDS_ATTN_GATHER", "1", 1);
    unsetenv("MYDS_ATTN_GATHER_MIN_KEYS");
    setenv("MYDS_PREFILL_ATTN", "matrix", 1);

    int prev_hist = g_hist_enabled;
    g_hist_enabled = 0;
    myds_metal_set_concurrent(1);
    myds_metal_set_auto_barrier(0);

    printf("\n=== GPU Hardware Timestamp Microbenchmark (GPUEndTime - GPUStartTime, device=%s) ===\n",
           g_device.name.UTF8String);

    for (size_t ic = 0; ic < sizeof(bench_clens)/sizeof(*bench_clens); ++ic) {
        uint32_t c_len = bench_clens[ic];
        uint32_t reps = (c_len >= 2048u) ? (base_reps >= 4u ? base_reps / 4u : 4u)
                      : (c_len >= 512u)  ? (base_reps >= 2u ? base_reps / 2u : 8u)
                      : base_reps;
        if (reps < 4u) reps = 4u;

        struct chunk_fixture f;
        chunk_fixture_init(&f, c_len, 1280u, 2u); /* full 640 gathered keys */
        myds_v41_gpu_scratch_t *s = myds_v41_gpu_scratch_create(8192);
        REQUIRE(s != NULL);

        /* Pre-stage gather, mask, and carry once via full chunk call so `f.gather`,
         * `s->mask_chunk`, and `s->attn_head_blocks` hold exact realistic 640-key data! */
        setenv("MYDS_ATTN_HEADS_MODE", "smem", 1);
        REQUIRE(myds_metal_begin_batch());
        REQUIRE(myds_v41_gpu_carry_stage(s, f.window, 128u));
        REQUIRE(myds_v41_gpu_attention_chunk_bf16(
            f.out[0].view, f.sinks, f.q, f.kv, f.packed, f.ids, f.gather,
            f.rows, f.ncomp, 512u, f.start, 0u, f.ratio, s));
        REQUIRE(myds_metal_commit_batch());

        /* Measure Scope A: pure `blk + heads` kernel pair (and pure `heads_only`) */
        struct flash_vec_args ma = {
            .ne01 = 64, .ne02 = (int32_t)c_len, .ne03 = 1,
            .nb01 = 2048u, .nb02 = 64u * 2048u, .nb03 = (uint64_t)c_len * 64u * 2048u,
            .ne11 = 640, .ne_12_2 = (int32_t)c_len, .ne_12_3 = 1,
            .ns10 = 512, .nb11 = 1024u, .nb12 = 640u * 1024u, .nb13 = (uint64_t)c_len * 640u * 1024u,
            .ns20 = 512, .nb21 = 1024u, .nb22 = 640u * 1024u, .nb23 = (uint64_t)c_len * 640u * 1024u,
            .ne31 = 64, .ne32 = (int32_t)c_len, .ne33 = 1,
            .nb31 = 0u, .nb32 = (uint64_t)s->mask_u16 * 2u, .nb33 = (uint64_t)s->mask_u16 * 2u * c_len,
            .ne1 = 64, .ne2 = (int32_t)c_len, .ne3 = 1,
            .scale = 1.0f / sqrtf(512.0f),
        };
        struct { int32_t ne01, ne30, ne31, ne32, ne33; uint64_t nb31, nb32, nb33; } ba = {
            64, 640, 64, (int32_t)c_len, 1, 0u, (uint64_t)s->mask_u16 * 2u, (uint64_t)s->mask_u16 * 2u * c_len
        };

        for (int scope = 0; scope < 2; ++scope) {
            /* scope 0 = heads_only (blk precomputed per q_tile)
             * scope 1 = full_chunk_bf16 (gather + mask + blk + heads + carry) */
            double times_us[3][32];
            REQUIRE(samples <= 32u);

            /* Warmup + interleaved samples across the 3 modes to eliminate thermal drift */
            for (uint32_t samp = 0; samp <= samples; ++samp) {
                for (int m = 0; m < 3; ++m) {
                    uint32_t nblk1 = (uint32_t)(64 / q_tiles[m]);
                    if (scope == 0) {
                        /* Precompute matching `attn_head_blocks` for this q_tile outside timed CB */
                        const myds_metal_buffer_t *bb[] = { s->mask_chunk, s->attn_head_blocks };
                        const myds_metal_constant_t bc[] = { {224, false, q_tiles[m]}, {225, false, 64} };
                        REQUIRE(myds_metal_begin_batch());
                        REQUIRE(specialized("kernel_flash_attn_ext_blk", &ba, sizeof(ba), bb, 2,
                                            10u, nblk1, c_len, 32, 1, 1, 0, bc, 2));
                        REQUIRE(myds_metal_commit_batch());

                        const myds_metal_buffer_t *mb[] = {
                            f.q, f.gather, f.gather, s->mask_chunk, f.sinks, s->pad, s->attn_head_blocks, f.out[m].view
                        };
                        const myds_metal_constant_t mc[] = {
                            {300, true, 1}, {301, true, 1}, {302, true, 0}, {303, true, 0},
                            {304, true, 0}, {310, true, 0}, {320, false, 512}, {321, false, 512}, {322, false, nsgs[m]}
                        };

                        REQUIRE(myds_metal_begin_batch());
                        id<MTLCommandBuffer> cb = g_batch_cb;
                        for (uint32_t r = 0; r < reps; ++r) {
                            REQUIRE(specialized(knames[m], &ma, 192, mb, 8,
                                                nblk1, c_len, 1, 32, (uint32_t)nsgs[m], 1, smems[m], mc, 9));
                            myds_metal_barrier();
                        }
                        REQUIRE(myds_metal_commit_batch());
                        REQUIRE(cb.status == MTLCommandBufferStatusCompleted && cb.GPUEndTime > cb.GPUStartTime);
                        if (samp > 0) times_us[m][samp - 1] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / (double)reps;
                    } else {
                        setenv("MYDS_ATTN_HEADS_MODE", modes[m], 1);
                        REQUIRE(myds_metal_begin_batch());
                        id<MTLCommandBuffer> cb = g_batch_cb;
                        for (uint32_t r = 0; r < reps; ++r) {
                            REQUIRE(myds_v41_gpu_carry_stage(s, f.window, 128u));
                            REQUIRE(myds_v41_gpu_attention_chunk_bf16(
                                f.out[m].view, f.sinks, f.q, f.kv, f.packed, f.ids, f.gather,
                                f.rows, f.ncomp, 512u, f.start, 0u, f.ratio, s));
                            myds_metal_barrier();
                        }
                        REQUIRE(myds_metal_commit_batch());
                        REQUIRE(cb.status == MTLCommandBufferStatusCompleted && cb.GPUEndTime > cb.GPUStartTime);
                        if (samp > 0) times_us[m][samp - 1] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / (double)reps;
                    }
                }
            }

            double med[3], minv[3];
            for (int m = 0; m < 3; ++m) {
                qsort(times_us[m], samples, sizeof(double), cmp_double);
                minv[m] = times_us[m][0];
                med[m]  = times_us[m][samples / 2];
            }
            printf("BENCH %-15s c_len=%4u keys=640 reps=%2u samples=%u | "
                   "smem: med=%8.2f us (min=%8.2f) | "
                   "q8_rego: med=%8.2f us (min=%8.2f, %+.2f%%, %.3fx) | "
                   "q16_rego: med=%8.2f us (min=%8.2f, %+.2f%%, %.3fx) | "
                   "q16_vs_q8=%+.2f%%\n",
                   scope == 0 ? "heads_only" : "full_chunk_bf16",
                   c_len, reps, samples,
                   med[0], minv[0],
                   med[1], minv[1], (med[1] - med[0]) / med[0] * 100.0, med[0] / med[1],
                   med[2], minv[2], (med[2] - med[0]) / med[0] * 100.0, med[0] / med[2],
                   (med[2] - med[1]) / med[1] * 100.0);
        }

        myds_v41_gpu_scratch_free(s);
        chunk_fixture_free(&f);
    }

    unsetenv("MYDS_ATTN_HEADS_MODE");
    g_hist_enabled = prev_hist;
}

int main(int argc, char **argv) {
    const char *archive_path = NULL;
    int skip_bench = 0;
    uint32_t samples = 7, reps = 16;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump-archive") == 0 && i + 1 < argc) {
            archive_path = argv[++i];
        } else if (strcmp(argv[i], "--no-bench") == 0) {
            skip_bench = 1;
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            samples = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "Usage: %s [--dump-archive path] [--no-bench] [--samples N] [--reps N]\n", argv[0]);
            return 2;
        }
    }

    setenv("MYDS_METAL_CONCURRENT", "0", 1);
    setenv("MYDS_DEBUG_DISPATCH", "1", 1);
    g_hist_enabled = 1;
    REQUIRE(myds_metal_init("metal"));

    printf("=== [P-A1] Reg-O Heads Verification & Benchmark (device=%s) ===\n", g_device.name.UTF8String);
    test_simdgroup_float8x8_mapping();
    check_pso_geometry(archive_path);
    test_direct_qrem_and_kvpad_parity();
    test_production_chunk_parity();

    if (!skip_bench) {
        benchmark_gpu_timestamps(samples, reps);
    }

    myds_metal_cleanup();
    printf("=== [P-A1] ALL GATES PASSED ===\n");
    return 0;
}
