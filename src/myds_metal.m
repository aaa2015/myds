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
static id<MTLComputePipelineState> g_pso_v41_hc_sum_rmsnorm = nil;
static id<MTLComputePipelineState> g_pso_v41_hc_expand = nil;
static id<MTLComputePipelineState> g_pso_v41_moe_add_hc_expand = nil;
static id<MTLComputePipelineState> g_pso_v41_engram_add = nil;
static id<MTLComputePipelineState> g_pso_swiglu = nil;
static id<MTLComputePipelineState> g_pso_add = nil;
static id<MTLComputePipelineState> g_pso_add_bf16 = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_f32 = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_f16 = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_q4_K = nil;
static id<MTLComputePipelineState> g_pso_mul_mv_q4_K_dense_bf16 = nil;
static id<MTLComputePipelineState> g_pso_softmax = nil;
static id<MTLComputePipelineState> g_pso_moe_mv_id_iq2_xxs = nil;
static id<MTLComputePipelineState> g_pso_moe_mv_id_q2_k = nil;
static id<MTLComputePipelineState> g_pso_moe_swiglu_weight = nil;
static id<MTLComputePipelineState> g_pso_moe_sum6 = nil;
static id<MTLComputePipelineState> g_pso_moe_pair_swiglu_iq2_xxs = nil;
static id<MTLComputePipelineState> g_pso_moe_sum6_q2_k = nil;
static id<MTLComputePipelineState> g_pso_moe_pair_swiglu_batch_iq2_xxs = nil;
static id<MTLComputePipelineState> g_pso_moe_sum6_batch_q2_k = nil;
static inline bool get_moe_batch_dedup_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("MYDS_MOE_BATCH_DEDUP");
        cached = (!v || strcmp(v, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}
static id<MTLComputePipelineState> g_pso_moe_mm_map0_ne20_6 = nil;
static id<MTLComputePipelineState> g_pso_moe_mm_iq2_pair_swiglu_f16 = nil;
static id<MTLComputePipelineState> g_pso_moe_mm_q2_k_f16 = nil;
static id<MTLComputePipelineState> g_pso_moe_mm_q2_k_f32 = nil;
static id<MTLBuffer>               g_moe_id_map_buffer = nil;
static NSUInteger                  g_moe_id_map_bytes = 0;
static int                         g_moe_nsg = 2;
static int                         g_moe_down_nsg = 2;
static bool                        g_moe_iq2_constant_lut = false;
static id<MTLComputePipelineState> g_pso_attn_out_low_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_attn_out_low_q8_0_bf16_r4 = nil;
static id<MTLComputePipelineState> g_pso_attn_out_low_q4_K = nil;
static id<MTLComputePipelineState> g_pso_attn_out_low_q4_K_bf16 = nil;
static id<MTLComputePipelineState> g_pso_attn_out_low_q8_0_bf16_shared2 = nil;
static id<MTLComputePipelineState> g_pso_attn_out_low_q4_K_bf16_shared2 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q8_0 = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q8_0_bco = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q4_K = nil;
static id<MTLComputePipelineState> g_pso_mul_mm_q4_K_bco = nil;
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
static bool g_indexer_head_slots = false;
static id<MTLComputePipelineState> g_pso_indexer_head_slots = nil;
static int indexer_head_slots_parse_mode(void) {
    const char *value = getenv("MYDS_INDEXER_HEAD_SLOTS");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr, "myds_metal: MYDS_INDEXER_HEAD_SLOTS must be exactly 0 or 1\n");
    return -1;
}
int myds_metal_indexer_head_slots_enabled(void) { return g_indexer_head_slots ? 1 : 0; }
static bool indexer_head_slots_geometry(void) {
    const NSUInteger dynamic_bytes = 640u;
    return g_device && g_pso_indexer_head_slots && g_pso_indexer_head_slots.device == g_device &&
        g_pso_indexer_head_slots.threadExecutionWidth == 32u &&
        g_pso_indexer_head_slots.maxTotalThreadsPerThreadgroup >= 128u &&
        g_device.maxThreadsPerThreadgroup.width >= 32u &&
        g_device.maxThreadsPerThreadgroup.height >= 4u &&
        g_device.maxThreadsPerThreadgroup.depth >= 1u &&
        g_device.maxThreadgroupMemoryLength >= dynamic_bytes &&
        g_pso_indexer_head_slots.staticThreadgroupMemoryLength <= g_device.maxThreadgroupMemoryLength - dynamic_bytes;
}
static int indexer_head_slots_initialize(void) {
    NSError *error = nil;
    id<MTLFunction> function = [g_library newFunctionWithName:@"kernel_myds_indexer_score_one_head_slots"];
    if (function) g_pso_indexer_head_slots = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!indexer_head_slots_geometry()) {
        fprintf(stderr, "myds_metal: explicit indexer head slots PSO/geometry unavailable (%s); refusing fallback\n",
            error.localizedDescription.UTF8String ?: "function or resource limit");
        return 0;
    }
    return 1;
}
static int g_qkv_finish_available = -1;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_native_pipelines = nil;
static bool g_indexer_topk_nan_shuffle = false;
static int indexer_topk_nan_shuffle_parse_mode(void) {
    const char *value = getenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr,"myds_metal: MYDS_INDEXER_TOPK_NAN_SHUFFLE must be exactly 0 or 1\n");
    return -1;
}
int myds_metal_indexer_topk_nan_shuffle_enabled(void) { return g_indexer_topk_nan_shuffle ? 1 : 0; }
int myds_metal_indexer_topk_nan_shuffle_preflight(bool causal,uint32_t reference_nth) {
    if (!g_device || !g_library || !reference_nth || reference_nth > 1024u ||
        (reference_nth & (reference_nth - 1u))) return 0;
    const char *names[] = {
        causal ? "kernel_myds_argsort_f32_i32_desc_causal_partition_nan_shuffle" :
                 "kernel_myds_argsort_f32_i32_desc_partition_nan_shuffle",
        causal ? "kernel_myds_argsort_merge_f32_i32_desc_causal_partition_prefix_guarded" :
                 "kernel_myds_argsort_merge_f32_i32_desc_partition_prefix_guarded"};
    if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
    for (unsigned i = 0; i < 2u; ++i) {
        NSString *name = [NSString stringWithUTF8String:names[i]];
        id<MTLComputePipelineState> p = g_native_pipelines[name];
        NSError *error = nil;
        if (!p) {
            id<MTLFunction> fn = [g_library newFunctionWithName:name];
            if (fn) p = [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (p) g_native_pipelines[name] = p;
        }
        const NSUInteger threads = i ? 512u : reference_nth;
        const NSUInteger dynamic = i ? 0u : (NSUInteger)reference_nth * 8u;
        if (!p || p.device != g_device || p.threadExecutionWidth != 32u ||
            threads > p.maxTotalThreadsPerThreadgroup ||
            threads > g_device.maxThreadsPerThreadgroup.width ||
            g_device.maxThreadsPerThreadgroup.height < 1u ||
            g_device.maxThreadsPerThreadgroup.depth < 1u ||
            dynamic > g_device.maxThreadgroupMemoryLength ||
            p.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - dynamic) {
            fprintf(stderr,"myds_metal: explicit topk NaN shuffle PSO/geometry unavailable: %s (%s); refusing fallback\n",
                names[i],error.localizedDescription.UTF8String ?: "function or resource limit");
            return 0;
        }
    }
    return 1;
}
static int indexer_topk_nan_shuffle_initialize(void) {
    for (unsigned causal = 0; causal < 2u; ++causal) {
        /* Initialization precedes g_initialized. Query the actual reference
         * PSO here; the public post-init query correctly rejects this phase. */
        NSString *name = causal ? @"kernel_argsort_f32_i32_desc_causal" :
                                  @"kernel_argsort_f32_i32_desc";
        if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
        id<MTLComputePipelineState> reference = g_native_pipelines[name];
        NSError *error = nil;
        if (!reference) {
            id<MTLFunction> fn = [g_library newFunctionWithName:name];
            if (fn) reference = [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (reference) g_native_pipelines[name] = reference;
        }
        if (!reference) {
            fprintf(stderr,"myds_metal: topk NaN shuffle reference PSO unavailable: %s (%s)\n",
                name.UTF8String,error.localizedDescription.UTF8String ?: "function unavailable");
            return 0;
        }
        const uint32_t max = (uint32_t)reference.maxTotalThreadsPerThreadgroup;
        if (!max) return 0;
        uint32_t nth = 1u;
        while (nth <= max / 2u) nth *= 2u;
        if (!myds_metal_indexer_topk_nan_shuffle_preflight(causal != 0u,nth)) return 0;
    }
    return 1;
}
static bool g_indexer_topk_threshold = false;
static myds_metal_buffer_t *g_indexer_topk_threshold_stats = NULL;
static int indexer_topk_threshold_parse_mode(void) {
    const char *value=getenv("MYDS_INDEXER_TOPK_THRESHOLD");
    if(!value||strcmp(value,"0")==0)return 0;
    if(strcmp(value,"1")==0)return 1;
    fprintf(stderr,"myds_metal: MYDS_INDEXER_TOPK_THRESHOLD must be exactly 0 or 1\n");
    return -1;
}
int myds_metal_indexer_topk_threshold_enabled(void) { return g_indexer_topk_threshold?1:0; }
int myds_metal_buffer_native_valid(const myds_metal_buffer_t *b) {
    return b&&b->mtl_buf&&b->mtl_buf.device==g_device&&b->ptr&&
        b->offset<=b->mtl_buf.length&&b->size<=b->mtl_buf.length-b->offset;
}
int myds_metal_indexer_topk_threshold_prepare(void) {
    if(!g_device||!g_library)return 0;
    const char *names[]={"kernel_myds_topk_threshold_range_parts","kernel_myds_topk_threshold_range_reduce",
        "kernel_myds_topk_threshold_hist_parts","kernel_myds_topk_threshold_hist_reduce",
        "kernel_myds_topk_threshold_compact","kernel_myds_topk_threshold_final",
        "kernel_myds_argsort_f32_i32_desc_threshold_guarded",
        "kernel_myds_argsort_merge_f32_i32_desc_threshold_guarded","kernel_argsort_f32_i32_desc"};
    if(!g_native_pipelines)g_native_pipelines=[NSMutableDictionary new];
    uint32_t original_nth=0u;
    for(int i=8;i>=0;i--) {
        NSString *name=[NSString stringWithUTF8String:names[i]];
        id<MTLComputePipelineState> p=g_native_pipelines[name];
        NSError *error=nil;
        if(!p) {
            id<MTLFunction> fn=[g_library newFunctionWithName:name];
            if(fn)p=[g_device newComputePipelineStateWithFunction:fn error:&error];
            if(p)g_native_pipelines[name]=p;
        }
        if(i==8&&p) { original_nth=1u;while(original_nth<=p.maxTotalThreadsPerThreadgroup/2u)original_nth*=2u; }
        const uint32_t threads=i==8?original_nth:i==6?original_nth:i==5||i==7?512u:256u;
        const NSUInteger dynamic=i==5?8192u:i==6?(NSUInteger)original_nth*8u:0u;
        if(!p||p.device!=g_device||p.threadExecutionWidth!=32u||!threads||
           threads>p.maxTotalThreadsPerThreadgroup||threads>g_device.maxThreadsPerThreadgroup.width||
           dynamic>g_device.maxThreadgroupMemoryLength||
           p.staticThreadgroupMemoryLength>g_device.maxThreadgroupMemoryLength-dynamic) {
            fprintf(stderr,"myds_metal: explicit topk threshold PSO/geometry unavailable: %s (%s); refusing fallback\n",
                names[i],error.localizedDescription.UTF8String?:"function or resource limit");
            return 0;
        }
    }
    if(!g_indexer_topk_threshold_stats) {
        g_indexer_topk_threshold_stats=myds_metal_buffer_alloc(9u*sizeof(uint32_t));
        if(!g_indexer_topk_threshold_stats)return 0;
        memset(g_indexer_topk_threshold_stats->ptr,0,9u*sizeof(uint32_t));
    }
    return myds_metal_buffer_native_valid(g_indexer_topk_threshold_stats);
}
myds_metal_buffer_t *myds_metal_indexer_topk_threshold_stats_buffer(void) {
    return g_indexer_topk_threshold_stats;
}
static atomic_uint_fast64_t g_batch2_q8_shared_requests = 0;
static atomic_uint_fast64_t g_batch2_q8_shared_encoded = 0;
static atomic_uint_fast64_t g_batch2_q8_shared_failures = 0;
static id<MTLComputePipelineState> batch2_q8_shared_pipeline(uint32_t nsg, uint32_t nr0);
static id<MTLComputePipelineState> batch2_q4_shared_pipeline(uint32_t nsg, uint32_t nr0);
static int g_lm_head_q8_nr0 = 2;
static id<MTLComputePipelineState> g_pso_lm_head_q8_nr4 = nil;
static atomic_uint_fast64_t g_lm_head_q8_requests = 0;
static atomic_uint_fast64_t g_lm_head_q8_encoded = 0;
static atomic_uint_fast64_t g_lm_head_q8_failures = 0;
static bool g_f16_gemv_bf16_fuse = false;
static id<MTLComputePipelineState> g_pso_f16_gemv_bf16[8] = {nil};
static atomic_uint_fast64_t g_f16_gemv_bf16_requests = 0;
static atomic_uint_fast64_t g_f16_gemv_bf16_encoded = 0;
static atomic_uint_fast64_t g_f16_gemv_bf16_failures = 0;

int myds_metal_f16_gemv_bf16_fuse_parse_mode(void) {
    const char *value = getenv("MYDS_F16_GEMV_BF16_FUSE");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr, "myds_metal: MYDS_F16_GEMV_BF16_FUSE must be exactly 0 or 1\n");
    return -1;
}

int myds_metal_f16_gemv_bf16_fuse_enabled(void) {
    return g_f16_gemv_bf16_fuse ? 1 : 0;
}

void myds_metal_f16_gemv_bf16_fuse_counters(
    myds_metal_f16_gemv_bf16_fuse_counters_t *out, bool reset) {
#define F16_BF16_COUNT(field) (reset ? atomic_exchange_explicit(&g_f16_gemv_bf16_##field, 0, memory_order_relaxed) \
                                  : atomic_load_explicit(&g_f16_gemv_bf16_##field, memory_order_relaxed))
    myds_metal_f16_gemv_bf16_fuse_counters_t value = {
        .requests = F16_BF16_COUNT(requests), .encoded = F16_BF16_COUNT(encoded),
        .failures = F16_BF16_COUNT(failures)};
    if (out) *out = value;
#undef F16_BF16_COUNT
}

static bool f16_gemv_bf16_geometry(id<MTLComputePipelineState> pso, uint32_t nsg) {
    const NSUInteger dynamic_bytes = 32u * 2u * sizeof(float);
    return g_device && pso && pso.device == g_device && nsg >= 1u && nsg <= 8u &&
        pso.threadExecutionWidth == 32u && pso.maxTotalThreadsPerThreadgroup >= 32u * nsg &&
        g_device.maxThreadsPerThreadgroup.width >= 32u &&
        g_device.maxThreadsPerThreadgroup.height >= nsg &&
        g_device.maxThreadsPerThreadgroup.depth >= 1u &&
        dynamic_bytes <= g_device.maxThreadgroupMemoryLength &&
        pso.staticThreadgroupMemoryLength <= g_device.maxThreadgroupMemoryLength - dynamic_bytes;
}

static int f16_gemv_bf16_initialize(void) {
    for (uint32_t nsg = 1u; nsg <= 8u; ++nsg) {
        MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
        short short_nsg = (short)nsg;
        [values setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
        NSError *error = nil;
        id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_mul_mv_f16_f32_4_bf16_nr2"
                                            constantValues:values error:&error];
        id<MTLComputePipelineState> pso = fn ?
            [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!f16_gemv_bf16_geometry(pso, nsg)) {
            fprintf(stderr, "myds_metal: explicit F16 GEMV BF16 fuse PSO/geometry unavailable (NSG=%u, %s); refusing fallback\n",
                nsg, error.localizedDescription.UTF8String ?: "function or resource limit");
            return 0;
        }
        g_pso_f16_gemv_bf16[nsg - 1u] = pso;
    }
    return 1;
}
static bool g_indexer_rope_quant_fuse = false;
static id<MTLComputePipelineState> g_pso_indexer_rope_quant = nil;
static atomic_uint_fast64_t g_indexer_rope_quant_requests = 0;
static atomic_uint_fast64_t g_indexer_rope_quant_encoded = 0;
static atomic_uint_fast64_t g_indexer_rope_quant_failures = 0;

int myds_metal_indexer_rope_quant_fuse_parse_mode(void) {
    const char *value = getenv("MYDS_INDEXER_ROPE_QUANT_FUSE");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr, "myds_metal: MYDS_INDEXER_ROPE_QUANT_FUSE must be exactly 0 or 1\n");
    return -1;
}

int myds_metal_indexer_rope_quant_fuse_enabled(void) {
    return g_indexer_rope_quant_fuse ? 1 : 0;
}

void myds_metal_indexer_rope_quant_fuse_counters(
    myds_metal_indexer_rope_quant_fuse_counters_t *out, bool reset) {
#define INDEXER_RQ_COUNT(field) (reset ? atomic_exchange_explicit(&g_indexer_rope_quant_##field, 0, memory_order_relaxed) \
                                     : atomic_load_explicit(&g_indexer_rope_quant_##field, memory_order_relaxed))
    myds_metal_indexer_rope_quant_fuse_counters_t value = {
        .requests = INDEXER_RQ_COUNT(requests), .encoded = INDEXER_RQ_COUNT(encoded),
        .failures = INDEXER_RQ_COUNT(failures)};
    if (out) *out = value;
#undef INDEXER_RQ_COUNT
}

static bool indexer_rope_quant_geometry(id<MTLComputePipelineState> pso) {
    return g_device && pso && pso.device == g_device &&
        pso.threadExecutionWidth == 32u && pso.maxTotalThreadsPerThreadgroup >= 32u &&
        g_device.maxThreadsPerThreadgroup.width >= 32u &&
        g_device.maxThreadsPerThreadgroup.height >= 1u &&
        g_device.maxThreadsPerThreadgroup.depth >= 1u &&
        pso.staticThreadgroupMemoryLength <= g_device.maxThreadgroupMemoryLength;
}

static int indexer_rope_quant_initialize(void) {
    NSError *error = nil;
    id<MTLFunction> function = [g_library newFunctionWithName:@"kernel_dsv41_indexer_rope_quant"];
    if (function) g_pso_indexer_rope_quant = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!indexer_rope_quant_geometry(g_pso_indexer_rope_quant)) {
        fprintf(stderr, "myds_metal: indexer RoPE quant candidate unavailable; refusing fallback (%s)\n",
            error.localizedDescription.UTF8String ?: "function or resource limit");
        return 0;
    }
    return 1;
}

static bool g_moe_q2_sum_cache = false;
static id<MTLComputePipelineState> g_pso_moe_q2_sum_cache_prepare = nil;
static id<MTLComputePipelineState> g_pso_moe_q2_sum6_cached = nil;
static id<MTLBuffer> g_moe_q2_sum_cache_buffer = nil;
static atomic_uint_fast64_t g_moe_q2_sum_cache_requests = 0;
static atomic_uint_fast64_t g_moe_q2_sum_cache_encoded = 0;
static atomic_uint_fast64_t g_moe_q2_sum_cache_failures = 0;
static bool g_moe_q2_sum_cache_fused = false;
static id<MTLComputePipelineState> g_pso_moe_iq2_pair_swiglu_sum_cache = nil;
static atomic_uint_fast64_t g_moe_q2_sum_cache_fused_requests = 0;
static atomic_uint_fast64_t g_moe_q2_sum_cache_fused_encoded = 0;
static atomic_uint_fast64_t g_moe_q2_sum_cache_fused_failures = 0;
#define MOE_Q2_SUM_CACHE_BYTES (6u * 9u * 32u * sizeof(float))

static int moe_q2_sum_cache_parse_mode(void) {
    const char *value = getenv("MYDS_MOE_Q2_SUM_CACHE");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr, "myds_metal: MYDS_MOE_Q2_SUM_CACHE must be exactly 0 or 1\n");
    return -1;
}

int myds_metal_moe_q2_sum_cache_enabled(void) { return g_moe_q2_sum_cache ? 1 : 0; }

void myds_metal_moe_q2_sum_cache_counters(myds_metal_moe_q2_sum_cache_counters_t *out, bool reset) {
#define Q2_COUNT(field) (reset ? atomic_exchange_explicit(&g_moe_q2_sum_cache_##field, 0, memory_order_relaxed) \
                             : atomic_load_explicit(&g_moe_q2_sum_cache_##field, memory_order_relaxed))
    myds_metal_moe_q2_sum_cache_counters_t value = {
        .requests = Q2_COUNT(requests), .encoded = Q2_COUNT(encoded), .failures = Q2_COUNT(failures)};
    if (out) *out = value;
#undef Q2_COUNT
}

static int moe_q2_sum_cache_fused_parse_mode(void) {
    const char *value = getenv("MYDS_MOE_Q2_SUM_CACHE_FUSED");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr, "myds_metal: MYDS_MOE_Q2_SUM_CACHE_FUSED must be exactly 0 or 1\n");
    return -1;
}

static bool moe_q2_sum_cache_modes_valid(int independent, int fused, bool constant_lut) {
    if (independent < 0 || fused < 0) return false;
    if (independent && fused) {
        fprintf(stderr, "myds_metal: Q2 sum cache and fused sum cache are mutually exclusive\n");
        return false;
    }
    if (fused && constant_lut) {
        fprintf(stderr, "myds_metal: fused Q2 sum cache requires reference IQ2 LUT0\n");
        return false;
    }
    return true;
}

int myds_metal_moe_q2_sum_cache_fused_enabled(void) { return g_moe_q2_sum_cache_fused ? 1 : 0; }

void myds_metal_moe_q2_sum_cache_fused_counters(myds_metal_moe_q2_sum_cache_fused_counters_t *out, bool reset) {
#define Q2_FUSED_COUNT(field) (reset ? atomic_exchange_explicit(&g_moe_q2_sum_cache_fused_##field, 0, memory_order_relaxed) \
                                   : atomic_load_explicit(&g_moe_q2_sum_cache_fused_##field, memory_order_relaxed))
    myds_metal_moe_q2_sum_cache_fused_counters_t value = {
        .requests = Q2_FUSED_COUNT(requests), .encoded = Q2_FUSED_COUNT(encoded), .failures = Q2_FUSED_COUNT(failures)};
    if (out) *out = value;
#undef Q2_FUSED_COUNT
}

static bool moe_q2_sum_cache_fused_geometry(void) {
    id<MTLComputePipelineState> pipelines[] = {
        g_pso_moe_iq2_pair_swiglu_sum_cache, g_pso_moe_q2_sum6_cached};
    const NSUInteger threads[] = {32u * (NSUInteger)g_moe_nsg, 32u * (NSUInteger)g_moe_down_nsg};
    const NSUInteger heights[] = {(NSUInteger)g_moe_nsg, (NSUInteger)g_moe_down_nsg};
    const NSUInteger dynamic_bytes[] = {2176u + 64u, 0u};
    if (!g_device || (g_moe_nsg != 2 && g_moe_nsg != 4) ||
        (g_moe_down_nsg != 2 && g_moe_down_nsg != 4)) return false;
    for (unsigned i = 0; i < 2u; ++i) {
        if (!pipelines[i] || pipelines[i].device != g_device || pipelines[i].threadExecutionWidth != 32u ||
            threads[i] > pipelines[i].maxTotalThreadsPerThreadgroup ||
            32u > g_device.maxThreadsPerThreadgroup.width ||
            heights[i] > g_device.maxThreadsPerThreadgroup.height ||
            dynamic_bytes[i] > g_device.maxThreadgroupMemoryLength ||
            pipelines[i].staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - dynamic_bytes[i])
            return false;
    }
    return g_device.maxThreadsPerThreadgroup.depth >= 1u;
}

static int moe_q2_sum_cache_fused_initialize(void) {
    NSError *error = nil;
    MTLFunctionConstantValues *gate_values = [MTLFunctionConstantValues new];
    short gate_nsg = (short)g_moe_nsg;
    [gate_values setConstantValue:&gate_nsg type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> producer = [g_library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_sum_cache_f32"
                                              constantValues:gate_values error:&error];
    if (producer) g_pso_moe_iq2_pair_swiglu_sum_cache = [g_device newComputePipelineStateWithFunction:producer error:&error];
    MTLFunctionConstantValues *down_values = [MTLFunctionConstantValues new];
    short down_nsg = (short)g_moe_down_nsg;
    [down_values setConstantValue:&down_nsg type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> down = [g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_cached_f32"
                                         constantValues:down_values error:&error];
    if (down) g_pso_moe_q2_sum6_cached = [g_device newComputePipelineStateWithFunction:down error:&error];
    if (!moe_q2_sum_cache_fused_geometry()) {
        fprintf(stderr, "myds_metal: explicit fused Q2 sum cache PSO/geometry unavailable (%s); refusing fallback\n",
                error.localizedDescription.UTF8String ?: "function or resource limit");
        return 0;
    }
    g_moe_q2_sum_cache_buffer = [g_device newBufferWithLength:MOE_Q2_SUM_CACHE_BYTES options:MTLResourceStorageModeShared];
    if (!g_moe_q2_sum_cache_buffer || !g_moe_q2_sum_cache_buffer.contents ||
        (uintptr_t)g_moe_q2_sum_cache_buffer.contents % 16u) {
        fprintf(stderr, "myds_metal: explicit fused Q2 sum cache scratch allocation failed; refusing fallback\n");
        return 0;
    }
    g_moe_q2_sum_cache_buffer.label = @"myds_q2_sum_cache_fused_f32";
    return 1;
}

static bool moe_q2_sum_cache_geometry(bool constant_lut) {
    const NSUInteger iq2_bytes = constant_lut ? 0u : 2176u;
    id<MTLComputePipelineState> pipelines[] = {
        g_pso_moe_pair_swiglu_iq2_xxs, g_pso_moe_q2_sum_cache_prepare, g_pso_moe_q2_sum6_cached};
    const NSUInteger threads[] = {32u * (NSUInteger)g_moe_nsg, 128u, 32u * (NSUInteger)g_moe_down_nsg};
    const NSUInteger widths[] = {32u, 128u, 32u};
    const NSUInteger heights[] = {(NSUInteger)g_moe_nsg, 1u, (NSUInteger)g_moe_down_nsg};
    const NSUInteger dynamic_bytes[] = {iq2_bytes, 0u, 0u};
    if (!g_device || (g_moe_nsg != 2 && g_moe_nsg != 4) ||
        (g_moe_down_nsg != 2 && g_moe_down_nsg != 4)) return false;
    for (unsigned i = 0; i < 3u; ++i) {
        if (!pipelines[i] || pipelines[i].threadExecutionWidth != 32u ||
            threads[i] > pipelines[i].maxTotalThreadsPerThreadgroup ||
            widths[i] > g_device.maxThreadsPerThreadgroup.width ||
            heights[i] > g_device.maxThreadsPerThreadgroup.height ||
            dynamic_bytes[i] > g_device.maxThreadgroupMemoryLength ||
            pipelines[i].staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - dynamic_bytes[i])
            return false;
    }
    return g_device.maxThreadsPerThreadgroup.depth >= 1u;
}

static int moe_q2_sum_cache_initialize(bool constant_lut) {
    NSError *error = nil;
    id<MTLFunction> prep = [g_library newFunctionWithName:@"kernel_dsv41_q2_sum_cache_prepare_f32"];
    if (prep) g_pso_moe_q2_sum_cache_prepare = [g_device newComputePipelineStateWithFunction:prep error:&error];
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    short nsg = (short)g_moe_down_nsg;
    [values setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> down = [g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_cached_f32"
                                         constantValues:values error:&error];
    if (down) g_pso_moe_q2_sum6_cached = [g_device newComputePipelineStateWithFunction:down error:&error];
    if (!moe_q2_sum_cache_geometry(constant_lut)) {
        fprintf(stderr, "myds_metal: explicit Q2 sum cache PSO/geometry unavailable (%s); refusing fallback\n",
                error.localizedDescription.UTF8String ?: "function or resource limit");
        return 0;
    }
    g_moe_q2_sum_cache_buffer = [g_device newBufferWithLength:MOE_Q2_SUM_CACHE_BYTES options:MTLResourceStorageModeShared];
    if (!g_moe_q2_sum_cache_buffer || !g_moe_q2_sum_cache_buffer.contents ||
        (uintptr_t)g_moe_q2_sum_cache_buffer.contents % 16u) {
        fprintf(stderr, "myds_metal: explicit Q2 sum cache scratch allocation failed; refusing fallback\n");
        return 0;
    }
    g_moe_q2_sum_cache_buffer.label = @"myds_q2_sum_cache_f32";
    return 1;
}

static int lm_head_q8_parse_mode(void) {
    const char *value = getenv("MYDS_LM_HEAD_Q8_NR0");
    if (!value || strcmp(value, "2") == 0) return 2;
    if (strcmp(value, "4") == 0) return 4;
    fprintf(stderr, "myds_metal: MYDS_LM_HEAD_Q8_NR0 must be exactly 2 or 4\n");
    return -1;
}

int myds_metal_lm_head_q8_nr0(void) { return g_lm_head_q8_nr0; }

void myds_metal_lm_head_q8_counters(myds_metal_lm_head_q8_counters_t *out, bool reset) {
#define LM_COUNT(field) (reset ? atomic_exchange_explicit(&g_lm_head_q8_##field, 0, memory_order_relaxed) \
                             : atomic_load_explicit(&g_lm_head_q8_##field, memory_order_relaxed))
    myds_metal_lm_head_q8_counters_t value = {
        .requests = LM_COUNT(requests), .encoded = LM_COUNT(encoded), .failures = LM_COUNT(failures)};
    if (out) *out = value;
#undef LM_COUNT
}

static id<MTLComputePipelineState> lm_head_q8_nr4_pipeline(void) {
    if (g_pso_lm_head_q8_nr4) return g_pso_lm_head_q8_nr4;
    if (!g_library || !g_device) return nil;
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    short nsg = 8;
    [values setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_mul_mv_q8_0_f32_r4"
                                        constantValues:values error:&error];
    id<MTLComputePipelineState> pipeline = fn ?
        [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
    const NSUInteger dynamic_bytes = 32u * 4u * sizeof(float);
    if (!pipeline || pipeline.threadExecutionWidth != 32u ||
        pipeline.maxTotalThreadsPerThreadgroup < 256u ||
        g_device.maxThreadsPerThreadgroup.width < 256u ||
        dynamic_bytes > g_device.maxThreadgroupMemoryLength ||
        pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - dynamic_bytes) {
        fprintf(stderr, "myds_metal: explicitly requested LM-head Q8 NR4 PSO/geometry unavailable (%s); refusing fallback\n",
                error.localizedDescription.UTF8String ?: "function or resource limit");
        return nil;
    }
    g_pso_lm_head_q8_nr4 = pipeline;
    return pipeline;
}

int myds_metal_batch2_q8_shared_mode(void) {
    const char *value = getenv("MYDS_BATCH2_Q8_SHARED");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") == 0) return 1;
    fprintf(stderr, "myds_metal: MYDS_BATCH2_Q8_SHARED must be exactly 0 or 1\n");
    return -1;
}

int myds_metal_batch2_q4_shared_mode(void) {
    const char *value = getenv("MYDS_BATCH2_Q4_SHARED");
    if (!value) value = getenv("MYDS_BATCH2_SHARED");
    if (!value || strcmp(value, "1") == 0) return 1;
    if (strcmp(value, "0") == 0) return 0;
    fprintf(stderr, "myds_metal: MYDS_BATCH2_Q4_SHARED / MYDS_BATCH2_SHARED must be exactly 0 or 1\n");
    return -1;
}

int myds_metal_batch2_dense_shared_mode(void) {
    const char *value = getenv("MYDS_BATCH2_DENSE_SHARED");
    if (!value) value = getenv("MYDS_BATCH2_SHARED");
    if (!value || strcmp(value, "1") == 0) return 1;
    if (strcmp(value, "0") == 0) return 0;
    fprintf(stderr, "myds_metal: MYDS_BATCH2_DENSE_SHARED / MYDS_BATCH2_SHARED must be exactly 0 or 1\n");
    return -1;
}

void myds_metal_batch2_q8_shared_counters(
    myds_metal_batch2_q8_shared_counters_t *out, bool reset) {
#define SHARED_COUNT(field) (reset ? atomic_exchange_explicit(&g_batch2_q8_shared_##field, 0, memory_order_relaxed) \
                                 : atomic_load_explicit(&g_batch2_q8_shared_##field, memory_order_relaxed))
    myds_metal_batch2_q8_shared_counters_t value = {
        .requests = SHARED_COUNT(requests), .encoded = SHARED_COUNT(encoded), .failures = SHARED_COUNT(failures)};
    if (out) *out = value;
#undef SHARED_COUNT
}
struct native_kernel_cache_entry {
    char name[64];
    id<MTLComputePipelineState> pso;
};
static struct native_kernel_cache_entry g_native_pso_cache[64];
static size_t g_native_pso_cache_count = 0;

/* Keep the ordered numeric tuple on the hot path. The dictionary remains the
 * backing store when this bounded cache replaces an entry or a name is long. */
struct specialized_kernel_cache_entry {
    char name[64];
    size_t n_constants;
    myds_metal_constant_t constants[64];
    id<MTLComputePipelineState> pso;
};
static struct specialized_kernel_cache_entry g_specialized_pso_cache[64];
static size_t g_specialized_pso_cache_count = 0;
static size_t g_specialized_pso_cache_next = 0;

static id<MTLComputePipelineState> specialized_pipeline_lookup(const char *name,
    const myds_metal_constant_t *constants, size_t n_constants) {
    for (size_t i = 0; i < g_specialized_pso_cache_count; ++i) {
        const struct specialized_kernel_cache_entry *entry = &g_specialized_pso_cache[i];
        if (entry->n_constants != n_constants || strcmp(entry->name, name) != 0) continue;
        size_t j = 0;
        for (; j < n_constants; ++j) {
            /* Compare fields rather than struct padding. Repeated indices are
             * order-sensitive because Metal uses the last assigned value. */
            if (entry->constants[j].index != constants[j].index ||
                entry->constants[j].is_bool != constants[j].is_bool ||
                entry->constants[j].value != constants[j].value) break;
        }
        if (j == n_constants) return entry->pso;
    }
    return nil;
}

static void specialized_pipeline_cache(const char *name,
    const myds_metal_constant_t *constants, size_t n_constants,
    id<MTLComputePipelineState> pipeline) {
    if (strlen(name) >= sizeof(g_specialized_pso_cache[0].name)) return;
    struct specialized_kernel_cache_entry *entry = &g_specialized_pso_cache[g_specialized_pso_cache_next];
    strcpy(entry->name, name);
    entry->n_constants = n_constants;
    for (size_t i = 0; i < n_constants; ++i) entry->constants[i] = constants[i];
    entry->pso = pipeline;
    if (g_specialized_pso_cache_count < 64) ++g_specialized_pso_cache_count;
    g_specialized_pso_cache_next = (g_specialized_pso_cache_next + 1u) % 64u;
}


/* Batch command buffer state */
static id<MTLCommandBuffer>         g_batch_cb = nil;
static id<MTLComputeCommandEncoder> g_batch_enc = nil;
/* An event can wake the CPU before the rest of its command buffer finishes. */
static NSMutableArray<id<MTLCommandBuffer>> *g_inflight_cbs = nil;
static NSMutableArray<id<MTLBuffer>>        *g_batch_released_buffers = nil;
static uint64_t                             g_batch_commit_count = 0;
static int                                  g_auto_barrier = 1;

static inline id<MTLCommandBuffer> make_command_buffer(void) {
    if (!g_queue) return nil;
    static int s_has_unretained = -1;
    if (__builtin_expect(s_has_unretained < 0, 0)) {
        s_has_unretained = [g_queue respondsToSelector:@selector(commandBufferWithUnretainedReferences)] ? 1 : 0;
    }
    if (s_has_unretained) {
        return [g_queue commandBufferWithUnretainedReferences];
    }
    return [g_queue commandBuffer];
}

int myds_metal_indexer_topk_threshold_stats(uint32_t words[9],bool reset) {
    if(!words)return 0;
    memset(words,0,9u*sizeof(uint32_t));
    if(g_batch_cb)return 0;
    for(id<MTLCommandBuffer> cb in g_inflight_cbs)
        if(cb.status!=MTLCommandBufferStatusCompleted)return 0;
    if(!g_indexer_topk_threshold_stats)return 1;
    memcpy(words,g_indexer_topk_threshold_stats->ptr,9u*sizeof(uint32_t));
    if(reset)memset(g_indexer_topk_threshold_stats->ptr,0,9u*sizeof(uint32_t));
    return 1;
}

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

int myds_metal_get_auto_barrier(void) {
    return g_auto_barrier;
}

static void close_compute_encoder(void) {
    if (g_batch_enc) {
        [g_batch_enc endEncoding];
        g_batch_enc = nil;
    }
}

static int g_concurrent_mode = 0;

void myds_metal_set_concurrent(int enable) {
    static int s_concurrent_env = -1;
    if (__builtin_expect(s_concurrent_env < 0, 0)) {
        const char *env = getenv("MYDS_METAL_CONCURRENT");
        s_concurrent_env = (env && atoi(env) != 0) ? 1 : 0;
    }
    const int new_use = (enable > 0) || (s_concurrent_env > 0);
    const int old_use = (g_concurrent_mode > 0) || (s_concurrent_env > 0);
    if (old_use != new_use) {
        close_compute_encoder();
    }
    g_concurrent_mode = enable;
}

int myds_metal_get_concurrent(void) {
    return g_concurrent_mode;
}

static id<MTLComputeCommandEncoder> get_compute_encoder(id<MTLCommandBuffer> cb) {
    if (cb == g_batch_cb && g_batch_enc) {
        return g_batch_enc;
    }
    static int s_concurrent_env = -1;
    if (s_concurrent_env < 0) {
        const char *env = getenv("MYDS_METAL_CONCURRENT");
        s_concurrent_env = (env && atoi(env) != 0) ? 1 : 0;
    }
    int use_concurrent = (g_concurrent_mode > 0) || (s_concurrent_env > 0);
    id<MTLComputeCommandEncoder> enc = nil;
    if (use_concurrent && [cb respondsToSelector:@selector(computeCommandEncoderWithDispatchType:)]) {
        enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    } else {
        enc = [cb computeCommandEncoder];
    }
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

void myds_metal_barrier_buffers(const myds_metal_buffer_t *const *buffers, size_t count) {
    if (!g_batch_cb || !buffers || count == 0) return;
    id<MTLComputeCommandEncoder> enc = get_compute_encoder(g_batch_cb);
    if (!enc) return;
    if (![enc respondsToSelector:@selector(memoryBarrierWithResources:count:)]) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return;
    }
    id<MTLResource> res[count];
    NSUInteger n = 0;
    for (size_t i = 0; i < count; i++) {
        if (buffers[i] && buffers[i]->mtl_buf) {
            id<MTLBuffer> mb = buffers[i]->mtl_buf;
            bool duplicate = false;
            for (NSUInteger j = 0; j < n; j++) {
                if (res[j] == mb) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                res[n++] = mb;
            }
        }
    }
    if (n > 0) {
        [enc memoryBarrierWithResources:res count:n];
    }
}

void myds_metal_barrier_buffer(const myds_metal_buffer_t *buf) {
    if (buf) {
        const myds_metal_buffer_t *bufs[1] = { buf };
        myds_metal_barrier_buffers(bufs, 1);
    }
}

static int command_buffer_result(id<MTLCommandBuffer> cb) {
    if (cb && cb.status == MTLCommandBufferStatusCompleted) return 1;
    fprintf(stderr, "myds_metal: command buffer failed: %s\n",
            cb.error ? [[cb.error localizedDescription] UTF8String] : "command buffer did not complete");
    return 0;
}

static int wait_inflight_commands(void) {
    if (!g_inflight_cbs || [g_inflight_cbs count] == 0) {
        if (g_batch_released_buffers && [g_batch_released_buffers count] > 0) {
            [g_batch_released_buffers removeAllObjects];
        }
        return 1;
    }
    int ok = 1;
    for (id<MTLCommandBuffer> cb in g_inflight_cbs) {
        if (cb.status != MTLCommandBufferStatusCompleted) {
            [cb waitUntilCompleted];
        }
        if (!command_buffer_result(cb)) ok = 0;
    }
    [g_inflight_cbs removeAllObjects];
    if (g_batch_released_buffers && [g_batch_released_buffers count] > 0) {
        [g_batch_released_buffers removeAllObjects];
    }
    return ok;
}

static int wait_command_buffer(id<MTLCommandBuffer> cb) {
    [cb waitUntilCompleted];
    int ok = command_buffer_result(cb);
    /* Keep errors from earlier asynchronous submissions observable, too. */
    if (!wait_inflight_commands()) ok = 0;
    if (g_batch_released_buffers && [g_batch_released_buffers count] > 0) {
        [g_batch_released_buffers removeAllObjects];
    }
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

    const int f16_gemv_bf16_fuse = myds_metal_f16_gemv_bf16_fuse_parse_mode();
    if (f16_gemv_bf16_fuse < 0) return 0;
    const int indexer_rope_quant_fuse = myds_metal_indexer_rope_quant_fuse_parse_mode();
    if (indexer_rope_quant_fuse < 0) return 0;

    const int indexer_head_slots = indexer_head_slots_parse_mode();
    if (indexer_head_slots < 0) return 0;
    const int indexer_topk_threshold=indexer_topk_threshold_parse_mode();
    if(indexer_topk_threshold<0)return 0;
    const int indexer_topk_nan_shuffle = indexer_topk_nan_shuffle_parse_mode();
    if (indexer_topk_nan_shuffle < 0) return 0;
    if (indexer_topk_threshold && indexer_topk_nan_shuffle) {
        fprintf(stderr,"myds_metal: threshold and NaN shuffle experiments must be isolated\n");
        return 0;
    }

    const int moe_q2_sum_cache = moe_q2_sum_cache_parse_mode();
    if (moe_q2_sum_cache < 0) return 0;
    const int moe_q2_sum_cache_fused = moe_q2_sum_cache_fused_parse_mode();
    if (!moe_q2_sum_cache_modes_valid(moe_q2_sum_cache, moe_q2_sum_cache_fused, false)) return 0;

    const int lm_head_q8_nr0 = lm_head_q8_parse_mode();
    if (lm_head_q8_nr0 < 0) return 0;
    const int batch2_q8_shared = myds_metal_batch2_q8_shared_mode();
    if (batch2_q8_shared < 0) return 0;
    const int batch2_q4_shared = myds_metal_batch2_q4_shared_mode();
    if (batch2_q4_shared < 0) return 0;

    /* PSO function constants are initialization-time settings. An absent down
     * override inherits the existing IQ2 setting, including its legacy parser.
     * Reject malformed overrides before allocating any Metal resources. */
    const char *nsg_env = getenv("MYDS_MOE_NSG");
    const int moe_nsg = (nsg_env && atoi(nsg_env) == 4) ? 4 : 2;
    const char *down_nsg_env = getenv("MYDS_MOE_DOWN_NSG");
    int down_nsg = moe_nsg;
    if (down_nsg_env) {
        if (strcmp(down_nsg_env, "2") == 0) down_nsg = 2;
        else if (strcmp(down_nsg_env, "4") == 0) down_nsg = 4;
        else {
            fprintf(stderr, "myds_metal: MYDS_MOE_DOWN_NSG must be 2 or 4\n");
            return 0;
        }
    }
    const char *iq2_lut_env = getenv("MYDS_MOE_IQ2_CONSTANT_LUT");
    if (iq2_lut_env && strcmp(iq2_lut_env, "0") != 0 && strcmp(iq2_lut_env, "1") != 0) {
        fprintf(stderr, "myds_metal: MYDS_MOE_IQ2_CONSTANT_LUT must be 0 or 1\n");
        return 0;
    }
    const bool moe_iq2_constant_lut = iq2_lut_env && strcmp(iq2_lut_env, "1") == 0;
    if (!moe_q2_sum_cache_modes_valid(moe_q2_sum_cache, moe_q2_sum_cache_fused, moe_iq2_constant_lut)) return 0;
    g_moe_nsg = moe_nsg;
    g_moe_down_nsg = down_nsg;

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
            @"topk_threshold.metal",
            @"cpy.metal",
            @"concat.metal",
            @"get_rows.metal",
            @"sum_rows.metal",
            @"softmax.metal",
            @"repeat.metal",
            @"glu.metal",
            @"norm.metal",
            @"bin.metal",
            @"set_rows.metal",
            @"myds_chunk.metal",
            /* ★ 这里是【显式清单】，不是扫目录 —— 新增 .metal 忘了登记，
             *   表象是「内核找不到」，很容易误判成内核写错。
             *   visenc = DeepSeek V4.1 视觉编码器专用内核。 */
            @"visenc.metal"
        ];

        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *exec_path = [[NSProcessInfo processInfo] arguments].firstObject;
        NSString *exec_dir = [exec_path stringByDeletingLastPathComponent];

        /* Check for precompiled metallib to eliminate ~500ms runtime shader compile jitter */
        NSArray<NSString *> *metallib_search_paths = @[
            [NSString stringWithFormat:@"%s/myds.metallib", dir],
            [NSString stringWithFormat:@"./bin/myds.metallib"],
            [NSString stringWithFormat:@"./build/myds.metallib"],
            [NSString stringWithFormat:@"./metal/myds.metallib"],
            [NSString stringWithFormat:@"%@/myds.metallib", exec_dir],
            [NSString stringWithFormat:@"%@/../bin/myds.metallib", exec_dir]
        ];

        for (NSString *lib_path in metallib_search_paths) {
            if ([fm fileExistsAtPath:lib_path]) {
                NSError *lib_err = nil;
                g_library = [g_device newLibraryWithURL:[NSURL fileURLWithPath:lib_path] error:&lib_err];
                if (g_library) {
                    /* Verify essential kernels exist in the precompiled library */
                    NSArray<NSString *> *required_kernels = @[
                        @"kernel_add2_bf16_f32",
                        @"kernel_dsv41_bf16_linear",
                        @"kernel_dsv41_pack_bf16",
                        @"kernel_dsv41_stage_bf16_f16",
                        @"kernel_dsv41_get_rows_bf16",
                        @"kernel_rms_norm_mul_f32_4",
                        @"kernel_rms_norm_f32_4",
                        @"kernel_swiglu_flat_f32",
                        @"kernel_add2_f32",
                        @"kernel_dsv4_hc_split_sinkhorn",
                        @"kernel_dsv4_hc_weighted_sum",
                        @"kernel_dsv4_hc_expand4",
                        @"kernel_dsv41_engram_add",
                        @"kernel_dsv41_rope",
                        @"kernel_dsv41_rope_round",
                        @"kernel_compute_copy_u8",
                        @"kernel_compute_copy_u32",
                        @"kernel_compute_copy_u32x4",
                        @"kernel_soft_max_f32",
                        @"kernel_dsv4_moe_swiglu_weight",
                        @"kernel_dsv4_moe_sum6_f32",
                        /* V4.1 prefill, indexer & flash attention */
                        @"kernel_myds_mask_chunk_window",
                        @"kernel_myds_mask_chunk_fill",
                        @"kernel_myds_mask_chunk_scatter",
                        @"kernel_myds_attn_gather_comp",
                        @"kernel_myds_attn_gather_comp_bf16",
                        @"kernel_dsv41_candidate_filter",
                        @"kernel_dsv41_candidate_blocks",
                        @"kernel_dsv41_quantize",
                        @"kernel_dsv41_pool2",
                        @"kernel_glm_indexer_score_one",
                        @"kernel_glm_indexer_score_one_direct",
                        @"kernel_glm_indexer_score_chunk_direct",
                        @"kernel_dsv4_indexer_scores_tiled_f32",
                        @"kernel_dsv4_indexer_scores_tiled_bf16",
                        @"kernel_argsort_f32_i32_desc",
                        @"kernel_argsort_f32_i32_desc_causal",
                        @"kernel_argsort_merge_f32_i32_desc",
                        @"kernel_argsort_merge_f32_i32_desc_causal",
                        @"kernel_dsv4_topk_mask",
                        @"kernel_dsv4_topk_mask_scatter",
                        @"kernel_get_rows_f32",
                        @"kernel_cpy_contig_f32_f16_4",
                        @"kernel_flash_attn_ext_vec_f16_dk512_dv512",
                        @"kernel_flash_attn_ext_pad",
                        @"kernel_flash_attn_ext_vec_reduce",
                        @"kernel_dsv41_copy_window_f32_to_f16",
                        @"kernel_dsv41_get_rows_bf16_to_f16",
                        /* V4.1 MoE & routing */
                        @"kernel_mul_mv_id_iq2_xxs_f32",
                        @"kernel_mul_mv_id_q2_K_f32",
                        @"kernel_myds_v41_route_transform",
                        @"kernel_myds_v41_route_prepare",
                        @"kernel_myds_v41_route_sum",
                        @"kernel_myds_v41_route_divide",
                        @"kernel_myds_v41_route_scale",
                        @"kernel_myds_v41_route_finish",
                        @"kernel_myds_v41_ids_f32",
                        @"kernel_myds_dspark_route_fused_128",
                        /* V4.1 Vision encoder */
                        @"kernel_visenc_round",
                        @"kernel_visenc_bias",
                        @"kernel_visenc_rmsnorm",
                        @"kernel_visenc_qkv_rope",
                        @"kernel_visenc_attention",
                        @"kernel_visenc_bias_residual41",
                        @"kernel_visenc_swiglu_split41",
                        @"kernel_visenc_add_residual",
                        @"kernel_visenc_aligner_reorder",
                        @"kernel_visenc_gelu_bias"
                    ];
                    BOOL all_present = YES;
                    for (NSString *kname in required_kernels) {
                        id<MTLFunction> test_fn = [g_library newFunctionWithName:kname];
                        if (!test_fn) {
                            myds_log(MYDS_LOG_WARN, "myds_metal: precompiled library %s is missing %s, rebuilding from source",
                                     [lib_path UTF8String], [kname UTF8String]);
                            all_present = NO;
                            break;
                        }
                    }
                    if (!all_present) {
                        g_library = nil;
                    } else {
                        myds_log(MYDS_LOG_INFO, "myds_metal: loaded precompiled library from %s (all required V4.1 kernels verified)", [lib_path UTF8String]);
                        break;
                    }
                }
            }
        }

        if (!g_library) {
            NSMutableString *full_source = [NSMutableString stringWithUTF8String:g_metal_base_preamble];
            NSString *dir_used = nil;
            for (NSString *filename in shader_files) {
                NSString *path = [NSString stringWithFormat:@"%s/%@", dir, filename];
                if (![fm fileExistsAtPath:path]) {
                    path = [NSString stringWithFormat:@"./metal/%@", filename];
                }
                if (![fm fileExistsAtPath:path]) {
                    path = [NSString stringWithFormat:@"../metal/%@", filename];
                }
                if (![fm fileExistsAtPath:path]) {
                    path = [NSString stringWithFormat:@"%@/metal/%@", exec_dir, filename];
                    if (![fm fileExistsAtPath:path]) {
                        path = [NSString stringWithFormat:@"%@/../metal/%@", exec_dir, filename];
                    }
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
                if (!dir_used) dir_used = [[path stringByDeletingLastPathComponent] copy];
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
            if (g_library)
                myds_log(MYDS_LOG_INFO, "myds_metal: shaders loaded from %s (%lu files)",
                         dir_used ? [dir_used UTF8String] : "?", (unsigned long)[shader_files count]);
            if (!g_library) {
                fprintf(stderr, "myds_metal: Metal shader compilation failed: %s\n",
                        [[compile_err localizedDescription] UTF8String]);
                g_queue = nil;
                g_device = nil;
                return 0;
            }
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
        fn = [g_library newFunctionWithName:@"kernel_myds_v41_hc_sum_rmsnorm"];
        if (fn) g_pso_v41_hc_sum_rmsnorm = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_dsv4_hc_expand4"];
        if (fn) g_pso_v41_hc_expand = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_dsv4_moe_add_hc_expand4"];
        if (fn) g_pso_v41_moe_add_hc_expand = [g_device newComputePipelineStateWithFunction:fn error:nil];
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
        fn = [g_library newFunctionWithName:@"kernel_add2_bf16_f32"];
        if (fn) {
            g_pso_add_bf16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
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
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_q4_K_dense_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_mul_mv_q4_K = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_q4_K_dense_f32_bf16" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_mul_mv_q4_K_dense_bf16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        /* 9. Build Routed MoE pipelines */
        MTLFunctionConstantValues *moe_mv_id_constants = [[MTLFunctionConstantValues alloc] init];
        short moe_id_nsg = (short)g_moe_nsg;
        [moe_mv_id_constants setConstantValue:&moe_id_nsg type:MTLDataTypeShort atIndex:600];
        MTLFunctionConstantValues *moe_down_constants = [[MTLFunctionConstantValues alloc] init];
        short moe_down_nsg = (short)g_moe_down_nsg;
        [moe_down_constants setConstantValue:&moe_down_nsg type:MTLDataTypeShort atIndex:600];

        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_f32" constantValues:moe_mv_id_constants error:nil];
        if (fn) {
            g_pso_moe_mv_id_iq2_xxs = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_f32" constantValues:moe_down_constants error:nil];
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
        NSString *iq2_pair_name = moe_iq2_constant_lut ?
            @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_constant_lut_f32" :
            @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32";
        fn = [g_library newFunctionWithName:iq2_pair_name constantValues:moe_mv_id_constants error:&pso_err];
        if (fn) {
            g_pso_moe_pair_swiglu_iq2_xxs = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_pair_swiglu_iq2_xxs) {
                printf("myds_metal: %s PSO error: %s\n", [iq2_pair_name UTF8String], [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: %s fn error: %s\n", [iq2_pair_name UTF8String], [[pso_err localizedDescription] UTF8String]);
        }
        /* Explicit opt-in must not report successful initialization while
         * silently selecting the reference kernel or an unrelated MM path. */
        if (moe_iq2_constant_lut && !g_pso_moe_pair_swiglu_iq2_xxs) {
            fprintf(stderr, "myds_metal: explicitly requested IQ2 constant LUT kernel unavailable\n");
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }

        pso_err = nil;
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32" constantValues:moe_down_constants error:&pso_err];
        if (fn) {
            g_pso_moe_sum6_q2_k = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_sum6_q2_k) {
                printf("myds_metal: kernel_mul_mv_id_q2_K_sum6_f32 PSO error: %s\n", [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: kernel_mul_mv_id_q2_K_sum6_f32 fn error: %s\n", [[pso_err localizedDescription] UTF8String]);
        }

        pso_err = nil;
        NSString *iq2_pair_batch_name = moe_iq2_constant_lut ?
            @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_batch_constant_lut_f32" :
            @"kernel_mul_mv_id_iq2_xxs_pair_swiglu_batch_f32";
        fn = [g_library newFunctionWithName:iq2_pair_batch_name constantValues:moe_mv_id_constants error:&pso_err];
        if (fn) {
            g_pso_moe_pair_swiglu_batch_iq2_xxs = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_pair_swiglu_batch_iq2_xxs) {
                printf("myds_metal: %s PSO error: %s\n", [iq2_pair_batch_name UTF8String], [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: %s fn error: %s\n", [iq2_pair_batch_name UTF8String], [[pso_err localizedDescription] UTF8String]);
        }

        pso_err = nil;
        fn = [g_library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_batch_f32" constantValues:moe_down_constants error:&pso_err];
        if (fn) {
            g_pso_moe_sum6_batch_q2_k = [g_device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!g_pso_moe_sum6_batch_q2_k) {
                printf("myds_metal: kernel_mul_mv_id_q2_K_sum6_batch_f32 PSO error: %s\n", [[pso_err localizedDescription] UTF8String]);
            }
        } else {
            printf("myds_metal: kernel_mul_mv_id_q2_K_sum6_batch_f32 fn error: %s\n", [[pso_err localizedDescription] UTF8String]);
        }

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_id_map0_ne20_6"];
        if (fn) {
            g_pso_moe_mm_map0_ne20_6 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        const char *moe_tail_env = getenv("MYDS_MOE_IQ2_TAIL_CULL");
        const bool moe_tail = !moe_tail_env || strcmp(moe_tail_env, "1") == 0;
        NSString *moe_pair_name = moe_tail ?
            @"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_tail_cull" :
            @"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16";
        fn = [g_library newFunctionWithName:moe_pair_name];
        if (fn) {
            g_pso_moe_mm_iq2_pair_swiglu_f16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }
        if (moe_tail && !moe_tail_env && !g_pso_moe_mm_iq2_pair_swiglu_f16) {
            fn = [g_library newFunctionWithName:@"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16"];
            if (fn) g_pso_moe_mm_iq2_pair_swiglu_f16 =
                [g_device newComputePipelineStateWithFunction:fn error:nil];
        }
        if (moe_tail_env && moe_tail && !g_pso_moe_mm_iq2_pair_swiglu_f16) {
            fprintf(stderr, "myds_metal: requested IQ2 MoE tail kernel unavailable\n");
            /* Initialization has not encoded any work. Let the common cleanup
             * release every partially created PSO/cache/queue before returning,
             * including when the caller retries with another library. */
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
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

        fn = [g_library newFunctionWithName:@"kernel_dsv4_attn_out_low_q8_0_bf16_r4" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_attn_out_low_q8_0_bf16_r4 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        fn = [g_library newFunctionWithName:@"kernel_dsv4_attn_out_low_q4_K_f32" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_attn_out_low_q4_K = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }
        fn = [g_library newFunctionWithName:@"kernel_dsv4_attn_out_low_q4_K_bf16" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_attn_out_low_q4_K_bf16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }
        fn = [g_library newFunctionWithName:@"kernel_dsv4_attn_out_low_q8_0_bf16_shared2" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_attn_out_low_q8_0_bf16_shared2 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }
        fn = [g_library newFunctionWithName:@"kernel_dsv4_attn_out_low_q4_K_bf16_shared2" constantValues:mv_constants error:nil];
        if (fn) {
            g_pso_attn_out_low_q4_K_bf16_shared2 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        }

        MTLFunctionConstantValues *mm_c_bco1 = [[MTLFunctionConstantValues alloc] init];
        [mm_c_bco1 setConstantValue:&b_false type:MTLDataTypeBool atIndex:700];
        [mm_c_bco1 setConstantValue:&b_true type:MTLDataTypeBool atIndex:701];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_q8_0_f32" constantValues:mm_c_bco0 error:nil];
        if (fn) g_pso_mul_mm_q8_0 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_mul_mm_q8_0_f32" constantValues:mm_c_bco1 error:nil];
        if (fn) g_pso_mul_mm_q8_0_bco = [g_device newComputePipelineStateWithFunction:fn error:nil];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_q4_K_f32" constantValues:mm_c_bco0 error:nil];
        if (fn) g_pso_mul_mm_q4_K = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_mul_mm_q4_K_f32" constantValues:mm_c_bco1 error:nil];
        if (fn) g_pso_mul_mm_q4_K_bco = [g_device newComputePipelineStateWithFunction:fn error:nil];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f16_f32" constantValues:mm_c_bco0 error:nil];
        if (fn) g_pso_mul_mm_f16 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f16_f32" constantValues:mm_c_bco1 error:nil];
        if (fn) g_pso_mul_mm_f16_bco = [g_device newComputePipelineStateWithFunction:fn error:nil];

        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f32_f32_full" constantValues:mm_c_bco0 error:nil];
        if (fn) g_pso_mul_mm_f32 = [g_device newComputePipelineStateWithFunction:fn error:nil];
        fn = [g_library newFunctionWithName:@"kernel_mul_mm_f32_f32_full" constantValues:mm_c_bco1 error:nil];
        if (fn) g_pso_mul_mm_f32_bco = [g_device newComputePipelineStateWithFunction:fn error:nil];

        /* Verify all essential pipelines were successfully created */
        if (!g_pso_rms_norm || !g_pso_v41_norm_plain || !g_pso_v41_bf16 ||
            !g_pso_v41_hc_split || !g_pso_v41_hc_sum || !g_pso_v41_hc_expand ||
            !g_pso_v41_engram_add || !g_pso_compute_copy_u8 || !g_pso_compute_copy_u32 ||
            !g_pso_compute_copy_u32x4 || !g_pso_swiglu || !g_pso_add ||
            !g_pso_add_bf16 || !g_pso_softmax || !g_pso_moe_swiglu_weight ||
            !g_pso_moe_sum6 || !g_pso_moe_mv_id_iq2_xxs || !g_pso_moe_mv_id_q2_k) {
            fprintf(stderr, "myds_metal: failed to create one or more required compute pipeline states\n");
            /* Match explicit-tail failure cleanup so a retry cannot retain PSOs
             * created from the previous device/library. No work was encoded. */
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }

        /* An explicit experiment requires all of its named PSOs. Default0
         * never asks an older library for these optional symbols. */
        if (batch2_q8_shared) {
            for (uint32_t nsg = 4; nsg <= 8; nsg += 4)
                for (uint32_t nr0 = 2; nr0 <= 4; nr0 += 2)
                    if (!batch2_q8_shared_pipeline(nsg, nr0)) {
                        g_initialized = true;
                        myds_metal_cleanup();
                        return 0;
                    }
        }
        if (batch2_q4_shared) {
            for (uint32_t nsg = 4; nsg <= 8; nsg += 4)
                for (uint32_t nr0 = 2; nr0 <= 4; nr0 += 2)
                    (void)batch2_q4_shared_pipeline(nsg, nr0);
        }
        if (lm_head_q8_nr0 == 4 && !lm_head_q8_nr4_pipeline()) {
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }
        if (f16_gemv_bf16_fuse && !f16_gemv_bf16_initialize()) {
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }
        if (indexer_rope_quant_fuse && !indexer_rope_quant_initialize()) {
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }
        if (moe_q2_sum_cache && !moe_q2_sum_cache_initialize(moe_iq2_constant_lut)) {
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }
        if (moe_q2_sum_cache_fused && !moe_q2_sum_cache_fused_initialize()) {
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }
        if (indexer_head_slots && !indexer_head_slots_initialize()) {
            g_initialized = true;
            myds_metal_cleanup();
            return 0;
        }
        g_indexer_head_slots = indexer_head_slots != 0;
        if(indexer_topk_threshold&&!myds_metal_indexer_topk_threshold_prepare()) {
            g_initialized=true;myds_metal_cleanup();return 0;
        }
        g_indexer_topk_threshold=indexer_topk_threshold!=0;
        if (indexer_topk_nan_shuffle && !indexer_topk_nan_shuffle_initialize()) {
            g_initialized = true; myds_metal_cleanup(); return 0;
        }
        g_indexer_topk_nan_shuffle = indexer_topk_nan_shuffle != 0;
        g_lm_head_q8_nr0 = lm_head_q8_nr0;
        g_f16_gemv_bf16_fuse = f16_gemv_bf16_fuse != 0;
        g_indexer_rope_quant_fuse = indexer_rope_quant_fuse != 0;
        g_moe_iq2_constant_lut = moe_iq2_constant_lut;
        g_moe_q2_sum_cache = moe_q2_sum_cache != 0;
        g_moe_q2_sum_cache_fused = moe_q2_sum_cache_fused != 0;
        g_initialized = true;
        MYDS_LOG_I("myds_metal: Initialized successfully on GPU: %s", [[g_device name] UTF8String]);
        MYDS_LOG_I("Indexer head slots: %d (%s; explicit ordinary scalar scope)",
                   g_indexer_head_slots, g_indexer_head_slots ? "F32 candidate" : "reference");
        MYDS_LOG_I("Indexer topk threshold: %d (%s; explicit ordinary scalar scope)",
                   g_indexer_topk_threshold,g_indexer_topk_threshold?"F32 candidate":"reference");
        MYDS_LOG_I("Indexer topk NaN shuffle: %d (%s; reference partition geometry)",
                   g_indexer_topk_nan_shuffle,g_indexer_topk_nan_shuffle?"SIMD candidate":"reference");
        MYDS_LOG_I("Routed MoE geometry: IQ2 NSG=%d, Q2 down NSG=%d (%s)",
                   g_moe_nsg, g_moe_down_nsg, down_nsg_env ? "explicit" : "inherited");
        MYDS_LOG_I("Routed MoE IQ2 LUT: %s", g_moe_iq2_constant_lut ? "constant (explicit)" : "threadgroup (reference)");
        MYDS_LOG_I("Batch2 Q8 shared weights: %s", batch2_q8_shared ? "enabled (explicit; TAIL, SHARED, QKV)" : "disabled (scalar reference)");
        MYDS_LOG_I("LM-head Q8 rows per threadgroup: %d (%s; single token, large output, NSG8)",
                   g_lm_head_q8_nr0, lm_head_q8_nr0 == 4 ? "explicit candidate" : "reference");
        MYDS_LOG_I("F16 GEMV BF16 fuse: %d (%s; explicit ordinary scalar scope)",
                   g_f16_gemv_bf16_fuse, g_f16_gemv_bf16_fuse ? "fused store" : "reference");
        MYDS_LOG_I("Indexer RoPE quant fuse: %d (%s; explicit ordinary scalar scope)",
                   g_indexer_rope_quant_fuse, g_indexer_rope_quant_fuse ? "SIMD candidate" : "reference");
        MYDS_LOG_I("Routed MoE Q2 sum cache: %d (%s; explicit ordinary scalar scope)",
                   g_moe_q2_sum_cache, g_moe_q2_sum_cache ? "F32 candidate" : "reference");
        MYDS_LOG_I("Routed MoE Q2 sum cache fused: %d (%s; explicit ordinary scalar scope)",
                   g_moe_q2_sum_cache_fused, g_moe_q2_sum_cache_fused ? "F32 fused candidate" : "reference");
        return 1;
    }
}

void myds_metal_cleanup(void) {
    if (!g_initialized) return;
    (void)myds_metal_synchronize();
    g_pso_indexer_head_slots = nil;
    g_indexer_head_slots = false;
    g_indexer_topk_nan_shuffle = false;
    g_indexer_topk_threshold=false;
    myds_metal_buffer_free(g_indexer_topk_threshold_stats);
    g_indexer_topk_threshold_stats=NULL;
    g_pso_rms_norm = nil;
    g_pso_v41_norm_plain = nil;
    g_pso_v41_bf16 = nil;
    g_pso_v41_hc_split = nil;
    g_pso_v41_hc_sum = nil;
    g_pso_v41_hc_sum_rmsnorm = nil;
    g_pso_v41_hc_expand = nil;
    g_pso_v41_moe_add_hc_expand = nil;
    g_pso_v41_engram_add = nil;
    g_pso_swiglu = nil;
    g_pso_add = nil;
    g_pso_add_bf16 = nil;
    g_pso_mul_mv_q8_0 = nil;
    g_pso_mul_mv_f32 = nil;
    g_pso_mul_mv_f16 = nil;
    g_pso_mul_mv_q4_K = nil;
    g_pso_mul_mv_q4_K_dense_bf16 = nil;
    g_pso_softmax = nil;
    g_pso_moe_mv_id_iq2_xxs = nil;
    g_pso_moe_mv_id_q2_k = nil;
    g_pso_moe_swiglu_weight = nil;
    g_pso_moe_sum6 = nil;
    g_pso_moe_pair_swiglu_iq2_xxs = nil;
    g_pso_moe_sum6_q2_k = nil;
    g_pso_moe_pair_swiglu_batch_iq2_xxs = nil;
    g_pso_moe_sum6_batch_q2_k = nil;
    g_pso_moe_q2_sum_cache_prepare = nil;
    g_pso_moe_q2_sum6_cached = nil;
    g_moe_q2_sum_cache_buffer = nil;
    g_moe_q2_sum_cache = false;
    g_pso_moe_iq2_pair_swiglu_sum_cache = nil;
    g_moe_q2_sum_cache_fused = false;
    g_pso_moe_mm_map0_ne20_6 = nil;
    g_pso_moe_mm_iq2_pair_swiglu_f16 = nil;
    g_pso_moe_mm_q2_k_f16 = nil;
    g_pso_moe_mm_q2_k_f32 = nil;
    g_moe_id_map_buffer = nil;
    g_moe_id_map_bytes = 0;
    g_moe_nsg = 2;
    g_moe_down_nsg = 2;
    g_moe_iq2_constant_lut = false;
    g_pso_attn_out_low_q8_0 = nil;
    g_pso_attn_out_low_q8_0_bf16_r4 = nil;
    g_pso_attn_out_low_q4_K = nil;
    g_pso_attn_out_low_q4_K_bf16 = nil;
    g_pso_attn_out_low_q8_0_bf16_shared2 = nil;
    g_pso_attn_out_low_q4_K_bf16_shared2 = nil;
    g_pso_mul_mm_q8_0 = nil;
    g_pso_mul_mm_q8_0_bco = nil;
    g_pso_mul_mm_q4_K = nil;
    g_pso_mul_mm_q4_K_bco = nil;
    g_pso_mul_mm_f16 = nil;
    g_pso_mul_mm_f16_bco = nil;
    g_pso_mul_mm_f32 = nil;
    g_pso_mul_mm_f32_bco = nil;
    g_pso_compute_copy_u8 = nil;
    g_pso_compute_copy_u32 = nil;
    g_pso_compute_copy_u32x4 = nil;
    g_inflight_cbs = nil;
    g_batch_released_buffers = nil;
    g_shared_event = nil;
    g_shared_event_val = 0;
    g_library = nil;
    g_pso_lm_head_q8_nr4 = nil;
    g_lm_head_q8_nr0 = 2;
    g_f16_gemv_bf16_fuse = false;
    for (unsigned i = 0; i < 8u; ++i) g_pso_f16_gemv_bf16[i] = nil;
    g_indexer_rope_quant_fuse = false;
    g_pso_indexer_rope_quant = nil;
    g_native_pipelines = nil;
    g_queue = nil;
    for (size_t i = 0; i < g_native_pso_cache_count; i++) {
        g_native_pso_cache[i].pso = nil;
        g_native_pso_cache[i].name[0] = '\0';
    }
    g_native_pso_cache_count = 0;
    for (size_t i = 0; i < g_specialized_pso_cache_count; ++i) {
        g_specialized_pso_cache[i].pso = nil;
        g_specialized_pso_cache[i].name[0] = '\0';
    }
    g_specialized_pso_cache_count = 0;
    g_specialized_pso_cache_next = 0;
    g_device = nil;
    g_qkv_finish_available = -1;
    g_initialized = false;
}


static uint32_t g_inject_copy_fail_at = 0;
static uint32_t g_inject_copy_calls = 0;
static uint32_t g_inject_commit_fail_at = 0;
static uint32_t g_inject_commit_calls = 0;

void myds_metal_inject_copy_failure(uint32_t fail_at_call_index) {
    g_inject_copy_fail_at = fail_at_call_index;
    g_inject_copy_calls = 0;
}

void myds_metal_inject_commit_failure(uint32_t fail_at_call_index) {
    g_inject_commit_fail_at = fail_at_call_index;
    g_inject_commit_calls = 0;
}

uint32_t myds_metal_get_copy_call_count(void) {
    return g_inject_copy_calls;
}

uint32_t myds_metal_get_commit_call_count(void) {
    return g_inject_commit_calls;
}

int myds_metal_begin_batch(void) {
    if (!g_queue && !myds_metal_init(NULL)) return 0;
    if (g_batch_cb) return 1;
    close_compute_encoder();
    g_batch_cb = make_command_buffer();
    return g_batch_cb != nil;
}

int myds_metal_commit_batch(void) {
    if (g_inject_commit_fail_at) {
        g_inject_commit_calls++;
        if (g_inject_commit_calls == g_inject_commit_fail_at) {
            if (g_batch_cb) {
                close_compute_encoder();
                id<MTLCommandBuffer> cb = g_batch_cb;
                g_batch_cb = nil;
                [cb commit];
                [cb waitUntilCompleted];
            }
            return 0;
        }
    }
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
    if (!g_inflight_cbs) g_inflight_cbs = [[NSMutableArray alloc] initWithCapacity:64];
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
    if (!g_inflight_cbs) g_inflight_cbs = [[NSMutableArray alloc] initWithCapacity:64];
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
    id<MTLBuffer> old_buf = view->mtl_buf;
    if (old_buf && old_buf != parent->mtl_buf && (g_batch_cb || (g_inflight_cbs && [g_inflight_cbs count] > 0))) {
        if (!g_batch_released_buffers) g_batch_released_buffers = [[NSMutableArray alloc] initWithCapacity:16];
        [g_batch_released_buffers addObject:old_buf];
    }
    view->mtl_buf = parent->mtl_buf;
    view->size = bytes;
    view->offset = parent->offset + offset;
    view->ptr = (void *)((uintptr_t)parent->ptr + offset);
    view->is_view = true;
    return 1;
}

void myds_metal_buffer_free(myds_metal_buffer_t *buf) {
    if (!buf) return;
    if (buf->mtl_buf && (g_batch_cb || (g_inflight_cbs && [g_inflight_cbs count] > 0))) {
        if (!g_batch_released_buffers) g_batch_released_buffers = [[NSMutableArray alloc] initWithCapacity:16];
        [g_batch_released_buffers addObject:buf->mtl_buf];
    }
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

/* 诊断用（参谋长 2026-09-30）：统计 GPU dispatch 次数。
 * 用途：判定 prefill 的 GPU 时间是被"启动开销 x 海量小 kernel"主导，还是被算力/带宽主导。 */
static uint64_t g_dispatch_count = 0;
/* 诊断：按内核名统计 dispatch 次数，用于定位"每 token 一串小 kernel"里的大头。 */
#define MYDS_HIST_SLOTS 128
static struct { const char *name; uint64_t n; } g_hist[MYDS_HIST_SLOTS];
static int g_hist_used = 0;
/* 默认关闭：开启时每次 dispatch 都要做线性查找 + strcmp，会明显拖慢 CPU 侧
 * （实测在计时对比里造成 ~0.8s 的系统误差）。只在 MYDS_DISPATCH_HIST=1 时启用。 */
static int g_hist_enabled = -1;
static int hist_on(void) {
    if (g_hist_enabled < 0) g_hist_enabled = getenv("MYDS_DISPATCH_HIST") ? 1 : 0;
    return g_hist_enabled;
}
static void hist_add(const char *name) {
    if (!hist_on()) return;
    for (int i = 0; i < g_hist_used; i++) {
        if (g_hist[i].name == name || (g_hist[i].name && name && !strcmp(g_hist[i].name, name))) {
            g_hist[i].n++; return;
        }
    }
    if (g_hist_used < MYDS_HIST_SLOTS) { g_hist[g_hist_used].name = name; g_hist[g_hist_used].n = 1; g_hist_used++; }
}
int myds_metal_dispatch_hist_dump(FILE *out) {
    if (!out) return 0;
    /* 简单选择排序前 24 名 */
    int idx[MYDS_HIST_SLOTS];
    for (int i = 0; i < g_hist_used; i++) idx[i] = i;
    for (int i = 0; i < g_hist_used; i++)
        for (int j = i+1; j < g_hist_used; j++)
            if (g_hist[idx[j]].n > g_hist[idx[i]].n) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    uint64_t tot = 0;
    for (int i = 0; i < g_hist_used; i++) tot += g_hist[i].n;
    fprintf(out, "--- DISPATCH BY KERNEL (%d kinds, %llu total) ---\n", g_hist_used, (unsigned long long)tot);
    for (int i = 0; i < g_hist_used && i < 24; i++)
        fprintf(out, "%-46s %10llu  %5.2f%%\n", g_hist[idx[i]].name,
                (unsigned long long)g_hist[idx[i]].n,
                tot ? 100.0*(double)g_hist[idx[i]].n/(double)tot : 0.0);
    fflush(out);
    return 1;
}

uint64_t myds_metal_dispatch_count(void) { return g_dispatch_count; }
void myds_metal_dispatch_count_reset(void) { g_dispatch_count = 0; }

static inline void metal_dispatch(id<MTLComputeCommandEncoder> enc,
                                  const char *name,
                                  MTLSize threadgroups,
                                  MTLSize threadsPerThreadgroup) {
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    g_dispatch_count++;
    hist_add(name);
}

static inline void metal_dispatch_threads(id<MTLComputeCommandEncoder> enc,
                                          const char *name,
                                          MTLSize threads,
                                          MTLSize threadsPerThreadgroup) {
    [enc dispatchThreads:threads threadsPerThreadgroup:threadsPerThreadgroup];
    g_dispatch_count++;
    hist_add(name);
}

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
            metal_dispatch(enc, "rms_norm_ref", MTLSizeMake(1, 1, 1),
                MTLSizeMake(threads, 1, 1));

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
        if (g_auto_barrier) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        return 1;
    }
    [enc endEncoding];
    [cb commit];
    return wait_command_buffer(cb);
}

static bool indexer_head_slots_range(const myds_metal_buffer_t *buffer,uint64_t bytes,size_t alignment) {
    return buffer && buffer->mtl_buf && buffer->mtl_buf.device == g_device && bytes &&
        bytes <= buffer->size && buffer->offset <= buffer->mtl_buf.length &&
        bytes <= buffer->mtl_buf.length - buffer->offset && buffer->offset % alignment == 0;
}
static bool indexer_head_slots_overlap(const myds_metal_buffer_t *a,uint64_t a_bytes,
    const myds_metal_buffer_t *b,uint64_t b_bytes) {
    if (a->mtl_buf == b->mtl_buf) {
        return a->offset <= b->offset ? b->offset - a->offset < a_bytes : a->offset - b->offset < b_bytes;
    }
    /* Distinct no-copy handles can expose the same unified-memory range. */
    if (a->ptr && b->ptr) {
        const uintptr_t first=(uintptr_t)a->ptr,second=(uintptr_t)b->ptr;
        return first <= second ? second-first < a_bytes : first-second < b_bytes;
    }
    return false;
}
int myds_metal_indexer_head_slots_dispatch(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q,const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache,const myds_metal_buffer_t *mask,
    uint32_t n_comp,uint32_t n_heads,bool cache_bf16,bool *encoded,bool *unavailable) {
    if(encoded)*encoded=false;
    if(unavailable)*unavailable=false;
    if(!g_initialized || !g_indexer_head_slots || !indexer_head_slots_geometry()) {
        if(unavailable)*unavailable=true;
        return 0;
    }
    if(!n_comp || n_heads!=32u) return 0;
    const uint64_t score_bytes=(uint64_t)n_comp*4u;
    const uint64_t cache_bytes=(uint64_t)n_comp*128u*(cache_bf16?2u:4u);
    const uint64_t mask_bytes=((uint64_t)n_comp+7u)/8u*4u;
    if(!indexer_head_slots_range(scores,score_bytes,4u) ||
       !indexer_head_slots_range(q,32u*128u*4u,16u) ||
       !indexer_head_slots_range(weights,32u*4u,4u) ||
       !indexer_head_slots_range(cache,cache_bytes,cache_bf16?2u:4u) ||
       (mask && !indexer_head_slots_range(mask,mask_bytes,4u))) return 0;
    if(indexer_head_slots_overlap(scores,score_bytes,q,32u*128u*4u) ||
       indexer_head_slots_overlap(scores,score_bytes,weights,32u*4u) ||
       indexer_head_slots_overlap(scores,score_bytes,cache,cache_bytes) ||
       (mask && indexer_head_slots_overlap(scores,score_bytes,mask,mask_bytes))) return 0;
    struct {uint32_t n_rows,n_head,head_dim,cache_f16;float scale;uint32_t has_mask;} args={
        n_comp,n_heads,128u,cache_bf16?2u:0u,1.0f/64.0f,mask?1u:0u};
    const myds_metal_buffer_t *buffers[]={q,weights,cache,scores,mask?mask:scores};
    @autoreleasepool {
        id<MTLCommandBuffer> cb=g_batch_cb?g_batch_cb:[g_queue commandBuffer];
        if(!cb)return 0;
        id<MTLComputeCommandEncoder> enc=get_compute_encoder(cb);
        if(!enc)return 0;
        /* Preserve the original generic scorer bridge's auto1 pre/post barriers.
         * Auto0 callers retain the same upstream/score-consumer/reuse joins. */
        if(g_batch_cb && g_auto_barrier)[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_indexer_head_slots];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        for(size_t i=0;i<5u;++i)[enc setBuffer:buffers[i]->mtl_buf offset:buffers[i]->offset atIndex:i+1];
        [enc setThreadgroupMemoryLength:640u atIndex:0];
        metal_dispatch(enc,"kernel_myds_indexer_score_one_head_slots",MTLSizeMake(n_comp,1u,1u),MTLSizeMake(32u,4u,1u));
        if(encoded)*encoded=true;
        return v41_finish(cb,enc);
    }
}

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

/* 派发失败原因：设 MYDS_DEBUG_DISPATCH=1 时打印【哪一条校验】挡下了这次 dispatch，
 * 以及 newFunctionWithName / newComputePipelineState 的 NSError。
 * 2026-09-30 加 —— 当时一个新内核在 macstudio 上「FUNCTION NOT FOUND」，靠这条日志
 * 立刻定位到「二进制按 CWD 相对路径读 metal/ 源码，而测试脚本没 cd 到构建目录」。
 * 未设该环境变量时无任何额外开销（只在失败路径上 getenv）。 */
#define MYDS_DBG_DISPATCH_FAIL(why) do { \
    if (getenv("MYDS_DEBUG_DISPATCH")) myds_log(MYDS_LOG_ERROR, "DBGDISP %s: %s", name, why); \
    return 0; } while (0)
int myds_metal_dispatch_kernel_constants(const char *name, const void *args, size_t args_size,
    const myds_metal_buffer_t *const *buffers, size_t n_buffers,
    const uint32_t groups[3], const uint32_t threads[3], size_t threadgroup_memory,
    const myds_metal_constant_t *constants, size_t n_constants) {
    if (!g_initialized || !name || !*name || !args || !args_size || args_size > 4096 ||
        !buffers || !n_buffers || n_buffers > 30 || !groups || !threads ||
        (n_constants && !constants) || n_constants>64) MYDS_DBG_DISPATCH_FAIL("precheck args/buffers");
    uint64_t n_threads = 1;
    const MTLSize maximum = g_device.maxThreadsPerThreadgroup;
    const NSUInteger limits[3] = {maximum.width, maximum.height, maximum.depth};
    for (unsigned i = 0; i < 3; ++i) {
        if (!groups[i] || !threads[i] || threads[i] > limits[i] ||
            n_threads > UINT64_MAX / threads[i]) MYDS_DBG_DISPATCH_FAIL("threads/groups invalid");
        n_threads *= threads[i];
    }
    for (size_t i = 0; i < n_buffers; ++i)
        if (!buffers[i] || !buffers[i]->mtl_buf || !buffers[i]->size) MYDS_DBG_DISPATCH_FAIL("null/empty buffer");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = nil;
        if (n_constants == 0) {
            for (size_t i = 0; i < g_native_pso_cache_count; i++) {
                if (strcmp(g_native_pso_cache[i].name, name) == 0) {
                    pipeline = g_native_pso_cache[i].pso;
                    break;
                }
            }
        } else {
            pipeline = specialized_pipeline_lookup(name, constants, n_constants);
        }
        if (!pipeline) {
            NSString *function_name = [NSString stringWithUTF8String:name];
            if (!function_name) MYDS_DBG_DISPATCH_FAIL("bad name");
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
                NSError *ferr = nil;
                id<MTLFunction> fn = n_constants ? [g_library newFunctionWithName:function_name constantValues:values error:&ferr]
                                               : [g_library newFunctionWithName:function_name];
                if (!fn) {
                    if (getenv("MYDS_DEBUG_DISPATCH"))
                        myds_log(MYDS_LOG_ERROR, "DBGDISP %s: FUNCTION NOT FOUND (%s)", name,
                                 ferr ? [[ferr localizedDescription] UTF8String] : "no error");
                    return 0;
                }
                NSError *perr = nil;
                pipeline = [g_device newComputePipelineStateWithFunction:fn error:&perr];
                if (!pipeline) {
                    if (getenv("MYDS_DEBUG_DISPATCH"))
                        myds_log(MYDS_LOG_ERROR, "DBGDISP %s: PIPELINE FAILED (%s)", name,
                                 perr ? [[perr localizedDescription] UTF8String] : "no error");
                    return 0;
                }
                g_native_pipelines[key] = pipeline;
            }
            if (n_constants == 0 && g_native_pso_cache_count < 64 && strlen(name) < 64) {
                strncpy(g_native_pso_cache[g_native_pso_cache_count].name, name, 63);
                g_native_pso_cache[g_native_pso_cache_count].name[63] = '\0';
                g_native_pso_cache[g_native_pso_cache_count].pso = pipeline;
                g_native_pso_cache_count++;
            } else if (n_constants != 0) {
                specialized_pipeline_cache(name, constants, n_constants, pipeline);
            }
        }
        if (n_threads > pipeline.maxTotalThreadsPerThreadgroup)
            MYDS_DBG_DISPATCH_FAIL("n_threads > maxTotalThreadsPerThreadgroup");
        if (threadgroup_memory > g_device.maxThreadgroupMemoryLength)
            MYDS_DBG_DISPATCH_FAIL("threadgroup_memory > device max");
        if (pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - threadgroup_memory)
            MYDS_DBG_DISPATCH_FAIL("static+dynamic tg mem > device max");
        if (threadgroup_memory % 16) MYDS_DBG_DISPATCH_FAIL("threadgroup_memory not 16-aligned");
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:args length:args_size atIndex:0];
        for (size_t i = 0; i < n_buffers; ++i)
            [enc setBuffer:buffers[i]->mtl_buf offset:buffers[i]->offset atIndex:i+1];
        if (threadgroup_memory) [enc setThreadgroupMemoryLength:threadgroup_memory atIndex:0];
        metal_dispatch(enc, name, MTLSizeMake(groups[0], groups[1], groups[2]),
            MTLSizeMake(threads[0], threads[1], threads[2]));
        return v41_finish(cb, enc);
    }
}

int myds_metal_buffer_copy(myds_metal_buffer_t *dst, size_t dst_offset,
    const myds_metal_buffer_t *src, size_t src_offset, size_t bytes) {
    if (g_inject_copy_fail_at) {
        g_inject_copy_calls++;
        if (g_inject_copy_calls == g_inject_copy_fail_at) return 0;
    }
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

            if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
            metal_dispatch(enc, "buffer_copy", MTLSizeMake(ngr, 1, 1),
                MTLSizeMake(nth, 1, 1));
            if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_bf16];
        [enc setBytes:&n length:sizeof(n) atIndex:0];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:1];
        metal_dispatch(enc, "v41_bf16", MTLSizeMake((work + nth - 1u) / nth, 1, 1),
            MTLSizeMake(nth, 1, 1));
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:1];
        [enc setBuffer:w->mtl_buf offset:w->offset atIndex:2];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:3];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:4];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        metal_dispatch(enc, "v41_rmsnorm", MTLSizeMake(1, 1, 1), MTLSizeMake(nth, 1, 1));
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
    /* 诊断：线程组尺寸直接给定（0 = 用上面的启发式）。
     * 这条 norm 在剖析里是 Step 1 的最大单项（~336 ms），而它只是纯访存操作，
     * 实测有效带宽只有 ~110 GB/s ⇒ 怀疑 1024 线程/组的占用率或访存模式。 */
    {
        static int ov = -2;
        if (__builtin_expect(ov == -2, 0)) {
            const char *v = getenv("MYDS_RMS_NTH");
            long n = v ? atol(v) : 0;
            ov = (n > 0 && n <= 1024) ? (int)n : -1;
        }
        if (ov > 0) nth = (NSUInteger)ov;
    }
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:1];
        [enc setBuffer:w->mtl_buf offset:w->offset atIndex:2];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:3];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:4];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        metal_dispatch(enc, "v41_rmsnorm_batch", MTLSizeMake(n_tok, 1, 1), MTLSizeMake(nth, 1, 1));
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
    if (!myds_metal_v41_rmsnorm(flat_norm, residual, NULL, 4u * dim, eps, false)) return 0;
    if (!g_auto_barrier) myds_metal_barrier_buffer(flat_norm);
    if (!(fn_type == GGML_TYPE_F16
            ? myds_metal_v41_gemv(mix, fn, flat_norm, 24, 4u * dim, fn_type)
            : myds_metal_gemv(mix, fn, flat_norm, 24, 4u * dim, fn_type))) return 0;
    if (!g_auto_barrier) myds_metal_barrier_buffer(mix);
    @autoreleasepool {
        struct v41_hc_split_args args = {4, 20, 1, 24, 24 * sizeof(float), 24 * sizeof(float), hc_eps};
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_split];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mix->mtl_buf offset:mix->offset atIndex:1];
        [enc setBuffer:scale->mtl_buf offset:scale->offset atIndex:2];
        [enc setBuffer:base->mtl_buf offset:base->offset atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset atIndex:4];
        metal_dispatch(enc, "v41_hc_split", MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_split];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mix->mtl_buf offset:mix->offset atIndex:1];
        [enc setBuffer:scale->mtl_buf offset:scale->offset atIndex:2];
        [enc setBuffer:base->mtl_buf offset:base->offset atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset atIndex:4];
        NSUInteger nth = 32;
        NSUInteger ngr = (n_tokens + nth - 1) / nth;
        metal_dispatch(enc, "v41_hc_split_batch", MTLSizeMake(ngr, 1, 1), MTLSizeMake(nth, 1, 1));
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_sum];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:1];
        [enc setBuffer:weights->mtl_buf offset:weights->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        metal_dispatch(enc, "v41_hc_sum", MTLSizeMake((dim+nth-1)/nth, 1, 1), MTLSizeMake(nth, 1, 1));
        return v41_finish(cb, enc);
    }
}

struct v41_hc_sum_rmsnorm_args {
    uint32_t dim;
    float    eps;
    int32_t  round_bf16;
    int32_t  has_sum_out;
};

int myds_metal_v41_hc_sum_rmsnorm(
    myds_metal_buffer_t *norm_out,
    myds_metal_buffer_t *sum_out,
    const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *norm_weight,
    uint32_t dim,
    float eps,
    bool round_bf16) {
    if (!g_initialized || !g_pso_v41_hc_sum_rmsnorm || !dim || dim % 128 ||
        !isfinite(eps) || eps < 0 || !v41_floats(norm_out, dim) ||
        !v41_floats(residual, 4ull * dim) || !v41_floats(weights, 4) ||
        !v41_floats(norm_weight, dim) ||
        (sum_out && !v41_floats(sum_out, dim))) return 0;
    @autoreleasepool {
        struct v41_hc_sum_rmsnorm_args args = {
            .dim = dim,
            .eps = eps,
            .round_bf16 = round_bf16 ? 1 : 0,
            .has_sum_out = sum_out ? 1 : 0,
        };
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_sum_rmsnorm];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:1];
        [enc setBuffer:weights->mtl_buf offset:weights->offset atIndex:2];
        [enc setBuffer:norm_weight->mtl_buf offset:norm_weight->offset atIndex:3];
        const myds_metal_buffer_t *s_buf = sum_out ? sum_out : norm_out;
        [enc setBuffer:s_buf->mtl_buf offset:s_buf->offset atIndex:4];
        [enc setBuffer:norm_out->mtl_buf offset:norm_out->offset atIndex:5];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
        NSUInteger nth = 256;
        metal_dispatch(enc, "v41_hc_sum_rmsnorm", MTLSizeMake(1, 1, 1), MTLSizeMake(nth, 1, 1));
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_expand];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:block->mtl_buf offset:block->offset atIndex:1];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:2];
        [enc setBuffer:split->mtl_buf offset:split->offset+16 atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset+32 atIndex:4];
        [enc setBuffer:block->mtl_buf offset:block->offset atIndex:5];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:6];
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        metal_dispatch(enc, "v41_hc_expand", MTLSizeMake((dim+nth-1)/nth, 1, 1), MTLSizeMake(nth, 1, 1));
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_moe_add_hc_expand(myds_metal_buffer_t *out,
    myds_metal_buffer_t *ffn_out, const myds_metal_buffer_t *moe_out,
    const myds_metal_buffer_t *shared_out, const myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *split, uint32_t dim) {
    if (!g_initialized || !v41_floats(out, 4ull*dim) ||
        !v41_floats(moe_out, dim) || !v41_floats(shared_out, dim) ||
        !v41_floats(residual, 4ull*dim) || !v41_floats(split, 24)) return 0;
    if (ffn_out && !v41_floats(ffn_out, dim)) return 0;
    if (!g_pso_v41_moe_add_hc_expand) {
        myds_metal_buffer_t *fout = ffn_out ? ffn_out : out;
        if (!myds_metal_v41_add_bf16(fout, moe_out, shared_out, dim)) return 0;
        return myds_metal_v41_hc_expand(out, fout, residual, split, dim);
    }
    @autoreleasepool {
        struct v41_hc_expand_args args = {
            .n_embd=dim, .n_hc=4, .n_tokens=1,
            .nb_block0=4, .nb_block1=4ull*dim, .nb_add0=4, .nb_add1=4ull*dim,
            .nb_res0=4, .nb_res1=4ull*dim, .nb_res2=16ull*dim,
            .nb_post0=4, .nb_post1=96, .nb_comb0=4, .nb_comb1=16, .nb_comb2=96,
            .nb0=4, .nb1=4ull*dim, .nb2=16ull*dim,
            .has_add = (ffn_out != NULL) ? 1 : 0,
            .round_bf16 = 1,
        };
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_moe_add_hc_expand];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:1];
        [enc setBuffer:shared_out->mtl_buf offset:shared_out->offset atIndex:2];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:3];
        [enc setBuffer:split->mtl_buf offset:split->offset+16 atIndex:4];
        [enc setBuffer:split->mtl_buf offset:split->offset+32 atIndex:5];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:6];
        if (ffn_out) {
            [enc setBuffer:ffn_out->mtl_buf offset:ffn_out->offset atIndex:7];
        } else {
            [enc setBuffer:out->mtl_buf offset:out->offset atIndex:7];
        }
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        metal_dispatch(enc, "v41_moe_add_hc_expand", MTLSizeMake((dim+nth-1)/nth, 1, 1), MTLSizeMake(nth, 1, 1));
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_hc_sum];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:1];
        [enc setBuffer:weights->mtl_buf offset:weights->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        NSUInteger nth = MIN((NSUInteger)256, (NSUInteger)dim);
        uint64_t total_elements = (uint64_t)dim * n_tok;
        metal_dispatch(enc, "v41_hc_sum_batch", MTLSizeMake((total_elements + nth - 1) / nth, 1, 1),
            MTLSizeMake(nth, 1, 1));
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
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
        metal_dispatch(enc, "v41_hc_expand_batch", MTLSizeMake((total_elements + nth - 1) / nth, 1, 1),
            MTLSizeMake(nth, 1, 1));
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_engram_add_batch(myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *projected, const myds_metal_buffer_t *q_weight,
    const myds_metal_buffer_t *k_weight, uint32_t dim, uint32_t rows, float eps) {
    if (!g_initialized || !g_pso_v41_engram_add || !isfinite(eps) || eps <= 0 || !rows ||
        !v41_floats(residual, 4ull*dim*rows) || !v41_floats(projected, 5ull*dim*rows) ||
        !v41_floats(q_weight, 4ull*dim) || !v41_floats(k_weight, 4ull*dim)) return 0;
    @autoreleasepool {
        struct {uint32_t width, rows; float eps; uint32_t masked;} args = {dim,rows,eps,0};
        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return 0;
        if (g_batch_cb && g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:g_pso_v41_engram_add];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:1];
        [enc setBuffer:projected->mtl_buf offset:projected->offset atIndex:2];
        [enc setBuffer:q_weight->mtl_buf offset:q_weight->offset atIndex:3];
        [enc setBuffer:k_weight->mtl_buf offset:k_weight->offset atIndex:4];
        [enc setBuffer:residual->mtl_buf offset:residual->offset atIndex:5];
        metal_dispatch(enc, "v41_engram_add", MTLSizeMake(rows, 4, 1), MTLSizeMake(32, 1, 1));
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_engram_add(myds_metal_buffer_t *residual,
    const myds_metal_buffer_t *projected, const myds_metal_buffer_t *q_weight,
    const myds_metal_buffer_t *k_weight, uint32_t dim, float eps) {
    return myds_metal_v41_engram_add_batch(residual, projected, q_weight, k_weight, dim, 1, eps);
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
            metal_dispatch_threads(enc, "swiglu",
                MTLSizeMake(dim, 1, 1),
                MTLSizeMake(threads, 1, 1));

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

static uint32_t get_gemv_batch_size(void) {
    const char *env = getenv("MYDS_GEMV_BATCH_SIZE");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "0") == 0)) return 1;
    return 4;
}

static int lm_head_q8_failure(const char *reason) {
    atomic_fetch_add_explicit(&g_lm_head_q8_failures, 1, memory_order_relaxed);
    fprintf(stderr, "myds_metal: LM-head Q8 NR4 rejected: %s; refusing fallback\n", reason);
    return 0;
}

static bool lm_head_q8_range(const myds_metal_buffer_t *buffer, uint64_t bytes) {
    return buffer && buffer->mtl_buf && bytes <= buffer->size &&
           buffer->offset <= buffer->mtl_buf.length && bytes <= buffer->mtl_buf.length - buffer->offset;
}

static bool lm_head_q8_overlap(const myds_metal_buffer_t *a, uint64_t a_bytes,
                               const myds_metal_buffer_t *b, uint64_t b_bytes) {
    if (a->mtl_buf != b->mtl_buf) return false;
    return a->offset <= b->offset ? b->offset - a->offset < a_bytes
                                  : a->offset - b->offset < b_bytes;
}

static int lm_head_q8_nr4_gemv(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim) {
    atomic_fetch_add_explicit(&g_lm_head_q8_requests, 1, memory_order_relaxed);
    if (!g_initialized || out_dim <= 65536u || out_dim > INT32_MAX - 3u ||
        !in_dim || in_dim > INT32_MAX || in_dim % 32u)
        return lm_head_q8_failure("invalid runtime or dimensions");
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = row_bytes * out_dim;
    const uint64_t input_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t output_bytes = (uint64_t)out_dim * sizeof(float);
    if (!lm_head_q8_range(weight, weight_bytes) || weight->offset % 2u ||
        !lm_head_q8_range(input, input_bytes) || input->offset % 4u ||
        !lm_head_q8_range(out, output_bytes) || out->offset % 4u)
        return lm_head_q8_failure("invalid buffer range/alignment");
    if (lm_head_q8_overlap(out, output_bytes, weight, weight_bytes) ||
        lm_head_q8_overlap(out, output_bytes, input, input_bytes))
        return lm_head_q8_failure("output overlaps weights or activation");
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = lm_head_q8_nr4_pipeline();
        if (!pipeline) return lm_head_q8_failure("candidate PSO unavailable");
        const NSUInteger scratch_bytes = 32u * 4u * sizeof(float);
        if (pipeline.threadExecutionWidth != 32u ||
            pipeline.maxTotalThreadsPerThreadgroup < 256u ||
            g_device.maxThreadsPerThreadgroup.width < 256u ||
            scratch_bytes > g_device.maxThreadgroupMemoryLength ||
            pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - scratch_bytes)
            return lm_head_q8_failure("candidate geometry exceeds PSO/device limits");
        /* Complete previous writers of x and readers/writers of out even in
         * a concurrent encoder with automatic barriers disabled. Validation
         * above finishes before any barrier/encoder or output mutation. */
        if (g_batch_cb && !g_auto_barrier) {
            const myds_metal_buffer_t *deps[] = { weight, input, out };
            myds_metal_barrier_buffers(deps, 3);
        }
        struct ds4_metal_args_mul_mv args = {0};
        args.ne00 = (int32_t)in_dim; args.ne01 = (int32_t)out_dim; args.ne02 = 1;
        args.nb00 = 34; args.nb01 = row_bytes;
        args.nb02 = weight_bytes; args.nb03 = weight_bytes;
        args.ne10 = (int32_t)in_dim; args.ne11 = 1; args.ne12 = 1;
        args.nb10 = sizeof(float); args.nb11 = input_bytes;
        args.nb12 = input_bytes; args.nb13 = input_bytes;
        args.ne0 = (int32_t)out_dim; args.ne1 = 1;
        args.nr0 = 4; args.r2 = 1; args.r3 = 1;
        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return lm_head_q8_failure("command buffer unavailable");
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return lm_head_q8_failure("encoder unavailable");
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weight->mtl_buf offset:weight->offset atIndex:1];
        [enc setBuffer:input->mtl_buf offset:input->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        metal_dispatch(enc, "lm_head_q8_nr4", MTLSizeMake(((NSUInteger)out_dim + 3u) / 4u, 1, 1),
                       MTLSizeMake(32, 8, 1));
        atomic_fetch_add_explicit(&g_lm_head_q8_encoded, 1, memory_order_relaxed);
        if (!v41_finish(cb, enc)) return lm_head_q8_failure("command completion failed");
        return 1;
    }
}

int myds_metal_gemv_batch(
    myds_metal_buffer_t       *y,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *x,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    uint32_t                   n_tok,
    ggml_type                  weight_type) {
    /* Default/explicit2 leaves the complete original selector and dispatch
     * below unchanged. Only eligible explicit4 enters the fail-closed helper. */
    if (g_lm_head_q8_nr0 == 4 && weight_type == GGML_TYPE_Q8_0 &&
        n_tok == 1u && out_dim > 65536u)
        return lm_head_q8_nr4_gemv(y, W, x, out_dim, in_dim);
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
    } else if (weight_type == GGML_TYPE_Q4_K) {
        if (in_dim % 256u) return 0;
        block_bytes = 144;
        row_bytes = (uint64_t)(in_dim / 256u) * block_bytes;
        nsg = 4u;
        NSString *key = [NSString stringWithFormat:@"q4_k_%u", nsg];
        pipeline = g_native_pipelines[key];
        if (!pipeline) {
            MTLFunctionConstantValues *mv_constants = [MTLFunctionConstantValues new];
            short short_nsg = (short)nsg;
            [mv_constants setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
            id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_mul_mv_q4_K_dense_f32" constantValues:mv_constants error:nil];
            if (fn) {
                pipeline = [g_device newComputePipelineStateWithFunction:fn error:nil];
                if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
                g_native_pipelines[key] = pipeline;
            }
        }
        if (!pipeline) pipeline = g_pso_mul_mv_q4_K;
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
            if (weight_type == GGML_TYPE_Q4_K) {
                NSUInteger ngr = ((NSUInteger)out_dim + nsg * 2 - 1) / (nsg * 2);
                metal_dispatch(enc, "gemv_q4_k_batch", MTLSizeMake(ngr, n_tok, 1),
                    MTLSizeMake(32, nsg, 1));
            } else {
                metal_dispatch(enc, "gemv_q8_0_batch", MTLSizeMake(((NSUInteger)out_dim + 1) / 2, n_tok, 1),
                    MTLSizeMake(32, nsg, 1));
            }
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
    typedef struct {
        uint16_t d;
        uint16_t dmin;
        uint8_t  scales[12];
        uint8_t  qs[128];
    } block_q4_K_t;

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
            } else if (weight_type == GGML_TYPE_Q4_K) {
                const block_q4_K_t *blocks = (const block_q4_K_t *)w_row;
                for (uint32_t b = 0; b < in_dim / 256u; b++) {
                    const block_q4_K_t *blk = &blocks[b];
                    float d = fp16_to_fp32(blk->d);
                    float dmin = fp16_to_fp32(blk->dmin);
                    for (uint32_t group = 0; group < 8; group++) {
                        uint8_t sc, m;
                        if (group < 4) {
                            sc = blk->scales[group] & 63;
                            m  = blk->scales[group + 4] & 63;
                        } else {
                            sc = (blk->scales[group + 4] & 0x0F) | ((blk->scales[group - 4] & 0xC0) >> 2);
                            m  = (blk->scales[group + 4] >> 4)   | ((blk->scales[group] & 0xC0) >> 2);
                        }
                        float d_sc = d * (float)sc;
                        float dmin_m = dmin * (float)m;
                        const uint8_t *qs = blk->qs + (group >> 1) * 32;
                        const uint32_t shift = (group & 1) * 4;
                        const float *x_grp = xv + b * 256u + group * 32u;
                        for (uint32_t l = 0; l < 32; l++) {
                            uint8_t q = (qs[l] >> shift) & 0x0F;
                            float w = d_sc * (float)q - dmin_m;
                            sum += w * x_grp[l];
                        }
                    }
                }
            } else if (weight_type == GGML_TYPE_F32) {
                const float *weights = (const float *)w_row;
                for (uint32_t col = 0; col < in_dim; col++) sum += weights[col] * xv[col];
            } else if (weight_type == GGML_TYPE_F16) {
                const uint16_t *weights = (const uint16_t *)w_row;
                for (uint32_t col = 0; col < in_dim; col++) sum += fp16_to_fp32(weights[col]) * xv[col];
            } else {
                return 0;
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
        const uint32_t b_size = (n_tok > 1u) ? get_gemv_batch_size() : 1u;
        const bool use_b4 = (b_size == 4u && n_tok > 1u);
        NSString *funcName = nil;
        if (use_b4) {
            funcName = weight_type == GGML_TYPE_F32 ? @"kernel_mul_mv_f32_f32_4_b4" : @"kernel_mul_mv_f16_f32_4_b4";
        } else {
            funcName = weight_type == GGML_TYPE_F32 ? @"kernel_mul_mv_f32_f32_4" : @"kernel_mul_mv_f16_f32_4";
        }
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
        NSUInteger shmem_len = use_b4 ? (32u * 2u * 4u * sizeof(float)) : (32u * 2u * sizeof(float));
        [enc setThreadgroupMemoryLength:shmem_len atIndex:0];
        NSUInteger grid_y = use_b4 ? ((n_tok + 3u) / 4u) : n_tok;
        metal_dispatch(enc, [funcName UTF8String], MTLSizeMake((out_dim + 1u) / 2u, grid_y, 1),
             MTLSizeMake(32, nsg, 1));
        return v41_finish(cb, enc);
    }
}

int myds_metal_v41_gemv(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim, ggml_type weight_type) {
    return myds_metal_v41_gemv_batch(out, weight, input, out_dim, in_dim, 1, weight_type);
}

static int f16_gemv_bf16_failure(const char *reason) {
    atomic_fetch_add_explicit(&g_f16_gemv_bf16_failures, 1, memory_order_relaxed);
    fprintf(stderr, "myds_metal: F16 GEMV BF16 fuse rejected: %s; refusing fallback\n", reason);
    return 0;
}

static bool f16_gemv_bf16_range(const myds_metal_buffer_t *b, uint64_t bytes, size_t alignment) {
    if (!b || !b->mtl_buf || b->mtl_buf.device != g_device || !b->ptr ||
        bytes > b->size || b->offset > b->mtl_buf.length ||
        bytes > b->mtl_buf.length - b->offset || b->offset % alignment ||
        (uintptr_t)b->ptr % alignment || bytes > UINTPTR_MAX - (uintptr_t)b->ptr)
        return false;
    const uintptr_t base = (uintptr_t)b->mtl_buf.contents;
    return base && b->offset <= UINTPTR_MAX - base &&
        (uintptr_t)b->ptr == base + b->offset;
}

static bool f16_gemv_bf16_overlap(const myds_metal_buffer_t *a, uint64_t a_bytes,
                                 const myds_metal_buffer_t *b, uint64_t b_bytes) {
    /* Resource identity catches views; CPU ranges also catch two different
     * no-copy MTLBuffer wrappers around overlapping unified-memory pages. */
    if (a->mtl_buf == b->mtl_buf &&
        (a->offset <= b->offset ? b->offset - a->offset < a_bytes
                               : a->offset - b->offset < b_bytes)) return true;
    const uintptr_t ap = (uintptr_t)a->ptr, bp = (uintptr_t)b->ptr;
    return ap <= bp ? bp - ap < a_bytes : ap - bp < b_bytes;
}

int myds_metal_v41_gemv_f16_bf16(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t out_dim, uint32_t in_dim) {
    atomic_fetch_add_explicit(&g_f16_gemv_bf16_requests, 1, memory_order_relaxed);
    if (!g_initialized || !g_f16_gemv_bf16_fuse || !out_dim || !in_dim ||
        out_dim > INT32_MAX || in_dim > INT32_MAX || in_dim % 4u)
        return f16_gemv_bf16_failure("invalid immutable mode or dimensions");
    const uint64_t row_bytes = (uint64_t)in_dim * 2u;
    const uint64_t weight_bytes = row_bytes * (((uint64_t)out_dim + 1u) / 2u * 2u);
    const uint64_t input_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t output_bytes = (uint64_t)out_dim * sizeof(float);
    if (!f16_gemv_bf16_range(weight, weight_bytes, 8u) ||
        !f16_gemv_bf16_range(input, input_bytes, 16u) ||
        !f16_gemv_bf16_range(out, output_bytes, 16u))
        return f16_gemv_bf16_failure("invalid device, range, padding or alignment");
    if (f16_gemv_bf16_overlap(out, output_bytes, weight, weight_bytes) ||
        f16_gemv_bf16_overlap(out, output_bytes, input, input_bytes))
        return f16_gemv_bf16_failure("output overlaps weights or activation");
    const uint32_t nsg = (in_dim + 127u) / 128u > 8u ? 8u : (in_dim + 127u) / 128u;
    @autoreleasepool {
        id<MTLComputePipelineState> pso = g_pso_f16_gemv_bf16[nsg - 1u];
        if (!f16_gemv_bf16_geometry(pso, nsg))
            return f16_gemv_bf16_failure("candidate PSO or geometry unavailable");
        struct ds4_metal_args_mul_mv args = {0};
        args.ne00 = (int32_t)in_dim; args.ne01 = (int32_t)out_dim; args.ne02 = 1;
        args.nb00 = 2u; args.nb01 = row_bytes;
        args.nb02 = row_bytes * out_dim; args.nb03 = args.nb02;
        args.ne10 = (int32_t)in_dim; args.ne11 = 1; args.ne12 = 1;
        args.nb10 = sizeof(float); args.nb11 = input_bytes;
        args.nb12 = input_bytes; args.nb13 = input_bytes;
        args.ne0 = (int32_t)out_dim; args.ne1 = 1;
        args.nr0 = 2; args.r2 = 1; args.r3 = 1;
        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return f16_gemv_bf16_failure("command buffer unavailable");
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return f16_gemv_bf16_failure("encoder unavailable");
        [enc setComputePipelineState:pso];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weight->mtl_buf offset:weight->offset atIndex:1];
        [enc setBuffer:input->mtl_buf offset:input->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
        metal_dispatch(enc, "kernel_mul_mv_f16_f32_4_bf16_nr2",
            MTLSizeMake(((NSUInteger)out_dim + 1u) / 2u, 1, 1), MTLSizeMake(32, nsg, 1));
        atomic_fetch_add_explicit(&g_f16_gemv_bf16_encoded, 1, memory_order_relaxed);
        if (!v41_finish(cb, enc)) return f16_gemv_bf16_failure("command completion failed");
        return 1;
    }
}

static int indexer_rope_quant_failure(const char *reason) {
    atomic_fetch_add_explicit(&g_indexer_rope_quant_failures, 1, memory_order_relaxed);
    fprintf(stderr, "myds_metal: indexer RoPE quant fuse rejected: %s; refusing fallback\n", reason);
    return 0;
}

int myds_metal_indexer_rope_quant_dispatch(myds_metal_buffer_t *x,
    uint32_t heads, uint32_t pos, const float frequencies[32]) {
    atomic_fetch_add_explicit(&g_indexer_rope_quant_requests, 1, memory_order_relaxed);
    if (!g_initialized || !g_indexer_rope_quant_fuse ||
        (heads != 1u && heads != 32u) || pos >= 1048576u || !frequencies)
        return indexer_rope_quant_failure("invalid immutable mode, shape or position");
    /* One intentionally in-place range; there is no second GPU input/output
     * range to alias. Reuse/WAR synchronization remains the caller's contract. */
    if (!f16_gemv_bf16_range(x, (uint64_t)heads * 128u * sizeof(float), 4u))
        return indexer_rope_quant_failure("invalid device, full range or alignment");
    if (!indexer_rope_quant_geometry(g_pso_indexer_rope_quant))
        return indexer_rope_quant_failure("candidate PSO or geometry unavailable");
    for (unsigned i = 0; i < 32u; ++i)
        if (!isfinite(frequencies[i]) || frequencies[i] <= 0.0f)
            return indexer_rope_quant_failure("invalid forward compressed frequency");
    struct { uint32_t heads, pos; float frequencies[32]; } args = { heads, pos, {0} };
    memcpy(args.frequencies, frequencies, sizeof(args.frequencies));
    @autoreleasepool {
        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return indexer_rope_quant_failure("command buffer unavailable");
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return indexer_rope_quant_failure("encoder unavailable");
        [enc setComputePipelineState:g_pso_indexer_rope_quant];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:x->mtl_buf offset:x->offset atIndex:1];
        metal_dispatch(enc, "kernel_dsv41_indexer_rope_quant",
            MTLSizeMake(heads, 1, 1), MTLSizeMake(32, 1, 1));
        atomic_fetch_add_explicit(&g_indexer_rope_quant_encoded, 1, memory_order_relaxed);
        if (!v41_finish(cb, enc)) return indexer_rope_quant_failure("command completion failed");
        return 1;
    }
}

static uint32_t qkv_nr0(void) {
    static uint32_t cached=0;
    if(!cached) {const char *v=getenv("MYDS_QKV_NR0");cached=!v||strcmp(v,"4")==0?4u:2u;}
    return cached;
}

static id<MTLComputePipelineState> batch2_shared_pipeline(uint32_t nsg, uint32_t nr0, ggml_type type) {
    if ((nsg != 4 && nsg != 8) || (nr0 != 2 && nr0 != 4) || !g_library) return nil;
    NSString *key = [NSString stringWithFormat:@"batch2_shared_%s_%u_%u",
                     type == GGML_TYPE_Q4_K ? "q4_K" : "q8", nsg, nr0];
    id<MTLComputePipelineState> pipeline = g_native_pipelines[key];
    if (pipeline) return pipeline;
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    short short_nsg = (short)nsg;
    [values setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
    NSString *name = [NSString stringWithFormat:@"kernel_dsv41_%s_shared2_bf16_r%u",
                      type == GGML_TYPE_Q4_K ? "q4_K" : "q8", nr0];
    NSError *error = nil;
    id<MTLFunction> function = [g_library newFunctionWithName:name constantValues:values error:&error];
    if (function) pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "myds_metal: Batch2 %s shared PSO %s NSG%u unavailable; refusing fallback (%s)\n",
                type == GGML_TYPE_Q4_K ? "Q4_K" : "Q8",
                name.UTF8String, nsg, error.localizedDescription.UTF8String ?: "no library/function");
        return nil;
    }
    if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
    g_native_pipelines[key] = pipeline;
    return pipeline;
}

static id<MTLComputePipelineState> batch2_q8_shared_pipeline(uint32_t nsg, uint32_t nr0) {
    return batch2_shared_pipeline(nsg, nr0, GGML_TYPE_Q8_0);
}

static id<MTLComputePipelineState> batch2_q4_shared_pipeline(uint32_t nsg, uint32_t nr0) {
    return batch2_shared_pipeline(nsg, nr0, GGML_TYPE_Q4_K);
}

static int batch2_shared_failure(const char *reason, ggml_type type) {
    atomic_fetch_add_explicit(&g_batch2_q8_shared_failures, 1, memory_order_relaxed);
    fprintf(stderr, "myds_metal: Batch2 %s shared projection rejected: %s; refusing fallback\n",
            type == GGML_TYPE_Q4_K ? "Q4_K" : "Q8", reason);
    return 0;
}

static bool batch2_buffer_range(const myds_metal_buffer_t *buffer, uint64_t bytes) {
    return buffer && buffer->mtl_buf && bytes <= buffer->size &&
           buffer->offset <= buffer->mtl_buf.length && bytes <= buffer->mtl_buf.length - buffer->offset;
}
static bool batch2_buffer_overlap(const myds_metal_buffer_t *a, uint64_t a_bytes,
                                 const myds_metal_buffer_t *b, uint64_t b_bytes) {
    /* Ranges were validated before this subtraction-based overlap check. */
    if (a->mtl_buf != b->mtl_buf) return false;
    return a->offset <= b->offset ? b->offset - a->offset < a_bytes
                                  : a->offset - b->offset < b_bytes;
}

int myds_metal_v41_gemv_bf16_shared2_type(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim, ggml_type type) {
    atomic_fetch_add_explicit(&g_batch2_q8_shared_requests, 1, memory_order_relaxed);
    if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q4_K)
        return batch2_shared_failure("unsupported quantization type", type);
    if (!g_initialized || !out || !input || !out_dim || !in_dim ||
        out_dim > INT32_MAX - 3u || in_dim > INT32_MAX)
        return batch2_shared_failure("invalid runtime or dimensions", type);

    const uint32_t block_size = type == GGML_TYPE_Q4_K ? 256u : 32u;
    const uint32_t block_bytes = type == GGML_TYPE_Q4_K ? 144u : 34u;
    if (in_dim % block_size)
        return batch2_shared_failure("input dimension not divisible by block size", type);

    const uint64_t row_bytes = (uint64_t)(in_dim / block_size) * (uint64_t)block_bytes;
    const uint64_t weight_bytes = row_bytes * out_dim, input_bytes = (uint64_t)in_dim * 4u;
    const uint64_t output_bytes = (uint64_t)out_dim * 4u;
    if (!batch2_buffer_range(weight, weight_bytes) || weight->offset % 2u)
        return batch2_shared_failure("invalid weight range/alignment", type);
    for (unsigned t = 0; t < 2; ++t)
        if (!batch2_buffer_range(input[t], input_bytes) || input[t]->offset % 4u ||
            !batch2_buffer_range(out[t], output_bytes) || out[t]->offset % 16u)
            return batch2_shared_failure("invalid input/output range/alignment", type);
    if (batch2_buffer_overlap(out[0], output_bytes, out[1], output_bytes))
        return batch2_shared_failure("overlapping output rows", type);
    for (unsigned t = 0; t < 2; ++t) {
        if (batch2_buffer_overlap(out[t], output_bytes, weight, weight_bytes))
            return batch2_shared_failure("output overlaps weights", type);
        for (unsigned s = 0; s < 2; ++s)
            if (batch2_buffer_overlap(out[t], output_bytes, input[s], input_bytes))
                return batch2_shared_failure("output overlaps input", type);
    }
    const uint32_t nsg = out_dim > 65536u ? 8u : 4u;
    const uint32_t nr0 = (type == GGML_TYPE_Q4_K) ? ((qkv_nr0() == 4) ? 4u : 2u) : qkv_nr0();
    const NSUInteger shmem_bytes = (type == GGML_TYPE_Q4_K) ? 0 : (32u * nr0 * 2u * sizeof(float));
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = batch2_shared_pipeline(nsg, nr0, type);
        if (!pipeline) return batch2_shared_failure("candidate PSO unavailable", type);
        if (32u * nsg > pipeline.maxTotalThreadsPerThreadgroup ||
            32u * nsg > g_device.maxThreadsPerThreadgroup.width ||
            shmem_bytes > g_device.maxThreadgroupMemoryLength ||
            pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - shmem_bytes)
            return batch2_shared_failure("candidate geometry exceeds PSO/device limits", type);
        /* A public entry may reuse either output while auto barriers are off.
         * Close previous readers/writers before this kernel overwrites it. */
        if (g_batch_cb && !g_auto_barrier) {
            const myds_metal_buffer_t *deps[] = {input[0], input[1], out[0], out[1]};
            myds_metal_barrier_buffers(deps, 4);
        }
        struct ds4_metal_args_mul_mv a = {0};
        a.ne00 = (int32_t)in_dim; a.ne01 = (int32_t)out_dim; a.ne02 = 1;
        a.nb00 = (uint64_t)block_bytes; a.nb01 = row_bytes; a.nb02 = weight_bytes; a.nb03 = weight_bytes;
        a.ne10 = (int32_t)in_dim; a.ne11 = 1; a.ne12 = 1;
        a.nb10 = 4; a.nb11 = input_bytes; a.nb12 = input_bytes; a.nb13 = input_bytes;
        a.ne0 = (int32_t)out_dim; a.ne1 = 1; a.nr0 = (int16_t)nr0; a.r2 = 1; a.r3 = 1;
        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return batch2_shared_failure("command buffer unavailable", type);
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return batch2_shared_failure("encoder unavailable", type);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        const myds_metal_buffer_t *buffers[] = {weight, input[0], input[1], out[0], out[1]};
        for (unsigned i = 0; i < 5; ++i) [enc setBuffer:buffers[i]->mtl_buf offset:buffers[i]->offset atIndex:i + 1u];
        if (shmem_bytes > 0) [enc setThreadgroupMemoryLength:shmem_bytes atIndex:0];
        const char *name = (type == GGML_TYPE_Q4_K) ?
            (nr0 == 4u ? "kernel_dsv41_q4_K_shared2_bf16_r4" : "kernel_dsv41_q4_K_shared2_bf16_r2") :
            (nr0 == 4u ? "kernel_dsv41_q8_shared2_bf16_r4" : "kernel_dsv41_q8_shared2_bf16_r2");
        if (type == GGML_TYPE_Q4_K) {
            NSUInteger ngr = ((NSUInteger)out_dim + nsg * nr0 - 1u) / (nsg * nr0);
            metal_dispatch(enc, name, MTLSizeMake(ngr, 1, 1), MTLSizeMake(32, nsg, 1));
        } else {
            metal_dispatch(enc, name, MTLSizeMake(((NSUInteger)out_dim + nr0 - 1u) / nr0, 1, 1), MTLSizeMake(32, nsg, 1));
        }
        atomic_fetch_add_explicit(&g_batch2_q8_shared_encoded, 1, memory_order_relaxed);
        if (!v41_finish(cb, enc)) return batch2_shared_failure("command completion failed", type);
        return 1;
    }
}

int myds_metal_v41_gemv_bf16_shared2(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim) {
    return myds_metal_v41_gemv_bf16_shared2_type(out, weight, input, out_dim, in_dim, GGML_TYPE_Q8_0);
}

static id<MTLComputePipelineState> batch2_dense_shared_pipeline(uint32_t nsg, uint32_t nr0, ggml_type type) {
    if ((nsg != 4 && nsg != 8) || (nr0 != 2 && nr0 != 4) || !g_library) return nil;
    NSString *key = [NSString stringWithFormat:@"batch2_dense_shared_%s_%u_%u",
                     type == GGML_TYPE_F16 ? "f16" : "f32", nsg, nr0];
    id<MTLComputePipelineState> pipeline = g_native_pipelines[key];
    if (pipeline) return pipeline;
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    short short_nsg = (short)nsg;
    [values setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
    NSString *name = [NSString stringWithFormat:@"kernel_dsv41_%s_shared2_r%u",
                      type == GGML_TYPE_F16 ? "f16" : "f32", nr0];
    NSError *error = nil;
    id<MTLFunction> function = [g_library newFunctionWithName:name constantValues:values error:&error];
    if (function) pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "myds_metal: Batch2 %s shared PSO %s NSG%u unavailable; refusing fallback (%s)\n",
                type == GGML_TYPE_F16 ? "F16" : "F32",
                name.UTF8String, nsg, error.localizedDescription.UTF8String ?: "no library/function");
        return nil;
    }
    if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
    g_native_pipelines[key] = pipeline;
    return pipeline;
}

int myds_metal_v41_gemv_shared2_dense(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim, ggml_type type) {
    atomic_fetch_add_explicit(&g_batch2_q8_shared_requests, 1, memory_order_relaxed);
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_F32)
        return batch2_shared_failure("unsupported dense type", type);
    if (!g_initialized || !out || !input || !weight || !out[0] || !out[1] || !input[0] || !input[1] ||
        !out_dim || !in_dim || out_dim > INT32_MAX - 3u || in_dim > INT32_MAX ||
        (in_dim % 32u != 0))
        return batch2_shared_failure("invalid runtime or dimensions", type);

    const uint64_t elem_bytes = (type == GGML_TYPE_F32) ? sizeof(float) : sizeof(uint16_t);
    const uint64_t row_bytes = (uint64_t)in_dim * elem_bytes;
    const uint64_t weight_bytes = row_bytes * out_dim;
    const uint64_t input_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t output_bytes = (uint64_t)out_dim * sizeof(float);

    if (!batch2_buffer_range(weight, weight_bytes) || weight->offset % elem_bytes)
        return batch2_shared_failure("invalid weight range/alignment", type);
    for (unsigned t = 0; t < 2; ++t)
        if (!batch2_buffer_range(input[t], input_bytes) || input[t]->offset % 4u ||
            !batch2_buffer_range(out[t], output_bytes) || out[t]->offset % 16u)
            return batch2_shared_failure("invalid input/output range/alignment", type);
    if (batch2_buffer_overlap(out[0], output_bytes, out[1], output_bytes))
        return batch2_shared_failure("overlapping output rows", type);
    for (unsigned t = 0; t < 2; ++t) {
        if (batch2_buffer_overlap(out[t], output_bytes, weight, weight_bytes))
            return batch2_shared_failure("output overlaps weights", type);
        for (unsigned s = 0; s < 2; ++s)
            if (batch2_buffer_overlap(out[t], output_bytes, input[s], input_bytes))
                return batch2_shared_failure("output overlaps input", type);
    }
    const uint32_t nsg = (in_dim >= 4096u) ? 8u : 4u;
    const uint32_t nr0 = (qkv_nr0() == 4 && (out_dim % 4u == 0)) ? 4u : 2u;
    const NSUInteger shmem_bytes = 32u * nr0 * 2u * sizeof(float);
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = batch2_dense_shared_pipeline(nsg, nr0, type);
        if (!pipeline) return batch2_shared_failure("candidate dense PSO unavailable", type);
        if (32u * nsg > pipeline.maxTotalThreadsPerThreadgroup ||
            shmem_bytes > g_device.maxThreadgroupMemoryLength ||
            pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - shmem_bytes)
            return batch2_shared_failure("candidate geometry exceeds PSO/device limits", type);

        if (g_batch_cb && !g_auto_barrier) {
            const myds_metal_buffer_t *deps[] = {input[0], input[1], out[0], out[1]};
            myds_metal_barrier_buffers(deps, 4);
        }
        struct ds4_metal_args_mul_mv a = {0};
        a.ne00 = (int32_t)in_dim; a.ne01 = (int32_t)out_dim; a.ne02 = 1;
        a.nb00 = elem_bytes; a.nb01 = row_bytes; a.nb02 = weight_bytes; a.nb03 = weight_bytes;
        a.ne10 = (int32_t)in_dim; a.ne11 = 1; a.ne12 = 1;
        a.nb10 = sizeof(float); a.nb11 = input_bytes; a.nb12 = input_bytes; a.nb13 = input_bytes;
        a.ne0 = (int32_t)out_dim; a.ne1 = 1; a.nr0 = (int16_t)nr0; a.r2 = 1; a.r3 = 1;

        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return batch2_shared_failure("command buffer unavailable", type);
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return batch2_shared_failure("encoder unavailable", type);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        const myds_metal_buffer_t *buffers[] = {weight, input[0], input[1], out[0], out[1]};
        for (unsigned i = 0; i < 5; ++i) [enc setBuffer:buffers[i]->mtl_buf offset:buffers[i]->offset atIndex:i + 1u];
        [enc setThreadgroupMemoryLength:shmem_bytes atIndex:0];
        const char *name = (type == GGML_TYPE_F16) ?
            (nr0 == 4u ? "kernel_dsv41_f16_shared2_r4" : "kernel_dsv41_f16_shared2_r2") :
            (nr0 == 4u ? "kernel_dsv41_f32_shared2_r4" : "kernel_dsv41_f32_shared2_r2");
        metal_dispatch(enc, name, MTLSizeMake(((NSUInteger)out_dim + nr0 - 1u) / nr0, 1, 1), MTLSizeMake(32, nsg, 1));
        atomic_fetch_add_explicit(&g_batch2_q8_shared_encoded, 1, memory_order_relaxed);
        if (!v41_finish(cb, enc)) return batch2_shared_failure("command completion failed", type);
        return 1;
    }
}

int myds_metal_v41_hc_mix_batch2(
    myds_metal_buffer_t *split_b2,
    myds_metal_buffer_t *split[2],
    const myds_metal_buffer_t *residual[2],
    const myds_metal_buffer_t *fn,
    ggml_type fn_type,
    const myds_metal_buffer_t *scale,
    const myds_metal_buffer_t *base,
    myds_metal_buffer_t *flat_norm[2],
    myds_metal_buffer_t *mix_b2,
    myds_metal_buffer_t *mix[2],
    uint32_t dim,
    float eps,
    float hc_eps) {
    if (!g_initialized || !dim || dim > INT32_MAX / 4 || dim % 128 ||
        !isfinite(hc_eps) || hc_eps < 0 ||
        !split_b2 || !mix_b2 || !fn || !scale || !base ||
        !split || !split[0] || !split[1] ||
        !residual || !residual[0] || !residual[1] ||
        !flat_norm || !flat_norm[0] || !flat_norm[1] ||
        !mix || !mix[0] || !mix[1]) return 0;

    /* 1. RMSNorm on both tokens */
    if (!myds_metal_v41_rmsnorm(flat_norm[0], residual[0], NULL, 4u * dim, eps, false)) return 0;
    if (!myds_metal_v41_rmsnorm(flat_norm[1], residual[1], NULL, 4u * dim, eps, false)) return 0;
    if (!g_auto_barrier) {
        const myds_metal_buffer_t *fn_deps[] = {flat_norm[0], flat_norm[1]};
        myds_metal_barrier_buffers(fn_deps, 2);
    }

    /* 2. Shared-weight GEMV across both tokens (reads fn weights from DRAM only once) */
    int ok = 0;
    if (fn_type == GGML_TYPE_F16 || fn_type == GGML_TYPE_F32) {
        ok = myds_metal_v41_gemv_shared2_dense(mix, fn, flat_norm, 24, 4u * dim, fn_type);
    }
    if (!ok) {
        for (unsigned t = 0; t < 2; ++t) {
            if (!(fn_type == GGML_TYPE_F16
                    ? myds_metal_v41_gemv(mix[t], fn, flat_norm[t], 24, 4u * dim, fn_type)
                    : myds_metal_gemv(mix[t], fn, flat_norm[t], 24, 4u * dim, fn_type))) return 0;
        }
    }
    if (!g_auto_barrier) {
        const myds_metal_buffer_t *mx_deps[] = {mix[0], mix[1]};
        myds_metal_barrier_buffers(mx_deps, 2);
    }

    /* 3. Batched HC Split across both tokens */
    return myds_metal_v41_hc_split_batch(split_b2, mix_b2, scale, base, 2, hc_eps);
}

static id<MTLComputePipelineState> batch2_q8_f32_shared_pipeline(uint32_t nsg, uint32_t nr0) {
    if ((nsg != 4 && nsg != 8) || (nr0 != 2 && nr0 != 4) || !g_library) return nil;
    NSString *key = [NSString stringWithFormat:@"batch2_shared_q8_f32_%u_%u", nsg, nr0];
    id<MTLComputePipelineState> pipeline = g_native_pipelines[key];
    if (pipeline) return pipeline;
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    short short_nsg = (short)nsg;
    [values setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
    NSString *name = [NSString stringWithFormat:@"kernel_dsv41_q8_shared2_f32_r%u", nr0];
    NSError *error = nil;
    id<MTLFunction> function = [g_library newFunctionWithName:name constantValues:values error:&error];
    if (function) pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "myds_metal: Batch2 Q8 F32 shared PSO %s NSG%u unavailable; refusing fallback (%s)\n",
                name.UTF8String, nsg, error.localizedDescription.UTF8String ?: "no library/function");
        return nil;
    }
    if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
    g_native_pipelines[key] = pipeline;
    return pipeline;
}

int myds_metal_v41_gemv_shared2_q8_f32(myds_metal_buffer_t *const out[2],
    const myds_metal_buffer_t *weight, myds_metal_buffer_t *const input[2],
    uint32_t out_dim, uint32_t in_dim) {
    atomic_fetch_add_explicit(&g_batch2_q8_shared_requests, 1, memory_order_relaxed);
    if (!g_initialized || !out || !input || !weight || !out[0] || !out[1] || !input[0] || !input[1] ||
        !out_dim || !in_dim || out_dim > INT32_MAX - 3u || in_dim > INT32_MAX)
        return batch2_shared_failure("invalid runtime or dimensions", GGML_TYPE_Q8_0);

    const uint32_t block_size = 32u;
    const uint32_t block_bytes = 34u;
    if (in_dim % block_size)
        return batch2_shared_failure("input dimension not divisible by block size", GGML_TYPE_Q8_0);

    const uint64_t row_bytes = (uint64_t)(in_dim / block_size) * (uint64_t)block_bytes;
    const uint64_t weight_bytes = row_bytes * out_dim, input_bytes = (uint64_t)in_dim * 4u;
    const uint64_t output_bytes = (uint64_t)out_dim * 4u;
    if (!batch2_buffer_range(weight, weight_bytes) || weight->offset % 2u)
        return batch2_shared_failure("invalid weight range/alignment", GGML_TYPE_Q8_0);
    for (unsigned t = 0; t < 2; ++t)
        if (!batch2_buffer_range(input[t], input_bytes) || input[t]->offset % 4u ||
            !batch2_buffer_range(out[t], output_bytes) || out[t]->offset % 16u)
            return batch2_shared_failure("invalid input/output range/alignment", GGML_TYPE_Q8_0);
    if (batch2_buffer_overlap(out[0], output_bytes, out[1], output_bytes))
        return batch2_shared_failure("overlapping output rows", GGML_TYPE_Q8_0);
    for (unsigned t = 0; t < 2; ++t) {
        if (batch2_buffer_overlap(out[t], output_bytes, weight, weight_bytes))
            return batch2_shared_failure("output overlaps weights", GGML_TYPE_Q8_0);
        for (unsigned s = 0; s < 2; ++s)
            if (batch2_buffer_overlap(out[t], output_bytes, input[s], input_bytes))
                return batch2_shared_failure("output overlaps input", GGML_TYPE_Q8_0);
    }
    const uint32_t nsg = out_dim > 65536u ? 8u : 4u;
    const uint32_t nr0 = (out_dim > 65536u) ? 4u : ((qkv_nr0() == 4 && (out_dim % 4u == 0)) ? 4u : 2u);
    const NSUInteger shmem_bytes = 32u * nr0 * 2u * sizeof(float);
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = batch2_q8_f32_shared_pipeline(nsg, nr0);
        if (!pipeline) return batch2_shared_failure("candidate PSO unavailable", GGML_TYPE_Q8_0);
        if (32u * nsg > pipeline.maxTotalThreadsPerThreadgroup ||
            32u * nsg > g_device.maxThreadsPerThreadgroup.width ||
            shmem_bytes > g_device.maxThreadgroupMemoryLength ||
            pipeline.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength - shmem_bytes)
            return batch2_shared_failure("candidate geometry exceeds PSO/device limits", GGML_TYPE_Q8_0);
        if (g_batch_cb && !g_auto_barrier) {
            const myds_metal_buffer_t *deps[] = {input[0], input[1], out[0], out[1]};
            myds_metal_barrier_buffers(deps, 4);
        }
        struct ds4_metal_args_mul_mv a = {0};
        a.ne00 = (int32_t)in_dim; a.ne01 = (int32_t)out_dim; a.ne02 = 1;
        a.nb00 = (uint64_t)block_bytes; a.nb01 = row_bytes; a.nb02 = weight_bytes; a.nb03 = weight_bytes;
        a.ne10 = (int32_t)in_dim; a.ne11 = 1; a.ne12 = 1;
        a.nb10 = 4; a.nb11 = input_bytes; a.nb12 = input_bytes; a.nb13 = input_bytes;
        a.ne0 = (int32_t)out_dim; a.ne1 = 1; a.nr0 = (int16_t)nr0; a.r2 = 1; a.r3 = 1;
        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return batch2_shared_failure("command buffer unavailable", GGML_TYPE_Q8_0);
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return batch2_shared_failure("encoder unavailable", GGML_TYPE_Q8_0);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        const myds_metal_buffer_t *buffers[] = {weight, input[0], input[1], out[0], out[1]};
        for (unsigned i = 0; i < 5; ++i) [enc setBuffer:buffers[i]->mtl_buf offset:buffers[i]->offset atIndex:i + 1u];
        if (shmem_bytes > 0) [enc setThreadgroupMemoryLength:shmem_bytes atIndex:0];
        const char *name = (nr0 == 4u ? "kernel_dsv41_q8_shared2_f32_r4" : "kernel_dsv41_q8_shared2_f32_r2");
        metal_dispatch(enc, name, MTLSizeMake(((NSUInteger)out_dim + nr0 - 1u) / nr0, 1, 1), MTLSizeMake(32, nsg, 1));
        atomic_fetch_add_explicit(&g_batch2_q8_shared_encoded, 1, memory_order_relaxed);
        if (!v41_finish(cb, enc)) return batch2_shared_failure("command completion failed", GGML_TYPE_Q8_0);
        return 1;
    }
}



int myds_metal_v41_gemv_bf16(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight,const myds_metal_buffer_t *input,
    uint32_t out_dim,uint32_t in_dim,ggml_type type) {
    if (type == GGML_TYPE_Q4_K) {
        if (!out || !weight || !input || !out_dim || !in_dim || out_dim > INT32_MAX ||
            in_dim > INT32_MAX || in_dim % 256u || out->offset % 16u) return 0;
        const uint64_t row_bytes = (uint64_t)(in_dim / 256u) * 144u;
        if (weight->size / row_bytes < out_dim || input->size / 4u < in_dim || out->size / 4u < out_dim) return 0;
        const uint32_t nsg = 4u, nr0 = 2u;
        id<MTLComputePipelineState> pipeline = g_pso_mul_mv_q4_K_dense_bf16;
        if (!pipeline) {
            NSString *key = [NSString stringWithFormat:@"q4_k_bf16_%u", nsg];
            pipeline = g_native_pipelines[key];
            if (!pipeline && g_library) {
                MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
                short short_nsg = (short)nsg;
                [values setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
                id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_mul_mv_q4_K_dense_f32_bf16"
                    constantValues:values error:nil];
                if (fn) pipeline = [g_device newComputePipelineStateWithFunction:fn error:nil];
                if (pipeline) {
                    if (!g_native_pipelines) g_native_pipelines = [NSMutableDictionary new];
                    g_native_pipelines[key] = pipeline;
                }
            }
        }
        if (!pipeline || !out->mtl_buf || !weight->mtl_buf || !input->mtl_buf)
            return myds_metal_gemv(out, weight, input, out_dim, in_dim, type) && myds_metal_v41_bf16(out, out_dim);
        @autoreleasepool {
            struct ds4_metal_args_mul_mv a = {0};
            a.ne00 = (int32_t)in_dim; a.ne01 = (int32_t)out_dim; a.ne02 = 1;
            a.nb00 = 144; a.nb01 = row_bytes; a.nb02 = row_bytes * out_dim; a.nb03 = a.nb02;
            a.ne10 = (int32_t)in_dim; a.ne11 = 1; a.ne12 = 1; a.nb10 = 4; a.nb11 = (uint64_t)in_dim * 4;
            a.nb12 = a.nb11; a.nb13 = a.nb12; a.ne0 = (int32_t)out_dim; a.ne1 = 1; a.nr0 = (int16_t)nr0; a.r2 = 1; a.r3 = 1;
            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            if (!cb) return 0;
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            if (!enc || 32u * nsg > pipeline.maxTotalThreadsPerThreadgroup) return 0;
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            [enc setBuffer:weight->mtl_buf offset:weight->offset atIndex:1];
            [enc setBuffer:input->mtl_buf offset:input->offset atIndex:2];
            [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
            [enc setThreadgroupMemoryLength:32u * nr0 * sizeof(float) atIndex:0];
            NSUInteger ngr = ((NSUInteger)out_dim + nsg * nr0 - 1u) / (nsg * nr0);
            metal_dispatch(enc, "gemv_q4_k_bf16", MTLSizeMake(ngr, 1, 1), MTLSizeMake(32, nsg, 1));
            return v41_finish(cb, enc);
        }
    }
    if(type!=GGML_TYPE_Q8_0) {
        int ok=(type==GGML_TYPE_F16||type==GGML_TYPE_F32)?
            myds_metal_v41_gemv(out,weight,input,out_dim,in_dim,type):
            myds_metal_gemv(out,weight,input,out_dim,in_dim,type);
        return ok&&myds_metal_v41_bf16(out,out_dim);
    }
    if(!out||!weight||!input||!out_dim||!in_dim||out_dim>INT32_MAX||
       in_dim>INT32_MAX||in_dim%32u||out->offset%16u)return 0;
    const uint64_t row_bytes=(uint64_t)(in_dim/32u)*34u;
    if(weight->size/row_bytes<out_dim||input->size/4u<in_dim||out->size/4u<out_dim)return 0;
    const uint32_t nsg=out_dim>65536u?8u:4u,nr0=qkv_nr0();
    @autoreleasepool {
        NSString *key=[NSString stringWithFormat:@"q8_0_bf16_%u_%u",nsg,nr0];
        id<MTLComputePipelineState> pipeline=g_native_pipelines[key];
        if(!pipeline&&g_library) {
            MTLFunctionConstantValues *values=[MTLFunctionConstantValues new];
            short short_nsg=(short)nsg;
            [values setConstantValue:&short_nsg type:MTLDataTypeShort atIndex:600];
            NSString *name=nr0==4u?@"kernel_mul_mv_q8_0_f32_bf16_r4":@"kernel_mul_mv_q8_0_f32_bf16";
            id<MTLFunction> fn=[g_library newFunctionWithName:name
                constantValues:values error:nil];
            if(fn)pipeline=[g_device newComputePipelineStateWithFunction:fn error:nil];
            if(pipeline) {
                if(!g_native_pipelines)g_native_pipelines=[NSMutableDictionary new];
                g_native_pipelines[key]=pipeline;
            }
        }
        if(!pipeline||!out->mtl_buf||!weight->mtl_buf||!input->mtl_buf)
            return myds_metal_gemv(out,weight,input,out_dim,in_dim,type)&&myds_metal_v41_bf16(out,out_dim);
        struct ds4_metal_args_mul_mv a={0};
        a.ne00=(int32_t)in_dim;a.ne01=(int32_t)out_dim;a.ne02=1;
        a.nb00=34;a.nb01=row_bytes;a.nb02=row_bytes*out_dim;a.nb03=a.nb02;
        a.ne10=(int32_t)in_dim;a.ne11=1;a.ne12=1;a.nb10=4;a.nb11=(uint64_t)in_dim*4;
        a.nb12=a.nb11;a.nb13=a.nb12;a.ne0=(int32_t)out_dim;a.ne1=1;a.nr0=(int16_t)nr0;a.r2=1;a.r3=1;
        id<MTLCommandBuffer> cb=g_batch_cb?g_batch_cb:[g_queue commandBuffer];
        if(!cb)return 0;
        id<MTLComputeCommandEncoder> enc=get_compute_encoder(cb);
        if(!enc||32u*nsg>pipeline.maxTotalThreadsPerThreadgroup)return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:weight->mtl_buf offset:weight->offset atIndex:1];
        [enc setBuffer:input->mtl_buf offset:input->offset atIndex:2];
        [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
        [enc setThreadgroupMemoryLength:32u*nr0*sizeof(float) atIndex:0];
        metal_dispatch(enc,"gemv_q8_0_bf16",MTLSizeMake(((NSUInteger)out_dim+nr0-1u)/nr0,1,1),MTLSizeMake(32,nsg,1));
        return v41_finish(cb,enc);
    }
}

int myds_metal_v41_qkv_fusion_available(void) {
    if(!g_initialized||g_inject_copy_fail_at)return 0;
    if(g_qkv_finish_available<0) {
        @autoreleasepool {
            NSString *key=@"kernel_dsv41_qkv_finish";
            id<MTLComputePipelineState> p=g_native_pipelines[key];
            if(!p) {
                id<MTLFunction> fn=[g_library newFunctionWithName:key];
                if(fn)p=[g_device newComputePipelineStateWithFunction:fn error:nil];
            }
            if(p) {
                if(!g_native_pipelines)g_native_pipelines=[NSMutableDictionary new];
                g_native_pipelines[key]=p;
            }
            g_qkv_finish_available=p&&p.maxTotalThreadsPerThreadgroup>=32u;
        }
    }
    return g_qkv_finish_available;
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
    /* 追加在末尾：0 表示原来的紧凑布局（dst 行步长 = ne0）。 */
    uint64_t nb_out;
    uint64_t nb_out_batch;
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
    } else if (weight_type == GGML_TYPE_Q4_K) {
        if (in_dim % 256u != 0) return 0;
        pipeline = bc_out ? g_pso_mul_mm_q4_K_bco : g_pso_mul_mm_q4_K;
        row_bytes = (in_dim / 256u) * 144u;
    } else if (weight_type == GGML_TYPE_F16) {
        pipeline = bc_out ? g_pso_mul_mm_f16_bco : g_pso_mul_mm_f16;
        row_bytes = in_dim * sizeof(uint16_t);
    } else if (weight_type == GGML_TYPE_F32) {
        pipeline = bc_out ? g_pso_mul_mm_f32_bco : g_pso_mul_mm_f32;
        row_bytes = in_dim * sizeof(float);
    }

    if (row_bytes == 0 ||
        W->size / row_bytes < out_dim ||
        src1->size / sizeof(float) / in_dim < n_tok ||
        dst->size / sizeof(float) / out_dim < n_tok) return 0;

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
            metal_dispatch(enc, "gemm_batch", MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                                  ((NSUInteger)out_dim + 63u) / 64u,
                                                  1),
                 MTLSizeMake(128, 1, 1));

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

/* 分组矩阵乘：一组一个权重矩阵，所有组在【一次】派发里完成。
 *
 * 布局（DS4 的 attention 输出投影就是这种）：
 *   src1 : [n_tok][n_groups * in_dim]   —— 每行里第 g 组占 [g*in_dim, (g+1)*in_dim)
 *   W    : [n_groups][out_dim][in_dim]  —— 第 g 组的权重紧跟在上一组之后
 *   dst  : [n_tok][n_groups * out_dim]  —— 每行里第 g 组写 [g*out_dim, (g+1)*out_dim)
 *
 * 这和逐 (组, token) 的 matvec 是同一个数学，区别只在重用方式：
 * matvec 那条路每个 threadgroup 都完整读一遍该组的权重，权重被读 n_tok 遍
 * （本模型是 512 遍，17 GB / layer-chunk）；这里权重按块读进 threadgroup 后
 * 同时服务 32 个 token。
 *
 * 各组之间的偏移分别落在 nb13（输入）、nb03（权重）、nb_out_batch（输出）上，
 * 因此 tgpig.z 就是组号，内核本身不需要循环。 */
int myds_metal_mul_mm_grouped(
    myds_metal_buffer_t       *dst,
    const myds_metal_buffer_t *W,
    const myds_metal_buffer_t *src1,
    uint32_t                   out_dim,
    uint32_t                   in_dim,
    uint32_t                   n_groups,
    uint32_t                   n_tok,
    ggml_type                  weight_type) {
    if (!dst || !W || !src1 || !out_dim || !in_dim || !n_groups || !n_tok) return 0;

    const bool bc_inp = (in_dim % 32u) != 0;
    const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;

    id<MTLComputePipelineState> pipeline = nil;
    uint64_t row_bytes = 0;

    if (weight_type == GGML_TYPE_Q8_0) {
        pipeline = bc_out ? g_pso_mul_mm_q8_0_bco : g_pso_mul_mm_q8_0;
        row_bytes = (in_dim / 32u) * 34u;
    } else if (weight_type == GGML_TYPE_Q4_K) {
        if (in_dim % 256u != 0) return 0;
        pipeline = bc_out ? g_pso_mul_mm_q4_K_bco : g_pso_mul_mm_q4_K;
        row_bytes = (in_dim / 256u) * 144u;
    } else if (weight_type == GGML_TYPE_F16) {
        pipeline = bc_out ? g_pso_mul_mm_f16_bco : g_pso_mul_mm_f16;
        row_bytes = in_dim * sizeof(uint16_t);
    } else if (weight_type == GGML_TYPE_F32) {
        pipeline = bc_out ? g_pso_mul_mm_f32_bco : g_pso_mul_mm_f32;
        row_bytes = in_dim * sizeof(float);
    }

    if (!pipeline || !W->mtl_buf || !src1->mtl_buf || !dst->mtl_buf || bc_inp) return 0;
    if (row_bytes == 0 ||
        W->size / row_bytes / out_dim < n_groups ||
        src1->size / sizeof(float) / in_dim / n_groups < n_tok ||
        dst->size / sizeof(float) / out_dim / n_groups < n_tok) return 0;

    @autoreleasepool {
        struct ds4_metal_args_mul_mm args;
        memset(&args, 0, sizeof(args));
        args.ne00 = (int32_t)in_dim;
        args.ne02 = 1;
        args.nb01 = row_bytes;                          /* 权重一行（in_dim 宽） */
        args.nb02 = row_bytes * out_dim;                /* 未用：ne12 == 1 ⇒ i12 == 0 */
        args.nb03 = row_bytes * out_dim;                /* 每组权重的字节数 */
        args.ne12 = 1;
        args.nb10 = sizeof(float);
        args.nb11 = (uint64_t)n_groups * in_dim * sizeof(float);  /* 输入的 token 行步长 */
        args.nb12 = 0;                                  /* 未用 */
        args.nb13 = (uint64_t)in_dim * sizeof(float);             /* 输入里组的步长 */
        args.ne0 = (int32_t)out_dim;
        args.ne1 = (int32_t)n_tok;
        args.r2 = 1;
        args.r3 = 1;
        args.nb_out = (uint64_t)n_groups * out_dim;     /* 输出的 token 行步长 */
        args.nb_out_batch = out_dim;                    /* 输出行内组的偏移 */

        id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:W->mtl_buf offset:W->offset atIndex:1];
        [enc setBuffer:src1->mtl_buf offset:src1->offset atIndex:2];
        [enc setBuffer:dst->mtl_buf offset:dst->offset atIndex:3];
        uint32_t tg_mem = (weight_type == GGML_TYPE_F32) ? 12288u : (bc_out ? 8192u : 6144u);
        [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
        metal_dispatch(enc, "gemm_grouped", MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              (NSUInteger)n_groups),
             MTLSizeMake(128, 1, 1));

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

/* 诊断：MoE 四步分步计时（MYDS_MOE_STEP_PROF=1 才开）。
 * 在每步 dispatch 后 commit+wait，从而把「MoE GEMMs」这个大窗口拆成
 * map0 / pair_swiglu / down / sum6 四段。默认零开销。 */
static int    g_moe_split = -1;
static double g_moe_t_map0 = 0, g_moe_t_swiglu = 0, g_moe_t_down = 0, g_moe_t_sum = 0;
static long   g_moe_split_calls = 0;

static void moe_split_point(id<MTLCommandBuffer> *cbp,
                            id<MTLComputeCommandEncoder> *encp, double *acc) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    myds_metal_commit_batch();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *acc += (t1.tv_sec - t0.tv_sec) * 1000.0 + (double)(t1.tv_nsec - t0.tv_nsec) / 1.0e6;
    /* 与仓库其它剖析点同一模式：commit 之后必须 begin，否则 encoder 生命周期错乱
     * （首次实现漏了这步，实测直接 abort：encoder released without endEncoding）。 */
    myds_metal_begin_batch();
    *cbp = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
    *encp = get_compute_encoder(*cbp);
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
    if (n_activated_experts != 6 && n_activated_experts != 3) {
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
            metal_dispatch(enc, "moe_map0", MTLSizeMake(1, 1, 1),
                MTLSizeMake((NSUInteger)n_routed_experts, 1, 1));
            if (g_moe_split < 0) g_moe_split = getenv("MYDS_MOE_STEP_PROF") ? 1 : 0;
            if (g_moe_split) moe_split_point(&cb, &enc, &g_moe_t_map0);

            if (g_auto_barrier) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                id<MTLResource> res = g_moe_id_map_buffer;
                [enc memoryBarrierWithResources:&res count:1];
            }

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
            metal_dispatch(enc, "moe_pair_swiglu", MTLSizeMake((NSUInteger)work_cap,
                                                  ((NSUInteger)expert_dim + 63u) / 64u,
                                                  1),
                MTLSizeMake(128, 1, 1));
            if (g_moe_split) moe_split_point(&cb, &enc, &g_moe_t_swiglu);

            if (g_auto_barrier) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                id<MTLResource> res = scratch_mid->mtl_buf;
                [enc memoryBarrierWithResources:&res count:1];
            }

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
            metal_dispatch(enc, "moe_down_q2_k", MTLSizeMake((NSUInteger)work_cap,
                                                  ((NSUInteger)dim + 63u) / 64u,
                                                  1),
                MTLSizeMake(128, 1, 1));
            if (g_moe_split) moe_split_point(&cb, &enc, &g_moe_t_down);

            if (g_auto_barrier) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                id<MTLResource> res = scratch_down->mtl_buf;
                [enc memoryBarrierWithResources:&res count:1];
            }

            /* Step 4: Sum Experts dispatch */
            NSUInteger nth = g_pso_moe_sum6.maxTotalThreadsPerThreadgroup;
            if (nth > 256u) nth = 256u;
            if (nth > dim) nth = dim;
            if (nth == 0) nth = 1u;
            [enc setComputePipelineState:g_pso_moe_sum6];
            [enc setBytes:&sum_args length:sizeof(sum_args) atIndex:0];
            [enc setBuffer:scratch_down->mtl_buf offset:scratch_down->offset atIndex:1];
            [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:2];
            metal_dispatch(enc, "moe_sum6", MTLSizeMake((NSUInteger)n_tok, 1, 1),
                MTLSizeMake(nth, 1, 1));
            if (g_moe_split) {
                moe_split_point(&cb, &enc, &g_moe_t_sum);
                g_moe_split_calls++;
                if ((g_moe_split_calls % 20) == 0)
                    printf("[MOE-STEP] n=%ld  map0=%.0f  swiglu=%.0f  down=%.0f  sum6=%.0f ms\n",
                           g_moe_split_calls, g_moe_t_map0, g_moe_t_swiglu, g_moe_t_down, g_moe_t_sum);
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
        const NSUInteger nsg_gate = (NSUInteger)g_moe_nsg;
        const NSUInteger rows_per_group_gate = nr0_gate * nsg_gate;
        const NSUInteger row_groups_gate = (expert_dim + rows_per_group_gate - 1) / rows_per_group_gate;
        const NSUInteger pairs = (NSUInteger)n_activated_experts * (NSUInteger)n_tok;
        const NSUInteger smem_iq2 = g_moe_iq2_constant_lut ? 0u :
            256u * sizeof(uint64_t) + 128u * sizeof(uint8_t); /* reference: 2176 */
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
        const NSUInteger nsg_down = (NSUInteger)g_moe_down_nsg;
        const NSUInteger rows_per_group_down = nr0_down * nsg_down;
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
            bool use_batch_dedup = (n_tok == 2) && get_moe_batch_dedup_enabled() &&
                                   g_pso_moe_pair_swiglu_batch_iq2_xxs && g_pso_moe_sum6_batch_q2_k;
            static bool logged_moe_dedup = false;
            if (!logged_moe_dedup && use_batch_dedup) {
                MYDS_LOG_I("MoE batch-2 path: DEDUP (2-token expert weight sharing) — active");
                logged_moe_dedup = true;
            }

            if (use_batch_dedup) {
                /* Dedup Batch-2 Gate + Up + SwiGLU + Scaling dispatch (IQ2_XXS) across 2 tokens */
                [enc setComputePipelineState:g_pso_moe_pair_swiglu_batch_iq2_xxs];
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
                metal_dispatch(enc, "fused_moe_swiglu_batch", MTLSizeMake(row_groups_gate, 1, pairs),
                    MTLSizeMake(32, nsg_gate, 1));
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                } else {
                    id<MTLResource> res = scratch_mid->mtl_buf;
                    [enc memoryBarrierWithResources:&res count:1];
                }

                /* Dedup Batch-2 Down + Sum6 direct accumulation dispatch (Q2_K) across 2 tokens */
                [enc setComputePipelineState:g_pso_moe_sum6_batch_q2_k];
                [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
                [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
                [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
                [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:3];
                [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
                [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:5];
                metal_dispatch(enc, "fused_moe_sum6_batch_q2_k", MTLSizeMake(row_groups_down, 1, 1),
                    MTLSizeMake(32, nsg_down, 1));
            } else {
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
                metal_dispatch(enc, "fused_moe_swiglu", MTLSizeMake(row_groups_gate, 1, pairs),
                    MTLSizeMake(32, nsg_gate, 1));
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                } else {
                    id<MTLResource> res = scratch_mid->mtl_buf;
                    [enc memoryBarrierWithResources:&res count:1];
                }

                /* Fused Down + Sum6 direct accumulation dispatch (Q2_K) across all tokens */
                [enc setComputePipelineState:g_pso_moe_sum6_q2_k];
                [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
                [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
                [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
                [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:3];
                [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
                [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:5];
                metal_dispatch(enc, "fused_moe_sum6_q2_k", MTLSizeMake(row_groups_down, (NSUInteger)n_tok, 1),
                    MTLSizeMake(32, nsg_down, 1));
            }

            if (g_batch_cb) {
                if (g_auto_barrier) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                } else {
                    id<MTLResource> res = moe_out->mtl_buf;
                    [enc memoryBarrierWithResources:&res count:1];
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

static int moe_q2_sum_cache_failure(const char *reason) {
    atomic_fetch_add_explicit(&g_moe_q2_sum_cache_failures, 1, memory_order_relaxed);
    fprintf(stderr, "myds_metal: Q2 sum cache rejected: %s; refusing fallback\n", reason);
    return 0;
}

static bool moe_q2_sum_cache_range(const myds_metal_buffer_t *buffer, uint64_t bytes, size_t alignment) {
    return buffer && buffer->mtl_buf && buffer->mtl_buf.device == g_device && bytes <= buffer->size &&
           buffer->offset <= buffer->mtl_buf.length && bytes <= buffer->mtl_buf.length - buffer->offset &&
           buffer->offset % alignment == 0 && (!buffer->ptr || (uintptr_t)buffer->ptr % alignment == 0);
}

static bool moe_q2_sum_cache_overlap(const myds_metal_buffer_t *a, uint64_t a_bytes,
                                     const myds_metal_buffer_t *b, uint64_t b_bytes) {
    if (a->mtl_buf == b->mtl_buf)
        return a->offset <= b->offset ? b->offset - a->offset < a_bytes : a->offset - b->offset < b_bytes;
    // Different no-copy MTLBuffer objects may still expose overlapping shared
    // memory. Ranges were checked before these subtraction-based comparisons.
    if (a->ptr && b->ptr) {
        const uintptr_t first = (uintptr_t)a->ptr, second = (uintptr_t)b->ptr;
        return first <= second ? second - first < a_bytes : first - second < b_bytes;
    }
    return false;
}

static int moe_q2_sum_cache_fused_failure(const char *reason) {
    atomic_fetch_add_explicit(&g_moe_q2_sum_cache_fused_failures, 1, memory_order_relaxed);
    fprintf(stderr, "myds_metal: fused Q2 sum cache rejected: %s; refusing fallback\n", reason);
    return 0;
}

static int moe_q2_sum_cache_fused_scoped(
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
    float                      clamp_val,
    bool                       ordinary_decode_allowed) {
    (void)ordinary_decode_allowed; // Entry is gated by the scoped public dispatcher.
    atomic_fetch_add_explicit(&g_moe_q2_sum_cache_fused_requests, 1, memory_order_relaxed);
    if (!g_initialized || g_moe_iq2_constant_lut || dim != 5120u || expert_dim != 2304u || n_activated_experts != 6u ||
        n_routed_experts < 6u || n_routed_experts > INT32_MAX ||
        gate_type != GGML_TYPE_IQ2_XXS || up_type != GGML_TYPE_IQ2_XXS || down_type != GGML_TYPE_Q2_K ||
        !isfinite(clamp_val) || clamp_val < 0)
        return moe_q2_sum_cache_fused_failure("invalid runtime, shape, types or clamp");
    /* A separately enabled one-token MM experiment uses half activations and
     * another reduction tree. This candidate preserves the existing GEMV path
     * and must not silently replace an eligible MM reference configuration. */
    const int min_mm_tokens = get_moe_mm_min_tokens();
    if (min_mm_tokens > 0 && min_mm_tokens <= 1 && g_pso_moe_mm_map0_ne20_6 &&
        g_pso_moe_mm_iq2_pair_swiglu_f16 && g_pso_moe_mm_q2_k_f16 && g_pso_moe_sum6)
        return moe_q2_sum_cache_fused_failure("one-token MM reference is outside candidate scope");
    if (!moe_q2_sum_cache_fused_geometry() || !g_moe_q2_sum_cache_buffer ||
        g_moe_q2_sum_cache_buffer.device != g_device ||
        g_moe_q2_sum_cache_buffer.length < MOE_Q2_SUM_CACHE_BYTES || !g_moe_q2_sum_cache_buffer.contents ||
        (uintptr_t)g_moe_q2_sum_cache_buffer.contents % 16u)
        return moe_q2_sum_cache_fused_failure("candidate PSO/geometry or scratch unavailable");

    const uint64_t gate_row_bytes = (uint64_t)(dim / 256u) * 66u;
    const uint64_t gate_exp_bytes = (uint64_t)expert_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)(expert_dim / 256u) * 84u;
    const uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;
    const uint64_t mid_bytes = (uint64_t)n_activated_experts * expert_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)dim * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_activated_experts * sizeof(float);
    const myds_metal_buffer_t *readers[] = {x_in, gate_exps, up_exps, down_exps, selected_ids, selected_weights};
    const uint64_t reader_bytes[] = {out_bytes, gate_exp_bytes * n_routed_experts,
        gate_exp_bytes * n_routed_experts, down_exp_bytes * n_routed_experts, selected_bytes, selected_bytes};
    const size_t reader_align[] = {16u, 2u, 2u, 2u, 4u, 4u};
    if (!moe_q2_sum_cache_range(moe_out, out_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_mid, mid_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_gate, mid_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_up, mid_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_down, (uint64_t)n_activated_experts * out_bytes, 4u))
        return moe_q2_sum_cache_fused_failure("invalid output or scratch range/alignment");
    for (unsigned i = 0; i < 6u; ++i)
        if (!moe_q2_sum_cache_range(readers[i], reader_bytes[i], reader_align[i]))
            return moe_q2_sum_cache_fused_failure("invalid input range/alignment");
    myds_metal_buffer_t cache_view = {.mtl_buf = g_moe_q2_sum_cache_buffer,
        .size = MOE_Q2_SUM_CACHE_BYTES, .offset = 0, .ptr = g_moe_q2_sum_cache_buffer.contents, .is_view = true};
    const myds_metal_buffer_t *writers[] = {moe_out, scratch_mid, &cache_view};
    const uint64_t writer_bytes[] = {out_bytes, mid_bytes, MOE_Q2_SUM_CACHE_BYTES};
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = i + 1u; j < 3u; ++j)
            if (moe_q2_sum_cache_overlap(writers[i], writer_bytes[i], writers[j], writer_bytes[j]))
                return moe_q2_sum_cache_fused_failure("writable buffers overlap");
        for (unsigned j = 0; j < 6u; ++j)
            if (moe_q2_sum_cache_overlap(writers[i], writer_bytes[i], readers[j], reader_bytes[j]))
                return moe_q2_sum_cache_fused_failure("writable buffer overlaps a reader");
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return moe_q2_sum_cache_fused_failure("command buffer unavailable");
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return moe_q2_sum_cache_fused_failure("encoder unavailable");
        const uint32_t n_tok = 1u;
        const NSUInteger nsg_gate = (NSUInteger)g_moe_nsg;
        const NSUInteger nsg_down = (NSUInteger)g_moe_down_nsg;
        const NSUInteger row_groups_gate = (expert_dim + 4u * nsg_gate - 1u) / (4u * nsg_gate);
        const NSUInteger row_groups_down = (dim + 2u * nsg_down - 1u) / (2u * nsg_down);
        const NSUInteger pairs = n_activated_experts;
        const NSUInteger smem_iq2 = 2176u;
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

        // Inputs retain the original caller's upstream producer dependencies.
        // All validation above occurs before an IQ2, mid or cache write.
        [enc setComputePipelineState:g_pso_moe_iq2_pair_swiglu_sum_cache];
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
        [enc setBuffer:g_moe_q2_sum_cache_buffer offset:0 atIndex:10];
        [enc setThreadgroupMemoryLength:smem_iq2 atIndex:0];
        [enc setThreadgroupMemoryLength:64u atIndex:1];
        metal_dispatch(enc, "fused_moe_swiglu_sum_cache", MTLSizeMake(row_groups_gate, 1, pairs), MTLSizeMake(32, nsg_gate, 1));
        if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        else {
            id<MTLResource> produced[] = {scratch_mid->mtl_buf, g_moe_q2_sum_cache_buffer};
            [enc memoryBarrierWithResources:produced count:2];
        }

        [enc setComputePipelineState:g_pso_moe_q2_sum6_cached];
        [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
        [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
        [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
        [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:3];
        [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
        [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:5];
        [enc setBuffer:g_moe_q2_sum_cache_buffer offset:0 atIndex:6];
        metal_dispatch(enc, "fused_moe_sum6_q2_k_cached", MTLSizeMake(row_groups_down, 1, 1), MTLSizeMake(32, nsg_down, 1));
        // Match the reference's batch output dependency contract. With auto
        // barriers disabled, the caller must join OUT before consuming it or
        // reusing mid/cache; that join completes all prior dispatches, including
        // the cached down readers. Standalone completion still waits below.
        if (g_batch_cb) {
            if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            [enc endEncoding]; [cb commit];
            if (!wait_command_buffer(cb)) return moe_q2_sum_cache_fused_failure("command completion failed");
        }
        atomic_fetch_add_explicit(&g_moe_q2_sum_cache_fused_encoded, 1, memory_order_relaxed);
        return 1;
    }
}

int myds_metal_routed_moe_scoped(
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
    float                      clamp_val,
    bool                       ordinary_decode_allowed) {
    if (g_moe_q2_sum_cache_fused && ordinary_decode_allowed)
        return moe_q2_sum_cache_fused_scoped(moe_out, x_in, gate_exps, up_exps, down_exps,
            selected_ids, selected_weights, scratch_gate, scratch_up, scratch_mid, scratch_down,
            dim, expert_dim, n_routed_experts, n_activated_experts,
            gate_type, up_type, down_type, clamp_val, ordinary_decode_allowed);

    if (!g_moe_q2_sum_cache || !ordinary_decode_allowed)
        return myds_metal_routed_moe(moe_out, x_in, gate_exps, up_exps, down_exps,
            selected_ids, selected_weights, scratch_gate, scratch_up, scratch_mid, scratch_down,
            dim, expert_dim, n_routed_experts, n_activated_experts,
            gate_type, up_type, down_type, clamp_val);

    atomic_fetch_add_explicit(&g_moe_q2_sum_cache_requests, 1, memory_order_relaxed);
    if (!g_initialized || dim != 5120u || expert_dim != 2304u || n_activated_experts != 6u ||
        n_routed_experts < 6u || n_routed_experts > INT32_MAX ||
        gate_type != GGML_TYPE_IQ2_XXS || up_type != GGML_TYPE_IQ2_XXS || down_type != GGML_TYPE_Q2_K ||
        !isfinite(clamp_val) || clamp_val < 0)
        return moe_q2_sum_cache_failure("invalid runtime, shape, types or clamp");
    /* A separately enabled one-token MM experiment uses half activations and
     * another reduction tree. This candidate preserves the existing GEMV path
     * and must not silently replace an eligible MM reference configuration. */
    const int min_mm_tokens = get_moe_mm_min_tokens();
    if (min_mm_tokens > 0 && min_mm_tokens <= 1 && g_pso_moe_mm_map0_ne20_6 &&
        g_pso_moe_mm_iq2_pair_swiglu_f16 && g_pso_moe_mm_q2_k_f16 && g_pso_moe_sum6)
        return moe_q2_sum_cache_failure("one-token MM reference is outside candidate scope");
    if (!moe_q2_sum_cache_geometry(g_moe_iq2_constant_lut) || !g_moe_q2_sum_cache_buffer ||
        g_moe_q2_sum_cache_buffer.device != g_device ||
        g_moe_q2_sum_cache_buffer.length < MOE_Q2_SUM_CACHE_BYTES || !g_moe_q2_sum_cache_buffer.contents ||
        (uintptr_t)g_moe_q2_sum_cache_buffer.contents % 16u)
        return moe_q2_sum_cache_failure("candidate PSO/geometry or scratch unavailable");

    const uint64_t gate_row_bytes = (uint64_t)(dim / 256u) * 66u;
    const uint64_t gate_exp_bytes = (uint64_t)expert_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)(expert_dim / 256u) * 84u;
    const uint64_t down_exp_bytes = (uint64_t)dim * down_row_bytes;
    const uint64_t mid_bytes = (uint64_t)n_activated_experts * expert_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)dim * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_activated_experts * sizeof(float);
    const myds_metal_buffer_t *readers[] = {x_in, gate_exps, up_exps, down_exps, selected_ids, selected_weights};
    const uint64_t reader_bytes[] = {out_bytes, gate_exp_bytes * n_routed_experts,
        gate_exp_bytes * n_routed_experts, down_exp_bytes * n_routed_experts, selected_bytes, selected_bytes};
    const size_t reader_align[] = {16u, 2u, 2u, 2u, 4u, 4u};
    if (!moe_q2_sum_cache_range(moe_out, out_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_mid, mid_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_gate, mid_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_up, mid_bytes, 4u) ||
        !moe_q2_sum_cache_range(scratch_down, (uint64_t)n_activated_experts * out_bytes, 4u))
        return moe_q2_sum_cache_failure("invalid output or scratch range/alignment");
    for (unsigned i = 0; i < 6u; ++i)
        if (!moe_q2_sum_cache_range(readers[i], reader_bytes[i], reader_align[i]))
            return moe_q2_sum_cache_failure("invalid input range/alignment");
    myds_metal_buffer_t cache_view = {.mtl_buf = g_moe_q2_sum_cache_buffer,
        .size = MOE_Q2_SUM_CACHE_BYTES, .offset = 0, .ptr = g_moe_q2_sum_cache_buffer.contents, .is_view = true};
    const myds_metal_buffer_t *writers[] = {moe_out, scratch_mid, &cache_view};
    const uint64_t writer_bytes[] = {out_bytes, mid_bytes, MOE_Q2_SUM_CACHE_BYTES};
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = i + 1u; j < 3u; ++j)
            if (moe_q2_sum_cache_overlap(writers[i], writer_bytes[i], writers[j], writer_bytes[j]))
                return moe_q2_sum_cache_failure("writable buffers overlap");
        for (unsigned j = 0; j < 6u; ++j)
            if (moe_q2_sum_cache_overlap(writers[i], writer_bytes[i], readers[j], reader_bytes[j]))
                return moe_q2_sum_cache_failure("writable buffer overlaps a reader");
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = g_batch_cb ?: [g_queue commandBuffer];
        if (!cb) return moe_q2_sum_cache_failure("command buffer unavailable");
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
        if (!enc) return moe_q2_sum_cache_failure("encoder unavailable");
        const uint32_t n_tok = 1u;
        const NSUInteger nsg_gate = (NSUInteger)g_moe_nsg;
        const NSUInteger nsg_down = (NSUInteger)g_moe_down_nsg;
        const NSUInteger row_groups_gate = (expert_dim + 4u * nsg_gate - 1u) / (4u * nsg_gate);
        const NSUInteger row_groups_down = (dim + 2u * nsg_down - 1u) / (2u * nsg_down);
        const NSUInteger pairs = n_activated_experts;
        const NSUInteger smem_iq2 = g_moe_iq2_constant_lut ? 0u : 2176u;
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

        // Inputs retain the original caller's upstream producer dependencies.
        // All validation above occurs before an IQ2, mid or cache write.
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
        metal_dispatch(enc, "fused_moe_swiglu", MTLSizeMake(row_groups_gate, 1, pairs), MTLSizeMake(32, nsg_gate, 1));
        if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        else {id<MTLResource> mid = scratch_mid->mtl_buf; [enc memoryBarrierWithResources:&mid count:1];}

        [enc setComputePipelineState:g_pso_moe_q2_sum_cache_prepare];
        [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
        [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:1];
        [enc setBuffer:g_moe_q2_sum_cache_buffer offset:0 atIndex:2];
        [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:3];
        metal_dispatch(enc, "moe_q2_sum_cache_prepare", MTLSizeMake(4, 1, 1), MTLSizeMake(128, 1, 1));
        if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        else {id<MTLResource> cache = g_moe_q2_sum_cache_buffer; [enc memoryBarrierWithResources:&cache count:1];}

        [enc setComputePipelineState:g_pso_moe_q2_sum6_cached];
        [enc setBytes:&down_args length:sizeof(down_args) atIndex:0];
        [enc setBuffer:down_exps->mtl_buf offset:down_exps->offset atIndex:1];
        [enc setBuffer:scratch_mid->mtl_buf offset:scratch_mid->offset atIndex:2];
        [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:3];
        [enc setBuffer:selected_ids->mtl_buf offset:selected_ids->offset atIndex:4];
        [enc setBuffer:moe_out->mtl_buf offset:moe_out->offset atIndex:5];
        [enc setBuffer:g_moe_q2_sum_cache_buffer offset:0 atIndex:6];
        metal_dispatch(enc, "fused_moe_sum6_q2_k_cached", MTLSizeMake(row_groups_down, 1, 1), MTLSizeMake(32, nsg_down, 1));
        // Match the reference's batch output dependency contract. With auto
        // barriers disabled, the caller must join OUT before consuming it or
        // reusing mid/cache; that join completes all prior dispatches, including
        // the cached down readers. Standalone completion still waits below.
        if (g_batch_cb) {
            if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            [enc endEncoding]; [cb commit];
            if (!wait_command_buffer(cb)) return moe_q2_sum_cache_failure("command completion failed");
        }
        atomic_fetch_add_explicit(&g_moe_q2_sum_cache_encoded, 1, memory_order_relaxed);
        return 1;
    }
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
            metal_dispatch(enc, "attn_out_low_q8_0", MTLSizeMake(row_groups, 1, pairs),
                MTLSizeMake(32, 4, 1));

            if (g_batch_cb) {
                if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    } else if (weight_type == GGML_TYPE_Q4_K && g_pso_attn_out_low_q4_K && out_a->mtl_buf && heads->mtl_buf && low->mtl_buf && (group_dim % 256 == 0)) {
        @autoreleasepool {
            const uint64_t a_group_bytes = (uint64_t)rank * (group_dim / 256u) * 144u;
            struct ds4_metal_args_mul_mv_id args;
            memset(&args, 0, sizeof(args));
            args.nei0 = (int32_t)n_groups;
            args.nei1 = (int32_t)n_tok;
            args.nbi1 = (uint64_t)n_groups * sizeof(int32_t);
            args.ne00 = (int32_t)group_dim;
            args.ne01 = (int32_t)rank;
            args.ne02 = (int32_t)n_groups;
            args.nb00 = 144u;
            args.nb01 = (group_dim / 256u) * 144u;
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

            const NSUInteger rows_per_group = 8; // NSG=4 * nr0=2
            const NSUInteger row_groups = (rank + rows_per_group - 1) / rows_per_group;
            const NSUInteger pairs = (NSUInteger)n_groups * (NSUInteger)n_tok;
            const NSUInteger smem_bytes = 256;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_attn_out_low_q4_K];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:out_a->mtl_buf offset:out_a->offset atIndex:1];
            [enc setBuffer:heads->mtl_buf offset:heads->offset atIndex:2];
            [enc setBuffer:low->mtl_buf offset:low->offset atIndex:3];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            metal_dispatch(enc, "attn_out_low_q4_K", MTLSizeMake(row_groups, 1, pairs),
                MTLSizeMake(32, 4, 1));

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

int myds_metal_attn_output_a_bf16_batch(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    uint32_t                   n_tok,
    ggml_type                  weight_type) {
    if (!low || !out_a || !heads || n_groups == 0 || group_dim == 0 || rank == 0 || n_tok == 0) return 0;
    static int cached_fuse = -1;
    if (__builtin_expect(cached_fuse < 0, 0)) {
        const char *v = getenv("MYDS_ATTN_OUT_A_FUSE_BF16");
        cached_fuse = (v && strcmp(v, "0") == 0) ? 0 : 1;
    }
    static int cached_shared2 = -1;
    if (__builtin_expect(cached_shared2 < 0, 0)) {
        const char *v = getenv("MYDS_ATTN_OUT_A_SHARED2");
        cached_shared2 = (v && strcmp(v, "0") == 0) ? 0 : 1;
    }

    if (cached_fuse && cached_shared2 && weight_type == GGML_TYPE_Q8_0 && n_tok == 2 && g_pso_attn_out_low_q8_0_bf16_shared2 &&
        out_a->mtl_buf && heads->mtl_buf && low->mtl_buf && (group_dim % 32 == 0) && (low->offset % 16 == 0)) {
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
            args.nr0 = 4;

            const NSUInteger rows_per_group = 4;
            const NSUInteger row_groups = (rank + rows_per_group - 1) / rows_per_group;
            const NSUInteger smem_bytes = 1024;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_attn_out_low_q8_0_bf16_shared2];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:out_a->mtl_buf offset:out_a->offset atIndex:1];
            [enc setBuffer:heads->mtl_buf offset:heads->offset atIndex:2];
            [enc setBuffer:low->mtl_buf offset:low->offset atIndex:3];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            metal_dispatch(enc, "attn_out_low_q8_0_bf16_shared2", MTLSizeMake(row_groups, 1, n_groups),
                MTLSizeMake(32, 4, 1));

            if (g_batch_cb) {
                if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    } else if (cached_fuse && cached_shared2 && weight_type == GGML_TYPE_Q4_K && n_tok == 2 && g_pso_attn_out_low_q4_K_bf16_shared2 &&
               out_a->mtl_buf && heads->mtl_buf && low->mtl_buf && (group_dim % 256 == 0) && (low->offset % 16 == 0)) {
        @autoreleasepool {
            const uint64_t a_group_bytes = (uint64_t)rank * (group_dim / 256u) * 144u;
            struct ds4_metal_args_mul_mv_id args;
            memset(&args, 0, sizeof(args));
            args.nei0 = (int32_t)n_groups;
            args.nei1 = (int32_t)n_tok;
            args.nbi1 = (uint64_t)n_groups * sizeof(int32_t);
            args.ne00 = (int32_t)group_dim;
            args.ne01 = (int32_t)rank;
            args.ne02 = (int32_t)n_groups;
            args.nb00 = 144u;
            args.nb01 = (group_dim / 256u) * 144u;
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

            const NSUInteger rows_per_group = 8; // NSG=4 * nr0=2
            const NSUInteger row_groups = (rank + rows_per_group - 1) / rows_per_group;
            const NSUInteger smem_bytes = 256;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_attn_out_low_q4_K_bf16_shared2];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:out_a->mtl_buf offset:out_a->offset atIndex:1];
            [enc setBuffer:heads->mtl_buf offset:heads->offset atIndex:2];
            [enc setBuffer:low->mtl_buf offset:low->offset atIndex:3];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            metal_dispatch(enc, "attn_out_low_q4_K_bf16_shared2", MTLSizeMake(row_groups, 1, n_groups),
                MTLSizeMake(32, 4, 1));

            if (g_batch_cb) {
                if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    } else if (cached_fuse && weight_type == GGML_TYPE_Q8_0 && g_pso_attn_out_low_q8_0_bf16_r4 &&
        out_a->mtl_buf && heads->mtl_buf && low->mtl_buf && (group_dim % 32 == 0) && (low->offset % 16 == 0)) {
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
            args.nr0 = 4;

            const NSUInteger rows_per_group = 4;
            const NSUInteger row_groups = (rank + rows_per_group - 1) / rows_per_group;
            const NSUInteger pairs = (NSUInteger)n_groups * (NSUInteger)n_tok;
            const NSUInteger smem_bytes = 512;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_attn_out_low_q8_0_bf16_r4];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:out_a->mtl_buf offset:out_a->offset atIndex:1];
            [enc setBuffer:heads->mtl_buf offset:heads->offset atIndex:2];
            [enc setBuffer:low->mtl_buf offset:low->offset atIndex:3];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            metal_dispatch(enc, "attn_out_low_q8_0_bf16_r4", MTLSizeMake(row_groups, 1, pairs),
                MTLSizeMake(32, 4, 1));

            if (g_batch_cb) {
                if (g_auto_barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc endEncoding];
                [cb commit];
                if (!wait_command_buffer(cb)) return 0;
            }
            return 1;
        }
    } else if (cached_fuse && weight_type == GGML_TYPE_Q4_K && g_pso_attn_out_low_q4_K_bf16 &&
               out_a->mtl_buf && heads->mtl_buf && low->mtl_buf && (group_dim % 256 == 0) && (low->offset % 16 == 0)) {
        @autoreleasepool {
            const uint64_t a_group_bytes = (uint64_t)rank * (group_dim / 256u) * 144u;
            struct ds4_metal_args_mul_mv_id args;
            memset(&args, 0, sizeof(args));
            args.nei0 = (int32_t)n_groups;
            args.nei1 = (int32_t)n_tok;
            args.nbi1 = (uint64_t)n_groups * sizeof(int32_t);
            args.ne00 = (int32_t)group_dim;
            args.ne01 = (int32_t)rank;
            args.ne02 = (int32_t)n_groups;
            args.nb00 = 144u;
            args.nb01 = (group_dim / 256u) * 144u;
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

            const NSUInteger rows_per_group = 8; // NSG=4 * nr0=2
            const NSUInteger row_groups = (rank + rows_per_group - 1) / rows_per_group;
            const NSUInteger pairs = (NSUInteger)n_groups * (NSUInteger)n_tok;
            const NSUInteger smem_bytes = 256;

            id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
            [enc setComputePipelineState:g_pso_attn_out_low_q4_K_bf16];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:out_a->mtl_buf offset:out_a->offset atIndex:1];
            [enc setBuffer:heads->mtl_buf offset:heads->offset atIndex:2];
            [enc setBuffer:low->mtl_buf offset:low->offset atIndex:3];
            [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];
            metal_dispatch(enc, "attn_out_low_q4_K_bf16", MTLSizeMake(row_groups, 1, pairs),
                MTLSizeMake(32, 4, 1));

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

    return myds_metal_attn_output_a_batch(low, out_a, heads, n_groups, group_dim, rank, n_tok, weight_type) &&
           myds_metal_v41_bf16(low, (uint64_t)n_groups * rank * n_tok);
}

int myds_metal_attn_output_a_bf16(
    myds_metal_buffer_t       *low,
    const myds_metal_buffer_t *out_a,
    const myds_metal_buffer_t *heads,
    uint32_t                   n_groups,
    uint32_t                   group_dim,
    uint32_t                   rank,
    ggml_type                  weight_type) {
    return myds_metal_attn_output_a_bf16_batch(low, out_a, heads, n_groups, group_dim, rank, 1, weight_type);
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
            metal_dispatch_threads(enc, "add",
                MTLSizeMake(count, 1, 1),
                MTLSizeMake(threads, 1, 1));

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

int myds_metal_v41_add_bf16(
    myds_metal_buffer_t       *out,
    const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b,
    uint32_t                   count) {
    if (!out || !a || !b || count == 0) return 0;

    if (a->mtl_buf && b->mtl_buf && out->mtl_buf) {
        if (g_pso_add_bf16) {
            @autoreleasepool {
                struct ds4_metal_args_add3 args;
                args.n = count;
                args.r2 = 1;
                args.r3 = 1;

                id<MTLCommandBuffer> cb = g_batch_cb ? g_batch_cb : [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = get_compute_encoder(cb);
                [enc setComputePipelineState:g_pso_add_bf16];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:a->mtl_buf offset:(a ? a->offset : 0) atIndex:1];
                [enc setBuffer:b->mtl_buf offset:(b ? b->offset : 0) atIndex:2];
                [enc setBuffer:out->mtl_buf offset:(out ? out->offset : 0) atIndex:3];

                NSUInteger threads = (count < 256) ? count : 256;
                if (threads == 0) threads = 1;
                metal_dispatch_threads(enc, "v41_add_bf16",
                    MTLSizeMake(count, 1, 1),
                    MTLSizeMake(threads, 1, 1));

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
        /* Fallback: if fused kernel PSO is unavailable, run GPU add + GPU BF16 round within the active command buffer */
        return myds_metal_add(out, a, b, count) && myds_metal_v41_bf16(out, count);
    }

    if (g_batch_cb) {
        /* In batch mode with non-Metal buffers: cannot execute CPU fallback while command buffer is open */
        return 0;
    }

    /* CPU Fallback with exact BF16 round */
    float *ov = (float *)out->ptr;
    const float *av = (const float *)a->ptr;
    const float *bv = (const float *)b->ptr;
    for (uint32_t i = 0; i < count; i++) {
        float sum = av[i] + bv[i];
        uint32_t bits;
        memcpy(&bits, &sum, sizeof(bits));
        if ((bits & 0x7f800000u) != 0x7f800000u) {
            bits += 0x7fffu + ((bits >> 16u) & 1u);
        }
        bits &= 0xffff0000u;
        memcpy(&ov[i], &bits, sizeof(bits));
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
