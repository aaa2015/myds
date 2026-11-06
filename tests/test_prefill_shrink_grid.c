/* Exhaustively compare retained checkpoint sets and their per-layer causal
 * windows. Full-model arithmetic parity is a separate GPU acceptance gate. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/myds_v41.c"
#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); \
} } while (0)

static uint64_t cases, removed_rows;
static void check(uint32_t end,uint32_t shift) {
    const uint32_t effective=prefill_snapshot_grid_shift(end,shift);
    REQUIRE(effective<=shift && effective<=end);
    /* Enumerate legal periodic points independently of the rounded-up helper. */
    uint32_t earliest=end;
    const uint64_t low=end>shift?(uint64_t)end-shift:0u;
    for(uint64_t p=V41_SNAP_MIN;p<=end;p+=V41_SNAP_STRIDE) {
        const bool old=p>=low;
        const bool now=p>=(uint64_t)end-effective;
        REQUIRE(old==now);
        if(old && p<earliest) earliest=(uint32_t)p;
    }
    REQUIRE(effective==end-earliest);
    /* S(il-1)=S(il)-127: every needed input to a retained 128-key window
     * comes from a full output in the preceding layer, including the endpoint. */
    for(unsigned il=20;il<40;++il) {
        const uint64_t need=1u+(39u-il)*127u;
        const uint64_t old_sub=need+shift,new_sub=need+effective;
        const uint32_t old_start=end>old_sub?end-(uint32_t)old_sub:0u;
        const uint32_t new_start=end>new_sub?end-(uint32_t)new_sub:0u;
        REQUIRE(new_start>=old_start);
        removed_rows+=new_start-old_start;
        REQUIRE(!end || new_start<end);
        if(il>20) {
            const uint64_t previous_sub=new_sub+127u;
            const uint32_t previous_start=end>previous_sub?end-(uint32_t)previous_sub:0u;
            REQUIRE(previous_start==(new_start>127u?new_start-127u:0u));
        }
    }
    ++cases;
}

static void test_moe_batch_flush_and_shrink_alignment(void) {
    /* Regression case: mb_len=5014, chunk_size=2048, moe_batch=6144, shrink_shift=512.
     * Chunks in layers 0..19:
     *   [0, 2048)    -> moe_off=2048, next_rem=2966
     *   [2048, 4096) -> moe_off=4096, next_rem=918
     *   [4096, 4608) -> moe_off=4608, next_rem=406 (shrink_shift=512 aligns to 4608)
     *   [4608, 5014) -> moe_off=5014, next_rem=0
     * Previously, moe_off + chunk_size (4608 + 2048 = 6656 > 6144) flushed at 4608,
     * causing 2 MoE calls per layer across layers 0..19 (60 calls total instead of 40).
     * With next_chunk_max = min(2048, 406) = 406, 4608 + 406 = 5014 <= 6144: only 1 flush at 5014! */
    REQUIRE(!myds_v41_should_flush_moe_batch(2048u, 6144u, 0u, 2048u, 5014u, 2048u));
    REQUIRE(!myds_v41_should_flush_moe_batch(4096u, 6144u, 2048u, 2048u, 5014u, 2048u));
    REQUIRE(!myds_v41_should_flush_moe_batch(4608u, 6144u, 4096u, 512u, 5014u, 2048u));
    REQUIRE(myds_v41_should_flush_moe_batch(5014u, 6144u, 4608u, 406u, 5014u, 2048u));

    /* Exhaustive simulation of chunk slicing + MoE batch accumulation across
     * aligned and unaligned base_pos, shrink_shift in {0, 512}, and moe_batch sizes. */
    const uint32_t chunk_size = 2048u;
    const uint32_t moe_batches[] = {2048u, 4096u, 6144u, 8192u, 16384u};
    const uint32_t base_positions[] = {0u, 137u, 451u, 512u, 1597u};
    const uint32_t shrink_shifts[] = {0u, 512u};

    for (size_t bi = 0; bi < sizeof(base_positions) / sizeof(*base_positions); ++bi) {
        const uint32_t base_pos = base_positions[bi];
        for (size_t si = 0; si < sizeof(shrink_shifts) / sizeof(*shrink_shifts); ++si) {
            const uint32_t shrink_shift = shrink_shifts[si];
            for (size_t mi = 0; mi < sizeof(moe_batches) / sizeof(*moe_batches); ++mi) {
                const uint32_t moe_batch = moe_batches[mi];
                for (uint32_t mb_len = 1u; mb_len <= 16384u; mb_len += (mb_len < 1024u ? 19u : 113u)) {
                    uint32_t moe_off = 0u;
                    uint32_t flushes = 0u;
                    uint32_t flushed_tokens = 0u;
                    uint32_t c_len = 0u;
                    for (uint32_t c_start = 0u; c_start < mb_len; c_start += c_len) {
                        c_len = mb_len - c_start;
                        if (c_len > chunk_size) c_len = chunk_size;
                        if (c_start == 0u && (base_pos % V41_SNAP_STRIDE) != 0u) {
                            const uint32_t to_grid = V41_SNAP_STRIDE - (base_pos % V41_SNAP_STRIDE);
                            if (to_grid < c_len) c_len = to_grid;
                        }
                        if (shrink_shift > 0u && c_start + c_len == mb_len && c_len > shrink_shift) {
                            const uint32_t p_end = base_pos + mb_len;
                            const uint32_t g_last = (p_end / shrink_shift) * shrink_shift;
                            if (g_last >= shrink_shift && g_last > base_pos + c_start && g_last < p_end) {
                                c_len = g_last - (base_pos + c_start);
                            }
                        }
                        /* Invariant 1: accumulation must never overflow moe_batch */
                        REQUIRE(moe_off + c_len <= moe_batch);
                        moe_off += c_len;
                        if (myds_v41_should_flush_moe_batch(moe_off, moe_batch, c_start, c_len, mb_len, chunk_size)) {
                            flushes++;
                            flushed_tokens += moe_off;
                            moe_off = 0u;
                        }
                    }
                    /* Invariant 2: all tokens in the macroblock are flushed by macroblock end */
                    REQUIRE(moe_off == 0u);
                    REQUIRE(flushed_tokens == mb_len);
                    /* Invariant 3: if the entire macroblock fits in moe_batch, it flushes exactly once */
                    if (mb_len <= moe_batch) {
                        REQUIRE(flushes == 1u);
                    }
                }
            }
        }
    }
}

static void test_workspace_prealloc_resolution(void) {
    uint32_t ws_chunk_cap = 0, ws_moe_cap = 0, prompt_cap = 0;
    bool prealloc_full = false;

    /* 1. Short 1-token prompt does NOT trigger full multi-GiB pre-allocation */
    myds_v41_resolve_workspace_caps(1u, 2048u, 2u, 8u, 1u, 1u,
                                    &ws_chunk_cap, &ws_moe_cap, &prompt_cap, &prealloc_full);
    REQUIRE(!prealloc_full);
    REQUIRE(ws_chunk_cap == 2u);
    REQUIRE(ws_moe_cap == 2u);
    REQUIRE(prompt_cap == 512u);

    /* 2. Warmup probe (47 tokens >= 32) with MYDS_MOE_CHUNKS=8 pre-allocates full workspace once */
    myds_v41_resolve_workspace_caps(47u, 2048u, 48u, 8u, 1u, 47u,
                                    &ws_chunk_cap, &ws_moe_cap, &prompt_cap, &prealloc_full);
    REQUIRE(prealloc_full);
    REQUIRE(ws_chunk_cap == 2048u);
    REQUIRE(ws_moe_cap == 16384u);
    REQUIRE(prompt_cap == 16384u);

    /* 3. Subsequent 5,962-token and 16,384-token requests fit without reallocation */
    uint32_t req_chunk = 0, req_moe = 0, req_prompt = 0;
    bool req_prealloc = false;
    myds_v41_resolve_workspace_caps(5962u, 2048u, 2048u, 8u, 3u, 5962u,
                                    &req_chunk, &req_moe, &req_prompt, &req_prealloc);
    REQUIRE(req_prealloc);
    REQUIRE(req_chunk <= ws_chunk_cap);
    REQUIRE(req_moe <= ws_moe_cap);
    REQUIRE(req_prompt <= prompt_cap);

    /* 4. Single-chunk mode (cfg_moe_chunks == 1) does not force full pre-allocation */
    myds_v41_resolve_workspace_caps(64u, 2048u, 64u, 1u, 1u, 64u,
                                    &ws_chunk_cap, &ws_moe_cap, &prompt_cap, &prealloc_full);
    REQUIRE(!prealloc_full);
    REQUIRE(ws_chunk_cap == 64u);
    REQUIRE(ws_moe_cap == 64u);
    REQUIRE(prompt_cap == 512u);
}

int main(void) {
    const uint32_t shifts[]={0u,512u,1024u,4096u,8192u,16384u};
    for(uint32_t end=0;end<=65536u;end+=17u)
        for(size_t s=0;s<sizeof(shifts)/sizeof(*shifts);++s) check(end,shifts[s]);
    const uint32_t boundaries[]={1u,4u,94u,123u,890u,891u,1023u,1024u,1025u,1535u,1536u,1537u,4095u,4096u,4097u,4190u,4219u,8192u,1048576u,UINT32_MAX};
    for(size_t i=0;i<sizeof(boundaries)/sizeof(*boundaries);++i)
        for(size_t s=0;s<sizeof(shifts)/sizeof(*shifts);++s) check(boundaries[i],shifts[s]);
    REQUIRE(prefill_snapshot_grid_shift(890u,512u)==0u);
    REQUIRE(prefill_snapshot_grid_shift(1025u,512u)==1u);
    REQUIRE(prefill_snapshot_grid_shift(4219u,512u)==123u);
    test_moe_batch_flush_and_shrink_alignment();
    test_workspace_prealloc_resolution();
    printf("[PASS] prefill snapshot grid: %llu checkpoint-set/causal-cone cases, %llu redundant full rows removed in fixtures; no model/GPU claim\n",
        (unsigned long long)cases,(unsigned long long)removed_rows);
    return 0;
}
