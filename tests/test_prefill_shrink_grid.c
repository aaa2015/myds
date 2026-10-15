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
    printf("[PASS] prefill snapshot grid: %llu checkpoint-set/causal-cone cases, %llu redundant full rows removed in fixtures; no model/GPU claim\n",
        (unsigned long long)cases,(unsigned long long)removed_rows);
    return 0;
}
