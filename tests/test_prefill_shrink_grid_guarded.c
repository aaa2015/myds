/* CPU scheduling proof for guarded GRID. This includes the actual graph so
 * changes to its pure helpers are tested. No model/Metal API is called.
 *
 * The independent reference constructs sorted OLD boundary sets (absolute
 * 512 grid + old full start), rather than reproducing preserve_class's
 * arithmetic. It cannot prove split-GEMM byte parity or host/GPU integration;
 * the native raw-state/cache/resume oracle remains a separate adoption gate.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/myds_v41.c"

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s (begin=%u end=%u layer=%u)\n", \
            __FILE__,__LINE__,#c,where_begin,where_end,where_layer);exit(1); \
} } while (0)

enum { GRID=512, LAYERS=40, MIN_MM=32, MAX_BOUNDARIES=40 };
static uint32_t where_begin,where_end,where_layer;
static uint64_t cases,fragments,old_short_chunks,periodic_points,avoided_rows;
static uint64_t direct_cases,early_returns,partial_splits,unchanged_splits;
static unsigned attention_policy_cases;

/* Contract sentinels also catch helper writes outside either 40-entry array. */
struct starts { uint32_t before,value[LAYERS],after; };
static void init_starts(struct starts *s) {
    s->before=0x91c04e2bu;s->after=0xa531d8f7u;
    for (unsigned i=0;i<LAYERS;++i) s->value[i]=0xa5a5a5a5u;
}
static void intact(const struct starts *s) {
    REQUIRE(s->before==0x91c04e2bu && s->after==0xa531d8f7u);
}
static uint32_t reference_start(uint32_t begin,uint32_t end,unsigned layer,uint32_t shift) {
    if (layer<20u) return begin;
    /* Count backward from the last output independently in 64-bit space. */
    const uint64_t history=1u+127u*(uint64_t)(LAYERS-1u-layer)+(uint64_t)shift;
    return (uint64_t)end-(uint64_t)begin>history ? end-(uint32_t)history : begin;
}
static void insert(uint32_t *points,unsigned *count,uint32_t point) {
    unsigned i=0;
    while (i<*count && points[i]<point) ++i;
    if (i<*count && points[i]==point) return;
    REQUIRE(*count<MAX_BOUNDARIES);
    memmove(points+i+1u,points+i,(*count-i)*sizeof(*points));
    points[i]=point;++*count;
}
static unsigned old_partition(uint32_t begin,uint32_t end,uint32_t full,uint32_t points[MAX_BOUNDARIES]) {
    unsigned n=0;
    insert(points,&n,begin);insert(points,&n,end);insert(points,&n,full);
    /* A grid point is defined by its divisibility, independently of the
     * runtime chunk loop's round-down arithmetic. Only inspect this MB. */
    for (uint64_t p=(uint64_t)begin+1u;p<end;++p)
        if (!(p%GRID)) insert(points,&n,(uint32_t)p);
    return n;
}
static void check_partitions(uint32_t begin,uint32_t end,uint32_t old_full,uint32_t full) {
    uint32_t old[MAX_BOUNDARIES],now[MAX_BOUNDARIES];
    const unsigned no=old_partition(begin,end,old_full,old);
    memcpy(now,old,no*sizeof(*old));unsigned nn=no;insert(now,&nn,full);
    unsigned owner=0;
    for (unsigned i=0;i+1u<nn;++i) {
        const uint32_t a=now[i],b=now[i+1u];
        while (owner+1u<no && a>=old[owner+1u]) ++owner;
        REQUIRE(owner+1u<no && a<b && old[owner]<=a && b<=old[owner+1u]);
        const uint32_t original=old[owner+1u]-old[owner],length=b-a;
        REQUIRE((length>=MIN_MM)==(original>=MIN_MM));
        if (original<MIN_MM) {
            REQUIRE(a==old[owner] && b==old[owner+1u]);++old_short_chunks;
        }
        ++fragments;
    }
    for (unsigned i=0;i<no;++i) {
        bool found=false;
        for (unsigned j=0;j<nn;++j) if (old[i]==now[j]) found=true;
        REQUIRE(found);
    }
}
static void check(uint32_t begin,uint32_t end) {
    where_begin=begin;where_end=end;where_layer=0;
    REQUIRE(begin<end && (uint64_t)end-begin<=16384u);
    const uint32_t effective=prefill_snapshot_grid_shift(end,512u);
    REQUIRE(effective<=512u);
    struct starts old,now;init_starts(&old);init_starts(&now);
    REQUIRE(prefill_grid_guarded_starts(begin,end,512u,effective,old.value,now.value));
    intact(&old);intact(&now);
    for (unsigned layer=0;layer<LAYERS;++layer) {
        where_layer=layer;
        const uint32_t original=reference_start(begin,end,layer,512u);
        const uint32_t ideal=reference_start(begin,end,layer,effective);
        REQUIRE(old.value[layer]==original);
        REQUIRE(original<=now.value[layer] && now.value[layer]<=ideal && now.value[layer]<end);
        if (layer<20u) REQUIRE(now.value[layer]==begin);
        if (layer) {
            const uint32_t needed=(uint64_t)now.value[layer]>(uint64_t)begin+127u ? now.value[layer]-127u : begin;
            REQUIRE(now.value[layer-1u]<=needed);
        }
        /* Whole-MB routed MoE receives only remaining full rows. Preserve
         * its default >=32 batch class, independently of KV partitions. */
        REQUIRE((end-original>=MIN_MM)==(end-now.value[layer]>=MIN_MM));
        check_partitions(begin,end,original,now.value[layer]);
        avoided_rows+=(uint64_t)now.value[layer]-original;
    }
    /* Brute-force the configured 512-position suffix and the preceding
     * point: no ceiling/rounding formula is shared with the production helper.
     * Below this interval both old/new eligibility are necessarily false. */
    const uint32_t low=end>513u ? end-513u : 0u;
    uint32_t earliest=end;
    for (uint64_t p=low;p<=end;++p) {
        const bool legal=p>=1024u && !(p%512u);
        const bool was=legal && p+512u>=end;
        const bool present=legal && p+(uint64_t)effective>=end;
        REQUIRE(was==present);
        if (was && p<earliest) earliest=(uint32_t)p;
        if (!was || p<=begin) continue;
        for (unsigned layer=0;layer<LAYERS;++layer) REQUIRE(p>now.value[layer]);
        ++periodic_points;
    }
    REQUIRE(effective==end-earliest);
    /* The arbitrary prompt end must remain eligible even without a periodic
     * point. The first output needed by each retained point is in the cone. */
    for (unsigned layer=0;layer<LAYERS;++layer) REQUIRE(end>now.value[layer]);
    ++cases;
}
static void check_direct(uint32_t start,uint32_t end,uint32_t desired) {
    where_begin=start;where_end=end;where_layer=39;
    REQUIRE(start<end && desired>=start && desired<end);
    const uint32_t actual=prefill_grid_preserve_class(start,end,desired);
    REQUIRE(actual>=start && actual<=desired);
    check_partitions(start,end,start,actual);
    if (actual==desired) ++unchanged_splits; else ++partial_splits;
    ++direct_cases;
}
static void rejects(void) {
    static const uint32_t invalid[][4]={{0,0,512,0},{10,9,512,0},{0,512,0,0},{0,512,511,0},{0,512,1024,0},{0,512,512,513},{UINT32_MAX,UINT32_MAX,512,0}};
    for (unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);++i) {
        struct starts old,now;init_starts(&old);init_starts(&now);
        REQUIRE(!prefill_grid_guarded_starts(invalid[i][0],invalid[i][1],invalid[i][2],invalid[i][3],old.value,now.value));
        intact(&old);intact(&now);
        for (unsigned j=0;j<LAYERS;++j) REQUIRE(old.value[j]==0xa5a5a5a5u && now.value[j]==0xa5a5a5a5u);
        ++early_returns;
    }
}
static void check_attention_policy(void) {
    /* Explicit expectation tables prevent permissive atoi-style parsing from
     * admitting a policy whose slab/matrix selection depends on the new cut. */
    static const struct {const char *value;bool supported;} variants[]={
        {NULL,true},{"vec",true},{"auto",false},{"matrix",false},
        {"invalid",false},{"",false},{"VEC",false},{"vec ",false}
    };
    static const struct {const char *value;bool supported;} minimums[]={
        {NULL,true},{"0",true},{"1",false},{"32",false},
        {"invalid",false},{"",false},{"00",false},{"-1",false}
    };
    for (unsigned v=0;v<sizeof(variants)/sizeof(*variants);++v)
        for (unsigned m=0;m<sizeof(minimums)/sizeof(*minimums);++m) {
            REQUIRE(prefill_grid_attention_policy_supported(variants[v].value,minimums[m].value)==
                    (variants[v].supported && minimums[m].supported));
            ++attention_policy_cases;
        }
    /* Concrete auto-policy regression: ratio1 layer38, begin0/end600.
     * Old retained rows441..511 share the old p0=0/N512 vec chunk; GRID's
     * p0=441/N71 fragment meets auto's p0>=256/N>=32/n_comp>=512 matrix
     * threshold. Equal dense/MM classes alone cannot qualify this policy. */
    struct starts old,now;init_starts(&old);init_starts(&now);
    REQUIRE(prefill_grid_guarded_starts(0u,600u,512u,
            prefill_snapshot_grid_shift(600u,512u),old.value,now.value));
    REQUIRE(old.value[38]==0u && now.value[38]==441u);
    REQUIRE(!prefill_grid_attention_policy_supported("auto",NULL));
    REQUIRE(!prefill_grid_attention_policy_supported("vec","1"));
}
int main(void) {
    /* Offset 123 models the real anchor-aligned MB; 6153 is an odd resumed
     * prefix. Values near UINT32_MAX catch 32-bit overflow in dependency/grid
     * arithmetic, without constructing giant arrays or reading unused state. */
    static const uint32_t bases[]={0,1,4,31,123,127,128,511,512,6153,1048576,UINT32_MAX-16384u,UINT32_MAX-4096u,UINT32_MAX-512u,UINT32_MAX-1u};
    static const uint32_t lengths[]={1,2,3,23,24,25,31,32,33,63,64,65,126,127,128,129,378,379,511,512,513,766,890,891,1023,1024,1025,1529,4095,4096,4097,8192,16384};
    rejects();check_attention_policy();
    for (unsigned b=0;b<sizeof(bases)/sizeof(*bases);++b) {
        for (uint32_t n=1;n<=4096u;n+=17u)
            if ((uint64_t)bases[b]+n<=UINT32_MAX) check(bases[b],bases[b]+n);
        for (unsigned n=0;n<sizeof(lengths)/sizeof(*lengths);++n)
            if ((uint64_t)bases[b]+lengths[n]<=UINT32_MAX) check(bases[b],bases[b]+lengths[n]);
    }
    /* Sweep every absolute-grid residue at numerical policy boundaries. */
    static const uint32_t edge_lengths[]={31,32,33,63,64,65,511,512,513,1023,1024,1025,1529,4096};
    for (uint32_t residue=0;residue<512u;++residue)
        for (unsigned n=0;n<sizeof(edge_lengths)/sizeof(*edge_lengths);++n)
            check(4096u+residue,4096u+residue+edge_lengths[n]);
    /* Challenge the single-boundary rounding helper directly. This includes
     * short first/last partitions and both sides of 24/32/64 thresholds. */
    for (uint32_t residue=0;residue<512u;++residue) {
        const uint32_t start=1024u+residue;
        for (uint32_t n=1;n<=1025u;n+=17u) {
            const uint32_t offsets[]={0,1,23,24,25,31,32,33,63,64,n/2u,n>32u?n-32u:0u,n-1u};
            for (unsigned j=0;j<sizeof(offsets)/sizeof(*offsets);++j)
                if (offsets[j]<n) check_direct(start,start+n,start+offsets[j]);
        }
    }
    /* Regression examples: old eight-row GEMV chunks cannot become a 511-row
     * KV-prefix GEMM; real odd anchor still retains a useful, causal saving. */
    struct starts old,now;init_starts(&old);init_starts(&now);
    REQUIRE(prefill_grid_guarded_starts(0,1529,512,prefill_snapshot_grid_shift(1529,512),old.value,now.value));
    REQUIRE(old.value[39]==1016u && now.value[39]==1016u);
    REQUIRE(prefill_grid_guarded_starts(123,4219,512,prefill_snapshot_grid_shift(4219,512),old.value,now.value));
    REQUIRE(old.value[39]==3706u && now.value[39]==4064u);
    printf("[PASS] guarded prefill GRID: %llu all40 schedule cases, %llu direct-boundary cases, %llu fragments, %llu unchanged short chunks, %llu periodic points, %llu rejected inputs, %u attention policy cases; %llu fixture full rows avoided; %llu exact/%llu rounded boundaries. CPU schedule/class proof only; no model/GPU parity or performance claim.\n",
        (unsigned long long)cases,(unsigned long long)direct_cases,(unsigned long long)fragments,
        (unsigned long long)old_short_chunks,(unsigned long long)periodic_points,
        (unsigned long long)early_returns,attention_policy_cases,(unsigned long long)avoided_rows,
        (unsigned long long)unchanged_splits,(unsigned long long)partial_splits);
    return 0;
}
