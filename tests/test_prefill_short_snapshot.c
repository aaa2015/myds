/* Exercise the production short prompt-end checkpoint and existing transactional
 * copy/restore with real GPU buffers, plus host-return failure injection. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "myds_metal.h"
static int fixture_begin(void);
static int fixture_commit(void);
static int fixture_copy(myds_metal_buffer_t *, size_t, const myds_metal_buffer_t *, size_t, size_t);
#define myds_metal_begin_batch fixture_begin
#define myds_metal_commit_batch fixture_commit
#define myds_metal_buffer_copy fixture_copy
#include "../src/myds_v41.c"
#undef myds_metal_begin_batch
#undef myds_metal_commit_batch
#undef myds_metal_buffer_copy

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); \
} } while (0)

static unsigned begins, commits, copies, fail_begin, fail_commit, fail_copy, cases;
static int fixture_begin(void) {
    ++begins;
    return begins == fail_begin ? 0 : myds_metal_begin_batch();
}
static int fixture_commit(void) {
    ++commits;
    const int ok = myds_metal_commit_batch();
    return commits == fail_commit ? 0 : ok;
}
static int fixture_copy(myds_metal_buffer_t *dst, size_t dst_off,
                        const myds_metal_buffer_t *src, size_t src_off, size_t bytes) {
    ++copies;
    return copies == fail_copy ? 0 : myds_metal_buffer_copy(dst,dst_off,src,src_off,bytes);
}
static void reset_hooks(void) {
    begins=commits=copies=fail_begin=fail_commit=fail_copy=0;
}
static void mode(const char *flag, const char *prefix) {
    if (flag) REQUIRE(!setenv("MYDS_PREFILL_SHORT_END_SNAPSHOT",flag,1));
    else REQUIRE(!unsetenv("MYDS_PREFILL_SHORT_END_SNAPSHOT"));
    if (prefix) REQUIRE(!setenv("MYDS_PREFIX_CACHE",prefix,1));
    else REQUIRE(!unsetenv("MYDS_PREFIX_CACHE"));
}
static void pattern(myds_metal_buffer_t *b, unsigned salt) {
    float *p=myds_metal_buffer_data(b);
    REQUIRE(p);
    for(size_t i=0;i<myds_metal_buffer_size(b)/sizeof(float);++i)
        p[i]=(float)((int)((i*17u+salt*43u)%8191u)-4095)/4096.0f;
}
static void populate(myds_v41_context_t *g, uint32_t p, unsigned salt) {
    for(unsigned il=0;il<V41_LAYERS;++il) pattern(g->window[il],salt+il);
    for(unsigned o=0;o<4;++o) {
        pattern(g->previous_kv[o],salt+100u+o);
        pattern(g->previous_score[o],salt+200u+o);
    }
    myds_v41_snapshot_set_engram_history(g,salt+1u,salt+2u,salt+3u);
    myds_v41_snapshot_set_positions(g,p,p);
}
static void compare_state(myds_v41_context_t *g, uint32_t p, const myds_engram_history_t *history) {
    const size_t bytes=(size_t)V41_WINDOW*V41_HEAD_DIM*sizeof(float);
    const unsigned char *snapshot=myds_metal_buffer_data(g->snap_window[0]);
    REQUIRE(snapshot);
    for(unsigned il=0;il<V41_LAYERS;++il)
        REQUIRE(!memcmp(myds_metal_buffer_data(g->window[il]),snapshot+(size_t)il*bytes,bytes));
    for(unsigned o=0;o<4;++o) {
        REQUIRE(!memcmp(myds_metal_buffer_data(g->previous_kv[o]),myds_metal_buffer_data(g->snap_prev_kv[0][o]),512u*sizeof(float)));
        REQUIRE(!memcmp(myds_metal_buffer_data(g->previous_score[o]),myds_metal_buffer_data(g->snap_prev_score[0][o]),512u*sizeof(float)));
    }
    REQUIRE(!memcmp(&g->engram->history,history,sizeof(*history)));
    REQUIRE(g->engram->position==p && g->next_position==p && g->valid);
}
static void roundtrip(uint32_t p, const char *flag, const char *prefix) {
    mode(flag,prefix);
    myds_v41_context_t *g=myds_v41_context_create_snapshot_fixture(); REQUIRE(g);
    populate(g,p,p);
    myds_engram_history_t history=g->engram->history;
    reset_hooks();
    REQUIRE(myds_v41_snapshot_store_now(g,p));
    REQUIRE(begins==1 && commits==2 && copies==48);
    REQUIRE(g->snap_pos[0]==p && !g->snap_staged_pos[0]);
    REQUIRE(myds_v41_snapshot_best(g,p)==p && !myds_v41_snapshot_best(g,p-1u));
    REQUIRE(myds_v41_snapshot_allocated_bytes(g)==(uint64_t)V41_LAYERS*V41_WINDOW*V41_HEAD_DIM*4u+8u*512u*4u);
    compare_state(g,p,&history);
    populate(g,p+257u,p+77u);
    reset_hooks();
    REQUIRE(myds_v41_snapshot_restore(g,p));
    REQUIRE(copies==48);
    compare_state(g,p,&history);
    myds_v41_snapshot_clear(g,p);
    REQUIRE(myds_v41_snapshot_best(g,p)==p);
    myds_v41_snapshot_clear(g,p-1u);
    REQUIRE(!myds_v41_snapshot_best(g,p));
    myds_v41_context_free(g);++cases;
}
static void noop(uint32_t p,const char *flag,const char *prefix,bool snapshots) {
    mode(flag,prefix);
    myds_v41_context_t *g=myds_v41_context_create_snapshot_fixture();REQUIRE(g);
    g->snap_enabled=snapshots;
    reset_hooks();
    REQUIRE(myds_v41_snapshot_store_now(g,p));
    REQUIRE(!begins && !commits && !copies && !g->snap_pos[0]);
    REQUIRE(!myds_v41_snapshot_allocated_bytes(g));
    myds_v41_context_free(g);++cases;
}
static void failure(const char *flag,unsigned begin_at,unsigned commit_at,unsigned copy_at) {
    mode(flag,"1");
    myds_v41_context_t *g=myds_v41_context_create_snapshot_fixture();REQUIRE(g);
    populate(g,890u,31u);reset_hooks();
    REQUIRE(myds_v41_snapshot_store_now(g,890u));
    populate(g,891u,77u);reset_hooks();
    fail_begin=begin_at;fail_commit=commit_at;fail_copy=copy_at;
    REQUIRE(!myds_v41_snapshot_store_now(g,891u));
    REQUIRE(!g->snap_pos[0] && !g->snap_staged_pos[0] && !myds_v41_snapshot_best(g,891u));
    /* Retry must replace any partially written snapshot completely. */
    reset_hooks();REQUIRE(myds_v41_snapshot_store_now(g,891u));
    myds_engram_history_t history=g->engram->history;
    populate(g,999u,99u);reset_hooks();REQUIRE(myds_v41_snapshot_restore(g,891u));
    compare_state(g,891u,&history);
    myds_v41_context_free(g);++cases;
}
int main(void) {
    REQUIRE(myds_metal_init("metal"));
    const uint32_t positions[]={4u,5u,9u,31u,63u,127u,128u,129u,511u,512u,513u,890u,891u,1023u,1024u};
    /* Absent and exact "1" must both use the production default-on path. */
    const char *enabled_flags[]={NULL,"1"};
    for(size_t f=0;f<sizeof(enabled_flags)/sizeof(*enabled_flags);++f) {
        const char *flag=enabled_flags[f];
        for(size_t i=0;i<sizeof(positions)/sizeof(*positions);++i)
            roundtrip(positions[i],flag,"1");
        noop(0u,flag,"1",true);noop(1u,flag,"1",true);noop(3u,flag,"1",true);
        noop(890u,flag,"0",true);noop(890u,flag,NULL,true);noop(890u,flag,"invalid",true);
        noop(890u,flag,"1",false);
        failure(flag,1u,0u,0u);failure(flag,0u,1u,0u);failure(flag,0u,2u,0u);
        failure(flag,0u,0u,1u);failure(flag,0u,0u,24u);failure(flag,0u,0u,48u);
        /* Short RAM end coverage must not alter periodic/pinned staging policy. */
        mode(flag,"1");myds_v41_context_t *g=myds_v41_context_create_snapshot_fixture();REQUIRE(g);
        snapshot_store_engram(g,890u,true);REQUIRE(!g->snap_staged_pos[0]);
        REQUIRE(myds_v41_snapshot_store_now(NULL,890u));
        myds_v41_context_free(g);++cases;
    }
    /* Only the exact opt-in spelling overrides the enabled default. */
    const char *disabled_flags[]={"0","invalid","","01","1x"};
    for(size_t i=0;i<sizeof(disabled_flags)/sizeof(*disabled_flags);++i)
        noop(890u,disabled_flags[i],"1",true);
    /* The pre-existing >=1024 path does not depend on either short-only flag. */
    roundtrip(1024u,"0","0");roundtrip(1024u,"invalid",NULL);
    REQUIRE(cases==65u);
    myds_metal_cleanup();
    printf("[PASS] short prompt-end RAM snapshot: %u cases (32 roundtrips, 19 no-allocation guards, 12 fault retries, 2 periodic guards), default absent/exact-1, full window/previous/Engram byte roundtrip, unchanged periodic minimum\n",cases);
    return 0;
}
