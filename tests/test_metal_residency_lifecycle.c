/* Native Metal ownership/residency acceptance; no model, mocked GPU, or timing.
 * SDK contract: add/remove are staged until commit; subviews must keep their
 * allocation registered until their last wrapper and pending GPU use finish.
 * https://developer.apple.com/documentation/metal/mtlresidencyset
 * device.currentAllocatedSize is recorded, never asserted equal to a baseline:
 * driver caches and aliasing do not establish physical unique wired pages. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "myds_metal.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
enum { BYTES=65536 };
static myds_metal_residency_stats_t baseline;
static myds_metal_residency_stats_t stats(const char *phase) {
    myds_metal_residency_stats_t s;
    REQUIRE(myds_metal_residency_diagnostics(&s));
    printf("{\"phase\":\"%s\",\"allocation_count\":%" PRIu64 ",\"allocated_size\":%" PRIu64
        ",\"device_allocated_size\":%" PRIu64 ",\"live_mapped\":%" PRIu64 ",\"pending\":%" PRIu64
        ",\"active\":%s,\"heartbeat\":%s}\n",phase,s.allocation_count,s.allocated_size,
        s.device_allocated_size,s.live_mapped_allocations,s.pending_unregister,
        s.active?"true":"false",s.heartbeat_running?"true":"false");
    fflush(stdout);
    return s;
}
static void empty(const char *phase) {
    myds_metal_residency_stats_t s=stats(phase);
    REQUIRE(s.allocation_count==baseline.allocation_count && s.allocated_size==baseline.allocated_size);
    REQUIRE(s.live_mapped_allocations==0 && s.pending_unregister==0);
}
static void *mapping(unsigned salt) {
    void *p=mmap(NULL,BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    REQUIRE(p!=MAP_FAILED);
    for(size_t i=0;i<BYTES;i++)((unsigned char *)p)[i]=(unsigned char)(salt+17u*i);
    return p;
}
static myds_metal_buffer_t *mapped(void *p) {
    myds_metal_buffer_t *b=myds_metal_buffer_from_ptr(p,BYTES);REQUIRE(b);return b;
}
static void copied(const myds_metal_buffer_t *src,const void *reference) {
    myds_metal_buffer_t *out=myds_metal_buffer_alloc(64);REQUIRE(out);
    REQUIRE(myds_metal_begin_batch() && myds_metal_buffer_copy(out,0,src,0,32) && myds_metal_commit_batch());
    REQUIRE(!memcmp(myds_metal_buffer_data(out),reference,32));myds_metal_buffer_free(out);
}
static void prime(void) {
    void *p=mapping(1);myds_metal_buffer_t *b=mapped(p);
    myds_metal_residency_activate();
    myds_metal_residency_stats_t s=stats("prime_registered");
    REQUIRE(s.supported&&s.active&&s.heartbeat_running&&s.allocation_count==1);
    myds_metal_buffer_free(b);REQUIRE(munmap(p,BYTES)==0);
    baseline=stats("empty_set_baseline");
    REQUIRE(baseline.allocation_count==0&&baseline.live_mapped_allocations==0&&baseline.pending_unregister==0);
}
static void parent_first(void) {
    void *p=mapping(2);myds_metal_buffer_t *a=mapped(p);
    myds_metal_buffer_t *v=myds_metal_buffer_sub_view(a,16,64);
    myds_metal_buffer_t *child=myds_metal_buffer_sub_view(v,16,32);REQUIRE(v&&child);
    myds_metal_buffer_free(a);REQUIRE(stats("parent_freed_child_live").allocation_count==1);
    myds_metal_buffer_free(v);REQUIRE(stats("nested_child_live").allocation_count==1);
    copied(child,(unsigned char *)p+32);myds_metal_buffer_free(child);
    empty("nested_last_free");REQUIRE(munmap(p,BYTES)==0);
    /* Ordinary alloc has no residency owner, but its ARC subview must survive. */
    a=myds_metal_buffer_alloc(64);REQUIRE(a);memset(myds_metal_buffer_data(a),0x3d,64);
    unsigned char expected[32];memset(expected,0x3d,32);v=myds_metal_buffer_sub_view(a,0,32);REQUIRE(v);
    myds_metal_buffer_free(a);copied(v,expected);myds_metal_buffer_free(v);empty("ordinary_parent_first");
}
static void rebind(void) {
    void *p=mapping(3),*q=mapping(4);myds_metal_buffer_t *a=mapped(p),*b=mapped(q);
    myds_metal_buffer_t *v=myds_metal_buffer_sub_view(a,0,64);REQUIRE(v);
    myds_metal_buffer_free(a);REQUIRE(stats("rebind_two_allocations").allocation_count==2);
    REQUIRE(myds_metal_buffer_update_sub_view(v,b,16,64));
    REQUIRE(stats("rebind_released_old").allocation_count==1);
    REQUIRE(munmap(p,BYTES)==0);
    REQUIRE(myds_metal_buffer_update_sub_view(v,b,32,64)); /* same owner */
    REQUIRE(myds_metal_buffer_update_sub_view(v,v,16,32)); /* self rebind */
    copied(v,(unsigned char *)q+48);
    myds_metal_buffer_free(b);REQUIRE(stats("rebind_parent_freed").allocation_count==1);
    myds_metal_buffer_free(v);empty("rebind_last_free");REQUIRE(munmap(q,BYTES)==0);
    /* An empty view can acquire/release mapped ownership via an ordinary parent. */
    p=mapping(5);a=mapped(p);b=myds_metal_buffer_alloc(64);v=myds_metal_buffer_create_view();REQUIRE(b&&v);
    REQUIRE(myds_metal_buffer_update_sub_view(v,a,0,32));myds_metal_buffer_free(a);
    REQUIRE(myds_metal_buffer_update_sub_view(v,b,0,32));empty("rebind_to_unregistered");
    REQUIRE(munmap(p,BYTES)==0);myds_metal_buffer_free(b);myds_metal_buffer_free(v);
}
static void pending(bool older_inflight,bool injected) {
    void *p=mapping(6),*q=older_inflight?mapping(7):NULL;
    myds_metal_buffer_t *a=mapped(p),*out=myds_metal_buffer_alloc(64);REQUIRE(out);
    REQUIRE(myds_metal_begin_batch()&&myds_metal_buffer_copy(out,0,a,0,32));
    if(older_inflight) {
        REQUIRE(myds_metal_flush_batch());myds_metal_buffer_free(a);
        a=mapped(q);REQUIRE(myds_metal_begin_batch()&&myds_metal_buffer_copy(out,32,a,0,32));
    }
    myds_metal_buffer_free(a);
    myds_metal_residency_stats_t s=stats(older_inflight?"pending_older_and_active":"pending_active");
    REQUIRE(s.pending_unregister==(older_inflight?2u:1u)&&s.allocation_count==s.pending_unregister);
    if(injected)myds_metal_inject_commit_failure(1);
    REQUIRE(myds_metal_commit_batch()==(injected?0:1));
    myds_metal_inject_commit_failure(0);
    REQUIRE(!memcmp(myds_metal_buffer_data(out),p,32));
    if(older_inflight)REQUIRE(!memcmp((unsigned char *)myds_metal_buffer_data(out)+32,q,32));
    empty(injected?"injected_drain_released":"normal_drain_released");
    REQUIRE(munmap(p,BYTES)==0);if(q)REQUIRE(munmap(q,BYTES)==0);myds_metal_buffer_free(out);
    /* An inflight-only submission must also unregister at commit-without-batch. */
    p=mapping(8);a=mapped(p);out=myds_metal_buffer_alloc(64);REQUIRE(out);
    REQUIRE(myds_metal_begin_batch()&&myds_metal_buffer_copy(out,0,a,0,32)&&myds_metal_flush_batch());
    myds_metal_buffer_free(a);REQUIRE(stats("pending_inflight_only").pending_unregister==1);
    REQUIRE(myds_metal_commit_batch());empty("inflight_only_released");
    REQUIRE(!memcmp(myds_metal_buffer_data(out),p,32));myds_metal_buffer_free(out);REQUIRE(munmap(p,BYTES)==0);
}
static void cleanup_reinit(void) {
    void *p=mapping(9);myds_metal_buffer_t *old=mapped(p),*out=myds_metal_buffer_alloc(64);REQUIRE(out);
    REQUIRE(myds_metal_begin_batch()&&myds_metal_buffer_copy(out,0,old,0,32)&&myds_metal_flush_batch());
    /* Keep one old wrapper alive across cleanup to check the registration epoch. */
    void *q=mapping(10);myds_metal_buffer_t *pending_owner=mapped(q);
    REQUIRE(myds_metal_begin_batch()&&myds_metal_buffer_copy(out,32,pending_owner,0,32));
    myds_metal_buffer_free(pending_owner);myds_metal_inject_commit_failure(1);
    myds_metal_cleanup();myds_metal_inject_commit_failure(0);
    myds_metal_residency_stats_t s=stats("cleanup_joined");
    REQUIRE(!s.active&&!s.heartbeat_running&&s.allocation_count==0&&s.allocated_size==0&&s.pending_unregister==0);
    REQUIRE(!memcmp(myds_metal_buffer_data(out),p,32)&&!memcmp((unsigned char *)myds_metal_buffer_data(out)+32,q,32));
    myds_metal_buffer_free(out);REQUIRE(munmap(q,BYTES)==0);
    REQUIRE(myds_metal_init(NULL));
    q=mapping(11);myds_metal_buffer_t *fresh=mapped(q);myds_metal_residency_activate();
    REQUIRE(stats("reinit_registered").allocation_count==1);
    REQUIRE(!myds_metal_buffer_sub_view(old,0,32));
    myds_metal_buffer_t *view=myds_metal_buffer_create_view();REQUIRE(view);
    REQUIRE(!myds_metal_buffer_update_sub_view(view,old,0,32));myds_metal_buffer_free(view);
    myds_metal_buffer_free(old);REQUIRE(stats("old_epoch_freed").allocation_count==1);
    REQUIRE(munmap(p,BYTES)==0);copied(fresh,q);myds_metal_buffer_free(fresh);REQUIRE(munmap(q,BYTES)==0);
    REQUIRE(stats("reinit_empty").allocation_count==0);
    myds_metal_cleanup();s=stats("final_cleanup");
    REQUIRE(!s.active&&!s.heartbeat_running&&s.allocation_count==0&&s.pending_unregister==0&&s.live_mapped_allocations==0);
}
int main(void) {
    REQUIRE(unsetenv("MYDS_NO_RESIDENCY_HB")==0);
    REQUIRE(myds_metal_init(NULL));
    if(!stats("native_init").supported){fputs("SKIP: macOS 15 residency API required\n",stderr);myds_metal_cleanup();return 77;}
    prime();parent_first();rebind();pending(false,false);pending(true,false);pending(true,true);
    for(unsigned i=0;i<10;i++) {
        void *p=mapping(31u+i);myds_metal_buffer_t *a=mapped(p),*v=myds_metal_buffer_sub_view(a,0,32);REQUIRE(v);
        REQUIRE(stats("repeat_load").allocation_count==1);myds_metal_buffer_free(a);copied(v,p);
        myds_metal_buffer_free(v);empty("repeat_free");REQUIRE(munmap(p,BYTES)==0);
    }
    cleanup_reinit();puts("PASS: native residency ownership, pending submissions, ten load/free cycles, cleanup/reinit");return 0;
}
