/* CPU-only dispatch-contract test. Compiles the real decode wrapper and stubs
 * its Metal calls; GPU arithmetic and missing-PSO fallback are tested separately.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_decode.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local int generic_ok = 1, fused_ok = 1, round_ok = 1;
static _Thread_local char calls[8];
static _Thread_local size_t call_count;
static myds_metal_buffer_t *const out = (myds_metal_buffer_t *)0x10;
static const myds_metal_buffer_t *const weight = (myds_metal_buffer_t *)0x20;
static const myds_metal_buffer_t *const input = (myds_metal_buffer_t *)0x30;
static _Thread_local ggml_type expected_type;
static _Thread_local int forwarded_ok;
static _Thread_local myds_metal_buffer_t *expected_out;
static _Thread_local const myds_metal_buffer_t *expected_weight, *expected_input;
static _Thread_local uint32_t expected_rows, expected_width;
static _Thread_local size_t out_offset;

#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c); return 1; } } while (0)
static void record(char kind) {
    if (call_count + 1 < sizeof(calls)) calls[call_count++] = kind;
    calls[call_count] = 0;
}
static void forwarded(myds_metal_buffer_t *a, const myds_metal_buffer_t *b,
    const myds_metal_buffer_t *c, uint32_t rows, uint32_t width, ggml_type type) {
    forwarded_ok &= a == expected_out && b == expected_weight && c == expected_input &&
        rows == expected_rows && width == expected_width && type == expected_type;
}
int myds_metal_gemv(myds_metal_buffer_t *a, const myds_metal_buffer_t *b,
    const myds_metal_buffer_t *c, uint32_t rows, uint32_t width, ggml_type type) {
    forwarded(a,b,c,rows,width,type); record('G'); return generic_ok;
}
int myds_metal_v41_gemv_bf16(myds_metal_buffer_t *a, const myds_metal_buffer_t *b,
    const myds_metal_buffer_t *c, uint32_t rows, uint32_t width, ggml_type type) {
    forwarded(a,b,c,rows,width,type); record('F'); return fused_ok;
}
int myds_metal_v41_bf16(myds_metal_buffer_t *a, uint32_t rows) {
    forwarded_ok &= a == expected_out && rows == expected_rows; record('B');
    return round_ok && out_offset % 16u == 0;
}
size_t myds_metal_buffer_offset(const myds_metal_buffer_t *b) {
    forwarded_ok &= b == expected_out; return out_offset;
}
int myds_metal_get_auto_barrier(void) {
    return 1;
}
void myds_metal_barrier_buffer(const myds_metal_buffer_t *b) {
    (void)b;
}
/* The linked source also defines the independent QKV encoder scope, which has
 * real Metal coverage in test_qkv_fusion. Its linker stubs fail if this Q8
 * contract unexpectedly starts changing encoder state; they cannot hide such
 * a change behind successful no-op implementations. */
static _Noreturn void unexpected_scope_call(const char *name) {
    fprintf(stderr,"FAIL Q8 wrapper unexpectedly called %s\n",name);
    abort();
}
int myds_metal_get_concurrent(void) {
    unexpected_scope_call("myds_metal_get_concurrent");
}
void myds_metal_set_concurrent(int enable) {
    (void)enable;unexpected_scope_call("myds_metal_set_concurrent");
}
void myds_metal_set_auto_barrier(int enable) {
    (void)enable;unexpected_scope_call("myds_metal_set_auto_barrier");
}
void myds_metal_barrier(void) {
    unexpected_scope_call("myds_metal_barrier");
}
static int invoke_args(myds_metal_buffer_t *a, const myds_metal_buffer_t *b,
    const myds_metal_buffer_t *c, uint32_t rows, uint32_t width, ggml_type type, bool trace_on) {
    memset(calls,0,sizeof(calls)); call_count = 0; forwarded_ok = 1;
    expected_type = type; expected_out=a; expected_weight=b; expected_input=c;
    expected_rows=rows; expected_width=width;
    return myds_v41_decode_gemv_bf16(a,b,c,rows,width,type,trace_on);
}
static int invoke(ggml_type type, bool trace_on) {
    return invoke_args(out,weight,input,2304,5120,type,trace_on);
}
#define THREADS 8
static atomic_int ready, start_threads;
struct thread_args { unsigned index; int enabled, ok; };
static void *worker(void *opaque) {
    struct thread_args *a=opaque;
    myds_metal_buffer_t *o=(myds_metal_buffer_t *)(uintptr_t)(0x1000u+a->index*0x100u);
    const myds_metal_buffer_t *w=(const myds_metal_buffer_t *)(uintptr_t)(0x2000u+a->index*0x100u);
    const myds_metal_buffer_t *x=(const myds_metal_buffer_t *)(uintptr_t)(0x3000u+a->index*0x100u);
    const uint32_t rows=2304u+a->index, width=5120u+a->index*32u;
    atomic_fetch_add_explicit(&ready,1,memory_order_release);
    while(!atomic_load_explicit(&start_threads,memory_order_acquire)) sched_yield();
    a->ok=1;
    for(unsigned i=0;i<64;i++) {
        const bool trace_on=i%3u==1u;
        const ggml_type type=i%3u==2u?GGML_TYPE_F32:GGML_TYPE_Q8_0;
        const char *expected=a->enabled&&!trace_on&&type==GGML_TYPE_Q8_0?"F":"GB";
        if(!invoke_args(o,w,x,rows,width,type,trace_on) || !forwarded_ok || strcmp(calls,expected)) {
            a->ok=0; break;
        }
    }
    return NULL;
}
static int concurrent_first_calls(int enabled) {
    pthread_t threads[THREADS]; struct thread_args args[THREADS];
    for(unsigned i=0;i<THREADS;i++) {
        args[i]=(struct thread_args){.index=i,.enabled=enabled};
        CHECK(pthread_create(&threads[i],NULL,worker,&args[i])==0);
    }
    while(atomic_load_explicit(&ready,memory_order_acquire)<THREADS) sched_yield();
    atomic_store_explicit(&start_threads,1,memory_order_release);
    for(unsigned i=0;i<THREADS;i++) {CHECK(pthread_join(threads[i],NULL)==0); CHECK(args[i].ok);}
    return 0;
}
int main(int argc, char **argv) {
    CHECK(argc == 2 && (!strcmp(argv[1],"0") || !strcmp(argv[1],"1")));
    const int enabled = !strcmp(argv[1],"1");
    const char *q8_calls = enabled ? "F" : "GB";
    CHECK(concurrent_first_calls(enabled)==0);
    CHECK(invoke(GGML_TYPE_Q8_0,false) && forwarded_ok && !strcmp(calls,q8_calls));
    for(size_t offset=4;offset<16;offset+=4) {
        out_offset=offset;
        CHECK(!invoke(GGML_TYPE_Q8_0,false) && forwarded_ok && !strcmp(calls,"GB"));
    }
    out_offset=16;
    CHECK(invoke(GGML_TYPE_Q8_0,false) && forwarded_ok && !strcmp(calls,q8_calls));
    out_offset=0;
    CHECK(invoke(GGML_TYPE_Q8_0,true) && forwarded_ok && !strcmp(calls,"GB"));
    const ggml_type fallback[] = {GGML_TYPE_F16,GGML_TYPE_F32,GGML_TYPE_Q2_K};
    for (size_t i = 0; i < sizeof(fallback)/sizeof(*fallback); i++)
        CHECK(invoke(fallback[i],false) && forwarded_ok && !strcmp(calls,"GB"));
    generic_ok = 0;
    CHECK(!invoke(GGML_TYPE_Q8_0,true) && forwarded_ok && !strcmp(calls,"G"));
    CHECK(!invoke(GGML_TYPE_F16,false) && forwarded_ok && !strcmp(calls,"G"));
    generic_ok = 1; round_ok = 0;
    CHECK(!invoke(GGML_TYPE_Q8_0,true) && forwarded_ok && !strcmp(calls,"GB"));
    CHECK(!invoke(GGML_TYPE_F32,false) && forwarded_ok && !strcmp(calls,"GB"));
    round_ok = 1; fused_ok = 0;
    if (enabled)
        CHECK(!invoke(GGML_TYPE_Q8_0,false) && forwarded_ok && !strcmp(calls,"F"));
    fused_ok = 1;
    CHECK(setenv("MYDS_DECODE_Q8_FUSE_BF16", enabled ? "0" : "1",1) == 0);
    CHECK(invoke(GGML_TYPE_Q8_0,false) && forwarded_ok && !strcmp(calls,q8_calls));
    printf("PASS decode Q8 wrapper enabled=%d: 8 concurrent first-call threads with distinct arguments, forwarded arguments, output offsets 0/4/8/12/16, trace, 3 formats, GEMV/BF16/fused failures, read-once switch\n",enabled);
    return 0;
}
