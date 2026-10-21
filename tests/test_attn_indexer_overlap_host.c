/* Actual C scheduling entry for MYDS_ATTN_HC_OVERLAP and MYDS_INDEXER_DECODE_OVERLAP
 * with explicit CPU recording stubs only. No Metal device, model execution,
 * shader math or performance claim.
 * clang -O2 -Wall -Wextra -std=c11 -Iinclude -Ithird_party -Wl,-dead_strip
 *   tests/test_attn_indexer_overlap_host.c src/myds_v41_moe.c -o /tmp/test_attn_indexer_overlap_host
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_v41_decode.h"
#include "myds_v41_moe.h"
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_log.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct myds_metal_buffer { size_t bytes, offset; void *data; };
struct myds_v41_gpu_scratch { int dummy; };

static struct myds_metal_buffer buffers[128];
static struct myds_v41_gpu_scratch scratch_inst;
static unsigned used, events;
static const char *event_names[128], *failure;
static int auto_barrier = 1, concurrent = 0, batch = 1, trace = 0;
static int dense_q4 = 1, f16_bf16 = 1, idx_rope_quant = 1, head_slots = 1, topk_thresh = 0;

static int record(const char *name) {
    assert(events < 128);
    event_names[events++] = name;
    return !failure || strcmp(failure, name) != 0;
}

static myds_metal_buffer_t *buffer(size_t bytes) {
    assert(used < 120);
    struct myds_metal_buffer *b = &buffers[used++];
    *b = (struct myds_metal_buffer){
        bytes, 0, (void *)(uintptr_t)(UINT64_C(0x100000000) + used * UINT64_C(0x2000000))
    };
    return b;
}

size_t myds_metal_buffer_size(const myds_metal_buffer_t *b) { return b ? b->bytes : 0; }
size_t myds_metal_buffer_offset(const myds_metal_buffer_t *b) { return b ? b->offset : 0; }
void *myds_metal_buffer_data(myds_metal_buffer_t *b) { return b ? b->data : NULL; }
void myds_log(myds_log_level_t level, const char *fmt, ...) { (void)level; (void)fmt; }
bool myds_trace_active(void) { return trace != 0; }
bool myds_trace_f32(const char *s, uint32_t l, uint32_t p, const myds_metal_buffer_t *b, size_t n) {
    (void)s; (void)l; (void)p; (void)b; (void)n;
    return true;
}

int myds_metal_get_auto_barrier(void) { return auto_barrier; }
int myds_metal_get_concurrent(void) { return concurrent; }
void myds_metal_set_auto_barrier(int v) { auto_barrier = v; }
void myds_metal_set_concurrent(int v) { concurrent = v; }
int myds_metal_batch_active(void) { return batch; }

int myds_metal_dense_q4_v41_enabled(void) { return dense_q4; }
int myds_metal_f16_gemv_bf16_fuse_enabled(void) { return f16_bf16; }
int myds_metal_indexer_rope_quant_fuse_enabled(void) { return idx_rope_quant; }
int myds_v41_gpu_indexer_head_slots_enabled(void) { return head_slots; }
int myds_v41_gpu_topk_threshold_enabled(void) { return topk_thresh; }
static int kv_reuse = 0;
int myds_v41_attn_kv_reuse_enabled(void) { return kv_reuse; }
void myds_v41_attn_kv_reuse_record_prestage_failure(void) {}
void myds_v41_gpu_attn_kv_reuse_reset(myds_v41_gpu_scratch_t *s) { (void)s; }
int myds_v41_gpu_attn_kv_reuse_prepare_layer(myds_v41_gpu_scratch_t *s,
    uint32_t layer, uint32_t count, bool is_index_source) {
    (void)s; (void)layer; (void)count;
    return is_index_source ? 1 : 2;
}
int myds_v41_gpu_attn_kv_reuse_prestage_window(myds_v41_gpu_scratch_t *s,
    const myds_metal_buffer_t *window, uint32_t pos) {
    (void)s; (void)window; (void)pos;
    return record("prestage_window");
}
void myds_v41_gpu_attn_kv_reuse_mark_fenced(myds_v41_gpu_scratch_t *s) { (void)s; }
static uint32_t stub_qb_done = 0u;
void myds_v41_gpu_attn_kv_reuse_defer_q_rope(myds_v41_gpu_scratch_t *s,
    myds_metal_buffer_t *q, bool compressed) {
    (void)s; (void)q; (void)compressed;
    stub_qb_done = 0u;
    record("defer_q_rope");
}
void myds_v41_gpu_attn_kv_reuse_bind_topk_qb(myds_v41_gpu_scratch_t *s,
    const myds_metal_buffer_t *qb_weight, const myds_metal_buffer_t *qb_input,
    uint32_t pos, uint32_t qb_done_heads) {
    (void)s; (void)qb_weight; (void)qb_input; (void)pos;
    stub_qb_done = qb_done_heads ? 64u : 0u;
    record("bind_topk_qb");
}
uint32_t myds_v41_gpu_attn_kv_reuse_qb_done(const myds_v41_gpu_scratch_t *s) {
    (void)s;
    return stub_qb_done;
}
const myds_metal_buffer_t *myds_v41_gpu_scratch_kv_f16(const myds_v41_gpu_scratch_t *s) {
    (void)s;
    return &buffers[126];
}
const myds_metal_buffer_t *myds_v41_gpu_scratch_pad(const myds_v41_gpu_scratch_t *s) {
    (void)s;
    return &buffers[127];
}
myds_metal_buffer_t *myds_v41_gpu_scratch_partial(const myds_v41_gpu_scratch_t *s) {
    (void)s;
    return &buffers[125];
}
int myds_v41_gpu_attention_direct_pack_fuse_enabled(void) { return 0; }
int myds_v41_attention_rope_fuse_enabled(void) { return 1; }
int myds_v41_gpu_attention_packed_register_enabled(void) { return 1; }
int myds_metal_moe_down_overlap_enabled(void) { return 1; }
int myds_metal_moe_q2_v41_enabled(void) { return 1; }
int myds_metal_moe_iq2_v41_enabled(void) { return 0; }
int myds_metal_shared_q4_v41_nb_enabled(void) { return 0; }
int myds_metal_shared_q4_v41_nb_gate_up(myds_metal_buffer_t *o, const myds_metal_buffer_t *g,
    const myds_metal_buffer_t *u, const myds_metal_buffer_t *x, uint32_t w, uint32_t d, float c) {
    (void)o; (void)g; (void)u; (void)x; (void)w; (void)d; (void)c;
    return 1;
}
int myds_metal_routed_moe_down_overlap_scoped(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *g, const myds_metal_buffer_t *u, const myds_metal_buffer_t *d,
    const myds_metal_buffer_t *ids, const myds_metal_buffer_t *w, myds_metal_buffer_t *mg,
    myds_metal_buffer_t *mu, myds_metal_buffer_t *mm, myds_metal_buffer_t *md, uint32_t dim,
    uint32_t width, uint32_t n_exp, uint32_t k, ggml_type gt, ggml_type ut, ggml_type dt,
    float clamp, bool ordinary, myds_metal_buffer_t *so, const myds_metal_buffer_t *sd,
    const myds_metal_buffer_t *ss, ggml_type sdt) {
    (void)o; (void)x; (void)g; (void)u; (void)d; (void)ids; (void)w; (void)mg; (void)mu;
    (void)mm; (void)md; (void)dim; (void)width; (void)n_exp; (void)k; (void)gt; (void)ut;
    (void)dt; (void)clamp; (void)ordinary; (void)so; (void)sd; (void)ss; (void)sdt;
    return record("routed_moe_down_overlap_scoped");
}
int myds_metal_routed_moe_down_overlap_nbhc_scoped(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *g, const myds_metal_buffer_t *u, const myds_metal_buffer_t *d,
    const myds_metal_buffer_t *ids, const myds_metal_buffer_t *w, myds_metal_buffer_t *mg,
    myds_metal_buffer_t *mu, myds_metal_buffer_t *mm, myds_metal_buffer_t *md, uint32_t dim,
    uint32_t width, uint32_t n_exp, uint32_t k, ggml_type gt, ggml_type ut, ggml_type dt,
    float clamp, bool ordinary, myds_metal_buffer_t *so, const myds_metal_buffer_t *sd,
    const myds_metal_buffer_t *ss, ggml_type sdt) {
    (void)o; (void)x; (void)g; (void)u; (void)d; (void)ids; (void)w; (void)mg; (void)mu;
    (void)mm; (void)md; (void)dim; (void)width; (void)n_exp; (void)k; (void)gt; (void)ut;
    (void)dt; (void)clamp; (void)ordinary; (void)so; (void)sd; (void)ss; (void)sdt;
    return 1;
}
int myds_metal_v41_moe_add_hc_expand(myds_metal_buffer_t *r, myds_metal_buffer_t *fo,
    const myds_metal_buffer_t *mo, const myds_metal_buffer_t *so, const myds_metal_buffer_t *aa,
    const myds_metal_buffer_t *fs, uint32_t d) {
    (void)r; (void)fo; (void)mo; (void)so; (void)aa; (void)fs; (void)d;
    return record("moe_add_hc_expand");
}
int myds_metal_v41_route_fused(myds_metal_buffer_t *ids, myds_metal_buffer_t *w,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t n_exp, uint32_t k, float scale, int mode) {
    (void)ids; (void)w; (void)logits; (void)bias; (void)n_exp; (void)k; (void)scale; (void)mode;
    return record("route_fused");
}
int myds_metal_v41_route_simd_shuffle(myds_metal_buffer_t *ids, myds_metal_buffer_t *w,
    const myds_metal_buffer_t *logits, const myds_metal_buffer_t *bias,
    uint32_t n_exp, uint32_t k, float scale) {
    (void)ids; (void)w; (void)logits; (void)bias; (void)n_exp; (void)k; (void)scale;
    return record("route_simd_shuffle");
}
int myds_metal_v41_sigmoid_add_bias(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *b, uint32_t n) {
    (void)o; (void)x; (void)b; (void)n;
    return 1;
}
int myds_metal_v41_group_top2_sum(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    uint32_t g, uint32_t gs) {
    (void)o; (void)x; (void)g; (void)gs;
    return 1;
}
int myds_metal_v41_select_groups_mask(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *gs, uint32_t n, uint32_t g, uint32_t tg) {
    (void)o; (void)x; (void)gs; (void)n; (void)g; (void)tg;
    return 1;
}
int myds_metal_v41_topk_normalize(myds_metal_buffer_t *ids, myds_metal_buffer_t *w,
    const myds_metal_buffer_t *m, const myds_metal_buffer_t *r, uint32_t n, uint32_t k, float s) {
    (void)ids; (void)w; (void)m; (void)r; (void)n; (void)k; (void)s;
    return 1;
}
int myds_metal_v41_shared_gate_up_swiglu(myds_metal_buffer_t *o, const myds_metal_buffer_t *g,
    const myds_metal_buffer_t *u, const myds_metal_buffer_t *x, uint32_t w, uint32_t d,
    ggml_type gt, float c) {
    (void)o; (void)g; (void)u; (void)x; (void)w; (void)d; (void)gt; (void)c;
    return record("shared_gate_up_swiglu");
}
int myds_metal_v41_swiglu(myds_metal_buffer_t *o, const myds_metal_buffer_t *g,
    const myds_metal_buffer_t *u, uint32_t n, float c) {
    (void)o; (void)g; (void)u; (void)n; (void)c;
    return 1;
}
int myds_metal_routed_moe_gate_up(myds_metal_buffer_t *m, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *g, const myds_metal_buffer_t *u, const myds_metal_buffer_t *ids,
    uint32_t d, uint32_t w, uint32_t ne, uint32_t k, ggml_type gt, ggml_type ut, float c) {
    (void)m; (void)x; (void)g; (void)u; (void)ids; (void)d; (void)w; (void)ne; (void)k;
    (void)gt; (void)ut; (void)c;
    return 1;
}
int myds_metal_routed_moe_down(myds_metal_buffer_t *o, const myds_metal_buffer_t *m,
    const myds_metal_buffer_t *d, const myds_metal_buffer_t *ids, const myds_metal_buffer_t *w,
    uint32_t dim, uint32_t width, uint32_t ne, uint32_t k, ggml_type dt) {
    (void)o; (void)m; (void)d; (void)ids; (void)w; (void)dim; (void)width; (void)ne; (void)k; (void)dt;
    return 1;
}
int myds_metal_routed_moe_batch(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *g, const myds_metal_buffer_t *u, const myds_metal_buffer_t *d,
    const myds_metal_buffer_t *ids, const myds_metal_buffer_t *w, myds_metal_buffer_t *mg,
    myds_metal_buffer_t *mu, myds_metal_buffer_t *mm, myds_metal_buffer_t *md, uint32_t dim,
    uint32_t width, uint32_t ne, uint32_t k, uint32_t n_tok, ggml_type gt, ggml_type ut,
    ggml_type dt, float c) {
    (void)o; (void)x; (void)g; (void)u; (void)d; (void)ids; (void)w; (void)mg; (void)mu;
    (void)mm; (void)md; (void)dim; (void)width; (void)ne; (void)k; (void)n_tok; (void)gt;
    (void)ut; (void)dt; (void)c;
    return 1;
}
int myds_metal_v41_add(myds_metal_buffer_t *o, const myds_metal_buffer_t *a,
    const myds_metal_buffer_t *b, uint32_t n) {
    (void)o; (void)a; (void)b; (void)n;
    return 1;
}
int myds_metal_v41_hc_expand(myds_metal_buffer_t *o, const myds_metal_buffer_t *b,
    const myds_metal_buffer_t *r, const myds_metal_buffer_t *s, uint32_t d) {
    (void)o; (void)b; (void)r; (void)s; (void)d;
    return 1;
}
int myds_metal_shared_gate_up_swiglu_shared3(myds_metal_buffer_t *o, const myds_metal_buffer_t *g,
    const myds_metal_buffer_t *u, const myds_metal_buffer_t *x, uint32_t w, uint32_t d,
    ggml_type gt, float c) {
    (void)o; (void)g; (void)u; (void)x; (void)w; (void)d; (void)gt; (void)c;
    return 1;
}
int myds_metal_routed_moe_shared3(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *g, const myds_metal_buffer_t *u, const myds_metal_buffer_t *d,
    const myds_metal_buffer_t *ids, const myds_metal_buffer_t *w, myds_metal_buffer_t *mg,
    myds_metal_buffer_t *mu, myds_metal_buffer_t *mm, myds_metal_buffer_t *md, uint32_t dim,
    uint32_t width, uint32_t ne, uint32_t k, ggml_type gt, ggml_type ut, ggml_type dt, float c) {
    (void)o; (void)x; (void)g; (void)u; (void)d; (void)ids; (void)w; (void)mg; (void)mu;
    (void)mm; (void)md; (void)dim; (void)width; (void)ne; (void)k; (void)gt; (void)ut;
    (void)dt; (void)c;
    return 1;
}

void myds_metal_barrier(void) { record("barrier"); }
void myds_metal_barrier_buffer(const myds_metal_buffer_t *b) { assert(b); record("join_1"); }
void myds_metal_barrier_buffers(const myds_metal_buffer_t *const *b, size_t n) {
    assert(b && n > 0);
    switch (n) {
        case 2: record("join_2"); break;
        case 3: record("join_3"); break;
        case 4: record("join_4"); break;
        case 5: record("join_5"); break;
        case 6: record("join_6"); break;
        case 7: record("join_7"); break;
        case 9: record("join_9"); break;
        default: record("join_n"); break;
    }
}
int myds_metal_dispatch_kernel(const char *name, const void *args_ptr, size_t args_size,
    const myds_metal_buffer_t *const *bufs, size_t n_buffers,
    const uint32_t groups[3], const uint32_t threads[3], size_t threadgroup_memory) {
    (void)args_ptr; (void)args_size; (void)bufs; (void)n_buffers;
    (void)groups; (void)threads; (void)threadgroup_memory;
    return record(name);
}

int myds_metal_gemv(myds_metal_buffer_t *o, const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x, uint32_t r, uint32_t d, ggml_type t) {
    (void)o; (void)w; (void)x; (void)r; (void)d; (void)t;
    return record("gemv");
}

int myds_metal_v41_bf16(myds_metal_buffer_t *b, uint32_t n) {
    (void)b; (void)n;
    return record("bf16");
}

int myds_metal_v41_gemv(myds_metal_buffer_t *o, const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x, uint32_t r, uint32_t d, ggml_type t) {
    (void)o; (void)w; (void)x; (void)r; (void)d; (void)t;
    return record("v41_gemv_f16");
}

int myds_metal_v41_gemv_f16_bf16(myds_metal_buffer_t *o, const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x, uint32_t r, uint32_t d) {
    (void)o; (void)w; (void)x; (void)d;
    if (r == 512u && d == 5120u) return record("pool_kv_f16_bf16");
    if (r == 32u && d == 5120u) return record("index_weights_f16_bf16");
    if (r == 32u * 128u && d == 1280u) return record("index_q_f16_bf16");
    if (r == 128u && d == 512u) return record("index_k_f16_bf16");
    return record("gemv_f16_bf16");
}

int myds_metal_v41_gemv_bf16(myds_metal_buffer_t *o, const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x, uint32_t r, uint32_t d, ggml_type t) {
    (void)o; (void)w; (void)x; (void)d; (void)t;
    if (r == 1280u) return record("q_a");
    if (r == 512u) return record("kv_a");
    return record("gemv_bf16");
}

int myds_metal_dense_q4_v41(myds_metal_buffer_t *o, const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x, uint32_t r, uint32_t d, bool ordinary) {
    (void)o; (void)w; (void)x;
    assert(r == 64u * 512u && d == 1280u && ordinary);
    return record("q_b_dense");
}

int myds_metal_dense_q4_v41_qb_slice(myds_metal_buffer_t *o, const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *x, uint32_t row_offset, uint32_t rows, bool count_call) {
    (void)o; (void)w; (void)x;
    assert(row_offset + rows <= 64u * 512u && (row_offset % 512u) == 0u && (rows % 512u) == 0u);
    return record(count_call ? "q_b_slice_0" : "q_b_slice_1");
}

int myds_metal_v41_hc_sum_rmsnorm(myds_metal_buffer_t *o, myds_metal_buffer_t *x,
    const myds_metal_buffer_t *r, const myds_metal_buffer_t *p, const myds_metal_buffer_t *w,
    uint32_t d, float eps, bool round_bf16) {
    assert(o && x && r && p && w && d == 5120u && eps >= 0.0f && round_bf16);
    return record("hc_sum_rmsnorm");
}

int myds_metal_v41_hc_mix(myds_metal_buffer_t *s, const myds_metal_buffer_t *r,
    const myds_metal_buffer_t *f, ggml_type t, const myds_metal_buffer_t *sc,
    const myds_metal_buffer_t *b, myds_metal_buffer_t *n, myds_metal_buffer_t *m,
    uint32_t d, float e, float h) {
    (void)s; (void)r; (void)f; (void)t; (void)sc; (void)b; (void)n; (void)m; (void)d; (void)e; (void)h;
    return record("complete_hc");
}

int myds_metal_v41_hc_mix_stages_preflight(myds_metal_buffer_t *s, const myds_metal_buffer_t *r,
    const myds_metal_buffer_t *f, const myds_metal_buffer_t *sc, const myds_metal_buffer_t *b,
    myds_metal_buffer_t *n, myds_metal_buffer_t *m, uint32_t d, float e, float h) {
    assert(s && r && f && sc && b && n && m && d == 5120u && e >= 0.0f && h >= 0.0f);
    assert(!auto_barrier && concurrent && batch);
    return record("preflight");
}

int myds_metal_v41_hc_mix_norm_stage(myds_metal_buffer_t *n, const myds_metal_buffer_t *r, uint32_t d, float e) {
    assert(n && r && d == 5120u && e >= 0.0f && !auto_barrier && concurrent);
    return record("hc_mix_norm");
}

int myds_metal_v41_hc_mix_project_stage(myds_metal_buffer_t *m, const myds_metal_buffer_t *f,
    const myds_metal_buffer_t *n, uint32_t d) {
    assert(m && f && n && d == 5120u && !auto_barrier && concurrent);
    return record("hc_mix_project");
}

int myds_fuse_hc_norm_mix_mode(void) { return 0; }
void myds_metal_set_fuse_hc_norm_mix_mode(int mode) { (void)mode; }
int myds_metal_v41_hc_mix_norm_project_fused(myds_metal_buffer_t *m, const myds_metal_buffer_t *f,
    const myds_metal_buffer_t *r, uint32_t d, float e) {
    (void)m; (void)f; (void)r; (void)d; (void)e;
    return record("hc_mix_norm_project_fused");
}

int myds_metal_v41_hc_mix_split_stage(myds_metal_buffer_t *s, const myds_metal_buffer_t *m,
    const myds_metal_buffer_t *sc, const myds_metal_buffer_t *b, uint32_t d, float h) {
    assert(s && m && sc && b && d == 5120u && h >= 0.0f && !auto_barrier && concurrent);
    return record("hc_mix_split");
}

int myds_metal_v41_rmsnorm(myds_metal_buffer_t *o, const myds_metal_buffer_t *x,
    const myds_metal_buffer_t *w, uint32_t d, float eps, bool round_bf16) {
    (void)o; (void)x; (void)w; (void)eps; (void)round_bf16;
    if (d == 1280u) return record("qr_norm");
    if (d == 128u) return record("index_k_norm");
    return record("rmsnorm_512");
}

int myds_v41_gpu_qkv_finish(myds_metal_buffer_t *q, myds_metal_buffer_t *kv,
    myds_metal_buffer_t *win, uint32_t pos, bool compressed) {
    (void)q; (void)kv; (void)win; (void)pos; (void)compressed;
    return record("qkv_finish");
}

int myds_v41_gpu_pool2(myds_metal_buffer_t *out, const myds_metal_buffer_t *kv,
    const myds_metal_buffer_t *scores, myds_metal_buffer_t *prev_kv,
    myds_metal_buffer_t *prev_scores, uint32_t width, uint32_t pos) {
    (void)out; (void)kv; (void)scores; (void)prev_kv; (void)prev_scores; (void)width; (void)pos;
    return record("pool2");
}

int myds_metal_buffer_copy(myds_metal_buffer_t *dst, size_t doff,
    const myds_metal_buffer_t *src, size_t soff, size_t bytes) {
    (void)dst; (void)doff; (void)src; (void)soff; (void)bytes;
    return record("buffer_copy");
}

int myds_v41_gpu_indexer_rope_quant(myds_metal_buffer_t *x, uint32_t heads, uint32_t pos) {
    (void)x; (void)pos;
    return record(heads == 32u ? "idx_rope_quant_q" : "idx_rope_quant_k");
}

int myds_v41_gpu_rope(myds_metal_buffer_t *x, uint32_t dim, uint32_t heads,
    uint32_t pos, bool compressed, bool inverse) {
    (void)x; (void)dim; (void)heads; (void)pos; (void)compressed; (void)inverse;
    return record("rope");
}

int myds_v41_gpu_quant(myds_metal_buffer_t *x, uint32_t dim, uint32_t heads, uint32_t mode) {
    (void)x; (void)dim; (void)heads; (void)mode;
    return record("quant");
}

int myds_v41_gpu_pack_bf16(myds_metal_buffer_t *dst, size_t dst_elem_offset,
    const myds_metal_buffer_t *src, size_t src_elem_offset, uint32_t count) {
    (void)dst; (void)dst_elem_offset; (void)src; (void)src_elem_offset;
    return record(count == 128u ? "pack_index_cache" : "pack_compressed_kv");
}

int myds_v41_gpu_indexscores_head_slots(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q, const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache, const myds_metal_buffer_t *mask,
    uint32_t n_comp, uint32_t n_heads, bool cache_bf16) {
    (void)scores; (void)q; (void)weights; (void)cache; (void)mask; (void)n_comp; (void)n_heads; (void)cache_bf16;
    return record("indexscores_head_slots");
}

int myds_v41_gpu_indexscores_masked_bf16(myds_metal_buffer_t *scores,
    const myds_metal_buffer_t *q, const myds_metal_buffer_t *weights,
    const myds_metal_buffer_t *cache, const myds_metal_buffer_t *mask,
    uint32_t n_comp, uint32_t n_heads) {
    (void)scores; (void)q; (void)weights; (void)cache; (void)mask; (void)n_comp; (void)n_heads;
    return record("indexscores_masked_bf16");
}

int myds_v41_gpu_candidates(myds_metal_buffer_t *blocks, const myds_metal_buffer_t *scores,
    uint32_t n_comp, uint32_t pos, uint32_t ratio) {
    (void)blocks; (void)scores; (void)n_comp; (void)pos; (void)ratio;
    return record("candidates");
}

int myds_v41_gpu_filter(myds_metal_buffer_t *scores, const myds_metal_buffer_t *mask,
    uint32_t n_comp, uint32_t pos, uint32_t ratio) {
    (void)scores; (void)mask; (void)n_comp; (void)pos; (void)ratio;
    return record("filter");
}

int myds_v41_gpu_topk(myds_metal_buffer_t *ids, const myds_metal_buffer_t *scores,
    uint32_t n_cols, uint32_t top, myds_v41_gpu_scratch_t *s) {
    (void)ids; (void)scores; (void)n_cols; (void)top;
    assert(s != NULL);
    return record("topk");
}

int myds_v41_gpu_topk_threshold(myds_metal_buffer_t *ids, const myds_metal_buffer_t *scores,
    uint32_t cols, uint32_t top, myds_v41_gpu_scratch_t *s) {
    (void)ids; (void)scores; (void)cols; (void)top; (void)s;
    return record("topk_threshold");
}

int myds_v41_gpu_topkmask(myds_metal_buffer_t *mask, const myds_metal_buffer_t *ids,
    uint32_t n_cols, uint32_t top) {
    (void)mask; (void)ids; (void)n_cols; (void)top;
    return record("topkmask");
}

static myds_context_t context;
static myds_model_t model;
static myds_layer_t layers[40];
static myds_v41_decode_attn_args_t args;

static void reset(void) {
    events = 0;
    failure = NULL;
    auto_barrier = 1;
    concurrent = 0;
    batch = 1;
    trace = 0;
    dense_q4 = 1;
    f16_bf16 = 1;
    idx_rope_quant = 1;
    head_slots = 1;
    topk_thresh = 0;
    myds_v41_attn_hc_overlap_counters(NULL, true);
    myds_v41_indexer_decode_overlap_counters(NULL, true);
    myds_v41_attn_hc_overlap_parse_mode();
    myds_v41_indexer_decode_overlap_parse_mode();
}

static void assert_attn_counters(uint64_t r, uint64_t e, uint64_t f) {
    myds_metal_decode_candidate_counters_t c;
    myds_v41_attn_hc_overlap_counters(&c, false);
    assert(c.requests == r && c.encoded == e && c.failures == f);
}

static void assert_idx_counters(uint64_t r, uint64_t e, uint64_t f) {
    myds_metal_decode_candidate_counters_t c;
    myds_v41_indexer_decode_overlap_counters(&c, false);
    assert(c.requests == r && c.encoded == e && c.failures == f);
}

static int run_layer(uint32_t il, uint32_t pos, bool ordinary) {
    args.layer = il;
    args.pos = pos;
    return myds_v41_decode_attn_indexer_scoped(&context, &args, ordinary);
}

int main(void) {
    context.model = &model;
    context.cfg.max_seq_len = 4096u;
    model.layers = layers;
    model.n_layers = 40u;
    model.dim = 5120u;
    model.n_heads = 64u;
    model.head_dim = 512u;
    model.norm_eps = 1e-5f;

#define B(v) v = buffer(16u << 20)
    B(context.x);
    B(context.x_norm);
    B(context.qr);
    B(context.qr_norm);
    B(context.q);
    B(context.kv_act);
    myds_metal_buffer_t *attn_norm = buffer(16u << 20);
    myds_metal_buffer_t *attn_q_a = buffer(16u << 20);
    myds_metal_buffer_t *attn_q_a_norm = buffer(16u << 20);
    myds_metal_buffer_t *attn_q_b = buffer(16u << 20);
    myds_metal_buffer_t *attn_kv = buffer(16u << 20);
    myds_metal_buffer_t *attn_kv_a_norm = buffer(16u << 20);
    for (unsigned i = 0; i < 40u; ++i) {
        layers[i].attn_norm = attn_norm;
        layers[i].attn_q_a = attn_q_a;
        layers[i].attn_q_a_norm = attn_q_a_norm;
        layers[i].attn_q_b = attn_q_b;
        layers[i].attn_kv = attn_kv;
        layers[i].attn_kv_a_norm = attn_kv_a_norm;
        layers[i].attn_q_a_type = GGML_TYPE_Q8_0;
        layers[i].attn_q_b_type = GGML_TYPE_Q4_K;
        layers[i].attn_kv_type = GGML_TYPE_F16;
    }
    args.capacity = 4096u;
    args.hc_eps = 1e-4f;
    B(args.residual);
    B(args.hc_in_pre);
    B(args.attn_split);
    B(args.attn_fn);
    B(args.attn_scale);
    B(args.attn_base);
    B(args.flat_norm);
    B(args.mix);
    B(args.window);
    B(args.compressor_kv);
    B(args.compressor_gate);
    B(args.compressor_norm);
    B(args.index_k_weight);
    B(args.index_k_norm);
    B(args.index_q_weight);
    B(args.index_proj_weight);
    B(args.pool_kv);
    B(args.pool_score);
    B(args.latent);
    B(args.previous_kv);
    B(args.previous_score);
    B(args.compressed_kv);
    B(args.index_cache);
    B(args.index_k_buf);
    B(args.index_q_buf);
    B(args.index_weights);
    B(args.index_scores);
    B(args.selected);
    B(args.block_scores);
    B(args.block_selected);
    B(args.block_mask);
    args.gpu_scratch = &scratch_inst;
    buffers[125] = (struct myds_metal_buffer){
        16u << 20, 0, (void *)(uintptr_t)(UINT64_C(0x100000000) + 125u * UINT64_C(0x2000000))
    };
    buffers[126] = (struct myds_metal_buffer){
        16u << 20, 0, (void *)(uintptr_t)(UINT64_C(0x100000000) + 126u * UINT64_C(0x2000000))
    };
    buffers[127] = (struct myds_metal_buffer){
        16u << 20, 0, (void *)(uintptr_t)(UINT64_C(0x100000000) + 127u * UINT64_C(0x2000000))
    };
#undef B

    /* 1. Default modes (both 0): reference path executes cleanly. */
    unsetenv("MYDS_ATTN_HC_OVERLAP");
    unsetenv("MYDS_INDEXER_DECODE_OVERLAP");
    reset();
    assert(myds_v41_attn_hc_overlap_enabled() == 0);
    assert(myds_v41_indexer_decode_overlap_enabled() == 0);
    assert(run_layer(0u, 3u, true));
    assert_attn_counters(0, 0, 0);
    assert_idx_counters(0, 0, 0);
    assert(auto_barrier == 1 && concurrent == 0);

    /* 2. MYDS_ATTN_HC_OVERLAP=1, MYDS_INDEXER_DECODE_OVERLAP=0 on layer 0. */
    setenv("MYDS_ATTN_HC_OVERLAP", "1", 1);
    setenv("MYDS_INDEXER_DECODE_OVERLAP", "0", 1);
    reset();
    assert(run_layer(0u, 3u, true));
    assert_attn_counters(1, 1, 0);
    assert_idx_counters(0, 0, 0);
    const char *expected_attn_only[] = {
        "preflight", "hc_sum_rmsnorm", "hc_mix_norm", "join_2",
        "hc_mix_project", "q_a", "kv_a", "join_3",
        "hc_mix_split", "qr_norm", "rmsnorm_512", "join_2",
        "q_b_dense", "join_1", "qkv_finish", "join_4"
    };
    assert(events == sizeof(expected_attn_only) / sizeof(*expected_attn_only));
    for (unsigned i = 0; i < events; ++i) {
        assert(strcmp(event_names[i], expected_attn_only[i]) == 0);
    }
    assert(auto_barrier == 1 && concurrent == 0);

    /* 3. MYDS_ATTN_HC_OVERLAP=0, MYDS_INDEXER_DECODE_OVERLAP=1 on layer 2 (ratio=2, pos=3 => commit). */
    setenv("MYDS_ATTN_HC_OVERLAP", "0", 1);
    setenv("MYDS_INDEXER_DECODE_OVERLAP", "1", 1);
    reset();
    assert(run_layer(2u, 3u, true));
    assert_attn_counters(0, 0, 0);
    assert_idx_counters(1, 1, 0);
    const char *expected_idx_l2[] = {
        "hc_sum_rmsnorm", "join_1",
        "complete_hc", "q_a", "kv_a", "v41_gemv_f16", "v41_gemv_f16", "index_weights_f16_bf16", "join_6",
        "qr_norm", "rmsnorm_512", "pool2", "join_3",
        "q_b_dense", "index_q_f16_bf16", "rmsnorm_512", "join_3",
        "qkv_finish", "idx_rope_quant_q", "index_k_f16_bf16", "join_2",
        "index_k_norm", "rope", "join_2",
        "idx_rope_quant_k", "quant", "join_2",
        "pack_index_cache", "pack_compressed_kv",
        "join_7", "indexscores_head_slots", "join_1", "topk"
    };
    assert(events == sizeof(expected_idx_l2) / sizeof(*expected_idx_l2));
    for (unsigned i = 0; i < events; ++i) {
        assert(strcmp(event_names[i], expected_idx_l2[i]) == 0);
    }

    /* 4. Both MYDS_ATTN_HC_OVERLAP=1 and MYDS_INDEXER_DECODE_OVERLAP=1 across all 40 layers. */
    setenv("MYDS_ATTN_HC_OVERLAP", "1", 1);
    setenv("MYDS_INDEXER_DECODE_OVERLAP", "1", 1);
    reset();
    for (uint32_t il = 0; il < 40u; ++il) {
        events = 0;
        assert(run_layer(il, 3u, true));
    }
    assert_attn_counters(40, 40, 0);
    assert_idx_counters(8, 8, 0);

    /* 5. Stage failure injection in combined mode fails closed and restores state. */
    const char *stage_failures[] = {
        "preflight", "hc_sum_rmsnorm", "hc_mix_norm", "hc_mix_project",
        "q_a", "kv_a", "pool_kv_f16_bf16", "index_weights_f16_bf16",
        "hc_mix_split", "qr_norm", "buffer_copy", "q_b_dense",
        "index_q_f16_bf16", "qkv_finish", "idx_rope_quant_q",
        "index_k_f16_bf16", "index_k_norm", "rope", "idx_rope_quant_k",
        "quant", "pack_index_cache", "pack_compressed_kv",
        "indexscores_head_slots", "candidates", "topkmask", "topk"
    };
    for (unsigned i = 0; i < sizeof(stage_failures) / sizeof(*stage_failures); ++i) {
        reset();
        failure = stage_failures[i];
        assert(!run_layer(20u, 3u, true));
        assert_attn_counters(1, i > 8 ? 1 : 0, 1);
        assert_idx_counters(1, 0, 1);
        assert(auto_barrier == 1 && concurrent == 0);
    }

    /* 6. Invalid environment strings fail closed with 0 events. */
    const char *bad[] = {"", "01", "true", "2", " 1", "1 ", "-1"};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
        setenv("MYDS_ATTN_HC_OVERLAP", bad[i], 1);
        setenv("MYDS_INDEXER_DECODE_OVERLAP", "0", 1);
        reset();
        assert(myds_v41_attn_hc_overlap_enabled() == -1 && !run_layer(0u, 3u, true) && events == 0);
        assert_attn_counters(0, 0, 0);

        setenv("MYDS_ATTN_HC_OVERLAP", "0", 1);
        setenv("MYDS_INDEXER_DECODE_OVERLAP", bad[i], 1);
        reset();
        assert(myds_v41_indexer_decode_overlap_enabled() == -1 && !run_layer(2u, 3u, true) && events == 0);
        assert_idx_counters(0, 0, 0);
    }

    /* 7. Precondition exclusions and buffer aliasing guards fail closed. */
    setenv("MYDS_ATTN_HC_OVERLAP", "1", 1);
    setenv("MYDS_INDEXER_DECODE_OVERLAP", "1", 1);
    reset();
    batch = 0;
    assert(!run_layer(2u, 3u, true) && events == 0);
    assert_attn_counters(1, 0, 1);
    assert_idx_counters(1, 0, 1);

    reset();
    f16_bf16 = 0;
    assert(!run_layer(2u, 3u, true) && events == 0);
    assert_attn_counters(1, 0, 1);
    assert_idx_counters(1, 0, 1);

    reset();
    void *saved_ptr = args.mix->data;
    args.mix->data = context.qr->data;
    assert(!run_layer(2u, 3u, true) && events == 0);
    assert_attn_counters(1, 0, 1);
    assert_idx_counters(1, 0, 1);
    args.mix->data = saved_ptr;

    /* 8. MYDS_ATTN_KV_REUSE=1 prestages window on all layers and elides redundant compressor/block-mask steps. */
    kv_reuse = 1;
    reset();
    assert(run_layer(0u, 3u, true));
    assert(strcmp(event_names[events - 7], "hc_mix_split") == 0);
    assert(strcmp(event_names[events - 6], "qkv_finish") == 0);
    assert(strcmp(event_names[events - 5], "q_b_dense") == 0);
    assert(strcmp(event_names[events - 4], "join_5") == 0);
    assert(strcmp(event_names[events - 3], "rope") == 0);
    assert(strcmp(event_names[events - 2], "prestage_window") == 0);
    assert(strcmp(event_names[events - 1], "join_3") == 0);
    reset();
    assert(run_layer(2u, 3u, true));
    assert(strcmp(event_names[events - 9], "join_9") == 0);
    assert(strcmp(event_names[events - 8], "indexscores_head_slots") == 0);
    assert(strcmp(event_names[events - 7], "q_b_slice_1") == 0);
    assert(strcmp(event_names[events - 6], "join_2") == 0);
    assert(strcmp(event_names[events - 5], "hc_mix_split") == 0);
    assert(strcmp(event_names[events - 4], "defer_q_rope") == 0);
    assert(strcmp(event_names[events - 3], "bind_topk_qb") == 0);
    assert(strcmp(event_names[events - 2], "topk") == 0);
    assert(strcmp(event_names[events - 1], "join_3") == 0);
    reset();
    assert(run_layer(2u, 1200u, true));
    assert(strcmp(event_names[events - 9], "join_9") == 0);
    assert(strcmp(event_names[events - 8], "indexscores_head_slots") == 0);
    assert(strcmp(event_names[events - 7], "q_b_slice_1") == 0);
    assert(strcmp(event_names[events - 6], "join_1") == 0);
    assert(strcmp(event_names[events - 5], "hc_mix_split") == 0);
    assert(strcmp(event_names[events - 4], "defer_q_rope") == 0);
    assert(strcmp(event_names[events - 3], "topk") == 0);
    assert(strcmp(event_names[events - 2], "q_b_slice_1") == 0);
    assert(strcmp(event_names[events - 1], "join_3") == 0);
    reset();
    assert(run_layer(2u, 2400u, true));
    assert(strcmp(event_names[events - 9], "join_9") == 0);
    assert(strcmp(event_names[events - 8], "indexscores_head_slots") == 0);
    assert(strcmp(event_names[events - 7], "q_b_slice_1") == 0);
    assert(strcmp(event_names[events - 6], "join_2") == 0);
    assert(strcmp(event_names[events - 5], "hc_mix_split") == 0);
    assert(strcmp(event_names[events - 4], "defer_q_rope") == 0);
    assert(strcmp(event_names[events - 3], "bind_topk_qb") == 0);
    assert(strcmp(event_names[events - 2], "topk") == 0);
    assert(strcmp(event_names[events - 1], "join_3") == 0);
    reset();
    assert(run_layer(2u, 2u, true));
    for (unsigned i = 0; i < events; ++i) {
        assert(strcmp(event_names[i], "pool2") != 0);
    }
    reset();
    assert(run_layer(20u, 3u, true));
    for (unsigned i = 0; i < events; ++i) {
        assert(strcmp(event_names[i], "buffer_copy") != 0);
        assert(strcmp(event_names[i], "candidates") != 0);
        assert(strcmp(event_names[i], "topkmask") != 0);
    }
    kv_reuse = 0;

    /* 9. MYDS_FFN_HC_OVERLAP=1 overlaps FFN HC with router_gate/shared_gate_up_swiglu/route. */
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
        setenv("MYDS_FFN_HC_OVERLAP", bad[i], 1);
        myds_v41_ffn_hc_overlap_parse_mode();
        assert(myds_v41_ffn_hc_overlap_enabled() == -1);
    }
    setenv("MYDS_FFN_HC_OVERLAP", "1", 1);
    myds_v41_ffn_hc_overlap_parse_mode();
    assert(myds_v41_ffn_hc_overlap_enabled() == 1);
    myds_v41_ffn_hc_overlap_counters(NULL, true);
    reset();
    layers[0].ffn_norm = attn_norm;
    layers[0].router_gate = buffer(16u << 20);
    layers[0].router_bias = buffer(16u << 20);
    layers[0].shared_gate = buffer(16u << 20);
    layers[0].shared_up = buffer(16u << 20);
    layers[0].shared_down = buffer(16u << 20);
    layers[0].ffn_gate_exps = buffer(16u << 20);
    layers[0].ffn_up_exps = buffer(16u << 20);
    layers[0].ffn_down_exps = buffer(16u << 20);
    layers[0].router_gate_type = GGML_TYPE_F32;
    layers[0].shared_gate_type = GGML_TYPE_Q4_K;
    layers[0].shared_up_type = GGML_TYPE_Q4_K;
    layers[0].shared_down_type = GGML_TYPE_Q4_K;
    layers[0].ffn_gate_exps_type = GGML_TYPE_IQ2_XXS;
    layers[0].ffn_up_exps_type = GGML_TYPE_IQ2_XXS;
    layers[0].ffn_down_exps_type = GGML_TYPE_Q2_K;
    model.moe_expert_dim = 2304u;
    model.n_shared_experts = 1u;
    model.n_routed_experts = 384u;
    model.n_activated_experts = 6u;
    model.routed_scaling_factor = 1.0f;
    model.swiglu_clamp_exp = 7.0f;
    context.ffn_norm_out = buffer(16u << 20);
    context.router_logits = buffer(16u << 20);
    context.selected_ids = buffer(16u << 20);
    context.selected_weights = buffer(16u << 20);
    context.shared_swiglu_out = buffer(16u << 20);
    context.shared_out = buffer(16u << 20);
    context.moe_out = buffer(16u << 20);
    context.moe_gate = buffer(16u << 20);
    context.moe_up = buffer(16u << 20);
    context.moe_mid = buffer(16u << 20);
    context.moe_down = buffer(16u << 20);
    context.ffn_out = buffer(16u << 20);
    myds_metal_buffer_t *after_attn = buffer(16u << 20);
    myds_metal_buffer_t *ffn_split = buffer(16u << 20);
    events = 0;
    assert(myds_v41_moe_ffn_hc_overlap_scoped(&context, 0u, 1u, 3u,
        args.residual, after_attn, args.attn_split, ffn_split,
        args.attn_fn, args.attn_scale, args.attn_base, args.flat_norm, args.mix,
        1e-4f, true) == 1);
    myds_metal_decode_candidate_counters_t ffn_c;
    myds_v41_ffn_hc_overlap_counters(&ffn_c, false);
    assert(ffn_c.requests == 1 && ffn_c.encoded == 1 && ffn_c.failures == 0);
    const char *expected_ffn[] = {
        "preflight", "hc_sum_rmsnorm", "hc_mix_norm", "join_2",
        "hc_mix_project", "v41_gemv_f16", "join_2",
        "hc_mix_split", "kernel_myds_v41_route_fused", "kernel_myds_v41_shared_gate_up_swiglu_q4", "join_5",
        "routed_moe_down_overlap_scoped", "join_3",
        "moe_add_hc_expand", "join_1"
    };
    assert(events == sizeof(expected_ffn) / sizeof(*expected_ffn));
    for (unsigned i = 0; i < events; ++i) {
        assert(strcmp(event_names[i], expected_ffn[i]) == 0);
    }

    puts("PASS: attention HC overlap + indexer decode overlap + KV reuse + FFN HC overlap host schedule, cohorts, strict env, exclusions, aliasing guards, counters, and failure recovery");
    return 0;
}
