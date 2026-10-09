/* Model-free acceptance for the real single-token LM-head selector.
 * Includes the runtime only to select both initialized modes, retain completed
 * CB timestamps and inject an actually missing candidate symbol. Production
 * has no setter: mode remains fixed until cleanup/reinitialization.
 * Building and --host-only submit no GPU work. All other modes run GPU work
 * and belong in an exclusive validation window. Quantized weights are seeded
 * valid, nonzero fixtures, not a captured GGUF or a full-model benchmark. */
#include "myds_metal.m"
#include <errno.h>
#include <inttypes.h>
#include <sys/stat.h>

static const char *output_dir, *shader_dir = "bin";
static unsigned samples = 5, repetitions = 32;
static unsigned exported_cases;
static int no_timing, host_only, expect_missing;
static void failed(const char *file, int line, const char *condition) {
    fprintf(stderr, "FAIL %s:%d %s\n", file, line, condition);
    fflush(stdout); fflush(stderr); exit(1);
}
#define REQUIRE(c) do { if (!(c)) failed(__FILE__, __LINE__, #c); } while (0)

struct guarded { myds_metal_buffer_t *owner, *view; size_t bytes, offset; };
struct fixture { uint32_t rows, width, tokens; struct guarded weight, input, ref, out; };

static uint64_t fnv(const void *data, size_t bytes) {
    const uint8_t *p = data; uint64_t value = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i) { value ^= p[i]; value *= UINT64_C(1099511628211); }
    return value;
}

static struct guarded allocate(size_t bytes, size_t offset) {
    struct guarded b = {.bytes = bytes, .offset = offset};
    b.owner = myds_metal_buffer_alloc(bytes + offset + 16u); REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner), 0xa5, bytes + offset + 16u);
    b.view = myds_metal_buffer_sub_view(b.owner, offset, bytes); REQUIRE(b.view);
    return b;
}

static void canary(struct guarded b) {
    const uint8_t *p = myds_metal_buffer_data(b.owner);
    for (size_t i = 0; i < b.offset; ++i) REQUIRE(p[i] == 0xa5);
    for (size_t i = 0; i < 16u; ++i) REQUIRE(p[b.offset + b.bytes + i] == 0xa5);
}

static void release(struct guarded b) {
    canary(b); myds_metal_buffer_free(b.view); myds_metal_buffer_free(b.owner);
}

static void input_fill(struct fixture *f, unsigned seed) {
    float *x = myds_metal_buffer_data(f->input.view);
    for (size_t i = 0; i < f->input.bytes / 4u; ++i) {
        uint32_t z = (uint32_t)i * 747796405u + seed * 2891336453u;
        z = ((z >> ((z >> 28) + 4u)) ^ z) * 277803737u; z = (z >> 22) ^ z;
        x[i] = ((int)(z % 1023u) - 511) / 257.0f;
    }
}

static struct fixture create(uint32_t rows, uint32_t width, uint32_t tokens, unsigned seed) {
    REQUIRE(rows && width && width % 32u == 0 && tokens);
    struct fixture f = {.rows = rows, .width = width, .tokens = tokens};
    f.weight = allocate((size_t)rows * (width / 32u) * 34u, 18u);
    f.input = allocate((size_t)width * tokens * 4u, 20u);
    /* Plain F32 stores need only four-byte alignment; these views deliberately
     * are not all 16-byte aligned, unlike the separate BF16 shared experiment. */
    f.ref = allocate((size_t)rows * tokens * 4u, 36u);
    f.out = allocate(f.ref.bytes, 44u);
    uint8_t *w = myds_metal_buffer_data(f.weight.view);
    for (size_t block = 0; block < f.weight.bytes / 34u; ++block) {
        uint16_t scale = (uint16_t)(0x1800u + (block % 8u) * 0x180u);
        memcpy(w + block * 34u, &scale, 2u);
        for (unsigned j = 0; j < 32u; ++j)
            w[block * 34u + 2u + j] = (uint8_t)(int8_t)((int)((block * 37u + j * 29u + seed) % 255u) - 127);
    }
    input_fill(&f, seed); return f;
}

static void destroy(struct fixture *f) {
    release(f->weight); release(f->input); release(f->ref); release(f->out);
}

static void finite(struct guarded b) {
    const float *p = myds_metal_buffer_data(b.view); size_t nonzero = 0;
    for (size_t i = 0; i < b.bytes / 4u; ++i) {
        uint32_t bits; memcpy(&bits, p + i, 4u);
        REQUIRE(isfinite(p[i]) && bits != 0xa5a5a5a5u); nonzero += p[i] != 0;
    }
    REQUIRE(nonzero); canary(b);
}

static void exact(struct guarded a, struct guarded b) {
    REQUIRE(a.bytes == b.bytes);
    const float *pa = myds_metal_buffer_data(a.view), *pb = myds_metal_buffer_data(b.view);
    if (memcmp(pa, pb, a.bytes)) for (size_t i = 0; i < a.bytes / 4u; ++i) {
        if (!memcmp(pa + i, pb + i, 4u)) continue;
        uint32_t av, bv; memcpy(&av, pa + i, 4u); memcpy(&bv, pb + i, 4u);
        fprintf(stderr, "PARITY index=%zu NR2=%08x NR4=%08x\n", i, av, bv); break;
    }
    REQUIRE(!memcmp(pa, pb, a.bytes)); finite(a); finite(b);
}

static void export_blob(const char *name, struct guarded b) {
    if (!output_dir) return;
    char path[4096]; int n = snprintf(path, sizeof(path), "%s/%s.f32", output_dir, name);
    REQUIRE(n > 0 && (size_t)n < sizeof(path)); FILE *file = fopen(path, "wx"); REQUIRE(file);
    REQUIRE(fwrite(myds_metal_buffer_data(b.view), 1, b.bytes, file) == b.bytes);
    REQUIRE(!fclose(file));
}

static void mode(unsigned nr0) {
    REQUIRE(nr0 == 2 || nr0 == 4);
    /* Fixture-only selection of already validated PSOs. Every dispatch still
     * goes through the actual public runtime selector; no production setter. */
    g_lm_head_q8_nr0 = (int)nr0; REQUIRE(myds_metal_lm_head_q8_nr0() == (int)nr0);
}

static int projected(struct fixture *f, struct guarded output) {
    const bool candidate = myds_metal_lm_head_q8_nr0() == 4 &&
                           f->tokens == 1u && f->rows > 65536u;
    if (!candidate && g_batch_cb && !g_auto_barrier) {
        /* The old generic entry depends on its caller's graph fences. Give
         * the reference the same resources as NR4's defensive entry so a
         * component race or different synchronization cannot masquerade as
         * arithmetic parity or a geometry benefit. */
        const myds_metal_buffer_t *deps[] = { f->weight.view, f->input.view, output.view };
        myds_metal_barrier_buffers(deps, 3u);
    }
    return myds_metal_gemv_batch(output.view, f->weight.view, f->input.view,
                                f->rows, f->width, f->tokens, GGML_TYPE_Q8_0);
}

static uint64_t histogram(const char *name) {
    for (int i = 0; i < g_hist_used; ++i) if (!strcmp(g_hist[i].name, name)) return g_hist[i].n;
    return 0;
}

static void reference(struct fixture *f) {
    const int saved = g_lm_head_q8_nr0; mode(2);
    REQUIRE(myds_metal_begin_batch()); REQUIRE(projected(f, f->ref));
    REQUIRE(myds_metal_commit_batch()); g_lm_head_q8_nr0 = saved;
}

static void numeric(unsigned case_id, uint32_t rows, uint32_t width, uint32_t tokens) {
    struct fixture f = create(rows, width, tokens, 31u + case_id); reference(&f); mode(4);
    const bool eligible = tokens == 1 && rows > 65536u;
    const uint64_t prior_hist = histogram("lm_head_q8_nr4"), prior_dispatch = myds_metal_dispatch_count();
    myds_metal_lm_head_q8_counters_t a, b; myds_metal_lm_head_q8_counters(&a, false);
    REQUIRE(myds_metal_begin_batch()); id<MTLCommandBuffer> cb = g_batch_cb;
    REQUIRE(projected(&f, f.out)); REQUIRE(myds_metal_commit_batch());
    REQUIRE(cb.status == MTLCommandBufferStatusCompleted);
    char name[64]; snprintf(name, sizeof(name), "case%u-reference", case_id); export_blob(name, f.ref);
    snprintf(name, sizeof(name), "case%u-candidate", case_id); export_blob(name, f.out);
    exact(f.ref, f.out); canary(f.weight); canary(f.input);
    myds_metal_lm_head_q8_counters(&b, false);
    REQUIRE(b.requests - a.requests == eligible && b.encoded - a.encoded == eligible && b.failures == a.failures);
    REQUIRE(histogram("lm_head_q8_nr4") - prior_hist == eligible && myds_metal_dispatch_count() - prior_dispatch == 1u);
    printf("{\"kind\":\"parity\",\"case\":%u,\"rows\":%u,\"width\":%u,\"tokens\":%u,\"weight_bytes\":%zu,\"reference_nr0\":2,\"candidate_nr0\":%u,\"nsg\":%u,\"threads\":%u,\"dynamic_tg_bytes\":%u,\"candidate_encoded\":%u,\"concurrent\":%d,\"auto_barrier\":%d,\"output_floats\":%zu,\"output_fnv1a64\":\"%016" PRIx64 "\",\"full_f32_bit_exact\":true,\"finite\":true,\"canaries\":true,\"actual_kernel\":\"%s\",\"completed_cb\":true}\n",
           case_id, rows, width, tokens, f.weight.bytes, eligible ? 4u : 2u, rows > 65536u ? 8u : 4u,
           rows > 65536u ? 256u : 128u, eligible ? 512u : 256u, (unsigned)eligible,
           myds_metal_get_concurrent(), myds_metal_get_auto_barrier(), f.out.bytes / 4u,
           fnv(myds_metal_buffer_data(f.out.view), f.out.bytes),
           eligible ? "kernel_mul_mv_q8_0_f32_r4" : "kernel_mul_mv_q8_0_f32");
    ++exported_cases; destroy(&f);
}

static void reject_call(const char *reason, myds_metal_buffer_t *out,
                        const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
                        uint32_t rows, uint32_t width) {
    uint8_t *saved = NULL;
    if (out && out->ptr && out->size) { saved = malloc(out->size); REQUIRE(saved); memcpy(saved, out->ptr, out->size); }
    myds_metal_lm_head_q8_counters_t a, b; myds_metal_lm_head_q8_counters(&a, false);
    const uint64_t old_dispatch = myds_metal_dispatch_count();
    REQUIRE(!myds_metal_gemv(out, weight, input, rows, width, GGML_TYPE_Q8_0));
    myds_metal_lm_head_q8_counters(&b, false);
    REQUIRE(b.requests == a.requests + 1u && b.failures == a.failures + 1u && b.encoded == a.encoded);
    REQUIRE(myds_metal_dispatch_count() == old_dispatch);
    if (saved) { REQUIRE(!memcmp(saved, out->ptr, out->size)); free(saved); }
    printf("{\"kind\":\"rejected_call\",\"reason\":\"%s\",\"requests\":1,\"encoded\":0,\"failures\":1,\"dispatches\":0,\"output_untouched\":true}\n", reason);
}

/* Test the actual geometry rejection branch without claiming these are the
 * hardware's real limits. Every other property forwards to the real PSO. */
@interface LMRestrictedPipeline : NSProxy { id<MTLComputePipelineState> actual; }
+ (instancetype)withPipeline:(id<MTLComputePipelineState>)pipeline;
@end
@implementation LMRestrictedPipeline
+ (instancetype)withPipeline:(id<MTLComputePipelineState>)pipeline {
    LMRestrictedPipeline *proxy = [LMRestrictedPipeline alloc]; proxy->actual = pipeline; return proxy;
}
- (NSUInteger)maxTotalThreadsPerThreadgroup { return 128u; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)selector { return [(NSObject *)actual methodSignatureForSelector:selector]; }
- (void)forwardInvocation:(NSInvocation *)invocation { [invocation invokeWithTarget:actual]; }
@end

static void rejections(void) {
    struct fixture f = create(65537u, 32u, 1u, 79u); mode(4);
    myds_metal_buffer_t out = *f.out.view, input = *f.input.view, weight = *f.weight.view;
    out.size -= 4u; reject_call("short_output", &out, &weight, &input, f.rows, f.width);
    out = *f.out.view; input.size -= 4u; reject_call("short_input", &out, &weight, &input, f.rows, f.width);
    input = *f.input.view; weight.size -= 2u; reject_call("short_weight", &out, &weight, &input, f.rows, f.width);
    weight = *f.weight.view; ++out.offset; out.ptr = (char *)out.ptr + 1u;
    reject_call("unaligned_output", &out, &weight, &input, f.rows, f.width);
    out = *f.out.view; input.offset += 2u; input.ptr = (char *)input.ptr + 2u;
    reject_call("unaligned_input", &out, &weight, &input, f.rows, f.width);
    input = *f.input.view; ++weight.offset; weight.ptr = (char *)weight.ptr + 1u;
    reject_call("unaligned_weight", &out, &weight, &input, f.rows, f.width);
    weight = *f.weight.view;
    myds_metal_buffer_t alias_input = out; alias_input.size = f.input.bytes;
    reject_call("input_output_alias", &out, &weight, &alias_input, f.rows, f.width);
    myds_metal_buffer_t alias_output = weight; alias_output.size = f.out.bytes;
    reject_call("weight_output_alias", &alias_output, &weight, &input, f.rows, f.width);
    reject_call("width_not_block_aligned", &out, &weight, &input, f.rows, 31u);
    reject_call("zero_width", &out, &weight, &input, f.rows, 0u);
    reject_call("dimension_overflow", &out, &weight, &input, UINT32_MAX, 32u);
    reject_call("null_output", NULL, &weight, &input, f.rows, f.width);
    id<MTLComputePipelineState> saved_pso = g_pso_lm_head_q8_nr4;
    g_pso_lm_head_q8_nr4 = (id<MTLComputePipelineState>)[LMRestrictedPipeline withPipeline:saved_pso];
    reject_call("mock_restricted_geometry", &out, &weight, &input, f.rows, f.width);
    g_pso_lm_head_q8_nr4 = nil;
    id<MTLLibrary> saved_library = g_library; NSError *error = nil;
    g_library = [g_device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void unrelated(device float *p [[buffer(0)]],uint i [[thread_position_in_grid]]) {p[i]=0;}\n" options:nil error:&error];
    REQUIRE(g_library && ![g_library.functionNames containsObject:@"kernel_mul_mv_q8_0_f32_r4"]);
    reject_call("actual_missing_candidate_symbol", &out, &weight, &input, f.rows, f.width);
    g_library = saved_library; g_pso_lm_head_q8_nr4 = saved_pso;
    canary(f.out); canary(f.input); canary(f.weight); destroy(&f);
}

static void reuse(unsigned id) {
    const uint32_t widths[] = {32u, 256u, 96u}; struct fixture f[3];
    for (unsigned i = 0; i < 3u; ++i) { f[i] = create(65537u + i, widths[i], 1u, 13u + i); reference(f + i); }
    struct guarded shared_x = allocate(256u * 4u, 20u), shared_out = allocate(65539u * 4u, 44u);
    const unsigned counts[] = {1u, 3u, 3u}; mode(4);
    myds_metal_lm_head_q8_counters_t a, b; myds_metal_lm_head_q8_counters(&a, false);
    for (unsigned pass = 0; pass < 3u; ++pass) {
        struct guarded snapshots[3]; unsigned indices[3];
        for (unsigned j = 0; j < counts[pass]; ++j) {
            indices[j] = pass == 2u ? 2u - j : j;
            snapshots[j] = allocate(f[indices[j]].out.bytes, 36u);
        }
        REQUIRE(myds_metal_begin_batch());
        for (unsigned j = 0; j < counts[pass]; ++j) {
            struct fixture *v = f + indices[j];
            /* Actual GPU-produced activation and a single reused output view;
             * capture each complete result before the next overwrite. Copies
             * add observation dependencies, disclosed in the reuse record. */
            /* Compute copy does not add fences with automatic barriers off.
             * Complete the preceding GEMV's input read before overwriting x. */
            if (!g_auto_barrier) myds_metal_barrier_buffer(shared_x.view);
            REQUIRE(myds_metal_buffer_copy(shared_x.view, 0, v->input.view, 0, v->input.bytes));
            REQUIRE(myds_metal_gemv(shared_out.view, v->weight.view, shared_x.view,
                                   v->rows, v->width, GGML_TYPE_Q8_0));
            /* The helper's pre-GEMV fence orders earlier users, not this later
             * snapshot reader. Complete the output write before copying it. */
            if (!g_auto_barrier) myds_metal_barrier_buffer(shared_out.view);
            REQUIRE(myds_metal_buffer_copy(snapshots[j].view, 0, shared_out.view, 0, snapshots[j].bytes));
        }
        REQUIRE(myds_metal_commit_batch());
        for (unsigned j = 0; j < counts[pass]; ++j) {
            char name[96]; snprintf(name, sizeof(name), "reuse%u-pass%u-item%u", id, pass, j);
            export_blob(name, snapshots[j]); exact(f[indices[j]].ref, snapshots[j]); release(snapshots[j]);
        }
    }
    myds_metal_lm_head_q8_counters(&b, false);
    REQUIRE(b.requests - a.requests == 7u && b.encoded - a.encoded == 7u && b.failures == a.failures);
    printf("{\"kind\":\"reuse\",\"case\":%u,\"calls_per_batch\":[1,3,3],\"actual_nr4_encoded\":7,\"gpu_produced_activation\":true,\"shared_output\":true,\"concurrent\":%d,\"auto_barrier\":%d,\"snapshot_copies_add_dependencies\":true,\"bit_exact_full_outputs\":true,\"finite\":true,\"canaries\":true}\n",
           id, myds_metal_get_concurrent(), myds_metal_get_auto_barrier());
    release(shared_x); release(shared_out); for (unsigned i = 0; i < 3u; ++i) destroy(f + i);
}

static void timing(struct fixture *f, unsigned reps) {
    const char *names[] = {"A1", "B", "A2"}; const unsigned modes[] = {2u, 4u, 2u};
    for (unsigned arm = 0; arm < 3u; ++arm) for (unsigned sample = 0; sample <= samples; ++sample) {
        /* Each arm sees the same per-sample activation. Different inputs keep
         * this from being a constant-output loop; neither changed inputs nor
         * the 703MB matrix establish a physical cold-DRAM measurement. */
        input_fill(f, 1001u + sample); reference(f); mode(modes[arm]);
        myds_metal_lm_head_q8_counters_t a, b; myds_metal_lm_head_q8_counters(&a, false);
        uint64_t old_dispatch = myds_metal_dispatch_count();
        REQUIRE(myds_metal_begin_batch()); id<MTLCommandBuffer> cb = g_batch_cb;
        for (unsigned r = 0; r < reps; ++r) {
            REQUIRE(projected(f, f->out));
            /* Matched output dependencies for both arms; NR4's defensive
             * preflight barriers remain part of the actual measured helper. */
            myds_metal_barrier_buffer(f->out.view);
        }
        REQUIRE(myds_metal_commit_batch());
        char name[128]; snprintf(name, sizeof(name), "timing-r%u-%s-s%u-reference", reps, names[arm], sample); export_blob(name, f->ref);
        snprintf(name, sizeof(name), "timing-r%u-%s-s%u-output", reps, names[arm], sample); export_blob(name, f->out);
        exact(f->ref, f->out); canary(f->weight); canary(f->input);
        myds_metal_lm_head_q8_counters(&b, false);
        REQUIRE(b.requests - a.requests == (uint64_t)(arm == 1u) * reps &&
                b.encoded - a.encoded == (uint64_t)(arm == 1u) * reps && b.failures == a.failures);
        REQUIRE(myds_metal_dispatch_count() - old_dispatch == reps);
        REQUIRE(cb.status == MTLCommandBufferStatusCompleted && isfinite(cb.GPUStartTime) &&
                isfinite(cb.GPUEndTime) && cb.GPUEndTime > cb.GPUStartTime);
        const id<MTLComputePipelineState> pso = arm == 1u ? g_pso_lm_head_q8_nr4 : g_native_pipelines[@"q8_0_8"];
        printf("{\"kind\":\"gpu_timing\",\"arm\":\"%s\",\"nr0\":%u,\"nsg\":8,\"rows\":129280,\"width\":5120,\"weight_bytes\":%zu,\"sample\":%u,\"warmup\":%s,\"input_seed\":%u,\"input_fnv1a64\":\"%016" PRIx64 "\",\"repetitions\":%u,\"requests\":%" PRIu64 ",\"encoded\":%" PRIu64 ",\"failures\":0,\"dispatches\":%u,\"gpu_start_s\":%.9f,\"gpu_end_s\":%.9f,\"gpu_us_per_dispatch\":%.9f,\"threads\":256,\"groups\":%u,\"dynamic_tg_bytes\":%u,\"pso_max_threads\":%lu,\"static_tg_bytes\":%lu,\"output_fnv1a64\":\"%016" PRIx64 "\",\"full_f32_bit_exact\":true,\"finite\":true,\"canaries\":true,\"dispatch_hist_disabled\":true,\"fixture_not_fullmodel\":true,\"cold_dram_not_proven\":true,\"actual_kernel\":\"%s\",\"cb_status\":%lu,\"completed_cb\":true,\"matched_dependency_resources\":3}\n",
            names[arm], modes[arm], f->weight.bytes, sample, sample == 0 ? "true" : "false", 1001u + sample,
            fnv(myds_metal_buffer_data(f->input.view), f->input.bytes), reps, b.requests - a.requests, b.encoded - a.encoded,
            reps, cb.GPUStartTime, cb.GPUEndTime, (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps,
            (f->rows + modes[arm] - 1u) / modes[arm], 128u * modes[arm],
            (unsigned long)pso.maxTotalThreadsPerThreadgroup, (unsigned long)pso.staticThreadgroupMemoryLength,
            fnv(myds_metal_buffer_data(f->out.view), f->out.bytes),
            arm == 1u ? "kernel_mul_mv_q8_0_f32_r4" : "kernel_mul_mv_q8_0_f32", (unsigned long)cb.status);
    }
}

static void host_config(void) {
    REQUIRE(!g_initialized && !g_device && !g_library && myds_metal_lm_head_q8_nr0() == 2);
    unsetenv("MYDS_LM_HEAD_Q8_NR0"); REQUIRE(lm_head_q8_parse_mode() == 2);
    setenv("MYDS_LM_HEAD_Q8_NR0", "2", 1); REQUIRE(lm_head_q8_parse_mode() == 2);
    setenv("MYDS_LM_HEAD_Q8_NR0", "4", 1); REQUIRE(lm_head_q8_parse_mode() == 4);
    const char *bad[] = {"", "0", "1", "3", "8", "02", "04", "+4", "-2", " 4", "4 ", "4\n", "true", "0x4"};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); ++i) {
        setenv("MYDS_LM_HEAD_Q8_NR0", bad[i], 1);
        REQUIRE(!myds_metal_init("/path/not/read/invalid/env"));
        REQUIRE(!g_initialized && !g_device && !g_library && !g_queue && myds_metal_lm_head_q8_nr0() == 2);
    }
    myds_metal_lm_head_q8_counters(NULL, true); mode(4);
    REQUIRE(!myds_metal_gemv(NULL, NULL, NULL, 65537u, 32u, GGML_TYPE_Q8_0));
    myds_metal_lm_head_q8_counters_t count; myds_metal_lm_head_q8_counters(&count, true);
    REQUIRE(count.requests == 1 && count.encoded == 0 && count.failures == 1);
    myds_metal_lm_head_q8_counters(&count, false); REQUIRE(!count.requests && !count.encoded && !count.failures);
    mode(2); unsetenv("MYDS_LM_HEAD_Q8_NR0");
    puts("{\"kind\":\"host_config\",\"pass\":true,\"absent_default\":2,\"valid_values\":[2,4],\"invalid_values_rejected_before_metal\":14,\"counter_reset\":true,\"no_gpu_initialization\":true}");
}

static unsigned number(const char *text) {
    char *end; errno = 0; unsigned long n = strtoul(text, &end, 10);
    REQUIRE(!errno && end != text && !*end && n > 0 && n <= 10000u); return (unsigned)n;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--host-only")) host_only = 1;
        else if (!strcmp(argv[i], "--no-timing")) no_timing = 1;
        else if (!strcmp(argv[i], "--expect-missing-candidate")) expect_missing = 1;
        else if (!strcmp(argv[i], "--shader-dir") && i + 1 < argc) shader_dir = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i + 1 < argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = number(argv[++i]);
        else if (!strcmp(argv[i], "--repetitions") && i + 1 < argc) repetitions = number(argv[++i]);
        else { fprintf(stderr, "usage: %s [--shader-dir DIR] [--output-dir NEW_DIR] [--samples N] [--repetitions N] [--host-only] [--no-timing] [--expect-missing-candidate]\n", argv[0]); return 2; }
    }
    host_config(); if (host_only) return 0;
    if (output_dir) { struct stat st; errno = 0; REQUIRE(lstat(output_dir, &st) != 0 && errno == ENOENT); REQUIRE(!mkdir(output_dir, 0700)); }
    setenv("MYDS_BATCH2_Q8_SHARED", "0", 1); setenv("MYDS_MOE_IQ2_CONSTANT_LUT", "0", 1);
    setenv("MYDS_METAL_CONCURRENT", "0", 1); g_hist_enabled = 1;
    if (expect_missing) {
        setenv("MYDS_LM_HEAD_Q8_NR0", "4", 1); REQUIRE(!myds_metal_init(shader_dir));
        REQUIRE(!g_initialized && !g_device && !g_library && !g_queue && !g_pso_lm_head_q8_nr4 && myds_metal_lm_head_q8_nr0() == 2);
        unsetenv("MYDS_LM_HEAD_Q8_NR0"); REQUIRE(myds_metal_init(shader_dir));
        REQUIRE(![g_library.functionNames containsObject:@"kernel_mul_mv_q8_0_f32_r4"] && !g_pso_lm_head_q8_nr4);
        struct fixture f = create(65537u, 32u, 1u, 42u); reference(&f);
        REQUIRE(projected(&f, f.out)); exact(f.ref, f.out); destroy(&f); myds_metal_cleanup();
        setenv("MYDS_LM_HEAD_Q8_NR0", "2", 1); REQUIRE(myds_metal_init(shader_dir));
        REQUIRE(myds_metal_lm_head_q8_nr0() == 2 && !g_pso_lm_head_q8_nr4); myds_metal_cleanup();
        puts("{\"kind\":\"missing_init\",\"explicit4_rejected\":true,\"cleanup_complete\":true,\"absent_and_explicit2_compatible\":true,\"candidate_symbol_actually_absent\":true,\"pass\":true}"); return 0;
    }
    unsetenv("MYDS_LM_HEAD_Q8_NR0"); REQUIRE(myds_metal_init(shader_dir));
    REQUIRE(myds_metal_lm_head_q8_nr0() == 2 && !g_pso_lm_head_q8_nr4);
    myds_metal_cleanup(); REQUIRE(myds_metal_lm_head_q8_nr0() == 2 && !g_initialized && !g_pso_lm_head_q8_nr4);
    setenv("MYDS_LM_HEAD_Q8_NR0", "4", 1); REQUIRE(myds_metal_init(shader_dir));
    REQUIRE(myds_metal_lm_head_q8_nr0() == 4 && g_pso_lm_head_q8_nr4);
    setenv("MYDS_LM_HEAD_Q8_NR0", "2", 1); REQUIRE(myds_metal_init(shader_dir));
    REQUIRE(myds_metal_lm_head_q8_nr0() == 4); setenv("MYDS_LM_HEAD_Q8_NR0", "4", 1);
    printf("{\"kind\":\"identity\",\"device\":\"%s\",\"production_mode_fixed_until_reinit\":true,\"actual_kernel\":\"kernel_mul_mv_q8_0_f32_r4\",\"nr0\":4,\"nsg\":8,\"threads\":256,\"dynamic_tg_bytes\":512,\"pso_max_threads\":%lu,\"static_tg_bytes\":%lu,\"fixture_not_fullmodel\":true}\n",
           g_device.name.UTF8String, (unsigned long)g_pso_lm_head_q8_nr4.maxTotalThreadsPerThreadgroup,
           (unsigned long)g_pso_lm_head_q8_nr4.staticThreadgroupMemoryLength);
    myds_metal_lm_head_q8_counters(NULL, true);
    const uint32_t shapes[][3] = {{1u, 32u, 1u}, {65536u, 256u, 1u}, {65537u, 32u, 1u},
        {65539u, 96u, 1u}, {129281u, 32u, 1u}, {65537u, 32u, 2u}, {129280u, 5120u, 1u}};
    unsigned id = 0;
    for (unsigned c = 0; c < 2u; ++c) {
        myds_metal_set_concurrent((int)c);
        for (unsigned a = 0; a < 2u; ++a) {
            myds_metal_set_auto_barrier((int)a);
            for (unsigned i = 0; i < sizeof(shapes) / sizeof(*shapes); ++i)
                numeric(id++, shapes[i][0], shapes[i][1], shapes[i][2]);
            reuse(2u * c + a);
        }
        rejections();
    }
    myds_metal_set_concurrent(1); myds_metal_set_auto_barrier(0); g_hist_enabled = 0;
    if (!no_timing) {
        struct fixture f = create(129280u, 5120u, 1u, 42u); REQUIRE(f.weight.bytes == 703283200u);
        timing(&f, 1u); if (repetitions != 1u) timing(&f, repetitions); destroy(&f);
    }
    myds_metal_cleanup(); REQUIRE(myds_metal_lm_head_q8_nr0() == 2 && !g_pso_lm_head_q8_nr4);
    setenv("MYDS_LM_HEAD_Q8_NR0", "2", 1); REQUIRE(myds_metal_init(shader_dir));
    REQUIRE(myds_metal_lm_head_q8_nr0() == 2 && !g_pso_lm_head_q8_nr4);
    struct fixture f = create(65537u, 32u, 1u, 43u); reference(&f);
    REQUIRE(projected(&f, f.out)); exact(f.ref, f.out); destroy(&f); myds_metal_cleanup();
    printf("{\"kind\":\"suite_complete\",\"pass\":true,\"parity_cases\":%u,\"same_batch_reuse_modes\":4,\"samples_per_arm\":%u,\"timing_repetitions\":[1,%u],\"init2_cleanup_init4_cleanup_init2\":true,\"production_setter_added\":false}\n",
           exported_cases, samples, repetitions); return 0;
}
