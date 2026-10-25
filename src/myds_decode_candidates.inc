/* Included after both scalar GEMV argument ABIs. No new device arithmetic is
 * implemented here: the overlap branch encodes the original SharedDown PSO. */
typedef struct {
    myds_metal_buffer_t *out;
    const myds_metal_buffer_t *weight, *input;
    bool shared_q4_nb;
    bool input_pending; /* Candidate: join SharedGU MID with routed MID. */
} shared_down_job_t;
_Static_assert(sizeof(struct ds4_metal_args_mul_mv)==112u,"Q4 host/shader ABI");
_Static_assert(sizeof(struct ds4_metal_args_mul_mv_id)==136u,"IQ2/Q2 host/shader ABI");
_Static_assert(sizeof(struct ds4_metal_dsv4_moe_swiglu_weight_args)==48u,"SwiGLU host/shader ABI");

static int decode_candidate_parse(const char *name) {
    const char *v = getenv(name);
    if (!v || !strcmp(v, "0")) return 0;
    if (!strcmp(v, "1")) return 1;
    fprintf(stderr, "myds_metal: %s must be exactly 0 or 1\n", name);
    return -1;
}
int myds_metal_moe_down_overlap_enabled(void) { return g_moe_down_overlap; }
int myds_metal_moe_iq2_v41_enabled(void) { return g_moe_iq2_v41; }
int myds_metal_dense_q4_v41_enabled(void) { return g_dense_q4_v41; }
int myds_metal_dense_q4_v41_nb_enabled(void) { return g_dense_q4_v41_nb; }
int myds_metal_dense_q4_v41_head_reference_enabled(void) { return g_dense_q4_v41_head_reference; }
int myds_metal_shared_q4_v41_nb_enabled(void) { return g_shared_q4_v41_nb; }
const char *myds_metal_shared_q4_v41_nb_selected_kernel(unsigned shape) {
    if (!g_shared_q4_v41_nb || shape>1u) return NULL;
    return shape ? "kernel_myds_q4_v41_shared_down_nb_bf16" :
        "kernel_myds_v41_shared_gate_up_swiglu_q4_v41_nb";
}
const char *myds_metal_dense_q4_v41_selected_kernel(unsigned shape) {
    if (!g_dense_q4_v41 || shape >= 3u) return NULL;
    const uint32_t nr = dense_q4_nr_site(shape);
    if (!shape) return g_dense_q4_v41_head_reference ?
        (nr == 4u ? "kernel_mul_mv_q4_K_dense_f32_r4" : "kernel_mul_mv_q4_K_dense_f32") :
        (nr == 4u ? "kernel_myds_q4_v41_head_addr_f32_r4" : "kernel_myds_q4_v41_head_addr_f32");
    if (shape==1u) return g_dense_q4_v41_nb ?
        (nr == 4u ? "kernel_myds_q4_v41_qb_nb_bf16_r4" : "kernel_myds_q4_v41_qb_nb_bf16") :
        (nr == 4u ? "kernel_myds_q4_v41_qb_addr_bf16_r4" : "kernel_myds_q4_v41_qb_addr_bf16");
    return g_dense_q4_v41_nb ?
        (nr == 4u ? "kernel_myds_q4_v41_ob_nb_bf16_r4" : "kernel_myds_q4_v41_ob_nb_bf16") :
        (nr == 4u ? "kernel_myds_q4_v41_ob_addr_bf16_r4" : "kernel_myds_q4_v41_ob_addr_bf16");
}

static void decode_candidate_counts(unsigned which, myds_metal_decode_candidate_counters_t *out, bool reset) {
    uint64_t v[3];
    for (unsigned i=0; i<3; ++i) v[i] = reset
        ? atomic_exchange_explicit(&g_decode_candidates[which][i],0,memory_order_relaxed)
        : atomic_load_explicit(&g_decode_candidates[which][i],memory_order_relaxed);
    if (out) *out = (myds_metal_decode_candidate_counters_t){v[0],v[1],v[2]};
}
void myds_metal_moe_down_overlap_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
    decode_candidate_counts(DC_OVERLAP,out,reset);
}
void myds_metal_moe_iq2_v41_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
    decode_candidate_counts(DC_IQ2,out,reset);
}
void myds_metal_dense_q4_v41_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
    decode_candidate_counts(DC_Q4,out,reset);
}
void myds_metal_dense_q4_v41_nb_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
    decode_candidate_counts(DC_Q4_NB,out,reset);
}
void myds_metal_dense_q4_v41_head_reference_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
    decode_candidate_counts(DC_Q4_HEAD_REFERENCE,out,reset);
}
void myds_metal_shared_q4_v41_nb_counters(myds_metal_decode_candidate_counters_t *out, bool reset) {
    decode_candidate_counts(DC_SHARED_Q4_NB,out,reset);
}
static int decode_candidate_failure(unsigned which, const char *reason) {
    atomic_fetch_add_explicit(&g_decode_candidates[which][2],1,memory_order_relaxed);
    fprintf(stderr,"myds_metal: decode candidate %u rejected: %s; refusing fallback\n",which,reason);
    return 0;
}
static bool decode_candidate_geometry(id<MTLComputePipelineState> p, unsigned nsg, NSUInteger dynamic) {
    return g_device && p && p.device==g_device && p.threadExecutionWidth==32u &&
        p.maxTotalThreadsPerThreadgroup>=32u*nsg &&
        g_device.maxThreadsPerThreadgroup.width>=32u &&
        g_device.maxThreadsPerThreadgroup.height>=nsg && g_device.maxThreadsPerThreadgroup.depth>=1u &&
        dynamic<=g_device.maxThreadgroupMemoryLength &&
        p.staticThreadgroupMemoryLength<=g_device.maxThreadgroupMemoryLength-dynamic;
}
static id<MTLComputePipelineState> decode_candidate_pipeline(NSString *name, short nsg) {
    MTLFunctionConstantValues *values=[MTLFunctionConstantValues new];
    [values setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError *e=nil;
    id<MTLFunction> f=[g_library newFunctionWithName:name constantValues:values error:&e];
    id<MTLComputePipelineState> p=f?[g_device newComputePipelineStateWithFunction:f error:&e]:nil;
    if (!p) fprintf(stderr,"myds_metal: candidate PSO %s unavailable (%s)\n",
        name.UTF8String,e.localizedDescription.UTF8String?:"missing function");
    return p;
}
static bool shared_q4_nb_geometry(unsigned shape) {
    return shape<2u && decode_candidate_geometry(g_pso_shared_q4_v41_nb[shape],4u,shape ? 256u : 0u) &&
        (shape || g_device.maxThreadsPerThreadgroup.width>=128u);
}
static int decode_candidates_initialize(bool overlap, bool iq2, bool q4, bool nb, bool head_reference, bool shared_nb) {
    if ((overlap || q4) && ((dense_q4_nr0()!=2u && dense_q4_nr0()!=4u) || dense_q4_nsg()!=4u)) {
        fprintf(stderr,"myds_metal: overlap/Q4 candidates require existing Dense Q4 NR2/NR4 and NSG4\n");
        return 0;
    }
    if (iq2) {
        g_pso_moe_iq2_v41=decode_candidate_pipeline(@"kernel_myds_iq2_v41_shape32_f32",(short)g_moe_nsg);
        if (!decode_candidate_geometry(g_pso_moe_iq2_v41,(unsigned)g_moe_nsg,2176u)) return 0;
    }
    if (q4) {
        const uint32_t nr0 = dense_q4_nr_site(0u), nr1 = dense_q4_nr_site(1u), nr2 = dense_q4_nr_site(2u);
        NSArray<NSString *> *names=@[
            head_reference ? (nr0 == 4u ? @"kernel_mul_mv_q4_K_dense_f32_r4" : @"kernel_mul_mv_q4_K_dense_f32") :
                (nr0 == 4u ? @"kernel_myds_q4_v41_head_addr_f32_r4" : @"kernel_myds_q4_v41_head_addr_f32"),
            nb ? (nr1 == 4u ? @"kernel_myds_q4_v41_qb_nb_bf16_r4" : @"kernel_myds_q4_v41_qb_nb_bf16") :
                 (nr1 == 4u ? @"kernel_myds_q4_v41_qb_addr_bf16_r4" : @"kernel_myds_q4_v41_qb_addr_bf16"),
            nb ? (nr2 == 4u ? @"kernel_myds_q4_v41_ob_nb_bf16_r4" : @"kernel_myds_q4_v41_ob_nb_bf16") :
                 (nr2 == 4u ? @"kernel_myds_q4_v41_ob_addr_bf16_r4" : @"kernel_myds_q4_v41_ob_addr_bf16")];
        for (unsigned i=0;i<3u;++i) {
            g_pso_dense_q4_v41[i]=decode_candidate_pipeline(names[i],4);
            if (!decode_candidate_geometry(g_pso_dense_q4_v41[i],4u,256u)) return 0;
            fprintf(stderr,"myds_metal: Dense Q4 V41 provider %u = %s (NR%u/NSG4)\n",i,names[i].UTF8String,dense_q4_nr_site(i));
        }
    }
    if (overlap && !decode_candidate_geometry(g_pso_mul_mv_q4_K_dense_bf16,4u,256u)) return 0;
    if (shared_nb) {
        if (!overlap || iq2 || !q4 || !nb || head_reference) return 0;
        NSError *e=nil;
        id<MTLFunction> f=[g_library newFunctionWithName:@"kernel_myds_v41_shared_gate_up_swiglu_q4_v41_nb"];
        g_pso_shared_q4_v41_nb[0]=f ? [g_device newComputePipelineStateWithFunction:f error:&e] : nil;
        g_pso_shared_q4_v41_nb[1]=decode_candidate_pipeline(@"kernel_myds_q4_v41_shared_down_nb_bf16",4);
        if (!shared_q4_nb_geometry(0u) || !shared_q4_nb_geometry(1u)) {
            fprintf(stderr,"myds_metal: explicit Shared Q4 NB PSO/geometry unavailable; refusing fallback\n");
            return 0;
        }
        fprintf(stderr,"myds_metal: Shared Q4 NB providers GU=kernel_myds_v41_shared_gate_up_swiglu_q4_v41_nb (16B ABI), Down=kernel_myds_q4_v41_shared_down_nb_bf16 (112B ABI, FC600=4)\n");
    }
    g_moe_down_overlap=overlap; g_moe_iq2_v41=iq2; g_dense_q4_v41=q4;
    g_dense_q4_v41_nb=nb; g_dense_q4_v41_head_reference=head_reference;
    g_shared_q4_v41_nb=shared_nb;
    return 1;
}

/* Range checks include allocation, device, actual no-copy pointer and binding
 * base. Only relative offsets become uint32; a GGUF view may start above 4GiB. */
static bool decode_candidate_range(const myds_metal_buffer_t *b, uint64_t bytes, size_t alignment) {
    return f16_gemv_bf16_range(b,bytes,alignment);
}
static bool decode_candidate_overlap(const myds_metal_buffer_t *a,uint64_t na,
                                     const myds_metal_buffer_t *b,uint64_t nb) {
    return f16_gemv_bf16_overlap(a,na,b,nb);
}

struct shared_q4_nb_gu_args { uint32_t in_dim,out_dim,row_bytes; float clamp; };
_Static_assert(sizeof(struct shared_q4_nb_gu_args)==16u,"Shared Q4 GU ABI");
int myds_metal_shared_q4_v41_nb_gate_up(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *gate,const myds_metal_buffer_t *up,
    const myds_metal_buffer_t *input,uint32_t rows,uint32_t width,float clamp) {
    atomic_fetch_add_explicit(&g_decode_candidates[DC_SHARED_Q4_NB][0],1,memory_order_relaxed);
    if (!g_initialized || !g_shared_q4_v41_nb || !g_batch_cb || g_auto_barrier ||
        !myds_metal_get_concurrent() || g_kp_on || rows!=2304u || width!=5120u ||
        !isfinite(clamp) || clamp<0 || dense_q4_nr0()!=2u || dense_q4_nsg()!=4u ||
        !shared_q4_nb_geometry(0u))
        return decode_candidate_failure(DC_SHARED_Q4_NB,"GU scope/shape/PSO/geometry");
    if (!decode_candidate_range(out,9216u,16u) || !decode_candidate_range(input,20480u,4u) ||
        !decode_candidate_range(gate,6635520u,2u) || !decode_candidate_range(up,6635520u,2u) ||
        decode_candidate_overlap(out,9216u,input,20480u) ||
        decode_candidate_overlap(out,9216u,gate,6635520u) ||
        decode_candidate_overlap(out,9216u,up,6635520u))
        return decode_candidate_failure(DC_SHARED_Q4_NB,"GU range/alignment/alias");
    id<MTLComputeCommandEncoder> enc=get_compute_encoder(g_batch_cb);
    if (!enc || enc!=g_batch_enc) return decode_candidate_failure(DC_SHARED_Q4_NB,"GU encoder unavailable");
    const struct shared_q4_nb_gu_args a={5120u,2304u,2880u,clamp};
    [enc setComputePipelineState:g_pso_shared_q4_v41_nb[0]];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:gate->mtl_buf offset:gate->offset atIndex:1];
    [enc setBuffer:up->mtl_buf offset:up->offset atIndex:2];
    [enc setBuffer:input->mtl_buf offset:input->offset atIndex:3];
    [enc setBuffer:out->mtl_buf offset:out->offset atIndex:4];
    metal_dispatch(enc,"kernel_myds_v41_shared_gate_up_swiglu_q4_v41_nb",MTLSizeMake(288u,1u,1u),MTLSizeMake(128u,1u,1u));
    atomic_fetch_add_explicit(&g_decode_candidates[DC_SHARED_Q4_NB][1],1,memory_order_relaxed);
    return 1;
}

static int dense_q4_provider_failure(int provider, const char *reason) {
    if (provider>=0) atomic_fetch_add_explicit(&g_decode_candidates[provider][2],1,memory_order_relaxed);
    return decode_candidate_failure(DC_Q4,reason);
}
int myds_metal_dense_q4_v41(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t rows, uint32_t width, bool round_bf16) {
    atomic_fetch_add_explicit(&g_decode_candidates[DC_Q4][0],1,memory_order_relaxed);
    int index = rows==129280u && width==5120u && !round_bf16 ? 0 :
                rows==32768u && width==1280u && round_bf16 ? 1 :
                rows==5120u && width==8192u && round_bf16 ? 2 : -1;
    const int provider = index==0 && g_dense_q4_v41_head_reference ? DC_Q4_HEAD_REFERENCE :
        index>0 && g_dense_q4_v41_nb ? DC_Q4_NB : -1;
    if (provider>=0) atomic_fetch_add_explicit(&g_decode_candidates[provider][0],1,memory_order_relaxed);
    const uint32_t nr = (index >= 0 && dense_q4_nr_site((unsigned)index) == 4u && (rows % 4u == 0)) ? 4u : 2u;
    if (!g_initialized || !g_dense_q4_v41 || index<0 || (dense_q4_nr0()!=2u && dense_q4_nr0()!=4u) || dense_q4_nsg()!=4u ||
        !decode_candidate_geometry(g_pso_dense_q4_v41[index],4u,256u))
        return dense_q4_provider_failure(provider,"scope/shape/PSO/geometry");
    const uint64_t row_bytes=(uint64_t)(width/256u)*144u, weights=row_bytes*rows;
    const uint64_t in_bytes=(uint64_t)width*4u, out_bytes=(uint64_t)rows*4u;
    if (weights>UINT32_MAX || !decode_candidate_range(weight,weights,2u) ||
        !decode_candidate_range(input,in_bytes,4u) || !decode_candidate_range(out,out_bytes,16u) ||
        decode_candidate_overlap(out,out_bytes,weight,weights) ||
        decode_candidate_overlap(out,out_bytes,input,in_bytes))
        return dense_q4_provider_failure(provider,"range/alignment/alias");
    struct ds4_metal_args_mul_mv a={0};
    a.ne00=(int32_t)width; a.ne01=(int32_t)rows; a.ne02=1;
    a.nb00=144u; a.nb01=row_bytes; a.nb02=weights; a.nb03=weights;
    a.ne10=(int32_t)width; a.ne11=1; a.ne12=1; a.nb10=4u;
    a.nb11=in_bytes; a.nb12=in_bytes; a.nb13=in_bytes;
    a.ne0=(int32_t)rows; a.ne1=1; a.nr0=(int)nr; a.r2=1; a.r3=1;
    id<MTLCommandBuffer> cb=g_batch_cb?:[g_queue commandBuffer];
    if (!cb) return dense_q4_provider_failure(provider,"command buffer unavailable");
    id<MTLComputeCommandEncoder> enc=get_compute_encoder(cb);
    if (!enc) return dense_q4_provider_failure(provider,"encoder unavailable");
    [enc setComputePipelineState:g_pso_dense_q4_v41[index]];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:weight->mtl_buf offset:weight->offset atIndex:1];
    [enc setBuffer:input->mtl_buf offset:input->offset atIndex:2];
    [enc setBuffer:out->mtl_buf offset:out->offset atIndex:3];
    [enc setThreadgroupMemoryLength:256u atIndex:0];
    metal_dispatch(enc,myds_metal_dense_q4_v41_selected_kernel((unsigned)index),MTLSizeMake(rows/(4u*nr),1u,1u),MTLSizeMake(32u,4u,1u));
    if (!v41_finish(cb,enc)) return dense_q4_provider_failure(provider,"command completion failed");
    atomic_fetch_add_explicit(&g_decode_candidates[DC_Q4][1],1,memory_order_relaxed);
    if (provider>=0) atomic_fetch_add_explicit(&g_decode_candidates[provider][1],1,memory_order_relaxed);
    return 1;
}

int myds_metal_dense_q4_v41_qb_slice(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t row_offset, uint32_t slice_rows, bool count_call) {
    if (count_call) atomic_fetch_add_explicit(&g_decode_candidates[DC_Q4][0],1,memory_order_relaxed);
    const int index = 1;
    const int provider = g_dense_q4_v41_nb ? DC_Q4_NB : -1;
    if (count_call && provider>=0) atomic_fetch_add_explicit(&g_decode_candidates[provider][0],1,memory_order_relaxed);
    const uint32_t nr = (dense_q4_nr_site(1u) == 4u && (slice_rows % 16u == 0) && (row_offset % 16u == 0)) ? 4u : 2u;
    if (!g_initialized || !g_dense_q4_v41 || !slice_rows || (slice_rows % (4u*nr)) != 0u ||
        (row_offset % (4u*nr)) != 0u || row_offset > 32768u || slice_rows > 32768u - row_offset ||
        (dense_q4_nr0()!=2u && dense_q4_nr0()!=4u) || dense_q4_nsg()!=4u ||
        !decode_candidate_geometry(g_pso_dense_q4_v41[index],4u,256u))
        return dense_q4_provider_failure(provider,"scope/shape/PSO/geometry");
    const uint32_t width = 1280u, total_rows = 32768u;
    const uint64_t row_bytes=(uint64_t)(width/256u)*144u, weights=row_bytes*total_rows;
    const uint64_t in_bytes=(uint64_t)width*4u, out_bytes=(uint64_t)total_rows*4u;
    if (!decode_candidate_range(weight,weights,2u) ||
        !decode_candidate_range(input,in_bytes,4u) || !decode_candidate_range(out,out_bytes,16u) ||
        decode_candidate_overlap(out,out_bytes,weight,weights) ||
        decode_candidate_overlap(out,out_bytes,input,in_bytes))
        return dense_q4_provider_failure(provider,"range/alignment/alias");
    const uint64_t slice_weights = row_bytes * slice_rows;
    struct ds4_metal_args_mul_mv a={0};
    a.ne00=(int32_t)width; a.ne01=(int32_t)slice_rows; a.ne02=1;
    a.nb00=144u; a.nb01=row_bytes; a.nb02=slice_weights; a.nb03=slice_weights;
    a.ne10=(int32_t)width; a.ne11=1; a.ne12=1; a.nb10=4u;
    a.nb11=in_bytes; a.nb12=in_bytes; a.nb13=in_bytes;
    a.ne0=(int32_t)slice_rows; a.ne1=1; a.nr0=(int)nr; a.r2=1; a.r3=1;
    id<MTLCommandBuffer> cb=g_batch_cb?:[g_queue commandBuffer];
    if (!cb) return dense_q4_provider_failure(provider,"command buffer unavailable");
    id<MTLComputeCommandEncoder> enc=get_compute_encoder(cb);
    if (!enc) return dense_q4_provider_failure(provider,"encoder unavailable");
    [enc setComputePipelineState:g_pso_dense_q4_v41[index]];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:weight->mtl_buf offset:weight->offset + (NSUInteger)(row_offset * row_bytes) atIndex:1];
    [enc setBuffer:input->mtl_buf offset:input->offset atIndex:2];
    [enc setBuffer:out->mtl_buf offset:out->offset + (NSUInteger)(row_offset * 4u) atIndex:3];
    [enc setThreadgroupMemoryLength:256u atIndex:0];
    metal_dispatch(enc,myds_metal_dense_q4_v41_selected_kernel((unsigned)index),MTLSizeMake(slice_rows/(4u*nr),1u,1u),MTLSizeMake(32u,4u,1u));
    if (!v41_finish(cb,enc)) return dense_q4_provider_failure(provider,"command completion failed");
    if (count_call) {
        atomic_fetch_add_explicit(&g_decode_candidates[DC_Q4][1],1,memory_order_relaxed);
        if (provider>=0) atomic_fetch_add_explicit(&g_decode_candidates[provider][1],1,memory_order_relaxed);
    }
    return 1;
}

/* Called after Q2 dispatch and before its existing output join. Exactly the
 * reference Q4 SharedDown ABI/PSO/grid. No helper reentry, encoder transition,
 * barrier or BF16 fallback is permitted between these two down dispatches. */
static int encode_shared_down_overlap(id<MTLComputeCommandEncoder> enc, const shared_down_job_t *job) {
    if (!job || !g_batch_cb || enc!=g_batch_enc || g_auto_barrier || g_kp_on ||
        !(job->shared_q4_nb ? g_shared_q4_v41_nb && shared_q4_nb_geometry(1u) :
            decode_candidate_geometry(g_pso_mul_mv_q4_K_dense_bf16,4u,256u))) return 0;
    struct ds4_metal_args_mul_mv a={0};
    a.ne00=2304; a.ne01=5120; a.ne02=1;
    a.nb00=144u; a.nb01=1296u; a.nb02=6635520u; a.nb03=a.nb02;
    a.ne10=2304; a.ne11=1; a.ne12=1; a.nb10=4u;
    a.nb11=9216u; a.nb12=a.nb11; a.nb13=a.nb12;
    a.ne0=5120; a.ne1=1; a.nr0=2; a.r2=1; a.r3=1;
    [enc setComputePipelineState:job->shared_q4_nb ? g_pso_shared_q4_v41_nb[1] : g_pso_mul_mv_q4_K_dense_bf16];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:job->weight->mtl_buf offset:job->weight->offset atIndex:1];
    [enc setBuffer:job->input->mtl_buf offset:job->input->offset atIndex:2];
    [enc setBuffer:job->out->mtl_buf offset:job->out->offset atIndex:3];
    [enc setThreadgroupMemoryLength:256u atIndex:0];
    metal_dispatch(enc,job->shared_q4_nb ? "kernel_myds_q4_v41_shared_down_nb_bf16" : "shared_down_overlap_q4_bf16",
        MTLSizeMake(640u,1u,1u),MTLSizeMake(32u,4u,1u));
    if (job->shared_q4_nb) atomic_fetch_add_explicit(&g_decode_candidates[DC_SHARED_Q4_NB][1],1,memory_order_relaxed);
    return enc==g_batch_enc;
}

