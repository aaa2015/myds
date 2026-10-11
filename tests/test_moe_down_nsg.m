/* Component-only GPU fixture. Includes the actual runtime so the otherwise
 * unused non-fused Q2 PSO and init/cleanup state can also be checked without
 * adding production test hooks. Quantized blocks are deterministic nonzero
 * fixtures, not captured GGUF weights or a full-model acceptance oracle. */
#include "myds_metal.m"
#include <inttypes.h>

#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

struct guarded { myds_metal_buffer_t *owner, *view; size_t bytes; };
static struct guarded allocate(size_t bytes) {
    struct guarded b = {.bytes=bytes};
    b.owner = myds_metal_buffer_alloc(bytes + 32); REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner), 0xa5, bytes + 32);
    b.view = myds_metal_buffer_sub_view(b.owner, 16, bytes); REQUIRE(b.view);
    return b;
}
static void canary(struct guarded b) {
    const uint8_t *p = myds_metal_buffer_data(b.owner);
    for (size_t i=0;i<16;i++) REQUIRE(p[i]==0xa5 && p[16+b.bytes+i]==0xa5);
}
static void release(struct guarded b) {
    canary(b); myds_metal_buffer_free(b.view); myds_metal_buffer_free(b.owner);
}
static void iq2_fill(struct guarded b, unsigned seed) {
    uint8_t *p = myds_metal_buffer_data(b.view);
    REQUIRE(b.bytes % 66 == 0);
    for (size_t block=0;block<b.bytes/66;block++) {
        uint16_t d=(uint16_t)(0x1400u+(block%4u)*0x100u); memcpy(p+block*66,&d,2);
        for (size_t j=0;j<64;j++) p[block*66+2+j]=(uint8_t)(block*37u+j*29u+seed);
    }
}
static void q2_fill(struct guarded b, unsigned seed) {
    uint8_t *p = myds_metal_buffer_data(b.view);
    REQUIRE(b.bytes % 84 == 0);
    for (size_t block=0;block<b.bytes/84;block++) {
        for (size_t j=0;j<16;j++) p[block*84+j]=(uint8_t)(1u+(block*13u+j*17u+seed)%255u);
        for (size_t j=0;j<64;j++) p[block*84+16+j]=(uint8_t)(block*23u+j*41u+seed);
        uint16_t d=(uint16_t)(0x1c00u+(block%3u)*0x80u), m=(uint16_t)(0x1000u+(block%5u)*0x80u);
        memcpy(p+block*84+80,&d,2); memcpy(p+block*84+82,&m,2);
    }
}
static void activation_fill(struct guarded b, unsigned seed) {
    float *p=myds_metal_buffer_data(b.view);
    for (size_t i=0;i<b.bytes/4;i++) p[i]=((int)((i*19u+seed)%257u)-128)/1024.0f;
}
static void selected_fill(struct guarded ids, struct guarded weights, uint32_t tokens, uint32_t active) {
    const uint32_t selected[6]={11,3,0,12,5,7};
    int32_t *id=myds_metal_buffer_data(ids.view);
    float *w=myds_metal_buffer_data(weights.view);
    for (uint32_t t=0;t<tokens;t++) for (uint32_t s=0;s<active;s++) {
        id[(size_t)t*active+s]=(int32_t)((selected[s]+t*5u)%13u);
        w[(size_t)t*active+s]=(float)(s+1u)/(float)(active*(active+1u)/2u);
    }
}
static void complete_nonzero(struct guarded b) {
    const float *p=myds_metal_buffer_data(b.view); size_t nonzero=0;
    for (size_t i=0;i<b.bytes/4;i++) {
        uint32_t bits; memcpy(&bits,p+i,4);
        REQUIRE(isfinite(p[i]) && bits!=UINT32_C(0xa5a5a5a5)); nonzero+=p[i]!=0;
    }
    REQUIRE(nonzero>0); canary(b);
}
static uint64_t fnv(const void *bytes,size_t length) {
    const uint8_t *p=bytes; uint64_t h=UINT64_C(14695981039346656037);
    for (size_t i=0;i<length;i++) { h^=p[i]; h*=UINT64_C(1099511628211); }
    return h;
}
static void export_buffers(const char *directory,unsigned phase,const char *name,
                           struct guarded a,struct guarded b) {
    complete_nonzero(a); complete_nonzero(b);
    char path[4096]; int n=snprintf(path,sizeof(path),"%s/phase%u-%s.bin",directory,phase,name);
    REQUIRE(n>0 && (size_t)n<sizeof(path)); FILE *f=fopen(path,"wb"); REQUIRE(f);
    REQUIRE(fwrite(myds_metal_buffer_data(a.view),1,a.bytes,f)==a.bytes);
    REQUIRE(fwrite(myds_metal_buffer_data(b.view),1,b.bytes,f)==b.bytes); REQUIRE(fclose(f)==0);
    printf("{\"kind\":\"case\",\"phase\":%u,\"name\":\"%s\",\"iq2_nsg\":%d,\"down_nsg\":%d,"
           "\"first_bytes\":%zu,\"second_bytes\":%zu,\"first_fnv1a64\":\"%016" PRIx64 "\","
           "\"second_fnv1a64\":\"%016" PRIx64 "\",\"finite\":true,\"nonzero\":true,\"canaries\":true}\n",
           phase,name,g_moe_nsg,g_moe_down_nsg,a.bytes,b.bytes,
           fnv(myds_metal_buffer_data(a.view),a.bytes),fnv(myds_metal_buffer_data(b.view),b.bytes));
}

static void routed_case(const char *directory,unsigned phase,uint32_t dim,uint32_t width,
                        uint32_t active,uint32_t tokens,int batch) {
    const uint32_t experts=13;
    struct guarded gate=allocate((size_t)experts*width*(dim/256u)*66u),
      up=allocate(gate.bytes),down=allocate((size_t)experts*dim*(width/256u)*84u),
      x=allocate((size_t)tokens*dim*4u),ids=allocate((size_t)tokens*active*4u),
      weights=allocate(ids.bytes),sg=allocate((size_t)tokens*active*width*4u),
      su=allocate(sg.bytes),mid=allocate(sg.bytes),sd=allocate((size_t)tokens*active*dim*4u),
      out=allocate((size_t)tokens*dim*4u);
    iq2_fill(gate,17); iq2_fill(up,71); q2_fill(down,29); activation_fill(x,43);
    selected_fill(ids,weights,tokens,active);
    struct guarded all[]={gate,up,down,x,ids,weights,sg,su,mid,sd,out};
    if (active==3u) {
        /* The target execution graph uses a fixed sum6 and cannot accept
         * top-3. Verify rejection before any dispatch or buffer mutation. */
        uint64_t before[sizeof(all)/sizeof(*all)];
        for (size_t i=0;i<sizeof(all)/sizeof(*all);i++)
            before[i]=fnv(myds_metal_buffer_data(all[i].owner),all[i].bytes+32u);
        if (batch) REQUIRE(myds_metal_begin_batch());
        myds_metal_dispatch_count_reset();
        REQUIRE(!myds_metal_routed_moe_batch(out.view,x.view,gate.view,up.view,down.view,
            ids.view,weights.view,sg.view,su.view,mid.view,sd.view,dim,width,experts,active,tokens,
            GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,10.0f));
        REQUIRE(myds_metal_dispatch_count()==0);
        if (batch) REQUIRE(myds_metal_commit_batch());
        for (size_t i=0;i<sizeof(all)/sizeof(*all);i++) {
            REQUIRE(fnv(myds_metal_buffer_data(all[i].owner),all[i].bytes+32u)==before[i]);
            release(all[i]);
        }
        printf("{\"kind\":\"top3_rejected\",\"phase\":%u,\"dim\":%u,\"width\":%u,"
               "\"active\":3,\"tokens\":%u,\"batch\":%d,\"dispatches\":0,"
               "\"all_buffers_unchanged\":true,\"canaries\":true,\"pass\":true}\n",
               phase,dim,width,tokens,batch);
        return;
    }
    if (batch) REQUIRE(myds_metal_begin_batch());
    myds_metal_dispatch_count_reset();
    REQUIRE(myds_metal_routed_moe_batch(out.view,x.view,gate.view,up.view,down.view,
        ids.view,weights.view,sg.view,su.view,mid.view,sd.view,dim,width,experts,active,tokens,
        GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,10.0f));
    REQUIRE(myds_metal_dispatch_count()==2); /* Actual release fused MV path. */
    if (batch) REQUIRE(myds_metal_commit_batch());
    char name[128]; snprintf(name,sizeof(name),"routed-%u-%u-%u-%u-batch%d",dim,width,active,tokens,batch);
    export_buffers(directory,phase,name,mid,out);
    for (size_t i=0;i<sizeof(all)/sizeof(*all);i++) release(all[i]);
}

/* Both compiled Q2 PSOs, nonzero selected experts and activations, plus row
 * tails. The old generic MV kernel reads its complete NR tile before masking
 * output, so fixture expert blobs deliberately pad to the larger NSG4 tile. */
static void direct_q2_case(const char *directory,unsigned phase,uint32_t rows,uint32_t width) {
    const uint32_t active=6,tokens=2,experts=13,padded=(rows+7u)&~7u;
    const uint64_t row_bytes=(uint64_t)(width/256u)*84u;
    struct guarded down=allocate((size_t)experts*padded*row_bytes),
      mid=allocate((size_t)tokens*active*width*4u),ids=allocate((size_t)tokens*active*4u),
      weights=allocate(ids.bytes),out=allocate((size_t)tokens*rows*4u),
      partial=allocate((size_t)tokens*active*rows*4u);
    q2_fill(down,83); activation_fill(mid,97); selected_fill(ids,weights,tokens,active);
    struct ds4_metal_args_mul_mv_id args={
      .nei0=active,.nei1=tokens,.nbi1=(uint64_t)active*4u,
      .ne00=width,.ne01=rows,.ne02=experts,.nb00=84u,.nb01=row_bytes,.nb02=(uint64_t)padded*row_bytes,
      .ne10=width,.ne11=active,.ne12=1,.ne13=1,.nb10=4,.nb11=(uint64_t)width*4u,
      .nb12=(uint64_t)active*width*4u,.ne0=rows,.ne1=tokens,.nb1=(uint64_t)rows*4u,.nr0=2,.tp_world=1};
    const NSUInteger groups=(rows+2u*(uint32_t)g_moe_down_nsg-1u)/(2u*(uint32_t)g_moe_down_nsg);
    REQUIRE(myds_metal_begin_batch());
    id<MTLComputeCommandEncoder> enc=get_compute_encoder(g_batch_cb); REQUIRE(enc);
    [enc setComputePipelineState:g_pso_moe_sum6_q2_k];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:down.view->mtl_buf offset:down.view->offset atIndex:1];
    [enc setBuffer:mid.view->mtl_buf offset:mid.view->offset atIndex:2];
    [enc setBuffer:out.view->mtl_buf offset:out.view->offset atIndex:3];
    [enc setBuffer:ids.view->mtl_buf offset:ids.view->offset atIndex:4];
    [enc setBuffer:out.view->mtl_buf offset:out.view->offset atIndex:5];
    metal_dispatch(enc,"fixture_q2_sum6",MTLSizeMake(groups,tokens,1),MTLSizeMake(32,g_moe_down_nsg,1));
    myds_metal_barrier();
    args.ne1=active;
    [enc setComputePipelineState:g_pso_moe_mv_id_q2_k];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:partial.view->mtl_buf offset:partial.view->offset atIndex:3];
    metal_dispatch(enc,"fixture_q2_id",MTLSizeMake(groups,1,active*tokens),MTLSizeMake(32,g_moe_down_nsg,1));
    REQUIRE(myds_metal_commit_batch());
    char name[80]; snprintf(name,sizeof(name),"q2-psos-tail-%u-%u",rows,width);
    export_buffers(directory,phase,name,out,partial);
    struct guarded all[]={down,mid,ids,weights,out,partial};
    for (size_t i=0;i<sizeof(all)/sizeof(*all);i++) release(all[i]);
}
static void cases(const char *directory,unsigned phase) {
    for (int batch=0;batch<2;batch++) {
        routed_case(directory,phase,256,256,6,1,batch);
        routed_case(directory,phase,512,768,6,3,batch);
        routed_case(directory,phase,512,256,3,2,batch);
    }
    routed_case(directory,phase,5120,2304,6,1,1);
    direct_q2_case(directory,phase,1,256);
    direct_q2_case(directory,phase,17,768);
    direct_q2_case(directory,phase,259,2304);
}
int main(int argc,char **argv) {
    if (argc==2 && strcmp(argv[1],"--reject")==0) {
        REQUIRE(!myds_metal_init(NULL));
        REQUIRE(!g_initialized && !g_device && !g_queue && !g_library);
        REQUIRE(!g_pso_moe_mv_id_q2_k && !g_pso_moe_sum6_q2_k);
        puts("{\"kind\":\"rejected_before_metal_resources\",\"pass\":true}"); return 0;
    }
    if (argc<5 || argc>6) {
        fprintf(stderr,"Usage: test_moe_down_nsg shader_dir output_dir expected_iq2 expected_down [--reinit]\n"); return 2;
    }
    const int expected_iq2=atoi(argv[3]),expected_down=atoi(argv[4]);
    REQUIRE(myds_metal_init(argv[1])); REQUIRE(g_moe_nsg==expected_iq2 && g_moe_down_nsg==expected_down);
    REQUIRE(g_pso_moe_mv_id_q2_k && g_pso_moe_sum6_q2_k);
    cases(argv[2],0);
    /* Updating getenv while initialized must not mismatch the existing PSOs. */
    REQUIRE(setenv("MYDS_MOE_DOWN_NSG",expected_down==2?"4":"2",1)==0);
    REQUIRE(myds_metal_init(argv[1])); REQUIRE(g_moe_down_nsg==expected_down);
    myds_metal_cleanup(); REQUIRE(!g_initialized && g_moe_nsg==2 && g_moe_down_nsg==2);
    REQUIRE(!g_pso_moe_mv_id_q2_k && !g_pso_moe_sum6_q2_k);
    if (argc==6) {
        REQUIRE(strcmp(argv[5],"--reinit")==0);
        const int next=expected_down==2?4:2;
        REQUIRE(myds_metal_init(argv[1])); REQUIRE(g_moe_down_nsg==next && g_moe_nsg==expected_iq2);
        cases(argv[2],1); myds_metal_cleanup();
        REQUIRE(unsetenv("MYDS_MOE_DOWN_NSG")==0);
        REQUIRE(myds_metal_init(argv[1])); REQUIRE(g_moe_down_nsg==expected_iq2);
        cases(argv[2],2); myds_metal_cleanup();
    }
    puts("{\"kind\":\"complete\",\"pass\":true,\"fixture\":\"seeded_nonzero_iq2_q2\"}");
    return 0;
}
