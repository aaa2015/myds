/* Private component probe. No production source or API is changed.
 * Build and --host-only submit no GPU work. Full mode requires root's exclusive window. */
#define main original_attention_fixture_main
#include "../tests/test_attention_packed.m"
#undef main
#define PARTIAL_BYTES (64u*32u*128u*16u)
#define STAT_BYTES (64u*32u*3u*4u)
static const char *probe_names[2][2]={{"probe_attention_tg","probe_attention_tg_debug"},{"probe_attention_register","probe_attention_register_debug"}};
static unsigned completed_pairs;
static const char *golden_path;
static id<MTLLibrary> golden_library;
static void golden_encode(struct fixture *f,myds_v41_gpu_scratch_t *s,unsigned output) {
    struct flash_vec_args a=vector_args(f->keys);myds_metal_constant_t c[10];constants(f->keys,c);
    const char *cache_name="probe_attention_golden_actual_production";
    if(!specialized_pipeline_lookup(cache_name,c,10)) {
        MTLFunctionConstantValues *values=[MTLFunctionConstantValues new];
        for(unsigned i=0;i<10;i++){bool bv=c[i].value!=0;int32_t iv=c[i].value;
            [values setConstantValue:c[i].is_bool?(const void*)&bv:(const void*)&iv type:c[i].is_bool?MTLDataTypeBool:MTLDataTypeInt atIndex:c[i].index];}
        NSError *error=nil;
        id<MTLFunction> fn=[golden_library newFunctionWithName:@"kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512" constantValues:values error:&error];
        if(!fn)fprintf(stderr,"Golden function error: %s\n",error.localizedDescription.UTF8String);REQUIRE(fn);
        id<MTLComputePipelineState> pso=[g_device newComputePipelineStateWithFunction:fn error:&error];
        if(!pso)fprintf(stderr,"Golden PSO error: %s\n",error.localizedDescription.UTF8String);REQUIRE(pso);
        specialized_pipeline_cache(cache_name,c,10,pso);
    }
    const myds_metal_buffer_t *b[]={f->q,s->kv_f16,s->kv_f16,s->mask,f->sinks,s->pad,f->out[output].view};
    REQUIRE(specialized(cache_name,&a,sizeof(a),b,7,64,1,1,256,1,1,24448,c,10));
}
static void golden_dispatch(struct fixture *f,myds_v41_gpu_scratch_t *s,unsigned output) {
    REQUIRE(myds_metal_begin_batch());golden_encode(f,s,output);REQUIRE(myds_metal_commit_batch());
}

static void dump_blob(const char *prefix,const char *kind,unsigned arm,struct guarded b) {
    if(!output_dir)return;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s-%s-%s.bin",output_dir,prefix,arm?"register":"tg",kind);
    REQUIRE(n>0&&(size_t)n<sizeof(path));FILE *file=fopen(path,"wb");REQUIRE(file);
    REQUIRE(fwrite(myds_metal_buffer_data(b.view),1,b.bytes,file)==b.bytes);REQUIRE(fclose(file)==0);
}
static void free_guard(struct guarded b){canary(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void bit_equal(struct guarded a,struct guarded b,const char *kind) {
    REQUIRE(a.bytes==b.bytes);canary(a);canary(b);
    const uint8_t *ap=myds_metal_buffer_data(a.view),*bp=myds_metal_buffer_data(b.view);
    if(memcmp(ap,bp,a.bytes)){
        for(size_t i=0;i<a.bytes;i++)if(ap[i]!=bp[i]){fprintf(stderr,"PARTIAL MISMATCH %s byte=%zu ref=%02x candidate=%02x\n",kind,i,ap[i],bp[i]);break;}
        REQUIRE(0);
    }
}
static int probe_dispatch(struct fixture *f,myds_v41_gpu_scratch_t *s,unsigned output,unsigned arm,int capture,struct guarded partial,struct guarded stats) {
    struct flash_vec_args a=vector_args(f->keys);myds_metal_constant_t c[10];constants(f->keys,c);
    const myds_metal_buffer_t *b[]={f->q,s->kv_f16,s->kv_f16,s->mask,f->sinks,s->pad,f->out[output].view,partial.view,stats.view};
    return specialized(probe_names[arm][!!capture],&a,sizeof(a),b,9,64,1,1,256,1,1,arm?22400:24448,c,10);
}
static void probe_numeric(uint32_t keys,uint32_t start,uint32_t mode,int direct) {
    struct fixture f;fixture_init(&f,keys,start,mode);
    myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);prepare_mask(s,&f);
    select_mode(8,0);REQUIRE(myds_metal_begin_batch());REQUIRE(call(&f,s,0,direct));REQUIRE(myds_metal_commit_batch());
    memcpy(myds_metal_buffer_data(f.out[4].view),myds_metal_buffer_data(f.out[0].view),f.out[0].bytes);
    golden_dispatch(&f,s,0);compare(&f,4); /* Actual production library vs private original symbol. */
    struct guarded partial[2]={allocate(PARTIAL_BYTES),allocate(PARTIAL_BYTES)},stats[2]={allocate(STAT_BYTES),allocate(STAT_BYTES)};
    for(unsigned arm=0;arm<2;arm++) {
        REQUIRE(myds_metal_begin_batch());REQUIRE(probe_dispatch(&f,s,arm+1,arm,1,partial[arm],stats[arm]));REQUIRE(myds_metal_commit_batch());compare(&f,arm+1);
        REQUIRE(myds_metal_begin_batch());REQUIRE(probe_dispatch(&f,s,arm+3,arm,0,partial[arm],stats[arm]));REQUIRE(myds_metal_commit_batch());compare(&f,arm+3);
    }
    bit_equal(partial[0],partial[1],"all 64x32x512 split value partials");bit_equal(stats[0],stats[1],"all S/M/sink_scale");
    char prefix[128];snprintf(prefix,sizeof(prefix),"keys%u-start%u-mode%u-%s-concurrent%d",keys,start,mode,direct?"direct":"staged",myds_metal_get_concurrent());
    dump_blob(prefix,"actual-production-golden-output",0,f.out[0]);
    for(unsigned arm=0;arm<2;arm++){dump_blob(prefix,"partial",arm,partial[arm]);dump_blob(prefix,"stats",arm,stats[arm]);dump_blob(prefix,"output",arm,f.out[arm+1]);}
    printf("{\"kind\":\"partial_parity\",\"case\":\"%s\",\"partial_bytes\":%u,\"stats_bytes\":%u,\"output_bytes\":131072,\"partial_fnv1a64\":\"%016" PRIx64 "\",\"stats_fnv1a64\":\"%016" PRIx64 "\",\"output_fnv1a64\":\"%016" PRIx64 "\",\"all_partial_and_stats_bytes_equal\":true,\"actual_golden_matches_private_original_symbol\":true,\"debug_and_uninstrumented_output_equal_to_production_packed8\":true,\"canaries\":true}\n",prefix,PARTIAL_BYTES,STAT_BYTES,fnv(myds_metal_buffer_data(partial[0].view),partial[0].bytes),fnv(myds_metal_buffer_data(stats[0].view),stats[0].bytes),fnv(myds_metal_buffer_data(f.out[0].view),f.out[0].bytes));
    completed_pairs++;for(unsigned arm=0;arm<2;arm++){free_guard(partial[arm]);free_guard(stats[arm]);}myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
static void probe_timing(uint32_t keys,unsigned arm,const char *arm_name) {
    struct fixture f;fixture_init(&f,keys,127,0);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);
    struct guarded partial=allocate(PARTIAL_BYTES),stats=allocate(STAT_BYTES);prepare_mask(s,&f);select_mode(8,0);
    REQUIRE(myds_metal_begin_batch());REQUIRE(call(&f,s,0,1));REQUIRE(myds_metal_commit_batch());golden_dispatch(&f,s,0);
    for(unsigned sample=0;sample<=samples;sample++) {
        REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned rep=0;rep<repetitions;rep++){if(!arm)golden_encode(&f,s,1);else REQUIRE(probe_dispatch(&f,s,1,arm,0,partial,stats));myds_metal_barrier();}
        REQUIRE(myds_metal_commit_batch());compare(&f,1);
        REQUIRE(cb.status==MTLCommandBufferStatusCompleted&&isfinite(cb.GPUStartTime)&&cb.GPUEndTime>cb.GPUStartTime);
        printf("{\"kind\":\"gpu_timing\",\"arm\":\"%s\",\"kernel\":\"%s\",\"keys\":%u,\"warmup\":%s,\"sample\":%u,\"repetitions\":%u,\"gpu_us_per_call\":%.9f,\"dynamic_bytes\":%u,\"scope\":\"uninstrumented_packed_kernel_plus_repeat_barriers\",\"full_output_bit_exact\":true}\n",arm_name,arm?probe_names[arm][0]:"kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512",keys,sample?"false":"true",sample,repetitions,(cb.GPUEndTime-cb.GPUStartTime)*1e6/repetitions,arm?22400u:24448u);
    }
    free_guard(partial);free_guard(stats);myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
int main(int argc,char **argv) {
    int host_only=0,timing_enabled=1,quick=0;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--host-only"))host_only=1;
        else if(!strcmp(argv[i],"--no-timing"))timing_enabled=0;
        else if(!strcmp(argv[i],"--quick"))quick=1;
        else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)output_dir=argv[++i];
        else if(!strcmp(argv[i],"--golden-metallib")&&i+1<argc)golden_path=argv[++i];
        else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader_dir=argv[++i];
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=parse_count(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)repetitions=parse_count(argv[++i]);
        else {fprintf(stderr,"usage: %s [--host-only] [--quick] [--no-timing] [--golden-metallib actual-production.metallib] [--shader-dir private-lib-dir] [--output-dir NEW] [--samples N] [--repetitions N]\n",argv[0]);return 2;}
    }
    host_configuration();if(host_only){puts("{\"kind\":\"probe_host_only\",\"gpu_submitted\":false}");return 0;}
    if(!golden_path){fprintf(stderr,"--golden-metallib must name actual production library\n");return 2;}
    if(output_dir){struct stat st;errno=0;REQUIRE(lstat(output_dir,&st)!=0&&errno==ENOENT);REQUIRE(mkdir(output_dir,0700)==0);}
    setenv("MYDS_ATTN_VEC_HEAD_FIRST","0",1);setenv("MYDS_DEBUG_DISPATCH","0",1);setenv("MYDS_METAL_CONCURRENT","0",1);g_hist_enabled=0;
    REQUIRE(myds_metal_init(shader_dir));
    NSError *golden_error=nil;golden_library=[g_device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:golden_path]] error:&golden_error];
    if(!golden_library)fprintf(stderr,"Golden library error: %s\n",golden_error.localizedDescription.UTF8String);REQUIRE(golden_library);
    printf("{\"kind\":\"golden_library\",\"path\":\"%s\",\"actual_production_symbol\":\"kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512\"}\n",golden_path);
    printf("{\"kind\":\"identity\",\"device\":\"%s\",\"fixture_not_fullmodel\":true,\"shared_memory_bytes\":[24448,22400],\"logical_splits\":32,\"physical_simdgroups\":8,\"remaining_threadgroup_barrier_sites\":4}\n",g_device.name.UTF8String);
    const uint32_t keys[]={1,31,32,128,512,640};
    for(unsigned concurrent=0;concurrent<2;concurrent++) {
        myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(1);
        for(unsigned k=0;k<6;k++) {if(quick&&keys[k]!=128&&keys[k]!=640)continue;
            for(unsigned start=0;start<(quick?1u:2u);start++)for(unsigned mode=0;mode<5;mode++)for(unsigned direct=0;direct<(quick?1u:2u);direct++)probe_numeric(keys[k],start?0:127,mode,quick?1:direct);
        }
    }
    if(timing_enabled){myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);
        for(unsigned k=0;k<6;k++){if(quick&&keys[k]!=128&&keys[k]!=640)continue;probe_timing(keys[k],0,"A1");probe_timing(keys[k],1,"B");probe_timing(keys[k],0,"A2");}
    }
    myds_metal_cleanup();printf("{\"kind\":\"result\",\"passed\":true,\"full_partial_parity_pairs\":%u,\"full_output_parity_pairs\":%u,\"fixture_not_fullmodel\":true}\n",completed_pairs,completed_pairs*4u);return 0;
}
