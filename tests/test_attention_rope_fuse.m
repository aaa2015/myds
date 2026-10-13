/* Exclusive GPU component fixture. --host-only never initializes Metal.
 * Reuses only input allocation/generation and ABI helpers from the old fixture.
 * All timed controls use actual production register + rope_round PSOs loaded
 * from --golden-metallib; private debug taps must first match those controls.
 * This compile-only private allowance is absent from production builds. */
#define MYDS_ATTN_ROPE_FUSE_PRIVATE_PROBE 1
#define main attention_packed_unused_main
#include "test_attention_packed.m"
#undef main
#include <CommonCrypto/CommonDigest.h>
static NSString *rf_directory;
static NSMutableArray *rf_cases,*rf_timings,*rf_rejections;
static id<MTLLibrary> rf_golden_library;
static const char *rf_golden,*rf_source;
static const char *rf_binary;
static bool rf_quick,rf_timing;
static int rf_diagnostic_case=-1;
static NSMutableArray *rf_diagnostics;
static NSDictionary *rf_failure;
static NSString *rf_sha(NSData *data) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data.bytes,(CC_LONG)data.length,hash);
    NSMutableString *text=[NSMutableString string];for(unsigned i=0;i<sizeof(hash);i++)[text appendFormat:@"%02x",hash[i]];return text;
}
static void rf_json(NSDictionary *d) {
    NSData *data=[NSJSONSerialization dataWithJSONObject:d options:0 error:nil];REQUIRE(data);
    fwrite(data.bytes,1,data.length,stdout);fputc('\n',stdout);
}
static void rf_save(NSString *name,const void *bytes,size_t count) {
    REQUIRE([[NSData dataWithBytes:bytes length:count] writeToFile:[rf_directory stringByAppendingPathComponent:name] atomically:NO]);
}
static void rf_guarded_save(NSString *name,struct guarded b) {
    /* Persist before validating: a failed numerical/canary gate must retain
     * the complete original payload, rather than terminate before writing. */
    rf_save(name,myds_metal_buffer_data(b.owner),b.bytes+32u);
}
static bool rf_valid(struct guarded b,bool bf16) {
    const uint8_t *raw=myds_metal_buffer_data(b.owner);
    for(unsigned i=0;i<16;i++)if(raw[i]!=0xa5||raw[b.bytes+16u+i]!=0xa5)return false;
    const uint32_t *u=myds_metal_buffer_data(b.view);
    for(size_t i=0;i<b.bytes/4u;i++){float v;memcpy(&v,u+i,4);if(!isfinite(v)||u[i]==0xa5a5a5a5u||(bf16&&(u[i]&0xffffu)))return false;}
    return true;
}
static void rf_validate(struct guarded b,bool bf16) {
    REQUIRE(rf_valid(b,bf16));
}
static NSDictionary *rf_diff(const void *reference,const void *candidate,size_t bytes) {
    const uint32_t *a=reference,*b=candidate;size_t count=0,first=SIZE_MAX;
    for(size_t i=0;i<bytes/4u;i++)if(a[i]!=b[i]){if(first==SIZE_MAX)first=i;count++;}
    return @{@"same":@(count==0),@"different_words":@(count),@"first_index":first==SIZE_MAX?@(-1):@(first),
        @"reference_bits":first==SIZE_MAX?@"none":[NSString stringWithFormat:@"%08x",a[first]],
        @"candidate_bits":first==SIZE_MAX?@"none":[NSString stringWithFormat:@"%08x",b[first]]};
}
static void rf_checkpoint(NSString *status) {
    NSMutableDictionary *files=[NSMutableDictionary dictionary];
    for(NSString *name in [[NSFileManager defaultManager] contentsOfDirectoryAtPath:rf_directory error:nil]) {
        if([name isEqualToString:@"artifact-manifest.json"])continue;
        NSData *b=[NSData dataWithContentsOfFile:[rf_directory stringByAppendingPathComponent:name]];REQUIRE(b);files[name]=@{@"bytes":@(b.length),@"sha256":rf_sha(b)};
    }
    NSData *gold=[NSData dataWithContentsOfFile:@(rf_golden)],*source=[NSData dataWithContentsOfFile:@(rf_source)],*binary=[NSData dataWithContentsOfFile:@(rf_binary)];REQUIRE(gold&&source&&binary);
    NSData *lib=[NSData dataWithContentsOfFile:[@(shader_dir) stringByAppendingPathComponent:@"myds.metallib"]];REQUIRE(lib);
    NSMutableDictionary *manifest=[@{@"schema_version":@1,@"status":status,@"files":files,@"cases":rf_cases,@"timings":rf_timings,@"rejections":rf_rejections,
        @"diagnostics":rf_diagnostics,@"diagnostic_case":@(rf_diagnostic_case),@"diagnostic_timings_accepted":@NO,
        @"samples":@(samples),@"repetitions":@(repetitions),@"quick":@(rf_quick),@"timing_enabled":@(rf_timing),
        @"golden_metallib_sha256":rf_sha(gold),@"candidate_metallib_sha256":rf_sha(lib),@"binary_sha256":rf_sha(binary),@"source_identity_sha256":rf_sha(source),
        @"activation_source":@"seeded BF16-representable F32 fixture",@"fullmodel_speedup_proven":@NO,@"gpu_physical_overlap_proven":@NO} mutableCopy];
    if(rf_failure)manifest[@"failure"]=rf_failure;
    NSData *data=[NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];REQUIRE(data);
    REQUIRE([data writeToFile:[rf_directory stringByAppendingPathComponent:@"artifact-manifest.json"] atomically:YES]);
}
static void rf_host(void) {
    setenv("MYDS_ATTN_PACKED_REGISTER","1",1);unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");
    unsetenv("MYDS_ATTN_ROPE_FUSE");REQUIRE(myds_v41_attention_rope_fuse_enabled()==0);
    setenv("MYDS_ATTN_ROPE_FUSE","0",1);REQUIRE(myds_v41_attention_rope_fuse_enabled()==0);
    setenv("MYDS_ATTN_ROPE_FUSE","1",1);REQUIRE(myds_v41_attention_rope_fuse_enabled()==1);
    const char *bad[]={"","2","01","true","1 ","-1"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++){setenv("MYDS_ATTN_ROPE_FUSE",bad[i],1);REQUIRE(myds_v41_attention_rope_fuse_enabled()==-1);}
    setenv("MYDS_ATTN_ROPE_FUSE","1",1);setenv("MYDS_ATTN_PACKED_REGISTER","0",1);REQUIRE(myds_v41_attention_rope_fuse_enabled()==-1);
    setenv("MYDS_ATTN_PACKED_REGISTER","1",1);myds_v41_attention_rope_fuse_counters(NULL,true);
    const uint64_t before=myds_metal_dispatch_count();bool consumed=true;
    REQUIRE(!myds_v41_gpu_attention_rope_fuse(NULL,NULL,NULL,NULL,1,0,NULL,0,NULL,0,false,true,false,&consumed));
    REQUIRE(!consumed && myds_metal_dispatch_count()==before);
    myds_metal_decode_candidate_counters_t count;myds_v41_attention_rope_fuse_counters(&count,false);
    REQUIRE(count.requests==1 && count.encoded==0 && count.failures==1);
    puts("{\"kind\":\"host_contract\",\"cpu_only\":true,\"strict_flag\":true,\"invalid_null_dispatch_delta\":0,\"args_bytes\":336}");
}
static void rf_bind_golden(void) {
    NSError *error=nil;id<MTLLibrary> library=[g_device newLibraryWithFile:@(rf_golden) error:&error];REQUIRE(library);
    rf_golden_library=library;
    const char *name="kernel_myds_flash_attn_vec_packed8_register_reduce_f16_dk512_dv512";
    for(unsigned pad=0;pad<2;pad++) {
        myds_metal_constant_t c[10];constants(pad?1:32,c);
        MTLFunctionConstantValues *values=[MTLFunctionConstantValues new];
        for(unsigned i=0;i<10;i++){bool b=c[i].value!=0;int32_t v=c[i].value;[values setConstantValue:c[i].is_bool?(const void *)&b:(const void *)&v type:c[i].is_bool?MTLDataTypeBool:MTLDataTypeInt atIndex:c[i].index];}
        id<MTLFunction> fn=[library newFunctionWithName:@(name) constantValues:values error:&error];REQUIRE(fn);
        id<MTLComputePipelineState> p=[g_device newComputePipelineStateWithFunction:fn error:&error];REQUIRE(p && p.maxTotalThreadsPerThreadgroup>=256);
        /* This reference name is 66 bytes, beyond the small 64-byte native
         * cache. Bind its exact ordered-constant dictionary key as well. */
        if(!g_native_pipelines)g_native_pipelines=[NSMutableDictionary new];
        NSMutableString *key=[@(name) mutableCopy];
        for(unsigned i=0;i<10;i++)[key appendFormat:@":%u:%d:%d",c[i].index,c[i].is_bool,c[i].value];
        g_native_pipelines[key]=p;
        specialized_pipeline_cache(name,c,10,p);
    }
    const char *rope="kernel_dsv41_rope_round";
    id<MTLFunction> fn=[library newFunctionWithName:@(rope)];REQUIRE(fn);
    id<MTLComputePipelineState> p=[g_device newComputePipelineStateWithFunction:fn error:&error];REQUIRE(p);
    if(!g_native_pipelines)g_native_pipelines=[NSMutableDictionary new];
    g_native_pipelines[@(rope)]=p;
    for(size_t i=0;i<g_native_pso_cache_count;i++)if(!strcmp(g_native_pso_cache[i].name,rope))g_native_pso_cache[i].pso=p;
}
static void rf_negative(unsigned kind) {
    struct fixture f;fixture_init(&f,129,127,0);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);prepare_mask(s,&f);
    const myds_metal_buffer_t *observe[]={f.q,f.window,f.sinks,f.comp,f.ids,f.gathered,f.out[3].owner,s->kv_f16,s->pad,s->partial,s->mask};
    NSMutableArray *snapshots=[NSMutableArray array];size_t lengths[11];
    for(unsigned i=0;i<11;i++){lengths[i]=myds_metal_buffer_size(observe[i]);[snapshots addObject:[NSData dataWithBytes:myds_metal_buffer_data((myds_metal_buffer_t *)observe[i]) length:lengths[i]]];}
    id<MTLLibrary> saved_library=g_library;NSMutableDictionary *saved_pipelines=g_native_pipelines;
    size_t saved_specialized_count=g_specialized_pso_cache_count,saved_native_count=g_native_pso_cache_count;
    myds_metal_buffer_t *saved_out=f.out[3].view,*saved_kv=s->kv_f16;
    size_t saved_qsize=f.q->size;void *saved_qptr=f.q->ptr;uint32_t pos=11,nraw=f.nraw,start=f.start,selected=f.selected;
    bool consumed=true;bool *consume=&consumed;
    const char *names[]={"selected_PSO_absent_in_real_golden_library","pos_limit","short_q","heads_alias_q","scratch_writers_alias","null_consumed","nraw_zero","nraw_too_large","raw_start_limit","selected_too_large","forged_cpu_pointer"};
    if(kind==0){g_library=rf_golden_library;g_native_pipelines=[NSMutableDictionary dictionary];g_specialized_pso_cache_count=0;g_native_pso_cache_count=0;}
    if(kind==1)pos=1048576;
    if(kind==2)f.q->size=4;
    if(kind==3)f.out[3].view=f.q;
    if(kind==4)s->kv_f16=s->pad;
    if(kind==5)consume=NULL;
    if(kind==6)nraw=0;
    if(kind==7)nraw=129;
    if(kind==8)start=128;
    if(kind==9)selected=513;
    if(kind==10)f.q->ptr=f.window->ptr;
    const uint64_t dispatch_before=myds_metal_dispatch_count();myds_metal_decode_candidate_counters_t before,after;
    myds_v41_attention_rope_fuse_counters(&before,false);
    REQUIRE(!myds_v41_gpu_attention_direct_rope_fuse(f.out[3].view,f.sinks,f.q,f.window,nraw,start,f.comp,f.ids,f.source,selected,s,pos,false,true,false,consume));
    myds_v41_attention_rope_fuse_counters(&after,false);
    REQUIRE(myds_metal_dispatch_count()==dispatch_before && after.requests==before.requests+1 && after.encoded==before.encoded && after.failures==before.failures+1 && (!consume||!consumed));
    f.out[3].view=saved_out;s->kv_f16=saved_kv;f.q->size=saved_qsize;f.q->ptr=saved_qptr;
    g_library=saved_library;g_native_pipelines=saved_pipelines;g_specialized_pso_cache_count=saved_specialized_count;g_native_pso_cache_count=saved_native_count;
    for(unsigned i=0;i<11;i++) {
        REQUIRE(!memcmp([snapshots[i] bytes],myds_metal_buffer_data((myds_metal_buffer_t *)observe[i]),lengths[i]));
        rf_save([NSString stringWithFormat:@"reject%u-buffer%u-before.raw",kind,i],[snapshots[i] bytes],lengths[i]);
        rf_save([NSString stringWithFormat:@"reject%u-buffer%u-after.raw",kind,i],myds_metal_buffer_data((myds_metal_buffer_t *)observe[i]),lengths[i]);
    }
    NSDictionary *r=@{@"kind":@"rejection",@"case":@(names[kind]),@"dispatch_delta":@0,@"encoded_delta":@0,@"requests_delta":@1,@"failures_delta":@1,@"all_observed_buffers_full_bytes_unchanged":@YES};[rf_rejections addObject:r];rf_json(r);
    myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
static int rf_call(struct fixture *f,myds_v41_gpu_scratch_t *s,unsigned out,bool direct,bool fused,uint32_t pos,bool compressed) {
    if(!fused) {
        if(!call(f,s,out,direct))return 0;
        myds_metal_barrier_buffer(f->out[out].view);
        if(!myds_v41_gpu_rope_round(f->out[out].view,512,64,pos,compressed,true))return 0;
    } else {
        bool consumed=false;
        int ok=direct?myds_v41_gpu_attention_direct_rope_fuse(f->out[out].view,f->sinks,f->q,f->window,f->nraw,f->start,
            f->comp,f->ids,f->source,f->selected,s,pos,compressed,true,false,&consumed):
            myds_v41_gpu_attention_rope_fuse(f->out[out].view,f->sinks,f->q,f->window,f->nraw,f->start,
            f->gathered,f->selected,s,pos,compressed,true,false,&consumed);
        if(!ok || !consumed)return 0;
    }
    myds_metal_barrier_buffer(f->out[out].view);return 1;
}
static void rf_debug(struct fixture *f,myds_v41_gpu_scratch_t *s,unsigned out,bool fused,uint32_t pos,bool compressed,
    struct guarded partial,struct guarded stats) {
    struct attention_rope_fuse_args args={.attention=vector_args(f->keys),.pos=pos};
    memcpy(args.frequencies,rope_frequencies[compressed?1:0],sizeof(args.frequencies));
    myds_metal_constant_t c[10];constants(f->keys,c);
    const myds_metal_buffer_t *b[]={f->q,s->kv_f16,s->kv_f16,s->mask,f->sinks,s->pad,f->out[out].view,partial.view,stats.view};
    REQUIRE(specialized(fused?"probe_attention_rope_fused":"probe_attention_rope_reference",&args,sizeof(args),b,9,64,1,1,256,1,1,22400,c,10));
}
static void rf_case(unsigned number,uint32_t keys,uint32_t pos,unsigned mask,bool compressed,bool direct,bool timing_enabled) {
    struct fixture f;fixture_init(&f,keys,127,mask);myds_v41_gpu_scratch_t *s=myds_v41_gpu_scratch_create(2048);REQUIRE(s);prepare_mask(s,&f);
    struct guarded partial[2]={allocate(64u*32u*512u*4u),allocate(64u*32u*512u*4u)};
    struct guarded stats[2]={allocate(64u*32u*3u*4u),allocate(64u*32u*3u*4u)};
    REQUIRE(myds_metal_begin_batch());REQUIRE(call(&f,s,0,direct));REQUIRE(myds_metal_commit_batch());
    rf_guarded_save([NSString stringWithFormat:@"case%u-production-unrotated.guarded",number],f.out[0]);
    NSData *unrotated=[NSData dataWithBytes:myds_metal_buffer_data(f.out[0].view) length:f.out[0].bytes];
    /* Run the real production fused PSO before any private debug candidate or
     * comparison. A debug failure must never prevent its payload capture. */
    myds_metal_decode_candidate_counters_t before,after;myds_v41_attention_rope_fuse_counters(&before,false);
    REQUIRE(myds_metal_begin_batch());REQUIRE(rf_call(&f,s,3,direct,true,pos,compressed));REQUIRE(myds_metal_commit_batch());
    myds_v41_attention_rope_fuse_counters(&after,false);
    REQUIRE(myds_metal_begin_batch());myds_metal_barrier_buffer(f.out[0].view);REQUIRE(myds_v41_gpu_rope_round(f.out[0].view,512,64,pos,compressed,true));myds_metal_barrier_buffer(f.out[0].view);REQUIRE(myds_metal_commit_batch());
    REQUIRE(myds_metal_begin_batch());rf_debug(&f,s,1,false,pos,compressed,partial[0],stats[0]);REQUIRE(myds_metal_commit_batch());
    REQUIRE(myds_metal_begin_batch());rf_debug(&f,s,2,true,pos,compressed,partial[1],stats[1]);REQUIRE(myds_metal_commit_batch());
    NSString *prefix=[NSString stringWithFormat:@"case%u",number];
    for(unsigned i=0;i<4;i++)rf_guarded_save([prefix stringByAppendingFormat:@"-out%u.guarded",i],f.out[i]);
    for(unsigned i=0;i<2;i++){rf_guarded_save([prefix stringByAppendingFormat:@"-partial%u.guarded",i],partial[i]);rf_guarded_save([prefix stringByAppendingFormat:@"-stats%u.guarded",i],stats[i]);}
    rf_save([prefix stringByAppendingString:@"-q.f32"],myds_metal_buffer_data(f.q),64u*512u*4u);
    rf_save([prefix stringByAppendingString:@"-window.f32"],myds_metal_buffer_data(f.window),128u*512u*4u);
    rf_save([prefix stringByAppendingString:@"-sinks.f32"],myds_metal_buffer_data(f.sinks),64u*4u);
    rf_save([prefix stringByAppendingString:@"-mask.f16"],myds_metal_buffer_data(s->mask),640u*2u);
    rf_save([prefix stringByAppendingString:@"-frequencies.f32"],rope_frequencies[compressed?1:0],128u);
    if(f.selected){rf_save([prefix stringByAppendingString:@"-comp.bf16"],myds_metal_buffer_data(f.comp),myds_metal_buffer_size(f.comp));rf_save([prefix stringByAppendingString:@"-ids.i32"],myds_metal_buffer_data(f.ids),myds_metal_buffer_size(f.ids));rf_save([prefix stringByAppendingString:@"-gathered.f32"],myds_metal_buffer_data(f.gathered),myds_metal_buffer_size(f.gathered));}
    NSDictionary *reference_debug=rf_diff(unrotated.bytes,myds_metal_buffer_data(f.out[1].view),f.out[1].bytes);
    NSDictionary *production=rf_diff(myds_metal_buffer_data(f.out[0].view),myds_metal_buffer_data(f.out[3].view),f.out[0].bytes);
    NSDictionary *debug=rf_diff(myds_metal_buffer_data(f.out[0].view),myds_metal_buffer_data(f.out[2].view),f.out[0].bytes);
    NSDictionary *partial_diff=rf_diff(myds_metal_buffer_data(partial[0].view),myds_metal_buffer_data(partial[1].view),partial[0].bytes);
    NSDictionary *stats_diff=rf_diff(myds_metal_buffer_data(stats[0].view),myds_metal_buffer_data(stats[1].view),stats[0].bytes);
    bool all_valid=true;
    for(unsigned i=0;i<4;i++)all_valid=all_valid&&rf_valid(f.out[i],i!=1);
    for(unsigned i=0;i<2;i++)all_valid=all_valid&&rf_valid(partial[i],false)&&rf_valid(stats[i],false);
    const bool exact=all_valid&&[reference_debug[@"same"] boolValue]&&[production[@"same"] boolValue]&&[debug[@"same"] boolValue]&&[partial_diff[@"same"] boolValue]&&[stats_diff[@"same"] boolValue];
    NSDictionary *diagnostic=@{@"kind":@"diagnostic_case",@"case":@(number),@"keys":@(keys),@"pos":@(pos),@"mask_mode":@(mask),@"mask_description":mask==4?@"unmasked_no_effective_sink":@"prepare_mask_mode",
        @"compressed":@(compressed),@"direct":@(direct),@"actual_production_fused_ran_before_debug":@YES,@"reference_debug_unrotated":reference_debug,@"production_fused":production,@"debug_fused":debug,@"partial":partial_diff,@"stats":stats_diff,
        @"case_exact_gate_passed":@(exact),@"candidate_accepted":@NO,@"guards_finite_poison_bf16_valid":@(all_valid),@"diagnostic_timings_accepted":@NO};
    [rf_diagnostics addObject:diagnostic];rf_json(diagnostic);
    if(!exact)rf_failure=diagnostic;
    rf_checkpoint(exact?@"in_progress":@"failed_parity");
    /* All complete raw values and a failure manifest exist before these gates. */
    REQUIRE(exact);
    REQUIRE(after.requests==before.requests+1 && after.encoded==before.encoded+1 && after.failures==before.failures);
    for(unsigned i=0;i<4;i++)rf_validate(f.out[i],i!=1);
    for(unsigned i=0;i<2;i++){rf_validate(partial[i],false);rf_validate(stats[i],false);}
    NSDictionary *record=@{@"kind":@"parity",@"case":@(number),@"keys":@(keys),@"n_raw":@(f.nraw),@"raw_start":@(f.start),@"selected":@(f.selected),@"source_rows":@(f.source),@"pos":@(pos),@"mask_mode":@(mask),@"compressed":@(compressed),@"direct":@(direct),@"partial_stats_full_bytes_same":@YES,@"full_output_same":@YES,@"guards":@YES,@"finite":@YES,@"production_golden":@YES};
    [rf_cases addObject:record];rf_json(record);
    if(timing_enabled)for(unsigned arm=0;arm<3;arm++)for(unsigned sample=0;sample<=samples;sample++) {
        uint64_t dispatch_before=myds_metal_dispatch_count();myds_v41_attention_rope_fuse_counters(&before,false);
        REQUIRE(myds_metal_begin_batch());id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned r=0;r<repetitions;r++)REQUIRE(rf_call(&f,s,4,direct,arm==1,pos,compressed));
        REQUIRE(myds_metal_commit_batch());myds_v41_attention_rope_fuse_counters(&after,false);
        REQUIRE(cb.status==MTLCommandBufferStatusCompleted && isfinite(cb.GPUStartTime) && cb.GPUEndTime>cb.GPUStartTime);
        REQUIRE(after.encoded-before.encoded==(arm==1?repetitions:0) && after.requests-before.requests==(arm==1?repetitions:0) && after.failures==before.failures);
        NSString *label=@[@"A1",@"B",@"A2"][arm];double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/repetitions;REQUIRE(isfinite(us)&&us>0);
        rf_guarded_save([prefix stringByAppendingFormat:@"-timing-%@-s%u.guarded",label,sample],f.out[4]);
        rf_checkpoint(@"in_progress");compare(&f,4);rf_validate(f.out[4],true);
        NSDictionary *t=@{@"kind":@"timing",@"case":@(number),@"arm":label,@"sample":@(sample),@"warmup":@(sample==0),@"repetitions":@(repetitions),@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"gpu_us":@(us),@"dispatches":@(myds_metal_dispatch_count()-dispatch_before),@"fused_encoded":@(after.encoded-before.encoded),@"full_output_same":@YES,@"guards":@YES,@"finite":@YES,@"scope":@"complete attention staging+reduce+inverse-rope pair"};[rf_timings addObject:t];rf_json(t);
    }
    for(unsigned i=0;i<2;i++){canary(partial[i]);canary(stats[i]);myds_metal_buffer_free(partial[i].view);myds_metal_buffer_free(partial[i].owner);myds_metal_buffer_free(stats[i].view);myds_metal_buffer_free(stats[i].owner);}
    myds_v41_gpu_scratch_free(s);fixture_free(&f);
}
int main(int argc,char **argv) {
    bool host=false,quick=false,time=true;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--host-only"))host=true;
        else if(!strcmp(argv[i],"--quick"))quick=true;
        else if(!strcmp(argv[i],"--no-timing"))time=false;
        else if(!strcmp(argv[i],"--diagnostic-case")&&i+1<argc){char *end=NULL;const char *text=argv[++i];unsigned long n=strtoul(text,&end,10);REQUIRE(end&&end!=text&&!*end&&n<120u);rf_diagnostic_case=(int)n;}
        else if(!strcmp(argv[i],"--shader-dir")&&i+1<argc)shader_dir=argv[++i];
        else if(!strcmp(argv[i],"--golden-metallib")&&i+1<argc)rf_golden=argv[++i];
        else if(!strcmp(argv[i],"--source-identity")&&i+1<argc)rf_source=argv[++i];
        else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)rf_directory=@(argv[++i]);
        else if(!strcmp(argv[i],"--samples")&&i+1<argc)samples=parse_count(argv[++i]);
        else if(!strcmp(argv[i],"--repetitions")&&i+1<argc)repetitions=parse_count(argv[++i]);
        else REQUIRE(0);
    }
    rf_host();if(host)return 0;if(rf_diagnostic_case>=0){quick=false;time=false;}rf_binary=argv[0];rf_quick=quick;rf_timing=time;
    REQUIRE(rf_directory && rf_golden && rf_source && samples>=3 && samples<=31 && repetitions<=64);
    REQUIRE(![[NSFileManager defaultManager] fileExistsAtPath:rf_directory]);REQUIRE([[NSFileManager defaultManager] createDirectoryAtPath:rf_directory withIntermediateDirectories:NO attributes:nil error:nil]);
    setenv("MYDS_ATTN_ROPE_FUSE","1",1);setenv("MYDS_ATTN_PACKED_REGISTER","1",1);unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");
    REQUIRE(myds_metal_init(shader_dir));myds_metal_set_concurrent(1);myds_metal_set_auto_barrier(0);rf_bind_golden();
    rf_cases=[NSMutableArray array];rf_timings=[NSMutableArray array];rf_rejections=[NSMutableArray array];rf_diagnostics=[NSMutableArray array];
    for(unsigned i=0;i<11;i++)rf_negative(i);
    const uint32_t positions[]={0,11,127,128,8191,32768},keys[]={1,33,128,129,639,640};
    unsigned number=0;
    for(unsigned p=0;p<(quick?2u:6u);p++)for(unsigned compressed=0;compressed<2;compressed++)for(unsigned direct=0;direct<2;direct++)for(unsigned mask=0;mask<(quick?2u:5u);mask++){if(rf_diagnostic_case<0||number==(unsigned)rf_diagnostic_case)rf_case(number,keys[p],positions[p],mask,compressed,direct,time);number++;}
    if(rf_diagnostic_case>=0){rf_checkpoint(@"diagnostic_complete");myds_metal_cleanup();return 0;}
    rf_checkpoint(@"complete");
    NSDictionary *final=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[rf_directory stringByAppendingPathComponent:@"artifact-manifest.json"]] options:0 error:nil];REQUIRE(final);
    rf_json(@{@"kind":@"complete",@"pass":@YES,@"cases":@(number),@"raw_files":@([final[@"files"] count]),@"fullmodel_speedup_proven":@NO});myds_metal_cleanup();return 0;
}
