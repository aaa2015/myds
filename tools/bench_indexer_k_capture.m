/* Full real-input replay of exact/TILED/K-resident indexer scores.
 * Compile only until explicitly run. --host-only validates all payloads and
 * native argument bounds without creating a device or queue. No service use.
 * This is a component gate, never a full-model/TopK/mask acceptance claim. */
#define main indexer_capture_attention_helpers_unused_main
#include "bench_attention_multihead.m"
#undef main
#include "myds_v41_gpu.h"
#include "myds_configfile.h"

struct real_index_args {
    uint32_t n_comp,n_tokens,n_head,head_dim,pos0,ratio;
    uint64_t q_token_stride,q_head_stride,weights_token_stride,index_row_stride,score_token_stride;
    float scale;
};
struct real_exact_args {uint32_t rows,heads,dim,cache_type;float scale;
    uint32_t tokens,pos,ratio,window;};
_Static_assert(sizeof(struct real_index_args)==72,"indexer tiled ABI");
_Static_assert(sizeof(struct real_exact_args)==36,"indexer exact ABI");
typedef struct {
    struct real_index_args args;
    unsigned layer,bf16;uint32_t grid[2];size_t smem;
    NSString *provider;
    NSData *native_args,*input[3],*captured,*captured_ids,*reference_tiled;
    NSDictionary *captured_topk;
    id<MTLBuffer> buffers[3],out[3];
} real_index_fixture;
static NSData *capture_payload(NSDictionary *entry,NSString *capture_root) {
    need([entry isKindOfClass:[NSDictionary class]],"capture payload descriptor");
    NSString *relative=entry[@"path"];
    need([relative isKindOfClass:[NSString class]]&&relative.length&&![relative isAbsolutePath]&&
         relative.pathComponents.count==1&&![@[@".",@".."]containsObject:relative],"local capture filename");
    NSData *d=read_file([capture_root stringByAppendingPathComponent:relative]);
    need(d.length==json_uint(entry[@"bytes"],SIZE_MAX,"payload bytes")&&
         [sha(d.bytes,d.length)isEqual:entry[@"sha256"]],"complete capture payload size/SHA");return d;
}
static real_index_fixture read_index_case(NSDictionary *c,NSString *root) {
    real_index_fixture f={0};NSDictionary *p=c[@"payloads"];
    NSData *a=capture_payload(p[@"args72"],root);need(a.length==72,"real args72");memcpy(&f.args,a.bytes,72);
    struct real_index_args *x=&f.args;
    need(x->n_comp>=15000&&x->n_comp<=16384&&x->n_tokens>=8&&x->n_tokens<=2048&&
         (uint64_t)x->n_comp*x->n_tokens<=2097152&&x->n_head==32&&x->head_dim==128&&
         x->pos0==30720&&x->ratio==2&&x->q_token_stride==16384&&x->q_head_stride==512&&
         x->weights_token_stride==128&&x->score_token_stride==(uint64_t)x->n_comp*4&&x->scale==1.f/64.f,
         "bounded real subgroup and dense strides");
    f.layer=(unsigned)json_uint(c[@"layer"],39,"real layer");
    need(f.layer==2||f.layer==8||f.layer==14,"real owner/source layer");
    f.bf16=[c[@"cache_bf16"]boolValue];need(x->index_row_stride==(f.bf16?256u:512u),"real cache stride");
    f.provider=c[@"provider"];
    BOOL tiled=[f.provider isEqual:@"kernel_dsv4_indexer_scores_tiled_bf16"]||
               [f.provider isEqual:@"kernel_dsv4_indexer_scores_tiled_f32"];
    BOOL exact=[f.provider isEqual:@"kernel_myds_indexer_score_chunk_exact"];
    BOOL direct=[f.provider isEqual:@"kernel_glm_indexer_score_chunk_direct"];
    need(tiled||exact||direct,"captured actual production provider");
    f.native_args=capture_payload(p[@"native-args"],root);
    if(tiled)need(f.native_args.length==72&&!memcmp(f.native_args.bytes,a.bytes,68),"native TILED args match canonical");
    else {struct real_exact_args e={x->n_comp,32,128,f.bf16?2u:0u,1.f/64.f,x->n_tokens,30720,2,0};
        need(f.native_args.length==36&&!memcmp(f.native_args.bytes,&e,36),"native exact/direct args match canonical");}
    f.grid[0]=direct?x->n_comp:tiled?(x->n_comp+31)/32:(x->n_comp+15)/16;
    f.grid[1]=direct?x->n_tokens:(x->n_tokens+7)/8;f.smem=tiled?21504:exact?5120:528;
    need(json_uint(c[@"grid"][0],UINT32_MAX,"actual gx")==f.grid[0]&&
         json_uint(c[@"grid"][1],UINT32_MAX,"actual gy")==f.grid[1]&&
         json_uint(c[@"dynamic_threadgroup_bytes"],32768,"actual smem")==f.smem,"actual dispatch geometry");
    NSString *names[]={@"q-f32",@"weights-f32",@"cache"};
    uint64_t bytes[]={(uint64_t)x->n_tokens*16384,(uint64_t)x->n_tokens*128,(uint64_t)x->n_comp*x->index_row_stride};
    for(unsigned i=0;i<3;i++){f.input[i]=capture_payload(p[names[i]],root);need(f.input[i].length==bytes[i],"full real input extent");}
    f.captured=capture_payload(p[@"scores-f32"],root);
    need(f.captured.length==(uint64_t)x->n_tokens*x->n_comp*4,"complete real scores");
    f.captured_ids=capture_payload(p[@"selected-i32"],root);
    need(f.captured_ids.length==(uint64_t)x->n_tokens*512u*4u,"complete real ordered TopK IDs");
    f.captured_topk=c[@"actual_topk_consumer"];
    need([f.captured_topk[@"ordered_ids_complete"]boolValue]&&[f.captured_topk[@"top"]unsignedIntValue]==512,
         "real TopK consumer provenance");return f;
}
/* Persist independent post-read witnesses for every original shader input.
 * Capture payloads provide the pre-image; both guard sides and the complete
 * current body are emitted even when equality/guard checks fail. */
static BOOL real_index_inputs(real_index_fixture *f,NSString *phase) {
    NSString *names[]={@"q-f32",@"weights-f32",@"cache"};BOOL all_ok=YES;
    for(unsigned i=0;i<3;i++) {
        const size_t bytes=f->input[i].length;
        if(!f->buffers[i]) {
            emit(@{@"kind":@"real_inputs",@"layer":@(f->layer),@"phase":phase,@"input":names[i],
                @"bytes":@(bytes),@"available":@NO,@"passed":@NO});all_ok=NO;continue;
        }
        const void *post=body(f->buffers[i]);
        const uint8_t *prefix=f->buffers[i].contents,*suffix=prefix+GUARD+bytes;
        BOOL equal=!memcmp(post,f->input[i].bytes,bytes),guard=guards(f->buffers[i],bytes);
        NSString *stem=[NSString stringWithFormat:@"layer-%u-%@-%@",f->layer,phase,names[i]];
        NSString *post_path=[stem stringByAppendingString:@"-post.bin"];
        NSString *prefix_path=[stem stringByAppendingString:@"-guard-prefix.bin"];
        NSString *suffix_path=[stem stringByAppendingString:@"-guard-suffix.bin"];
        raw(post_path,post,bytes);raw(prefix_path,prefix,GUARD);raw(suffix_path,suffix,GUARD);
        emit(@{@"kind":@"real_inputs",@"layer":@(f->layer),@"phase":phase,@"input":names[i],
            @"bytes":@(bytes),@"available":@YES,@"capture_sha256":sha(f->input[i].bytes,bytes),
            @"post_sha256":sha(post,bytes),@"guards_intact":@(guard),@"bytes_equal_capture":@(equal),
            @"post_raw":post_path,@"guard_prefix_raw":prefix_path,@"guard_suffix_raw":suffix_path,
            @"guard_bytes_per_side":@(GUARD),@"guard_prefix_sha256":sha(prefix,GUARD),
            @"guard_suffix_sha256":sha(suffix,GUARD),@"passed":@(equal&&guard)});
        all_ok&=equal&&guard;
    }
    return all_ok;
}
static real_index_fixture *real_visible_fixtures;
static BOOL real_timing_phase;
static void real_save_visible_inputs(NSString *phase) {
    if(!real_visible_fixtures)return;
    for(unsigned ci=0;ci<3;ci++)if(real_visible_fixtures[ci].buffers[0])
        (void)real_index_inputs(&real_visible_fixtures[ci],phase);
}

static id<MTLComputePipelineState> real_index_pso(id<MTLDevice> device,id<MTLLibrary> lib,NSString *name,size_t smem) {
    id<MTLFunction> fn=[lib newFunctionWithName:name];need(fn!=nil,"real replay selected symbol, no fallback");
    NSError *error=nil;id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:fn error:&error];
    if(!p)fprintf(stderr,"PSO %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&
         p.staticThreadgroupMemoryLength+smem<=device.maxThreadgroupMemoryLength,"real replay PSO geometry");return p;
}
static id<MTLCommandBuffer> real_index_run(id<MTLCommandQueue> queue,real_index_fixture *f,unsigned provider,id<MTLComputePipelineState> p) {
    memset(body(f->out[provider]),0xa5,f->captured.length);
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"real replay command buffer");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"real replay concurrent encoder");
    [e setComputePipelineState:p];
    if(!provider)[e setBytes:f->native_args.bytes length:f->native_args.length atIndex:0];
    else [e setBytes:&f->args length:72 atIndex:0];
    for(unsigned i=0;i<3;i++)[e setBuffer:f->buffers[i]offset:GUARD atIndex:i+1];
    [e setBuffer:f->out[provider]offset:GUARD atIndex:4];
    [e setThreadgroupMemoryLength:provider?21504:f->smem atIndex:0];
    uint32_t gx=provider?(f->args.n_comp+31)/32:f->grid[0],gy=provider?(f->args.n_tokens+7)/8:f->grid[1];
    [e dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted) {
        fprintf(stderr,"real GPU: %s\n",cb.error.localizedDescription.UTF8String);
        real_save_visible_inputs(real_timing_phase?@"phase2_gpu_failure":@"phase1_gpu_failure");
    }
    need(cb.status==MTLCommandBufferStatusCompleted,"real replay GPU completed");return cb;
}
static size_t real_index_diff(const void *a,const void *b,size_t bytes) {
    const uint32_t *x=a,*y=b;size_t bad=0;for(size_t i=0;i<bytes/4;i++)bad+=x[i]!=y[i];return bad;
}
static BOOL real_index_validate(real_index_fixture *f,unsigned provider,BOOL save) {
    const uint32_t *scores=body(f->out[provider]);size_t bytes=f->captured.length;
    size_t capture_diff=real_index_diff(scores,f->captured.bytes,bytes);
    size_t tiled_diff=f->reference_tiled?real_index_diff(scores,f->reference_tiled.bytes,bytes):0;
    BOOL guard=guards(f->out[provider],bytes),tail=YES,finite=YES,written_all=YES;
    size_t invisible=0;
    for(unsigned t=0;t<f->args.n_tokens;t++) {
        unsigned visible=(f->args.pos0+t+1)/f->args.ratio;if(visible>f->args.n_comp)visible=f->args.n_comp;
        for(unsigned k=0;k<f->args.n_comp;k++) {
            uint32_t b=scores[(size_t)t*f->args.n_comp+k];written_all&=b!=0xa5a5a5a5u;
            if(k>=visible){tail&=b==0xff800000u;invisible++;}
            else finite&=(b&0x7f800000u)!=0x7f800000u;
        }
    }
    BOOL ok=guard&&tail&&finite&&written_all&&(provider!=0||capture_diff==0)&&
        (provider==0||!f->reference_tiled||tiled_diff==0);
    if(save||!ok) {
        static unsigned failures;
        NSString *prefix=save?[NSString stringWithFormat:@"layer-%u-provider-%u",f->layer,provider]:
            [NSString stringWithFormat:@"layer-%u-provider-%u-failure-%u",f->layer,provider,failures++];
        raw([prefix stringByAppendingString:@"-scores.f32"],scores,bytes);
        uint8_t g[2*GUARD];memcpy(g,f->out[provider].contents,GUARD);
        memcpy(g+GUARD,(uint8_t*)f->out[provider].contents+GUARD+bytes,GUARD);
        raw([prefix stringByAppendingString:@"-guards.bin"],g,sizeof(g));
        emit(@{@"kind":@"real_parity",@"layer":@(f->layer),@"provider":@(provider),
            @"tokens":@(f->args.n_tokens),@"keys":@(f->args.n_comp),@"bf16":@(f->bf16),
            @"captured_actual_provider":f->provider,@"mismatches_to_actual_capture":@(capture_diff),
            @"candidate_mismatches_to_tiled":provider==2?@(tiled_diff):(id)[NSNull null],
            @"invisible_minus_inf_count":@(invisible),@"full_scores":@YES,@"guards_intact":@(guard),
            @"all_written":@(written_all),@"visible_finite":@(finite),@"causal_tail_valid":@(tail),@"passed":@(ok),
            @"topk_mask_validated":@NO,@"fullmodel_speedup_proven":@NO});
    }
    return ok;
}

/* The exact production TopK API consumes each provider's complete score table.
 * This extra consumer run is correctness-only and excluded from score timings. */
static BOOL real_index_topk(real_index_fixture *f,BOOL failure_copy) {
    const size_t sb=f->captured.length,ib=f->captured_ids.length;
    myds_v41_gpu_scratch_t *scratch=myds_v41_gpu_scratch_create(f->args.n_comp);
    need(scratch&&myds_v41_gpu_indexer_batch_floats(scratch)==2097152u,"production TopK scratch budget");
    myds_metal_buffer_t *sp=myds_metal_buffer_alloc(sb+2*GUARD),*ip=myds_metal_buffer_alloc(ib+2*GUARD);
    need(sp&&ip,"TopK guarded parent allocations");
    myds_metal_buffer_t *scores=myds_metal_buffer_sub_view(sp,GUARD,sb),*ids=myds_metal_buffer_sub_view(ip,GUARD,ib);
    need(scores&&ids,"TopK guarded views");
    NSData *tiled_ids=nil;BOOL all_ok=YES;
    for(unsigned provider=0;provider<3;provider++) {
        memset(myds_metal_buffer_data(sp),0xa5,sb+2*GUARD);memset(myds_metal_buffer_data(ip),0xa5,ib+2*GUARD);
        memcpy(myds_metal_buffer_data(scores),body(f->out[provider]),sb);
        myds_v41_gpu_topk_prefix_counters(NULL,true);myds_v41_gpu_topk_nan_fuse_counters(NULL,true);
        myds_v41_gpu_topk_nan_shuffle_counters(NULL,true);
        need(myds_metal_begin_batch(),"TopK batch");
        int encoded=myds_v41_gpu_topk_causal(ids,scores,f->args.n_comp,512u,f->args.n_tokens,
                                           f->args.pos0,f->args.ratio,scratch);
        int completed=myds_metal_commit_batch();
        void *actual=myds_metal_buffer_data(ids);
        size_t actual_diff=real_index_diff(actual,f->captured_ids.bytes,ib);
        size_t tiled_diff=provider==2?real_index_diff(actual,tiled_ids.bytes,ib):0;
        BOOL guard=YES;
        const uint8_t *parents[]={myds_metal_buffer_data(sp),myds_metal_buffer_data(ip)};
        const size_t extents[]={sb,ib};
        for(unsigned p=0;p<2;p++)for(unsigned j=0;j<GUARD;j++)
            guard&=parents[p][j]==0xa5&&parents[p][GUARD+extents[p]+j]==0xa5;
        BOOL input_unchanged=!memcmp(myds_metal_buffer_data(scores),body(f->out[provider]),sb);
        myds_v41_gpu_topk_prefix_counters_t prefix;myds_v41_gpu_topk_nan_fuse_counters_t nan;
        myds_v41_gpu_topk_nan_shuffle_counters_t shuffle;
        myds_v41_gpu_topk_prefix_counters(&prefix,false);myds_v41_gpu_topk_nan_fuse_counters(&nan,false);
        myds_v41_gpu_topk_nan_shuffle_counters(&shuffle,false);
        BOOL same_path=prefix.requests==[f->captured_topk[@"prefix_requests"]unsignedLongLongValue]&&
            prefix.encoded==[f->captured_topk[@"prefix_encoded"]unsignedLongLongValue]&&
            prefix.unavailable==[f->captured_topk[@"prefix_unavailable"]unsignedLongLongValue]&&
            nan.encoded==[f->captured_topk[@"nan_fuse_encoded"]unsignedLongLongValue]&&
            shuffle.encoded==[f->captured_topk[@"nan_shuffle_encoded"]unsignedLongLongValue]&&!shuffle.failures;
        BOOL ok=encoded&&completed&&guard&&input_unchanged&&same_path&&
            actual_diff==0&&(provider!=2||tiled_diff==0);
        NSString *artifact_prefix=[NSString stringWithFormat:@"layer-%u-provider-%u%@",f->layer,provider,failure_copy?@"-failure":@""];
        raw([artifact_prefix stringByAppendingString:@"-selected.i32"],actual,ib);
        uint8_t g[4*GUARD];for(unsigned p=0;p<2;p++){memcpy(g+2*p*GUARD,parents[p],GUARD);
            memcpy(g+(2*p+1)*GUARD,parents[p]+GUARD+extents[p],GUARD);}
        raw([artifact_prefix stringByAppendingString:@"-topk-guards.bin"],g,sizeof(g));
        emit(@{@"kind":@"real_topk",@"layer":@(f->layer),@"provider":@(provider),@"top":@512,
            @"rows":@(f->args.n_tokens),@"ordered_ids_complete":@YES,@"actual_capture_mismatches":@(actual_diff),
            @"candidate_mismatches_to_tiled":provider==2?@(tiled_diff):(id)[NSNull null],
            @"host_encoded":@(encoded),@"GPU_completed":@(completed),@"guards_intact":@(guard),
            @"score_input_unchanged":@(input_unchanged),@"same_consumer_counter_path_as_capture":@(same_path),
            @"prefix_requests":@(prefix.requests),@"prefix_encoded":@(prefix.encoded),@"prefix_unavailable":@(prefix.unavailable),
            @"nan_fuse_encoded":@(nan.encoded),@"nan_shuffle_encoded":@(shuffle.encoded),@"passed":@(ok),
            @"actual_mask_filter_for_these_layers":@NO,@"fullmodel_speedup_proven":@NO});
        all_ok&=ok;
        if(provider==1)tiled_ids=[NSData dataWithBytes:actual length:ib];
    }
    myds_metal_buffer_free(scores);myds_metal_buffer_free(ids);myds_metal_buffer_free(sp);myds_metal_buffer_free(ip);
    myds_v41_gpu_scratch_free(scratch);return all_ok;
}
int main(int argc,char **argv){@autoreleasepool {
    NSString *capture_path=nil,*probe_path=nil,*native_path=nil,*config_path=nil;BOOL host_only=NO;unsigned samples=9;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--host-only")){host_only=YES;continue;}
        need(i+1<argc,"option value");const char *key=argv[i++],*v=argv[i];
        NSString *s=canonical([NSString stringWithUTF8String:v]);
        if(!strcmp(key,"--capture-json"))capture_path=s;
        else if(!strcmp(key,"--metallib"))probe_path=s;
        else if(!strcmp(key,"--native-metallib"))native_path=s;
        else if(!strcmp(key,"--out-dir"))output=s;
        else if(!strcmp(key,"--config"))config_path=s;
        else if(!strcmp(key,"--samples"))samples=number(v,9,9);
        else need(NO,"known replay option");
    }
    need(capture_path&&probe_path&&native_path&&output&&config_path,"capture-json/metallib/native-metallib/config/out-dir required");
    need(myds_configfile_load(config_path.fileSystemRepresentation,true)>=0,"production config overwrite semantics");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new real replay output");
    NSDictionary *capture=[NSJSONSerialization JSONObjectWithData:read_file(capture_path)options:0 error:nil];
    need([capture[@"capture_kind"]isEqual:@"myds_real_prefill_indexer"]&&
         [capture[@"status"]isEqual:@"verified_cpu_only"]&&[capture[@"cases"]count]==3,"verified real capture schema");
    NSString *capture_root=canonical(capture[@"capture_directory"]);real_index_fixture fixtures[3];
    for(unsigned i=0;i<3;i++)fixtures[i]=read_index_case(capture[@"cases"][i],capture_root);
    real_visible_fixtures=fixtures;
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"real replay output creation");
    files=[NSMutableArray new];stored_payloads=[NSMutableDictionary new];
    manifest=[@{@"schema_version":@1,@"probe":@"indexer_k_resident_real_capture",@"fixture_kind":@"actual_fullmodel_prefill",
        @"capture_json_sha256":file_sha(capture_path),@"probe_sha256":file_sha(probe_path),
        @"native_sha256":file_sha(native_path),@"binary_sha256":file_sha(binary_path()),
        @"host_only":@(host_only),@"fullmodel":@NO,@"topk_replayed_with_actual_helper":@(!host_only),@"performance_from_capture":@NO,
        @"config_sha256":file_sha(config_path),
        @"cold_dram_proven":@NO,@"capture_default_provider_is_timing_control_0":@YES,
        @"samples":@(samples),@"warmups":@1,@"mode":@"concurrent",@"gpu_work_submitted":@NO}mutableCopy];
    if(host_only){finish(@"host_only_passed");puts("{\"host_only_pass\":true,\"GPU_submitted\":false}");return 0;}
    manifest[@"gpu_work_submitted"]=@YES;
    need([native_path.lastPathComponent isEqual:@"myds.metallib"],"native library must retain myds.metallib filename");
    need(myds_metal_init(native_path.stringByDeletingLastPathComponent.fileSystemRepresentation),"actual TopK runtime Metal init");
    NSString *loaded=myds_metal_loaded_library_path()?[NSString stringWithUTF8String:myds_metal_loaded_library_path()]:nil;
    need(loaded&&[canonical(loaded)isEqual:native_path]&&[file_sha(loaded)isEqual:file_sha(native_path)],"actual TopK loaded frozen native library");
    manifest[@"topk_loaded_library_path"]=loaded;manifest[@"topk_loaded_library_sha256"]=file_sha(loaded);
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();need(d!=nil,"real replay Metal device");
    NSError *error=nil;
    id<MTLLibrary> native=[d newLibraryWithURL:[NSURL fileURLWithPath:native_path]error:&error];need(native!=nil,"frozen native library");
    id<MTLLibrary> probe=[d newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];need(probe!=nil,"probe library");
    id<MTLCommandQueue> queue=[d newCommandQueue];need(queue!=nil,"real replay queue");
    manifest[@"gpu_work_submitted"]=@YES;manifest[@"device"]=d.name;
    id<MTLComputePipelineState> psos[3][3];BOOL all_numerical=YES;
    // Phase 1: every layer's complete scores and actual ordered Top512 consumer
    // gates finish before the first warmup or formal timing command buffer.
    for(unsigned ci=0;ci<3;ci++){@autoreleasepool {
        real_index_fixture *f=&fixtures[ci];NSString *suffix=f->bf16?@"bf16":@"f32";
        psos[ci][0]=real_index_pso(d,native,f->provider,f->smem);
        psos[ci][1]=real_index_pso(d,probe,[@"kernel_dsv4_indexer_scores_tiled_"stringByAppendingString:suffix],21504);
        psos[ci][2]=real_index_pso(d,probe,[@"kernel_myds_indexer_scores_k_resident_"stringByAppendingString:suffix],21504);
        for(unsigned i=0;i<3;i++){f->buffers[i]=alloc(d,f->input[i].length,YES);memcpy(body(f->buffers[i]),f->input[i].bytes,f->input[i].length);f->out[i]=alloc(d,f->captured.length,YES);}
        for(unsigned p=0;p<3;p++) {
            (void)real_index_run(queue,f,p,psos[ci][p]);all_numerical&=real_index_validate(f,p,YES);
            if(p==1)f->reference_tiled=[NSData dataWithBytes:body(f->out[1])length:f->captured.length];
        }
        all_numerical&=real_index_topk(f,NO);
        all_numerical&=real_index_inputs(f,@"phase1_post_topk");
    }}
    manifest[@"all_layers_numerical_gate_passed"]=@(all_numerical);
    emit(@{@"kind":@"all_layers_numerical_gate",@"passed":@(all_numerical),@"layers":@[@2,@8,@14],
        @"complete_scores":@YES,@"ordered_top512_all_providers_equal_capture":@(all_numerical),@"timing_started":@NO});
    need(all_numerical,"all three layers numerical gates before timing; outputs/IDs/guards already saved");
    real_timing_phase=YES;
    // Phase 2: one frozen cohort, two distinct A1/B/A2 controls. No sample
    // replacement and no averaging away of a failed layer/control gate.
    for(unsigned ci=0;ci<3;ci++){@autoreleasepool {
        real_index_fixture *f=&fixtures[ci];
        for(unsigned control=0;control<2;control++) {
            const unsigned providers[]={control,2,control};double times[3][32];
            for(unsigned sample=0;sample<=samples;sample++)for(unsigned arm=0;arm<3;arm++) {
                unsigned provider=providers[arm];id<MTLCommandBuffer> cb=real_index_run(queue,f,provider,psos[ci][provider]);
                if(!real_index_validate(f,provider,NO)) {
                    (void)real_index_topk(f,YES);
                    real_save_visible_inputs(@"phase2_score_failure");
                    need(NO,"formal score failure; complete failed scores/IDs/guards/input witnesses saved");
                }
                if(!(cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime))
                    real_save_visible_inputs(@"phase2_timestamp_failure");
                need(cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime,"real timestamps");
                double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6;if(sample)times[arm][sample-1]=us;
                emit(@{@"kind":@"real_timing",@"layer":@(f->layer),@"control":@(control),@"provider":@(provider),@"arm":@[@"A1",@"B",@"A2"][arm],
                    @"sample":@(sample),@"warmup":@(sample==0),@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),
                    @"gpu_us":@(us),@"provider_full_scores_bit_exact_to_tiled":@(!memcmp(body(f->out[provider]),f->reference_tiled.bytes,f->captured.length)),
                    @"full_scores_sha256":sha(body(f->out[provider]),f->captured.length),
                    @"candidate_full_scores_to_tiled_gate_passed":@YES,@"all_layers_numerical_gate_passed":@YES,@"fullmodel_speedup_proven":@NO});
            }
            double a1=median(times[0],samples),b=median(times[1],samples),a2=median(times[2],samples);
            emit(@{@"kind":@"real_summary",@"layer":@(f->layer),@"control":@(control),@"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),
                @"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),@"control_drift_percent":@((a2/a1-1)*100),
                @"ordered_top512_validated":@YES,@"fullmodel_speedup_proven":@NO});
        }
        BOOL inputs_ok=real_index_inputs(f,@"phase2_post_timing");
        if(!inputs_ok)real_save_visible_inputs(@"phase2_input_failure");
        need(inputs_ok,"real replay full input bytes/guards unchanged; complete post inputs/guards saved");
        for(unsigned i=0;i<3;i++){f->buffers[i]=nil;f->out[i]=nil;}
    }}
    myds_metal_cleanup();finish(@"passed_component_and_topk_only");return 0;
}}
