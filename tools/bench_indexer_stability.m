/* Diagnostic only. Helpers are frozen from bench_indexer_k_capture.m before
 * its main(), without editing that file. No runtime candidate is registered.
 * These samples cannot be used as a performance-adoption cohort. */
#include "indexer_replay_helpers.inc"

typedef struct { const char *name; unsigned conditioner, arms[3]; BOOL deferred, distinct; } stability_phase;
static const stability_phase stability_phases[]={
    {"sham_exact_inline",0,{0,0,0},NO,NO},
    {"sham_tiled_inline",1,{1,1,1},NO,NO},
    {"sham_candidate_inline",2,{2,2,2},NO,NO},
    {"legacy_tiled_inline",1,{1,2,1},NO,NO},
    {"balanced_tiled_inline",2,{1,2,1},NO,NO},
    {"legacy_tiled_deferred",1,{1,2,1},YES,NO},
    {"balanced_tiled_deferred",2,{1,2,1},YES,NO},
    {"legacy_tiled_distinct",1,{1,2,1},NO,YES},
    {"balanced_tiled_distinct",2,{1,2,1},NO,YES},
};
enum { STABILITY_PHASES=9, STABILITY_FORMAL_ROUNDS=12 };
_Static_assert(sizeof(stability_phases)/sizeof(stability_phases[0])==STABILITY_PHASES,"frozen diagnostic matrix");

static BOOL stability_validate(real_index_fixture *f,unsigned provider,NSData *scores,NSData *guard,
                               NSMutableDictionary *record) {
    const double started=wall();
    size_t actual_bad=real_index_diff(scores.bytes,f->captured.bytes,scores.length);
    size_t tiled_bad=real_index_diff(scores.bytes,f->reference_tiled.bytes,scores.length);
    BOOL finite=YES,tail=YES,written_all=YES,canary=guard.length==2*GUARD;
    const uint32_t *v=scores.bytes;
    for(unsigned t=0;t<f->args.n_tokens;t++) {
        unsigned visible=(f->args.pos0+t+1)/f->args.ratio;
        if(visible>f->args.n_comp)visible=f->args.n_comp;
        for(unsigned k=0;k<f->args.n_comp;k++) {
            uint32_t bits=v[(size_t)t*f->args.n_comp+k];written_all&=bits!=0xa5a5a5a5u;
            if(k>=visible)tail&=bits==0xff800000u;
            else finite&=(bits&0x7f800000u)!=0x7f800000u;
        }
    }
    const uint8_t *g=guard.bytes;
    for(unsigned j=0;j<guard.length;j++)canary&=g[j]==0xa5;
    BOOL pass=finite&&tail&&written_all&&canary&&(provider==0?actual_bad==0:tiled_bad==0);
    record[@"mismatches_to_actual_capture"]=@(actual_bad);
    record[@"mismatches_to_tiled"]=@(tiled_bad);
    record[@"scores_sha256"]=sha(scores.bytes,scores.length);
    record[@"guards_sha256"]=sha(guard.bytes,guard.length);
    record[@"visible_finite"]=@(finite);record[@"causal_tail_valid"]=@(tail);
    record[@"all_written"]=@(written_all);record[@"guards_intact"]=@(canary);
    record[@"validation_passed"]=@(pass);
    record[@"validation_host_begin_s"]=@(started);record[@"validation_host_end_s"]=@(wall());
    return pass;
}

static void stability_run(id<MTLCommandQueue> queue,real_index_fixture *f,unsigned provider,
                          id<MTLComputePipelineState> p,id<MTLBuffer> out,NSMutableDictionary *record) {
    record[@"host_poison_begin_s"]=@(wall());memset(body(out),0xa5,f->captured.length);
    record[@"host_poison_end_s"]=@(wall());
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"diagnostic command buffer");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"diagnostic encoder");
    [e setComputePipelineState:p];
    if(provider==0)[e setBytes:f->native_args.bytes length:f->native_args.length atIndex:0];
    else [e setBytes:&f->args length:sizeof(f->args) atIndex:0];
    for(unsigned i=0;i<3;i++)[e setBuffer:f->buffers[i] offset:GUARD atIndex:i+1];
    [e setBuffer:out offset:GUARD atIndex:4];
    [e setThreadgroupMemoryLength:provider?21504:f->smem atIndex:0];
    const unsigned gx=provider?(f->args.n_comp+31)/32:f->grid[0];
    const unsigned gy=provider?(f->args.n_tokens+7)/8:f->grid[1];
    [e dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
    [e endEncoding];record[@"host_commit_s"]=@(wall());[cb commit];[cb waitUntilCompleted];
    record[@"host_wait_completed_s"]=@(wall());
    if(cb.status!=MTLCommandBufferStatusCompleted)real_save_visible_inputs(@"diagnostic_gpu_failure");
    need(cb.status==MTLCommandBufferStatusCompleted,"diagnostic GPU completed");
    need(cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime,"diagnostic timestamps");
    record[@"gpu_start_s"]=@(cb.GPUStartTime);record[@"gpu_end_s"]=@(cb.GPUEndTime);
    record[@"gpu_us"]=@((cb.GPUEndTime-cb.GPUStartTime)*1e6);
}

int main(int argc,char **argv){@autoreleasepool {
    NSString *capture_path=nil,*probe_path=nil,*native_path=nil,*config_path=nil,*protocol_path=nil;
    BOOL host_only=NO;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--host-only")){host_only=YES;continue;}
        need(i+1<argc,"option value");const char *key=argv[i++];NSString *s=canonical([NSString stringWithUTF8String:argv[i]]);
        if(!strcmp(key,"--capture-json"))capture_path=s;
        else if(!strcmp(key,"--metallib"))probe_path=s;
        else if(!strcmp(key,"--native-metallib"))native_path=s;
        else if(!strcmp(key,"--config"))config_path=s;
        else if(!strcmp(key,"--protocol"))protocol_path=s;
        else if(!strcmp(key,"--out-dir"))output=s;
        else need(NO,"known diagnostic option");
    }
    need(capture_path&&probe_path&&native_path&&config_path&&protocol_path&&output,"all diagnostic paths supplied");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new diagnostic output");
    need(myds_configfile_load(config_path.fileSystemRepresentation,true)>=0,"frozen production configuration");
    NSDictionary *protocol=[NSJSONSerialization JSONObjectWithData:read_file(protocol_path)options:0 error:nil];
    need([protocol[@"kind"]isEqual:@"indexer_predecessor_stability_diagnostic"]&&
         [protocol[@"formal_rounds"]unsignedIntValue]==STABILITY_FORMAL_ROUNDS&&
         [protocol[@"phases"]count]==STABILITY_PHASES&&![protocol[@"adoption_cohort"]boolValue],"diagnostic-only frozen protocol");
    for(unsigned pi=0;pi<STABILITY_PHASES;pi++) {
        NSDictionary *phase=protocol[@"phases"][pi];const stability_phase *p=&stability_phases[pi];
        need([phase[@"name"]isEqual:[NSString stringWithUTF8String:p->name]]&&
             [phase[@"conditioner"]unsignedIntValue]==p->conditioner&&
             [phase[@"deferred_validation"]boolValue]==p->deferred&&
             [phase[@"distinct_outputs"]boolValue]==p->distinct&&[phase[@"arms"]count]==3,"frozen diagnostic phase");
        for(unsigned a=0;a<3;a++)need([phase[@"arms"][a]unsignedIntValue]==p->arms[a],"frozen diagnostic providers");
    }
    NSDictionary *capture=[NSJSONSerialization JSONObjectWithData:read_file(capture_path)options:0 error:nil];
    need([capture[@"status"]isEqual:@"verified_cpu_only"]&&[capture[@"cases"]count]==3,"verified real capture");
    need([file_sha(config_path)isEqual:capture[@"config_sha256"]],"captured configuration bytes");
    need([file_sha(capture_path)isEqual:protocol[@"capture_json_sha256"]]&&
         [file_sha(probe_path)isEqual:protocol[@"probe_sha256"]]&&
         [file_sha(native_path)isEqual:protocol[@"native_sha256"]],"frozen diagnostic inputs/libraries");
    NSString *capture_root=canonical(capture[@"capture_directory"]);
    real_index_fixture fixtures[3];for(unsigned i=0;i<3;i++)fixtures[i]=read_index_case(capture[@"cases"][i],capture_root);
    real_index_fixture *f=&fixtures[0];need(f->layer==2,"failed group selected before diagnostic");
    real_visible_fixtures=fixtures;
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"diagnostic output creation");
    files=[NSMutableArray new];stored_payloads=[NSMutableDictionary new];
    manifest=[@{@"schema_version":@1,@"probe":@"indexer_predecessor_stability_diagnostic",@"adoption_cohort":@NO,
        @"fullmodel":@NO,@"fullmodel_speedup_proven":@NO,@"production_adoption":@NO,@"host_only":@(host_only),
        @"protocol_sha256":file_sha(protocol_path),@"capture_json_sha256":file_sha(capture_path),
        @"probe_sha256":file_sha(probe_path),@"native_sha256":file_sha(native_path),@"binary_sha256":file_sha(binary_path()),
        @"config_sha256":file_sha(config_path),@"layer":@2,@"tokens":@(f->args.n_tokens),@"keys":@(f->args.n_comp),
        @"formal_rounds":@(STABILITY_FORMAL_ROUNDS),@"warmup_rounds":@1,@"phase_count":@(STABILITY_PHASES),
        @"timing_record_count_expected":@((STABILITY_FORMAL_ROUNDS+1)*STABILITY_PHASES*4),@"gpu_work_submitted":@NO}mutableCopy];
    if(host_only){finish(@"host_only_passed");puts("{\"host_only_passed\":true,\"GPU_submitted\":false}");return 0;}
    need([native_path.lastPathComponent isEqual:@"myds.metallib"],"frozen native library name");
    need(myds_metal_init(native_path.stringByDeletingLastPathComponent.fileSystemRepresentation),"diagnostic native runtime init");
    NSString *loaded=myds_metal_loaded_library_path()?[NSString stringWithUTF8String:myds_metal_loaded_library_path()]:nil;
    need(loaded&&[canonical(loaded)isEqual:native_path]&&[file_sha(loaded)isEqual:file_sha(native_path)],"actual native library path/SHA");
    manifest[@"loaded_native_library_path"]=loaded;manifest[@"gpu_work_submitted"]=@YES;
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();NSError *error=nil;
    id<MTLLibrary> native=[d newLibraryWithURL:[NSURL fileURLWithPath:native_path]error:&error];
    id<MTLLibrary> probe=[d newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];
    need(d&&native&&probe,"diagnostic Metal objects");
    id<MTLCommandQueue> queue=[d newCommandQueue];need(queue!=nil,"diagnostic queue");
    id<MTLComputePipelineState> psos[3];
    psos[0]=real_index_pso(d,native,f->provider,f->smem);
    psos[1]=real_index_pso(d,probe,@"kernel_dsv4_indexer_scores_tiled_bf16",21504);
    psos[2]=real_index_pso(d,probe,@"kernel_myds_indexer_scores_k_resident_bf16",21504);
    need(f->bf16,"frozen actual BF16 case");
    manifest[@"device"]=d.name;
    manifest[@"selected_symbols"]=@[f->provider,@"kernel_dsv4_indexer_scores_tiled_bf16",@"kernel_myds_indexer_scores_k_resident_bf16"];
    manifest[@"threads_per_threadgroup"]=@[@32,@4,@1];
    manifest[@"dynamic_threadgroup_bytes"]=@[@(f->smem),@21504,@21504];
    BOOL initial=YES;
    for(unsigned i=0;i<3;i++) {
        f->buffers[i]=alloc(d,f->input[i].length,YES);memcpy(body(f->buffers[i]),f->input[i].bytes,f->input[i].length);
        f->out[i]=alloc(d,f->captured.length,YES);
    }
    for(unsigned p=0;p<3;p++) {
        (void)real_index_run(queue,f,p,psos[p]);initial&=real_index_validate(f,p,YES);
        if(p==1)f->reference_tiled=[NSData dataWithBytes:body(f->out[1])length:f->captured.length];
    }
    initial&=real_index_topk(f,NO);initial&=real_index_inputs(f,@"diagnostic_before");
    manifest[@"initial_numerical_gate_passed"]=@(initial);need(initial,"diagnostic identity before timings");
    id<MTLBuffer> outputs[4];for(unsigned i=0;i<4;i++)outputs[i]=alloc(d,f->captured.length,YES);
    NSInteger previous_provider=-1,previous_slot=-1;unsigned sequence=0;
    for(unsigned round=0;round<=STABILITY_FORMAL_ROUNDS;round++)for(unsigned order=0;order<STABILITY_PHASES;order++){@autoreleasepool {
        const unsigned pi=(round>0&&(round%2))?STABILITY_PHASES-1-order:order;
        const stability_phase *phase=&stability_phases[pi];
        NSMutableDictionary *records[4];NSData *snapshots[4],*guards_saved[4];BOOL group_pass=YES;
        for(unsigned step=0;step<4;step++) {
            unsigned provider=step?phase->arms[step-1]:phase->conditioner;
            unsigned slot=phase->distinct?step:provider;
            records[step]=[@{@"kind":@"stability_timing",@"sequence":@(sequence++),@"round":@(round),@"warmup":@(round==0),
                @"phase_index":@(pi),@"phase_order":@(order),@"phase":[NSString stringWithUTF8String:phase->name],
                @"step":@(step),@"role":step?@[@"A1",@"B",@"A2"][step-1]:@"conditioner",
                @"provider":@(provider),@"output_slot":@(slot),@"previous_provider":@(previous_provider),
                @"previous_output_slot":@(previous_slot),@"deferred_validation":@(phase->deferred),@"distinct_outputs":@(phase->distinct),
                @"adoption_cohort":@NO,@"fullmodel_speedup_proven":@NO}mutableCopy];
            stability_run(queue,f,provider,psos[provider],outputs[slot],records[step]);
            snapshots[step]=[NSData dataWithBytes:body(outputs[slot])length:f->captured.length];
            uint8_t canary[2*GUARD];memcpy(canary,outputs[slot].contents,GUARD);
            memcpy(canary+GUARD,(uint8_t*)outputs[slot].contents+GUARD+f->captured.length,GUARD);
            guards_saved[step]=[NSData dataWithBytes:canary length:sizeof(canary)];
            records[step][@"host_snapshot_completed_s"]=@(wall());
            if(!phase->deferred)group_pass&=stability_validate(f,provider,snapshots[step],guards_saved[step],records[step]);
            previous_provider=provider;previous_slot=slot;
        }
        for(unsigned step=0;step<4;step++) {
            if(phase->deferred)group_pass&=stability_validate(f,[records[step][@"provider"]unsignedIntValue],snapshots[step],guards_saved[step],records[step]);
            NSString *stem=[NSString stringWithFormat:@"round-%02u-phase-%u-step-%u",round,pi,step];
            NSString *scores=[stem stringByAppendingString:@"-scores.f32"],*guards_file=[stem stringByAppendingString:@"-guards.bin"];
            raw(scores,snapshots[step].bytes,snapshots[step].length);raw(guards_file,guards_saved[step].bytes,guards_saved[step].length);
            records[step][@"scores_raw"]=scores;records[step][@"guards_raw"]=guards_file;emit(records[step]);
        }
        if(!group_pass)real_save_visible_inputs(@"diagnostic_score_failure");
        need(group_pass,"diagnostic complete scores/guards saved before refusal");
    }}
    need(real_index_inputs(f,@"diagnostic_after"),"diagnostic input bytes/guards unchanged");
    myds_metal_cleanup();finish(@"diagnostic_complete_not_adoption");return 0;
}}
