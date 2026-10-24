/* Independent functional failure scenarios. Helpers provide only file/guard
 * mechanics; this main controls real factory/selection/rollback transactions.
 * No timing or full-model cancellation claim. Never run without --run. */
#define main moe_workspace_normal_main
#include "bench_moe_workspace.m"
#undef main

static NSDictionary *live_allocations(void) {
 myds_metal_moe_workspace_allocation_stats_t s={0};myds_metal_moe_workspace_allocation_stats(&s);
 return @{@"owners":@(s.live_owners),@"views":@(s.live_views),@"owner_bytes":@(s.owner_bytes)};
}
static NSDictionary *pool_identity(NSDictionary *s) {
 NSDictionary *p=s[@"scratch"];NSMutableDictionary *v=[NSMutableDictionary new];
 for(NSString *key in @[@"capacity",@"moe_batch",@"compact",@"generation",@"owner_bytes",@"owners",@"views",@"legacy_gate_bytes",@"legacy_up_bytes",@"legacy_mid_bytes"])v[key]=s[key];
 for(NSString *key in @[@"mm_capacity",@"f32_capacity",@"generation",@"mm_mid_bytes",@"f32_gate_bytes",@"f32_up_bytes",@"f32_mid_bytes",@"retired_bytes",@"retired_owners",@"allocated_bytes"])v[[@"scratch_" stringByAppendingString:key]]=p[key];
 return v;
}
static NSArray *allocation_trace(void) {
 uint64_t n=myds_metal_moe_workspace_allocation_trace(NULL,0);need(n<100000,"bounded diagnostic trace");
 myds_metal_moe_workspace_allocation_t *t=calloc(n?n:1,sizeof(*t));need(t!=NULL,"diagnostic trace memory");
 need(myds_metal_moe_workspace_allocation_trace(t,n)==n,"complete diagnostic trace");NSMutableArray *rows=[NSMutableArray new];
 for(uint64_t i=0;i<n;i++){need(memchr(t[i].site,0,sizeof(t[i].site))!=NULL,"terminated allocation site");[rows addObject:@{@"ordinal":@(t[i].ordinal),@"bytes":@(t[i].bytes),@"kind":@(t[i].kind),@"index":@(t[i].index),@"success":@(t[i].success),@"site":[NSString stringWithUTF8String:t[i].site]}];}free(t);return rows;
}
static fixture derived_fixture(NSDictionary *record,NSData *x,NSData *ids,NSData *routeweights,unsigned rows) {
 fixture f={0};f.record=record;f.rows=rows;f.derived=YES;f.index=[record[@"case"]unsignedIntValue];
 need(rows>0&&x.length>=(size_t)rows*20480u&&ids.length>=(size_t)rows*24u&&routeweights.length>=(size_t)rows*24u,"real derived prefix extent");
 f.x=[x subdataWithRange:NSMakeRange(0,(size_t)rows*20480u)];f.ids=[ids subdataWithRange:NSMakeRange(0,(size_t)rows*24u)];f.weights=[routeweights subdataWithRange:NSMakeRange(0,(size_t)rows*24u)];
 f.xb=buffer(f.x.length,f.x.bytes);f.ib=buffer(f.ids.length,f.ids.bytes);f.wb=buffer(f.weights.length,f.weights.bytes);f.out=buffer(f.x.length,NULL);return f;
}
static void release_fixture(fixture *f){release_buffer(f->xb);release_buffer(f->ib);release_buffer(f->wb);release_buffer(f->out);}
static NSData *run_output(myds_v41_moe_workspace_probe_t *w,fixture *f,myds_metal_buffer_t **weights,float clamp,NSString *name) {
 poison(f->out);need(myds_metal_begin_batch(),"diagnostic begin");encode(w,f,weights,clamp);need(myds_metal_commit_batch(),"diagnostic complete");validate(f);dump_buffer(name,f->out);
 return [NSData dataWithBytes:myds_metal_buffer_data(f->out.body) length:f->out.bytes];
}
static NSDictionary *mask_case(fixture *f,myds_metal_buffer_t **weights,float clamp,uint32_t mask) {
 NSMutableArray *arms=[NSMutableArray new];NSData *oracle=nil;
 for(unsigned compact=0;compact<2;compact++) {
  need(myds_metal_moe_workspace_test_control(0,mask),"idle MM mask control");NSDictionary *live_before=live_allocations();
  myds_v41_moe_workspace_probe_t *w=myds_v41_moe_workspace_probe_create(32,2304,f->rows,compact);need(w!=NULL,"actual full masked factory");
  NSString *name=[NSString stringWithFormat:@"mask-%u-rows-%u-%s",mask,f->rows,compact?"compact":"native"];
  NSData *result=run_output(w,f,weights,clamp,name);NSDictionary *s=stats(w);BOOL exact=!compact||[result isEqual:oracle];if(!compact)oracle=result;
  need([s[@"mm_encoded"]unsignedLongLongValue]==0&&[s[@"mv_encoded"]unsignedLongLongValue]>0,"both matching-mask whole-batch providers must actually encode MV");
  if(compact){need([s[@"scratch"][@"last_provider"]unsignedIntValue]==MYDS_MOE_WS_MV_F32&&[s[@"scratch"][@"mm_encoded"]unsignedLongLongValue]==0&&[s[@"scratch"][@"mv_encoded"]unsignedLongLongValue]>0,"masked actual compact provider must select full MV before encoding");}
  [arms addObject:@{@"provider":compact?@"compact":@"native",@"output_stem":name,@"full_f32_bit_exact":@(exact),@"finite_guards_written":@YES,@"workspace":s}];
  need(myds_v41_moe_workspace_probe_quiescent(w,false),"masked retirement");myds_v41_moe_workspace_probe_free(w);need([live_before isEqual:live_allocations()],"masked full factory release accounting");need(exact,"masked baseline/compact full output parity");
 }
 NSDictionary *row=@{@"kind":@"mm_pso_mask",@"mask":@(mask),@"rows":@(f->rows),@"capacity":@32,@"same_mask_oracle":@YES,@"arms":arms,@"timing_measured":@NO};emit(row);return row;
}
static NSDictionary *lazy_failure(fixture *f,myds_metal_buffer_t **weights,float clamp,uint64_t ordinal) {
 need(myds_metal_moe_workspace_test_control(0,1),"initial masked MV control");
 myds_v41_moe_workspace_probe_t *native=myds_v41_moe_workspace_probe_create(32,2304,64,false);need(native!=NULL,"lazy native factory");
 NSString *oracle_name=[NSString stringWithFormat:@"lazy-%llu-native",(unsigned long long)ordinal];NSData *oracle=run_output(native,f,weights,clamp,oracle_name);myds_v41_moe_workspace_probe_free(native);
 /* Start with the real MM pool, then make MM unavailable. Otherwise the
  * factory would allocate F32 eagerly and the injected run would never hit. */
 need(myds_metal_moe_workspace_test_control(0,0),"clean compact MM factory control");NSDictionary *global_before=live_allocations();
 myds_v41_moe_workspace_probe_t *w=myds_v41_moe_workspace_probe_create(32,2304,64,true);need(w!=NULL,"lazy actual compact factory");NSDictionary *before=stats(w),*live_before=live_allocations();
 need(myds_metal_moe_workspace_test_control(ordinal,1),"arm idle relative lazy allocation failure");
 poison(f->out);need(myds_metal_begin_batch(),"lazy failed attempt begin");uint64_t dispatch_before=myds_metal_dispatch_count();
 int ok=myds_v41_moe_workspace_probe_run(w,f->out.body,f->xb.body,weights[0],weights[1],weights[2],f->ib.body,f->wb.body,f->rows,clamp);
 uint64_t dispatch_after=myds_metal_dispatch_count();need(myds_metal_commit_batch(),"lazy failed attempt existing batch completion");
 NSString *failed_name=[NSString stringWithFormat:@"lazy-%llu-failed-attempt",(unsigned long long)ordinal];dump_buffer(failed_name,f->out);NSArray *trace=allocation_trace();NSDictionary *after=stats(w);unsigned failures=0;NSString *site=nil;
 for(NSDictionary *a in trace)if(![a[@"success"]boolValue]){failures++;need([a[@"ordinal"]unsignedLongLongValue]==ordinal,"requested lazy ordinal hit");site=a[@"site"];}
 BOOL poison_unchanged=YES;const uint8_t *p=myds_metal_buffer_data(f->out.body);for(size_t i=0;i<f->out.bytes;i++)if(p[i]!=0xa5){poison_unchanged=NO;break;}
 need(!ok&&failures==1&&dispatch_before==dispatch_after&&poison_unchanged&&guards(f->out)&&guards(f->xb)&&guards(f->ib)&&guards(f->wb),"failed lazy allocation must not encode or write partial batch");
 need(!memcmp(myds_metal_buffer_data(f->xb.body),f->x.bytes,f->x.length)&&!memcmp(myds_metal_buffer_data(f->ib.body),f->ids.bytes,f->ids.length)&&!memcmp(myds_metal_buffer_data(f->wb.body),f->weights.bytes,f->weights.length),"failed allocation input immutability");
 need([pool_identity(before)isEqual:pool_identity(after)]&&[live_before isEqual:live_allocations()],"failed lazy pool transaction preserves all owners/views");
 need(myds_metal_moe_workspace_test_control(0,1),"disable lazy NULL before same-handle retry");NSString *retry_name=[NSString stringWithFormat:@"lazy-%llu-retry",(unsigned long long)ordinal];NSData *actual=run_output(w,f,weights,clamp,retry_name);BOOL exact=[actual isEqual:oracle];NSDictionary *retry=stats(w);need(exact,"same-handle lazy retry bit parity");
 need([retry[@"scratch"][@"mm_encoded"]unsignedLongLongValue]==0&&[retry[@"scratch"][@"mv_encoded"]unsignedLongLongValue]>0,"retry uses masked MV");
 need(myds_v41_moe_workspace_probe_quiescent(w,false),"lazy quiescent");myds_v41_moe_workspace_probe_free(w);need([global_before isEqual:live_allocations()],"lazy final free accounting");
 NSDictionary *row=@{@"kind":@"fallback_allocation_failure",@"rows":@(f->rows),@"mask":@1,@"ordinal":@(ordinal),@"site":site?:@"",@"failure_hits":@(failures),@"real_oom":@NO,@"simulated_null":@YES,@"dispatches_before":@(dispatch_before),@"dispatches_after":@(dispatch_after),@"failed_output_unchanged":@(poison_unchanged),@"before":before,@"after_failure":after,@"allocation_trace":trace,@"oracle_stem":oracle_name,@"failed_output_stem":failed_name,@"retry_output_stem":retry_name,@"retry":retry,@"full_f32_bit_exact":@(exact),@"finite_guards_written":@YES,@"owner_accounting_restored":@YES};emit(row);return row;
}
static NSDictionary *missing_shared_sum(fixture *f,myds_metal_buffer_t **weights,float clamp) {
 NSMutableArray *arms=[NSMutableArray new];NSData *oracle=nil;
 for(unsigned compact=0;compact<2;compact++) {
  need(myds_metal_moe_workspace_test_control(0,0),"prepare valid provider before shared-sum failure");NSDictionary *live=live_allocations();
  myds_v41_moe_workspace_probe_t *w=myds_v41_moe_workspace_probe_create(32,2304,64,compact);need(w!=NULL,"prepared shared-sum workspace");
  need(myds_metal_moe_workspace_test_control(0,8),"simulate shared sum6 unavailable while idle");
  poison(f->out);need(myds_metal_begin_batch(),"missing shared sum begin");uint64_t before=myds_metal_dispatch_count();
  int ok=myds_v41_moe_workspace_probe_run(w,f->out.body,f->xb.body,weights[0],weights[1],weights[2],f->ib.body,f->wb.body,f->rows,clamp);
  uint64_t after=myds_metal_dispatch_count();need(myds_metal_commit_batch(),"missing shared sum empty batch completion");
  NSString *failed=[NSString stringWithFormat:@"shared-sum-missing-%s-failed",compact?"compact":"native"];dump_buffer(failed,f->out);
  BOOL unchanged=YES;const uint8_t *p=myds_metal_buffer_data(f->out.body);for(size_t i=0;i<f->out.bytes;i++)if(p[i]!=0xa5){unchanged=NO;break;}
  need(!ok&&before==after&&unchanged&&guards(f->out)&&guards(f->xb)&&guards(f->ib)&&guards(f->wb),"shared sum6 absence must fail closed before any dispatch/write");
  need(!memcmp(myds_metal_buffer_data(f->xb.body),f->x.bytes,f->x.length)&&!memcmp(myds_metal_buffer_data(f->ib.body),f->ids.bytes,f->ids.length)&&!memcmp(myds_metal_buffer_data(f->wb.body),f->weights.bytes,f->weights.length),"missing shared sum inputs unchanged");
  need(myds_metal_moe_workspace_test_control(0,0),"restore shared sum availability");NSString *retry=[NSString stringWithFormat:@"shared-sum-restored-%s",compact?"compact":"native"];
  NSData *result=run_output(w,f,weights,clamp,retry);BOOL exact=!compact||[result isEqual:oracle];if(!compact)oracle=result;
  [arms addObject:@{@"provider":compact?@"compact":@"native",@"returned_failure":@YES,@"dispatches_before":@(before),@"dispatches_after":@(after),@"failed_output_unchanged":@(unchanged),@"failed_output_stem":failed,@"recovery_output_stem":retry,@"same_handle_recovery":@YES,@"full_f32_bit_exact":@(exact),@"finite_guards_written":@YES}];
  need(myds_v41_moe_workspace_probe_quiescent(w,false),"shared sum recovery quiescent");myds_v41_moe_workspace_probe_free(w);need([live isEqual:live_allocations()],"shared sum final ownership");need(exact,"shared sum same-handle recovered full parity");
 }
 NSDictionary *row=@{@"kind":@"shared_pso_failclosed",@"mask":@8,@"rows":@(f->rows),@"shared_dependency":@"sum6",@"provider_when_unavailable":@"NONE",@"arms":arms,@"timing_measured":@NO};emit(row);return row;
}
static NSDictionary *long_short(fixture *long_f,fixture *short_f,myds_metal_buffer_t **weights,float clamp) {
 need(myds_metal_moe_workspace_test_control(0,0),"long short reset");NSDictionary *live=live_allocations();
 myds_v41_moe_workspace_probe_t *reference=myds_v41_moe_workspace_probe_create(32,2304,64,false);need(reference!=NULL,"short authoritative factory");NSData *oracle=run_output(reference,short_f,weights,clamp,@"long-short-short-native");myds_v41_moe_workspace_probe_free(reference);
 myds_v41_moe_workspace_probe_t *w=myds_v41_moe_workspace_probe_create(32,2304,long_f->rows,true);need(w!=NULL,"long reused factory");
 (void)run_output(w,long_f,weights,clamp,@"long-short-long-compact");NSDictionary *long_stats=stats(w);
 NSData *short_result=run_output(w,short_f,weights,clamp,@"long-short-short-compact");NSDictionary *short_stats=stats(w);BOOL exact=[short_result isEqual:oracle];
 need(exact&&[long_stats[@"generation"]isEqual:short_stats[@"generation"]],"same-handle long then short exact output");
 need(myds_v41_moe_workspace_probe_quiescent(w,false),"long short retirement");myds_v41_moe_workspace_probe_free(w);need([live isEqual:live_allocations()],"long short final free");
 NSDictionary *row=@{@"kind":@"long_short",@"long_rows":@(long_f->rows),@"short_rows":@(short_f->rows),@"same_handle":@YES,@"long_workspace":long_stats,@"short_workspace":short_stats,@"native_short_stem":@"long-short-short-native",@"compact_short_stem":@"long-short-short-compact",@"full_f32_bit_exact":@(exact),@"finite_guards_written":@YES,@"timing_measured":@NO};emit(row);return row;
}
int main(int argc,char **argv){@autoreleasepool {
 NSString *model=nil,*capture_path=nil;BOOL run=NO,host=NO;files=[NSMutableArray new];manifest=[NSMutableDictionary new];
 for(int i=1;i<argc;i++){NSString *k=[NSString stringWithUTF8String:argv[i]];if([k isEqual:@"--run"]){run=YES;continue;}if([k isEqual:@"--host-only"]){host=YES;continue;}need(++i<argc,"diagnostic option value");NSString *v=[NSString stringWithUTF8String:argv[i]];if([k isEqual:@"--model"])model=canonical(v);else if([k isEqual:@"--capture-json"])capture_path=canonical(v);else if([k isEqual:@"--out-dir"])output=canonical(v);else need(NO,"unknown diagnostic option");}
 need(run!=host&&model.isAbsolutePath&&capture_path.isAbsolutePath&&output.isAbsolutePath,"explicit run/host and absolute model/capture/new out");need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:NO attributes:nil error:nil],"new diagnostic output");
 manifest[@"schema_version"]=@1;manifest[@"probe"]=@"real_prefill_moe_workspace_failures";manifest[@"status"]=@"running";manifest[@"files"]=files;manifest[@"binary_sha256"]=file_sha(binary());manifest[@"capture_json_sha256"]=file_sha(capture_path);manifest[@"real_oom"]=@NO;manifest[@"timing_measured"]=@NO;manifest[@"fullmodel_cancel_recovery_proven"]=@NO;
 NSDictionary *capture=json_file(capture_path);need([capture[@"capture_kind"]isEqual:@"myds_prefill_moe_workspace"]&&[capture[@"source"]isKindOfClass:NSDictionary.class],"real prefilling capture provenance");
 gguf_file_t *gf=gguf_open(model.fileSystemRepresentation);need(gf!=NULL,"diagnostic real model");struct stat ms;need(stat(model.fileSystemRepresentation,&ms)==0&&ms.st_size>=1048576&&[capture[@"model_bytes"]unsignedLongLongValue]==(uint64_t)ms.st_size&&[capture[@"model_header_bytes"]unsignedIntValue]==1048576&&[sha(gf->mmap_addr,1048576)isEqual:capture[@"model_header_sha256"]],"diagnostic exact actual model identity");
 NSDictionary *source=nil;for(NSDictionary *r in capture[@"cases"])if([r[@"layer"]unsignedIntValue]==0&&[r[@"rows"]unsignedIntValue]>=16384){source=r;break;}need(source!=nil,"real layer0 full B16384 source");unsigned rows=[source[@"rows"]unsignedIntValue];need([source[@"all_finite"]boolValue]&&![source[@"trace_on"]boolValue]&&[source[@"observer_batch_restored"]boolValue],"production provider source");
 NSString *base=capture_path.stringByDeletingLastPathComponent;NSData *x=payload(source,@"input_f32",base,(size_t)rows*20480u),*ids=payload(source,@"ids_i32",base,(size_t)rows*24u),*rw=payload(source,@"routeweights_f32",base,(size_t)rows*24u);need(finite_f32(x.bytes,x.length)&&finite_f32(rw.bytes,rw.length),"finite real diagnostic inputs");
 const int32_t *id=ids.bytes;for(unsigned i=0;i<rows;i++)for(unsigned j=0;j<6;j++){need(id[i*6+j]>=0&&id[i*6+j]<384,"diagnostic IDs range");for(unsigned k=0;k<j;k++)need(id[i*6+j]!=id[i*6+k],"diagnostic top6 unique");}
 const char *arch=gguf_get_str(gf,"general.architecture",NULL);need(arch!=NULL,"diagnostic architecture");char key[256];snprintf(key,sizeof(key),"%s.swiglu_limit",arch);float clamp=gguf_get_f32(gf,key,10.f),captured_clamp=[source[@"clamp"]floatValue];need(isfinite(clamp)&&clamp>0&&!memcmp(&clamp,&captured_clamp,4),"actual diagnostic clamp");
 manifest[@"model"]=@{@"path":model,@"bytes":@((uint64_t)ms.st_size),@"header_sha256":sha(gf->mmap_addr,1048576)};manifest[@"source_case"]=source;NSMutableArray *tensor_identity=[NSMutableArray new];manifest[@"model_backed_tensors"]=tensor_identity;
 const gguf_tensor_t *tensor[3];myds_metal_buffer_t *weights[3]={0};const char *kind[]={"gate","up","down"};
 for(unsigned k=0;k<3;k++){char name[128];snprintf(name,sizeof(name),"blk.0.ffn_%s_exps.weight",kind[k]);tensor[k]=gguf_find_tensor(gf,name);const gguf_tensor_t *t=tensor[k];need(t&&t->n_dims==3&&t->type==(k==2?GGML_TYPE_Q2_K:GGML_TYPE_IQ2_XXS)&&t->dims[0]==(k==2?2304u:5120u)&&t->dims[1]==(k==2?5120u:2304u)&&t->dims[2]==384,"diagnostic real expert shape");[tensor_identity addObject:@{@"name":[NSString stringWithUTF8String:name],@"bytes":@(t->size_bytes),@"file_offset":@(gf->tensor_data_offset+t->offset),@"sha256":sha(t->data,t->size_bytes)}];}
 NSMutableArray *results=[NSMutableArray new];manifest[@"results"]=results;
 if(run){need(getenv("MYDS_METAL_CONCURRENT")&&!strcmp(getenv("MYDS_METAL_CONCURRENT"),"1"),"actual production concurrent mode");need(myds_metal_init(NULL),"diagnostic Metal init");need(!myds_metal_kprof_exclusive(),"no kernel profiler");myds_metal_set_auto_barrier(0);manifest[@"metallib_path"]=[NSString stringWithUTF8String:myds_metal_loaded_library_path()];manifest[@"metallib_sha256"]=file_sha(manifest[@"metallib_path"]);
  for(unsigned k=0;k<3;k++){weights[k]=myds_metal_buffer_from_ptr(tensor[k]->data,tensor[k]->size_bytes);need(weights[k]!=NULL,"diagnostic tensor binding");}
  fixture small=derived_fixture(source,x,ids,rw,33),short_f=derived_fixture(source,x,ids,rw,31),large=derived_fixture(source,x,ids,rw,16384);
  for(uint32_t mask=1;mask<=4;mask<<=1)[results addObject:mask_case(&small,weights,clamp,mask)];
  [results addObject:missing_shared_sum(&small,weights,clamp)];
  [results addObject:mask_case(&large,weights,clamp,1)];
  for(uint64_t ordinal=1;ordinal<=3;ordinal++)[results addObject:lazy_failure(&small,weights,clamp,ordinal)];
  [results addObject:long_short(&large,&short_f,weights,clamp)];
  release_fixture(&small);release_fixture(&short_f);release_fixture(&large);need(myds_metal_moe_workspace_test_control(0,0),"diagnostic controls restored");
  for(unsigned k=0;k<3;k++){need([sha(tensor[k]->data,tensor[k]->size_bytes)isEqual:tensor_identity[k][@"sha256"]],"diagnostic complete expert input immutability");myds_metal_buffer_free(weights[k]);}myds_metal_cleanup();
 }
 manifest[@"status"]=@"complete";manifest[@"phase"]=run?@"functional_failures":@"host_only";manifest[@"full_f32_bit_exact"]=@(run);manifest[@"mode"]=@"concurrent";
 NSData *j=[NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];need([j writeToFile:[output stringByAppendingPathComponent:@"artifact-manifest.json"] options:NSDataWritingWithoutOverwriting error:nil],"diagnostic complete manifest");emit(@{@"kind":@"complete",@"cases":@(results.count),@"full_f32_bit_exact":@(run),@"real_oom":@NO,@"timing_measured":@NO,@"fullmodel_cancel_recovery_proven":@NO});gguf_close(gf);return 0;
}}
