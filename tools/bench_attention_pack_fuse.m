/* Standalone complete-buffer direct-pack component gate. Build/host-only is CPU
 * only. Full mode needs an exclusive GPU window. Inputs are seeded fixtures. */
#define main packed_attention_unused_main
#include "../tests/test_attention_packed.m"
#undef main
#include <CommonCrypto/CommonDigest.h>
static NSMutableDictionary *files;
static NSMutableArray *cases;
static NSString *gold_file,*candidate_file,*identity_file;
static id<MTLLibrary> pack_golden;
static id<MTLComputePipelineState> pack_reference[2];
static const size_t pack_capacity=640u*512u*2u;
static NSString *digest(NSData *data) {
    unsigned char h[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data.bytes,(CC_LONG)data.length,h);
    NSMutableString *s=[NSMutableString new];for(unsigned i=0;i<sizeof(h);i++)[s appendFormat:@"%02x",h[i]];return s;
}
static NSString *file_digest(NSString *path) {
    NSData *d=[NSData dataWithContentsOfFile:path];REQUIRE(d);return digest(d);
}
static void emit(NSDictionary *d) {
    NSData *j=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:nil];REQUIRE(j);
    fwrite(j.bytes,1,j.length,stdout);putchar('\n');fflush(stdout);
}
static void manifest(NSString *status) {
    if(!output_dir)return;
    NSDictionary *d=@{@"schema_version":@1,@"status":status,@"fixture_not_fullmodel":@YES,
      @"golden_metallib":@{@"path":gold_file,@"sha256":file_digest(gold_file)},
      @"candidate_metallib":@{@"path":candidate_file,@"sha256":file_digest(candidate_file)},
      @"source_identity":@{@"path":identity_file,@"sha256":file_digest(identity_file)},
      @"args_bytes":@16,@"threads":@[@128,@1,@1],@"dynamic_bytes":@0,
      @"golden_symbols":@[@"kernel_dsv41_copy_window_f32_to_f16",@"kernel_dsv41_get_rows_bf16_to_f16"],
      @"candidate_symbol":@(attention_pack_kernel),@"files":files,@"cases":cases};
    NSData *j=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];REQUIRE(j);
    NSString *path=[@(output_dir) stringByAppendingPathComponent:@"artifact-manifest.json"];REQUIRE([j writeToFile:path atomically:YES]);
}
static NSString *save(unsigned number,NSString *name,struct guarded b) {
    NSString *n=[NSString stringWithFormat:@"case-%03u-%@.bin",number,name];
    NSData *d=[NSData dataWithBytes:myds_metal_buffer_data(b.owner) length:b.bytes+32u];
    REQUIRE([d writeToFile:[@(output_dir) stringByAppendingPathComponent:n] options:NSDataWritingWithoutOverwriting error:nil]);
    files[n]=@{@"bytes":@(d.length),@"sha256":digest(d),@"fnv1a64":[NSString stringWithFormat:@"%016" PRIx64,fnv(d.bytes,d.length)],@"view_offset":@16,@"view_bytes":@(b.bytes)};
    return n;
}
static void destroy(struct guarded b) {canary(b);myds_metal_buffer_free(b.view);myds_metal_buffer_free(b.owner);}
static void host_reset(void) {attention_pack_once=(pthread_once_t)PTHREAD_ONCE_INIT;attention_pack_mode=0;}
static void host_only(void) {
    REQUIRE(!g_device&&!g_queue&&!g_library&&!g_batch_cb);setenv("MYDS_ATTN_PACKED_REGISTER","1",1);setenv("MYDS_ATTN_ROPE_FUSE","1",1);
    unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");unsetenv("MYDS_ATTN_DIRECT_PACK");
    unsetenv("MYDS_ATTN_DIRECT_PACK_FUSE");host_reset();REQUIRE(myds_v41_gpu_attention_direct_pack_fuse_enabled()==0);
    const char *valid[]={"0","1"};for(unsigned i=0;i<2;i++){setenv("MYDS_ATTN_DIRECT_PACK_FUSE",valid[i],1);host_reset();REQUIRE(myds_v41_gpu_attention_direct_pack_fuse_enabled()==(int)i);}
    const char *bad[]={"","2","01","true","1 ","-1"};
    for(unsigned i=0;i<6;i++){setenv("MYDS_ATTN_DIRECT_PACK_FUSE",bad[i],1);host_reset();REQUIRE(myds_v41_gpu_attention_direct_pack_fuse_enabled()==-1&&myds_v41_gpu_scratch_create(128)==NULL);}
    setenv("MYDS_ATTN_DIRECT_PACK_FUSE","1",1);
    const char *keys[]={"MYDS_ATTN_DIRECT_PACK","MYDS_ATTN_DIRECT_PACK","MYDS_ATTN_ROPE_FUSE","MYDS_ATTN_PACKED_REGISTER","MYDS_ATTN_PACKED_NSG","MYDS_ATTN_PACKED_REDUCE"};
    const char *vals[]={"0","2","0","0","4","0"};
    for(unsigned i=0;i<6;i++){
        setenv(keys[i],vals[i],1);host_reset();REQUIRE(myds_v41_gpu_attention_direct_pack_fuse_enabled()==-1&&myds_v41_gpu_scratch_create(128)==NULL);
        unsetenv("MYDS_ATTN_DIRECT_PACK");setenv("MYDS_ATTN_ROPE_FUSE","1",1);setenv("MYDS_ATTN_PACKED_REGISTER","1",1);unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");
    }
    host_reset();REQUIRE(myds_v41_gpu_attention_direct_pack_fuse_enabled()==1);setenv("MYDS_ATTN_DIRECT_PACK_FUSE","0",1);REQUIRE(myds_v41_gpu_attention_direct_pack_fuse_enabled()==1); /* fixed configuration */
    uint8_t storage[8][256],prior[sizeof(storage)];memset(storage,0xa5,sizeof(storage));memcpy(prior,storage,sizeof(storage));
    myds_metal_buffer_t b[8]={0};for(unsigned i=0;i<8;i++){b[i].ptr=storage[i]+16;b[i].size=224;}
    REQUIRE(!attention_direct_pack_preflight(&b[0],&b[1],1,0,&b[2],&b[3],33,1));
    myds_v41_gpu_scratch_t scratch={0},original;scratch.kv_f16=&b[4];scratch.pad=&b[5];scratch.partial=&b[6];scratch.mask=&b[7];original=scratch;
    /* Public direct has rope=NULL, staged has no pack selector, and trace/other
     * scopes take their existing public entries. Fake wrappers reject without
     * reaching any pack dispatch; all complete storage/scratch bytes remain. */
    REQUIRE(!myds_v41_gpu_attention_direct(&b[0],&b[1],&b[2],&b[3],1,0,NULL,NULL,0,0,&scratch));
    REQUIRE(!myds_v41_gpu_attention(&b[0],&b[1],&b[2],&b[3],1,0,NULL,0,&scratch));
    bool consumed=true;REQUIRE(!myds_v41_gpu_attention_direct_rope_fuse(&b[0],&b[1],&b[2],&b[3],1,0,NULL,NULL,0,0,&scratch,0,false,true,true,&consumed)&&!consumed);
    REQUIRE(!memcmp(storage,prior,sizeof(storage))&&!memcmp(&scratch,&original,sizeof(scratch))&&myds_metal_dispatch_count()==0);
    myds_metal_decode_candidate_counters_t c;myds_v41_gpu_attention_direct_pack_fuse_counters(&c,false);REQUIRE(!c.requests&&!c.encoded&&!c.failures);
    emit(@{@"kind":@"host_only",@"pass":@YES,@"GPU_work_submitted":@NO,@"strict_flags_and_dependency_conflicts":@YES,@"configuration_cached":@YES,@"public_staged_trace_pack_requests":@0,@"dispatch_delta":@0,@"full_storage_and_scratch_same":@YES});
}
static id<MTLComputePipelineState> make_pso(id<MTLLibrary> library,const char *name) {
    id<MTLFunction> f=[library newFunctionWithName:@(name)];REQUIRE(f);NSError *error=nil;
    id<MTLComputePipelineState> p=[g_device newComputePipelineStateWithFunction:f error:&error];REQUIRE(p&&p.device==g_device&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128);return p;
}
static void reference_encode(struct guarded dst,struct guarded window,struct guarded comp,struct guarded ids,uint32_t nr,uint32_t start,uint32_t source,uint32_t selected) {
    id<MTLComputeCommandEncoder> e=get_compute_encoder(g_batch_cb);REQUIRE(e);
    uint32_t a[2]={nr,start};[e setComputePipelineState:pack_reference[0]];[e setBytes:a length:sizeof(a) atIndex:0];
    [e setBuffer:window.view->mtl_buf offset:window.view->offset atIndex:1];[e setBuffer:dst.view->mtl_buf offset:dst.view->offset atIndex:2];metal_dispatch(e,"kernel_dsv41_copy_window_f32_to_f16",MTLSizeMake(1,nr,1),MTLSizeMake(128,1,1));
    if(selected){uint32_t b[4]={source,selected,nr,0};[e setComputePipelineState:pack_reference[1]];[e setBytes:b length:sizeof(b) atIndex:0];[e setBuffer:comp.view->mtl_buf offset:comp.view->offset atIndex:1];[e setBuffer:ids.view->mtl_buf offset:ids.view->offset atIndex:2];[e setBuffer:dst.view->mtl_buf offset:dst.view->offset atIndex:3];metal_dispatch(e,"kernel_dsv41_get_rows_bf16_to_f16",MTLSizeMake(1,selected,1),MTLSizeMake(128,1,1));}
}
static void gpu_preflight_negatives(void) {
    struct guarded w=allocate(128u*512u*4u),c=allocate(531u*512u*2u),ids=allocate(512u*4u),out=allocate(pack_capacity);
    const uint64_t before=myds_metal_dispatch_count();
    NSData *storage=[NSData dataWithBytes:myds_metal_buffer_data(out.owner) length:out.bytes+32u];
    REQUIRE(attention_direct_pack_preflight(out.view,w.view,128,127,c.view,ids.view,531,512));
    unsigned rejects=0;
#define REJECT(expr) do {REQUIRE(!(expr));rejects++;REQUIRE(myds_metal_dispatch_count()==before&&!memcmp(storage.bytes,myds_metal_buffer_data(out.owner),storage.length));}while(0)
    REJECT(attention_direct_pack_preflight(out.view,w.view,0,0,c.view,ids.view,531,1));
    REJECT(attention_direct_pack_preflight(out.view,w.view,129,0,c.view,ids.view,531,1));
    REJECT(attention_direct_pack_preflight(out.view,w.view,1,128,c.view,ids.view,531,1));
    REJECT(attention_direct_pack_preflight(out.view,w.view,1,0,c.view,ids.view,531,513));
    REJECT(attention_direct_pack_preflight(out.view,w.view,1,0,c.view,ids.view,0,1));
    REJECT(attention_direct_pack_preflight(out.view,w.view,1,0,c.view,ids.view,1,2));
    REJECT(attention_direct_pack_preflight(w.view,w.view,1,0,c.view,ids.view,531,1));
    REJECT(attention_direct_pack_preflight(c.view,w.view,1,0,c.view,ids.view,531,1));
    REJECT(attention_direct_pack_preflight(ids.view,w.view,1,0,c.view,ids.view,531,1));
    const size_t size=out.view->size;out.view->size=1024;REJECT(attention_direct_pack_preflight(out.view,w.view,1,0,c.view,ids.view,531,1));out.view->size=size;
    void *ptr=w.view->ptr;w.view->ptr=(char *)ptr+16;REJECT(attention_direct_pack_preflight(out.view,w.view,1,0,c.view,ids.view,531,1));w.view->ptr=ptr;
    /* Exact resolver caches must both be isolated to prove the absent golden
     * symbol rejection, rather than accidentally finding a cached candidate. */
    id<MTLLibrary> library=g_library;NSMutableDictionary *dictionary=g_native_pipelines;
    const size_t cached=g_native_pso_cache_count;g_native_pso_cache_count=0;g_native_pipelines=[NSMutableDictionary new];g_library=pack_golden;
    REJECT(attention_direct_pack_preflight(out.view,w.view,1,0,c.view,ids.view,531,1));
    g_library=library;g_native_pipelines=dictionary;g_native_pso_cache_count=cached;
#undef REJECT
    canary(w);canary(c);canary(ids);canary(out);
    NSString *blob=save(999,@"predispatch-poison-storage",out);manifest(@"pending");
    emit(@{@"kind":@"predispatch_rejections",@"rejects":@(rejects),@"zero_dispatch_delta":@YES,@"full_output_storage_same":@YES,@"missing_selected_symbol_rejected":@YES,@"poison_storage_file":blob});
    destroy(w);destroy(c);destroy(ids);destroy(out);
}
static void run_case(unsigned number,uint32_t nr,uint32_t selected,uint32_t start,unsigned invalid,unsigned concurrent) {
    const uint32_t source=selected?selected+19u:0u;struct guarded w=allocate(128u*512u*4u),c=allocate((size_t)(source?source:1u)*512u*2u),ids=allocate((selected?selected:1u)*4u),out[2]={allocate(pack_capacity),allocate(pack_capacity)};
    float *wp=myds_metal_buffer_data(w.view);uint16_t *cp=myds_metal_buffer_data(c.view);int32_t *ip=myds_metal_buffer_data(ids.view);
    for(size_t i=0;i<128u*512u;i++){uint32_t bits=0x3e800001u+(uint32_t)(i%1024u)*67u;if(i%7u==0)bits^=0x80000000u;if(i%19u==0)bits=0x80000000u;memcpy(wp+i,&bits,4);}
    for(size_t i=0;i<(size_t)(source?source:1u)*512u;i++)cp[i]=(uint16_t)(0x3e80u+(i%128u));
    for(uint32_t i=0;i<(selected?selected:1u);i++)ip[i]=!selected?(invalid?-1:0):invalid?(i%5u==0?-1:i%5u==1?(int32_t)source:i%5u==2?INT32_MAX:i%5u==3?INT32_MIN:(int32_t)(i%source)):(int32_t)((i*53u+7u)%source);
    NSData *ws=[NSData dataWithBytes:myds_metal_buffer_data(w.owner) length:w.bytes+32],*cs=[NSData dataWithBytes:myds_metal_buffer_data(c.owner) length:c.bytes+32],*is=[NSData dataWithBytes:myds_metal_buffer_data(ids.owner) length:ids.bytes+32];
    myds_metal_set_concurrent(concurrent);myds_metal_set_auto_barrier(0);REQUIRE(attention_direct_pack_preflight(out[1].view,w.view,nr,start,c.view,ids.view,source,selected));
    uint64_t dc=myds_metal_dispatch_count();myds_v41_gpu_attention_direct_pack_fuse_counters(NULL,true);
    REQUIRE(myds_metal_begin_batch());reference_encode(out[0],w,c,ids,nr,start,source,selected);myds_metal_barrier_buffer(out[0].view);REQUIRE(myds_metal_commit_batch());REQUIRE(myds_metal_dispatch_count()-dc==1u+!!selected);
    dc=myds_metal_dispatch_count();REQUIRE(myds_metal_begin_batch());atomic_fetch_add_explicit(&attention_pack_counters[AP_REQUESTS],1u,memory_order_relaxed);REQUIRE(attention_direct_pack_encode(out[1].view,w.view,nr,start,c.view,ids.view,source,selected));myds_metal_barrier_buffer(out[1].view);REQUIRE(myds_metal_commit_batch());REQUIRE(myds_metal_dispatch_count()-dc==1);
    NSString *wf=save(number,@"window-storage",w),*cf=save(number,@"compressed-storage",c),*ifile=save(number,@"ids-storage",ids),*a=save(number,@"golden-storage",out[0]),*b=save(number,@"candidate-storage",out[1]);
    NSDictionary *item=@{@"case":@(number),@"n_raw":@(nr),@"selected_rows":@(selected),@"raw_start":@(start),@"source_rows":@(source),@"invalid_ids":@(invalid),@"concurrent":@(concurrent),@"window":wf,@"compressed":cf,@"ids":ifile,@"golden":a,@"candidate":b};[cases addObject:item];manifest(@"pending");
    REQUIRE(!memcmp(myds_metal_buffer_data(out[0].owner),myds_metal_buffer_data(out[1].owner),pack_capacity+32));
    REQUIRE(!memcmp(ws.bytes,myds_metal_buffer_data(w.owner),ws.length)&&!memcmp(cs.bytes,myds_metal_buffer_data(c.owner),cs.length)&&!memcmp(is.bytes,myds_metal_buffer_data(ids.owner),is.length));
    const uint16_t *h=myds_metal_buffer_data(out[1].view);for(size_t i=0;i<(size_t)(nr+selected)*512u;i++)REQUIRE((h[i]&0x7c00u)!=0x7c00u&&h[i]!=0xa5a5u);
    myds_metal_decode_candidate_counters_t count;myds_v41_gpu_attention_direct_pack_fuse_counters(&count,false);REQUIRE(count.requests==1&&count.encoded==1&&!count.failures);
    emit(@{@"kind":@"pack_parity",@"case":@(number),@"full_storage_bytes":@(pack_capacity+32),@"full_pack_bytes_equal":@YES,@"finite_active_half":@YES,@"poison_free_active":@YES,@"inputs_full_storage_same":@YES,@"canaries":@YES,@"reference_dispatches":@(1u+!!selected),@"candidate_dispatches":@1,@"requests":@1,@"encoded":@1,@"failures":@0});
    destroy(w);destroy(c);destroy(ids);destroy(out[0]);destroy(out[1]);
}
int main(int argc,char **argv) {@autoreleasepool {
    bool host=false;for(int i=1;i<argc;i++){
      if(!strcmp(argv[i],"--host-only"))host=true;
      else if(!strcmp(argv[i],"--golden-metallib")&&i+1<argc)gold_file=@(argv[++i]);
      else if(!strcmp(argv[i],"--candidate-metallib")&&i+1<argc)candidate_file=@(argv[++i]);
      else if(!strcmp(argv[i],"--source-identity")&&i+1<argc)identity_file=@(argv[++i]);
      else if(!strcmp(argv[i],"--output-dir")&&i+1<argc)output_dir=argv[++i];
      else {fprintf(stderr,"usage: %s --host-only OR --golden-metallib FILE --candidate-metallib FILE --source-identity FILE --output-dir NEW\n",argv[0]);return 2;}}
    if(host){host_only();return 0;}REQUIRE(gold_file&&candidate_file&&identity_file&&output_dir);struct stat st;REQUIRE(lstat(output_dir,&st)!=0&&errno==ENOENT&&mkdir(output_dir,0700)==0);
    setenv("MYDS_ATTN_PACKED_REGISTER","1",1);setenv("MYDS_ATTN_ROPE_FUSE","1",1);setenv("MYDS_ATTN_DIRECT_PACK_FUSE","1",1);unsetenv("MYDS_ATTN_DIRECT_PACK");unsetenv("MYDS_ATTN_PACKED_NSG");unsetenv("MYDS_ATTN_PACKED_REDUCE");unsetenv("MYDS_DEBUG_DISPATCH");g_hist_enabled=0;
    NSString *dir=[candidate_file stringByDeletingLastPathComponent];REQUIRE([[candidate_file lastPathComponent]isEqual:@"myds.metallib"]);REQUIRE(myds_metal_init(dir.UTF8String));
    REQUIRE(myds_v41_gpu_attention_direct_pack_fuse_enabled()==1);NSError *error=nil;pack_golden=[g_device newLibraryWithURL:[NSURL fileURLWithPath:gold_file] error:&error];REQUIRE(pack_golden);
    pack_reference[0]=make_pso(pack_golden,"kernel_dsv41_copy_window_f32_to_f16");pack_reference[1]=make_pso(pack_golden,"kernel_dsv41_get_rows_bf16_to_f16");files=[NSMutableDictionary new];cases=[NSMutableArray new];
    emit(@{@"kind":@"library_identity",@"golden_path":gold_file,@"golden_sha256":file_digest(gold_file),@"candidate_path":candidate_file,@"candidate_sha256":file_digest(candidate_file),@"loaded_runtime_path":@(myds_metal_loaded_library_path()),@"candidate_symbol":@(attention_pack_kernel),@"source_identity_sha256":file_digest(identity_file),@"fixture_not_fullmodel":@YES});
    gpu_preflight_negatives();
    const uint32_t nr[]={1,33,128},sel[]={0,1,33,512},start[]={0,127};unsigned n=0;
    for(unsigned concurrent=0;concurrent<2;concurrent++)for(unsigned r=0;r<3;r++)for(unsigned s=0;s<4;s++)for(unsigned ring=0;ring<2;ring++)for(unsigned invalid=0;invalid<2;invalid++)run_case(n++,nr[r],sel[s],start[ring],invalid,concurrent);
    manifest(@"passed");emit(@{@"kind":@"result",@"pass":@YES,@"cases":@(n),@"fixture_not_fullmodel":@YES});myds_metal_cleanup();return 0;
}}
