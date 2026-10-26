/* Synthetic, isolated NE=NSG=NWG=1 prefill output-register experiment.
 * Build/run through build_prefill_output_register_probe.py.  The included
 * capture harness supplies SHA/raw/guard/dispatch helpers without starting
 * the inference runtime. Its original main is not called by this executable.
 * This fixture is deliberately synthetic; real-capture and full-model gates
 * are separate.  Native and candidate use the same metallib and FC values.
 */
#define main prefill_multihead_capture_main_unused
#include "bench_attention_multihead.m"
#undef main

static uint32_t fixture_word(uint64_t i,uint32_t seed) {
    uint32_t x=(uint32_t)(i+1u)*747796405u+seed*2891336453u;
    x=((x>>((x>>28)+4))^x)*277803737u;
    return (x>>22)^x;
}
static float fixture_float(uint64_t i,uint32_t seed) {
    return ((int32_t)(fixture_word(i,seed)%8191u)-4095)/4096.0f;
}
static fixture synthetic_fixture(id<MTLDevice> device,unsigned index,unsigned rows,unsigned pattern) {
    fixture f={0};f.index=index;f.rows=rows;
    f.args=(struct flash_args){.ne01=rows,.ne02=64,.ne03=1,.nb01=QUERY_BYTES,.nb02=2048,.nb03=QUERY_BYTES,
        .ne11=KEYS,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=1024,.nb12=KV_QUERY_BYTES,.nb13=KV_QUERY_BYTES,
        .ns20=512,.nb21=1024,.nb22=KV_QUERY_BYTES,.nb23=KV_QUERY_BYTES,.ne31=rows,.ne32=1,.ne33=1,
        .nb31=1280,.nb32=(uint64_t)1280*rows,.nb33=(uint64_t)1280*rows,.ne1=64,.ne2=rows,.ne3=1,
        .scale=1.0f/sqrtf(512.f),.nb_kvq=KV_QUERY_BYTES};
    need(valid_args(&f.args,rows),"synthetic narrow args");
    size_t qbytes=(size_t)rows*QUERY_BYTES,kvbytes=(size_t)rows*KV_QUERY_BYTES,maskbytes=(size_t)rows*1280;
    f.q=alloc(device,qbytes,NO);f.kv=alloc(device,kvbytes,NO);f.mask=alloc(device,maskbytes,NO);
    f.sinks=alloc(device,SINK_BYTES,NO);f.pad=alloc(device,16,YES);
    float *q=body(f.q),*s=body(f.sinks);__fp16 *kv=body(f.kv);uint16_t *m=body(f.mask);
    for(size_t i=0;i<qbytes/4;i++)q[i]=fixture_float(i,17u+index);
    for(size_t i=0;i<kvbytes/2;i++)kv[i]=(__fp16)fixture_float(i,37u+index);
    for(unsigned h=0;h<64;h++)s[h]=((int)h-31)/4.0f;
    for(unsigned r=0;r<rows;r++)for(unsigned k=0;k<KEYS;k++) {
        BOOL valid=YES;
        if(pattern==1)valid=k<(r*13u%640u)+1u;
        if(pattern==2)valid=(r%7u!=0u)&&(k%3u!=1u)&&(k<128u||k>=192u);
        if(pattern==3)valid=NO;
        m[(size_t)r*KEYS+k]=valid?0u:0xfc00u;
    }
    f.out[0]=alloc(device,qbytes,YES);f.out[2]=alloc(device,qbytes,YES);
    [manifest[@"cases"] addObject:@{@"index":@(index),@"rows":@(rows),@"mask_pattern":@(pattern),
        @"q_seed":@(17u+index),@"kv_seed":@(37u+index),@"q_sha256":sha(q,qbytes),
        @"kv_sha256":sha(kv,kvbytes),@"mask_sha256":sha(m,maskbytes),@"sinks_sha256":sha(s,SINK_BYTES),
        @"args_sha256":sha(&f.args,sizeof(f.args)),@"output_bytes_per_provider":@(qbytes)}];
    return f;
}
static void inputs_unchanged(fixture *f,NSDictionary *before) {
    const size_t qbytes=(size_t)f->rows*QUERY_BYTES,kvbytes=(size_t)f->rows*KV_QUERY_BYTES,maskbytes=(size_t)f->rows*1280;
    need([sha(body(f->q),qbytes)isEqual:before[@"q_sha256"]]&&
         [sha(body(f->kv),kvbytes)isEqual:before[@"kv_sha256"]]&&
         [sha(body(f->mask),maskbytes)isEqual:before[@"mask_sha256"]]&&
         [sha(body(f->sinks),SINK_BYTES)isEqual:before[@"sinks_sha256"]],"complete synthetic inputs unchanged");
    need(guards(f->q,qbytes)&&guards(f->kv,kvbytes)&&guards(f->mask,maskbytes)&&
         guards(f->sinks,SINK_BYTES)&&guards(f->pad,16),"all synthetic input guards");
}
int main(int argc,char **argv){@autoreleasepool {
    need(argc==5&&!strcmp(argv[1],"--metallib")&&!strcmp(argv[3],"--out-dir"),
         "usage: bench_prefill_output_register --metallib PATH --out-dir NEW");
    NSString *library=canonical([NSString stringWithUTF8String:argv[2]]);
    output=canonical([NSString stringWithUTF8String:argv[4]]);
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new output directory required");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output directory");
    files=[NSMutableArray new];stored_payloads=[NSMutableDictionary new];
    manifest=[@{@"schema_version":@1,@"probe":@"prefill_output_register_synthetic",
        @"fixture_kind":@"deterministic_synthetic",@"cases":[NSMutableArray new],@"fullmodel":@NO,
        @"production_registered":@NO,@"cold_dram_proven":@NO,@"mode":@"concurrent",
        @"metallib_path":library,@"metallib_sha256":file_sha(library),@"binary_path":binary_path(),
        @"binary_sha256":file_sha(binary_path()),@"samples":@5,@"warmups":@1}mutableCopy];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");
    manifest[@"device"]=device.name;
    NSError *error=nil;id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:library]error:&error];
    if(!lib)fprintf(stderr,"library: %s\n",error.localizedDescription.UTF8String);need(lib!=nil,"probe library");
    NSString *names[3]={@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",nil,
        @"kernel_myds_prefill_output_register_f16_dk512_dv512"};
    id<MTLComputePipelineState> p[3]={pipeline(device,lib,names[0],32,3328),nil,pipeline(device,lib,names[2],32,3328)};
    manifest[@"providers"]=@[@{@"index":@0,@"symbol":names[0]},@{@"index":@2,@"symbol":names[2]}];
    manifest[@"dynamic_threadgroup_bytes"]=@3328;
    manifest[@"candidate_static_threadgroup_bytes"]=@(p[2].staticThreadgroupMemoryLength);
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"command queue");
    const unsigned sizes[]={1,31,33,127,512,2048},patterns[]={0,1,2,3,0,0};
    for(unsigned i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++){@autoreleasepool {
        fixture f=synthetic_fixture(device,i,sizes[i],patterns[i]);
        NSDictionary *before=[manifest[@"cases"]lastObject];
        (void)run(queue,&f,0,1,p[0],1,3328);validate(&f,0,YES);
        (void)run(queue,&f,2,1,p[2],1,3328);validate(&f,2,YES);
        if(sizes[i]>=512) {
            double times[3][64];const unsigned providers[]={0,2,0};
            for(unsigned sample=0;sample<=5;sample++)for(unsigned arm=0;arm<3;arm++) {
                unsigned provider=providers[arm];
                id<MTLCommandBuffer> cb=run(queue,&f,provider,1,p[provider],1,3328);validate(&f,provider,NO);
                need(cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime,"GPU timestamp span");
                double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6;
                if(sample)times[arm][sample-1]=us;
                emit(@{@"kind":@"timing",@"case":@(i),@"rows":@(f.rows),@"provider":@(provider),
                    @"arm":@[@"A1",@"B",@"A2"][arm],@"sample":@(sample),@"warmup":@(sample==0),
                    @"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"gpu_us_per_chunk":@(us),
                    @"full_f32_bit_exact":@YES,@"component_only":@YES,@"device":device.name});
            }
            double a1=median(times[0],5),b=median(times[1],5),a2=median(times[2],5);
            emit(@{@"kind":@"summary",@"case":@(i),@"rows":@(f.rows),@"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),
                @"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),
                @"control_drift_percent":@((a2/a1-1)*100),@"fullmodel_speedup_proven":@NO});
        }
        inputs_unchanged(&f,before);
        f.q=nil;f.kv=nil;f.mask=nil;f.sinks=nil;f.pad=nil;f.out[0]=nil;f.out[2]=nil;
    }}
    need([file_sha(library)isEqual:manifest[@"metallib_sha256"]],"library unchanged during run");
    finish(@"complete");emit(@{@"kind":@"complete",@"cases":@6,@"full_f32_bit_exact":@YES,
        @"fixture_kind":@"deterministic_synthetic",@"fullmodel":@NO,@"device":device.name});
}return 0;}
