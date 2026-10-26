/* Isolated, synthetic tiled-indexer K-residency gate.
 * No inference runtime, model mapping, or service manipulation. Same-build
 * native and candidate always execute; --native-metallib adds a frozen native
 * library control. Timings are component GPU spans in concurrent encoders.
 */
#define main prefill_multihead_capture_main_unused
#include "bench_attention_multihead.m"
#undef main

struct index_args {
    uint32_t n_comp,n_tokens,n_head,head_dim,pos0,ratio;
    uint64_t q_token_stride,q_head_stride,weights_token_stride,index_row_stride,score_token_stride;
    float scale;
};
_Static_assert(sizeof(struct index_args)==72,"indexer argument ABI");
typedef struct {
    struct index_args args;unsigned case_index,bf16,special;
    id<MTLBuffer> q,w,k,out[3];
} index_fixture;
static uint32_t word(uint64_t i,uint32_t seed) {
    uint32_t x=(uint32_t)(i+1u)*747796405u+seed*2891336453u;
    x=((x>>((x>>28)+4))^x)*277803737u;return (x>>22)^x;
}
static float value(uint64_t i,uint32_t seed) {
    float v=((int32_t)(word(i,seed)%8191u)-4095)/4096.f;
    // Exactly BF16-representable common inputs let F32/BF16 exercise the same
    // data as test_v41_gpu.c without introducing a conversion-quality claim.
    uint32_t bits;memcpy(&bits,&v,4);bits&=0xffff0000u;memcpy(&v,&bits,4);return v;
}
static id<MTLComputePipelineState> index_pso(id<MTLDevice> d,id<MTLLibrary> l,NSString *name) {
    id<MTLFunction> fn=[l newFunctionWithName:name];need(fn!=nil,"selected indexer symbol, no fallback");
    NSError *error=nil;id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:fn error:&error];
    if(!p)fprintf(stderr,"PSO %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"indexer SIMD/threads");
    need(p.staticThreadgroupMemoryLength+21504<=d.maxThreadgroupMemoryLength,"indexer shared memory");return p;
}
static id<MTLCommandBuffer> index_run(id<MTLCommandQueue> queue,index_fixture *f,unsigned provider,id<MTLComputePipelineState> p) {
    size_t bytes=(size_t)f->args.n_tokens*f->args.score_token_stride;
    memset(body(f->out[provider]),0xa5,bytes);
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"indexer command buffer");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"indexer encoder");
    [e setComputePipelineState:p];[e setBytes:&f->args length:72 atIndex:0];
    id<MTLBuffer> bindings[]={f->q,f->w,f->k,f->out[provider]};
    for(unsigned i=0;i<4;i++)[e setBuffer:bindings[i]offset:GUARD atIndex:i+1];
    [e setThreadgroupMemoryLength:21504 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake((f->args.n_comp+31)/32,(f->args.n_tokens+7)/8,1)
        threadsPerThreadgroup:MTLSizeMake(32,4,1)];
    id<MTLResource> out=f->out[provider];[e memoryBarrierWithResources:&out count:1];
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU: %s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"indexer GPU completion");return cb;
}
static BOOL index_validate(index_fixture *f,unsigned provider,BOOL save) {
    const struct index_args *a=&f->args;size_t bytes=(size_t)a->n_tokens*a->score_token_stride;
    const uint32_t *gold=body(f->out[0]),*test=body(f->out[provider]);
    size_t bad=0,first=SIZE_MAX,nan_count=0,invisible=0,unwritten=0;
    BOOL causal=YES,padding=YES;
    for(unsigned t=0;t<a->n_tokens;t++) {
        const uint32_t *g=(const uint32_t*)((const uint8_t*)gold+(size_t)t*a->score_token_stride);
        const uint32_t *b=(const uint32_t*)((const uint8_t*)test+(size_t)t*a->score_token_stride);
        uint32_t visible=(a->pos0+t+1)/a->ratio;if(visible>a->n_comp)visible=a->n_comp;
        for(unsigned c=0;c<a->n_comp;c++) {
            if(g[c]!=b[c]){if(!bad)first=(size_t)t*a->n_comp+c;bad++;}
            unwritten+=b[c]==0xa5a5a5a5u;
            nan_count+=(b[c]&0x7f800000u)==0x7f800000u&&(b[c]&0x007fffffu)!=0;
            if(c>=visible){invisible++;causal&=b[c]==0xff800000u;}
            else if(!f->special){float v;memcpy(&v,b+c,4);causal&=isfinite(v);}
        }
        const uint8_t *tail=(const uint8_t*)b+a->n_comp*4;
        for(size_t i=0;i<a->score_token_stride-a->n_comp*4;i++)padding&=tail[i]==0xa5;
    }
    BOOL guard=guards(f->out[provider],bytes),exact=bad==0,ok=exact&&!unwritten&&causal&&padding&&guard;
    if(save){
        NSString *prefix=[NSString stringWithFormat:@"case-%03u-provider-%u",f->case_index,provider];
        raw([prefix stringByAppendingString:@".scores-strided.f32"],test,bytes);
        uint8_t g[2*GUARD];memcpy(g,f->out[provider].contents,GUARD);memcpy(g+GUARD,(uint8_t*)f->out[provider].contents+GUARD+bytes,GUARD);
        raw([prefix stringByAppendingString:@".guards"],g,sizeof(g));
        emit(@{@"kind":@"parity",@"case":@(f->case_index),@"provider":@(provider),@"bf16":@(f->bf16),
            @"tokens":@(a->n_tokens),@"keys":@(a->n_comp),@"mismatches":@(bad),
            @"first_mismatch":first==SIZE_MAX?(id)[NSNull null]:@(first),@"bit_exact":@(exact),
            @"unwritten":@(unwritten),@"nan_count":@(nan_count),@"invisible_minus_inf_count":@(invisible),
            @"causal_tail_valid":@(causal),@"padding_unchanged":@(padding),@"guards_intact":@(guard)});
    }
    need(ok,"complete indexer scores bits/causal tails/padding/guards");return ok;
}
static index_fixture index_fixture_create(id<MTLDevice> d,unsigned ci,unsigned tokens,unsigned keys,unsigned pos,unsigned ratio,unsigned bf16,unsigned special) {
    index_fixture f={0};f.case_index=ci;f.bf16=bf16;f.special=special;
    f.args=(struct index_args){.n_comp=keys,.n_tokens=tokens,.n_head=32,.head_dim=128,.pos0=pos,.ratio=ratio,
        .q_token_stride=32u*128u*4u+16u,.q_head_stride=512u,.weights_token_stride=32u*4u+16u,
        .index_row_stride=128u*(bf16?2u:4u)+16u,.score_token_stride=(keys+7u)*4u,.scale=1.f/64.f};
    size_t qb=(size_t)tokens*f.args.q_token_stride,wb=(size_t)tokens*f.args.weights_token_stride,kb=(size_t)keys*f.args.index_row_stride;
    f.q=alloc(d,qb,YES);f.w=alloc(d,wb,YES);f.k=alloc(d,kb,YES);
    for(unsigned t=0;t<tokens;t++) {
        float *q=(float*)((uint8_t*)body(f.q)+(size_t)t*f.args.q_token_stride);
        float *w=(float*)((uint8_t*)body(f.w)+(size_t)t*f.args.weights_token_stride);
        for(unsigned i=0;i<32u*128u;i++)q[i]=value((uint64_t)t*4096u+i,17u+ci);
        for(unsigned h=0;h<32;h++)w[h]=value((uint64_t)t*32u+h,29u+ci);
        if(special&&t%3==0){uint32_t qnan=0x7fc12345u;memcpy(w+7,&qnan,4);}
    }
    for(unsigned k=0;k<keys;k++)for(unsigned i=0;i<128;i++) {
        float v=value((uint64_t)k*128u+i,37u+ci);uint32_t bits;memcpy(&bits,&v,4);
        if(special&&k%13==0&&i==17)bits=0x7fc10000u;
        uint8_t *row=(uint8_t*)body(f.k)+(size_t)k*f.args.index_row_stride;
        if(bf16){uint16_t b=bits>>16;memcpy(row+i*2u,&b,2);}else memcpy(row+i*4u,&bits,4);
    }
    for(unsigned p=0;p<3;p++)f.out[p]=alloc(d,(size_t)tokens*f.args.score_token_stride,YES);
    NSDictionary *identity=@{@"case":@(ci),@"tokens":@(tokens),@"keys":@(keys),@"pos0":@(pos),@"ratio":@(ratio),
        @"bf16":@(bf16),@"special_nan_fixture":@(special),@"q_sha256":sha(body(f.q),qb),
        @"weights_sha256":sha(body(f.w),wb),@"cache_sha256":sha(body(f.k),kb),@"args_sha256":sha(&f.args,72)};
    [manifest[@"cases"]addObject:identity];
    raw([NSString stringWithFormat:@"case-%03u-args72.bin",ci],&f.args,72);
    raw([NSString stringWithFormat:@"case-%03u-q-strided.f32",ci],body(f.q),qb);
    raw([NSString stringWithFormat:@"case-%03u-w-strided.f32",ci],body(f.w),wb);
    raw([NSString stringWithFormat:@"case-%03u-cache-strided.bin",ci],body(f.k),kb);
    return f;
}
int main(int argc,char **argv){@autoreleasepool {
    NSString *probe_path=nil,*native_path=nil;
    for(int i=1;i<argc;i++){
        need(i+1<argc,"option value");const char *key=argv[i],*v=argv[++i];
        if(!strcmp(key,"--metallib"))probe_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(key,"--native-metallib"))native_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(key,"--out-dir"))output=canonical([NSString stringWithUTF8String:v]);
        else need(NO,"known option");
    }
    need(probe_path&&output,"--metallib PATH [--native-metallib PATH] --out-dir NEW");
    if(!native_path)native_path=probe_path;
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new output directory");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output creation");
    files=[NSMutableArray new];stored_payloads=[NSMutableDictionary new];
    manifest=[@{@"schema_version":@1,@"probe":@"indexer_k_resident",@"fixture_kind":@"deterministic_synthetic",
        @"cases":[NSMutableArray new],@"fullmodel":@NO,@"production_registered":@NO,@"cold_dram_proven":@NO,
        @"mode":@"concurrent",@"samples":@5,@"warmups":@1,@"probe_path":probe_path,@"native_path":native_path,
        @"probe_sha256":file_sha(probe_path),@"native_sha256":file_sha(native_path),
        @"binary_sha256":file_sha(binary_path()),@"binary_path":binary_path()}mutableCopy];
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();need(d!=nil,"Metal device");manifest[@"device"]=d.name;
    NSError *error=nil;id<MTLLibrary> probe=[d newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];need(probe!=nil,"probe library");
    id<MTLLibrary> gold=[d newLibraryWithURL:[NSURL fileURLWithPath:native_path]error:&error];need(gold!=nil,"native library");
    id<MTLComputePipelineState> psos[2][3];NSMutableArray *bindings=[NSMutableArray new];
    for(unsigned bf=0;bf<2;bf++)for(unsigned p=0;p<3;p++) {
        NSString *name=[NSString stringWithFormat:p==2?@"kernel_myds_indexer_scores_k_resident_%@":@"kernel_dsv4_indexer_scores_tiled_%@",bf?@"bf16":@"f32"];
        psos[bf][p]=index_pso(d,p==0?gold:probe,name);
        [bindings addObject:@{@"provider":@(p),@"bf16":@(bf),@"symbol":name,@"dynamic_threadgroup_bytes":@21504,
            @"static_threadgroup_bytes":@(psos[bf][p].staticThreadgroupMemoryLength),
            @"threads":@[@32,@4,@1],@"max_threads":@(psos[bf][p].maxTotalThreadsPerThreadgroup)}];
    }
    manifest[@"bindings"]=bindings;emit(@{@"kind":@"bindings",@"device":d.name,@"bindings":bindings});
    id<MTLCommandQueue> queue=[d newCommandQueue];need(queue!=nil,"command queue");
    const unsigned cases[][5]={{1,17,0,2,0},{7,31,60,2,0},{8,32,60,2,0},{9,33,0,2,0},
        {33,129,1024,1,0},{9,33,0,2,1},{128,4096,8192,2,0},{256,16384,32768,2,0}};
    unsigned ci=0;
    for(unsigned bf=0;bf<2;bf++)for(unsigned c=0;c<sizeof(cases)/sizeof(cases[0]);c++,ci++){@autoreleasepool {
        index_fixture f=index_fixture_create(d,ci,cases[c][0],cases[c][1],cases[c][2],cases[c][3],bf,cases[c][4]);
        NSDictionary *before=[manifest[@"cases"]lastObject];
        for(unsigned p=0;p<3;p++){(void)index_run(queue,&f,p,psos[bf][p]);index_validate(&f,p,YES);}
        if(c>=6)for(unsigned control=0;control<2;control++){
            double times[3][64];unsigned providers[]={control,2,control};
            for(unsigned sample=0;sample<=5;sample++)for(unsigned arm=0;arm<3;arm++){
                unsigned p=providers[arm];id<MTLCommandBuffer> cb=index_run(queue,&f,p,psos[bf][p]);index_validate(&f,p,NO);
                need(cb.GPUStartTime>0&&cb.GPUEndTime>cb.GPUStartTime,"valid GPU timestamps");
                double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6;if(sample)times[arm][sample-1]=us;
                emit(@{@"kind":@"timing",@"case":@(ci),@"bf16":@(bf),@"control":@(control),@"provider":@(p),
                    @"arm":@[@"A1",@"B",@"A2"][arm],@"sample":@(sample),@"warmup":@(sample==0),
                    @"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"gpu_us":@(us),@"full_scores_bit_exact":@YES});
            }
            double a1=median(times[0],5),b=median(times[1],5),a2=median(times[2],5);
            emit(@{@"kind":@"summary",@"case":@(ci),@"bf16":@(bf),@"control":@(control),
                @"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),@"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),
                @"control_drift_percent":@((a2/a1-1)*100),@"fullmodel_speedup_proven":@NO});
        }
        size_t qb=(size_t)f.args.n_tokens*f.args.q_token_stride,wb=(size_t)f.args.n_tokens*f.args.weights_token_stride,kb=(size_t)f.args.n_comp*f.args.index_row_stride;
        need([sha(body(f.q),qb)isEqual:before[@"q_sha256"]]&&[sha(body(f.w),wb)isEqual:before[@"weights_sha256"]]&&
            [sha(body(f.k),kb)isEqual:before[@"cache_sha256"]]&&guards(f.q,qb)&&guards(f.w,wb)&&guards(f.k,kb),"full inputs/guards unchanged");
        f.q=nil;f.w=nil;f.k=nil;for(unsigned p=0;p<3;p++)f.out[p]=nil;
    }}
    need([file_sha(probe_path)isEqual:manifest[@"probe_sha256"]]&&[file_sha(native_path)isEqual:manifest[@"native_sha256"]],"libraries unchanged");
    finish(@"complete");emit(@{@"kind":@"complete",@"cases":@(ci),@"full_scores_bit_exact":@YES,@"fullmodel":@NO,@"device":d.name});
}return 0;}
