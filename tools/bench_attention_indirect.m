/* Single-head indirect F16 layout replay. No runtime init or production wiring.
 * --model GGUF --capture-json JSON --probe-metallib PATH --production-metallib PATH
 * --out-dir NEW [--samples 9 --reps 1 --host-only] --correctness-only
 * Performance instead requires --correctness-manifest ABS_JSON and
 * --correctness-manifest-sha256 HEX from a completed correctness-only invocation
 * of this exact binary/model/capture/libraries. There is no ungated timing mode.
 * Schema 1, capture_kind=myds_prefill_attention_multihead. Each case stores
 * args200, Q F32, exact F16 KV row dictionary + u32 IDs + expanded SHA256,
 * compact 640-column F16 mask, and model F32 sinks. Payloads require path/SHA/bytes.
 * One full captured query chunk is processed at a time; no small hot fixture is
 * substituted for the captured working set. Timings use concurrent encoders only.
 * Baselines bind the complete expanded 1.34 GB KV. The candidate binds the exact
 * F16 row dictionary at 2/3 and the u32 row IDs at 8, with identical arithmetic,
 * args, 64 x rows geometry, 32 threads and 3328 dynamic scratch bytes.
 * A CPU-created dictionary proves only this layout replay; no production GPU
 * dictionary construction, end-to-end speedup or cold DRAM claim is made.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "myds_gguf.h"
#include <mach-o/dyld.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
enum { HEADS=64, DIM=512, KEYS=640, GUARD=64, QUERY_BYTES=131072,
       KV_ROW_BYTES=1024, KV_QUERY_BYTES=655360, SINK_BYTES=256 };
struct flash_args {
    int32_t ne01,ne02,ne03;uint64_t nb01,nb02,nb03;int32_t ne11,ne_12_2,ne_12_3,ns10;
    uint64_t nb11,nb12,nb13;int32_t ns20;uint64_t nb21,nb22,nb23;
    int32_t ne31,ne32,ne33;uint64_t nb31,nb32,nb33;int32_t ne1,ne2,ne3;
    float scale,max_bias,m0,m1;int32_t n_head_log2;float logit_softcap;uint64_t nb_kvq;
};
_Static_assert(sizeof(struct flash_args)==200,"prefill attention ABI");
_Static_assert(__builtin_offsetof(struct flash_args,nb_kvq)==192,"KV query stride ABI");
static NSMutableDictionary *manifest;
static NSMutableArray *files;
static NSString *output;
static NSMutableDictionary *stored_payloads;
static NSMutableArray *parity_records;
static NSMutableArray *input_records;
static NSString *phase;
static unsigned command_ordinal;
static void finish(NSString *status) {
    if(!manifest||!output)return;
    manifest[@"status"]=status; manifest[@"files"]=files;
    if(parity_records)manifest[@"parity"]=parity_records;
    if(input_records)manifest[@"input_integrity"]=input_records;
    NSData *d=[NSJSONSerialization dataWithJSONObject:manifest
        options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    if(d)[d writeToFile:[output stringByAppendingPathComponent:@"artifact-manifest.json"] atomically:YES];
}
static void need(BOOL v,const char *m) {
    if(v)return; fprintf(stderr,"FAIL: %s\n",m);
    if(manifest)manifest[@"failure"]=[NSString stringWithUTF8String:m];
    finish(@"failed"); exit(1);
}
static NSString *sha(const void *p,size_t n) {
    CC_SHA256_CTX c; CC_SHA256_Init(&c); const uint8_t *b=p;
    while(n){CC_LONG k=(CC_LONG)(n>0x10000000u?0x10000000u:n);CC_SHA256_Update(&c,b,k);b+=k;n-=k;}
    uint8_t d[32];CC_SHA256_Final(d,&c);char s[65];
    for(unsigned i=0;i<32;i++)snprintf(s+2*i,3,"%02x",d[i]);
    return [NSString stringWithUTF8String:s];
}
static NSData *read_file(NSString *p) {
    NSData *d=[NSData dataWithContentsOfFile:p options:NSDataReadingMappedIfSafe error:nil];
    if(!d)fprintf(stderr,"cannot read %s\n",p.fileSystemRepresentation);
    need(d!=nil,"input file read");return d;
}
static NSString *file_sha(NSString *p){NSData *d=read_file(p);return sha(d.bytes,d.length);}
static NSString *canonical(NSString *p){return [[p stringByStandardizingPath]stringByResolvingSymlinksInPath];}
static NSString *binary_path(void) {
    uint32_t n=0;_NSGetExecutablePath(NULL,&n);char *p=calloc(n,1);need(p!=NULL,"binary path allocation");
    need(!_NSGetExecutablePath(p,&n),"binary path");NSString *s=canonical([NSString stringWithUTF8String:p]);free(p);return s;
}
static void emit(NSDictionary *d) {
    NSData *j=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:nil];
    need(j!=nil,"JSON encoding");fwrite(j.bytes,1,j.length,stdout);putchar('\n');fflush(stdout);
}
static void raw(NSString *n,const void *p,size_t bytes) {
    NSString *digest=sha(p,bytes),*key=[NSString stringWithFormat:@"%zu:%@",bytes,digest];
    NSString *first=stored_payloads[key],*path=[output stringByAppendingPathComponent:n];
    need(![[NSFileManager defaultManager]fileExistsAtPath:path],"new raw artifact path");
    NSMutableDictionary *record=[@{@"path":n,@"bytes":@(bytes),@"sha256":digest}mutableCopy];
    if(first){
        need(!link([output stringByAppendingPathComponent:first].fileSystemRepresentation,path.fileSystemRepresentation),"identical payload hard link");
        record[@"storage_alias_of"]=first;
    }else{
        FILE *file=fopen(path.fileSystemRepresentation,"wbx");need(file!=NULL,"new raw artifact open");
        need(fwrite(p,1,bytes,file)==bytes&&!fclose(file),"complete raw artifact write");
        stored_payloads[key]=n;
    }
    [files addObject:record];
}
static unsigned number(const char *s,unsigned lo,unsigned hi) {
    char *end=NULL;unsigned long n=strtoul(s,&end,10);
    need(*s&&end&&!*end&&n>=lo&&n<=hi,"integer range");return(unsigned)n;
}
static uint64_t json_uint(id v,uint64_t hi,const char *m) {
    need([v isKindOfClass:[NSNumber class]],m);
    double x=[v doubleValue];need(isfinite(x)&&x>=0&&x<=hi&&floor(x)==x,m);
    return[v unsignedLongLongValue];
}
static BOOL finite_f32(const void *p,size_t n) {
    const float *v=p;for(size_t i=0;i<n/4;i++)if(!isfinite(v[i]))return NO;return YES;
}
static BOOL written(const void *p,size_t n) {
    const uint32_t *v=p;for(size_t i=0;i<n/4;i++)if(v[i]==0xa5a5a5a5u)return NO;return YES;
}
static size_t mismatches(const void *a,const void *b,size_t n,size_t *first) {
    const uint32_t *x=a,*y=b;size_t bad=0;*first=SIZE_MAX;
    for(size_t i=0;i<n/4;i++)if(x[i]!=y[i]){if(!bad)*first=i;bad++;}return bad;
}
static double wall(void) {
    struct timespec t;need(!clock_gettime(CLOCK_MONOTONIC,&t),"monotonic clock");return t.tv_sec+t.tv_nsec*1e-9;
}
static int cmp_double(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static double median(double *v,unsigned n){double c[64];need(n&&n<=64,"median count");memcpy(c,v,n*sizeof(double));qsort(c,n,sizeof(double),cmp_double);return n%2?c[n/2]:(c[n/2-1]+c[n/2])/2;}
static NSData *payload(NSDictionary *c,NSString *key,NSString *base,size_t bytes) {
    NSDictionary *v=c[key];need([v isKindOfClass:[NSDictionary class]],"capture payload descriptor");
    NSString *p=v[@"path"],*s=v[@"sha256"];
    need([p isKindOfClass:[NSString class]]&&p.length&&[s isKindOfClass:[NSString class]]&&s.length==64,"capture path/SHA");
    if(!p.isAbsolutePath)p=[base stringByAppendingPathComponent:p];
    NSData *d=read_file(canonical(p));need(d.length==bytes,"capture payload exact byte length");
    need(json_uint(v[@"bytes"],SIZE_MAX,"capture payload declared byte count")==bytes,"capture descriptor byte count");
    need([sha(d.bytes,d.length)isEqual:s],"capture payload SHA mismatch");return d;
}
static void *body(id<MTLBuffer> b){return(uint8_t*)b.contents+GUARD;}
static id<MTLBuffer> alloc(id<MTLDevice> d,size_t n,BOOL poison_body) {
    need(n+2*GUARD<=d.maxBufferLength,"buffer exceeds maxBufferLength");
    id<MTLBuffer> b=[d newBufferWithLength:n+2*GUARD options:MTLResourceStorageModeShared];
    need(b!=nil,"Metal buffer allocation");
    if(poison_body)memset(b.contents,0xa5,b.length);
    else{memset(b.contents,0xa5,GUARD);memset((uint8_t*)b.contents+GUARD+n,0xa5,GUARD);}
    return b;
}
static BOOL guards(id<MTLBuffer> b,size_t n) {
    const uint8_t *p=b.contents;for(unsigned i=0;i<GUARD;i++)if(p[i]!=0xa5||p[GUARD+n+i]!=0xa5)return NO;return YES;
}
static void dump(NSString *n,id<MTLBuffer> b,size_t bytes) {
    raw([n stringByAppendingString:@".f32"],body(b),bytes);
    uint8_t g[2*GUARD];memcpy(g,b.contents,GUARD);memcpy(g+GUARD,(uint8_t*)b.contents+GUARD+bytes,GUARD);
    raw([n stringByAppendingString:@".guards"],g,sizeof(g));
}
static BOOL finite_f16(const void *p,size_t count) {
    const uint16_t *h=p;for(size_t i=0;i<count;i++)if((h[i]&0x7c00u)==0x7c00u)return NO;return YES;
}
static BOOL valid_args(const struct flash_args *a,unsigned rows) {
    uint64_t mask=a->nb31;
    return rows>=1&&rows<=2048&&a->ne01==(int32_t)rows&&a->ne02==64&&a->ne03==1&&
        a->nb01==QUERY_BYTES&&a->nb02==2048&&a->nb03==QUERY_BYTES&&a->ne11==KEYS&&
        a->ne_12_2==1&&a->ne_12_3==1&&a->ns10==512&&a->nb11==1024&&
        a->nb12==KV_QUERY_BYTES&&a->nb13==KV_QUERY_BYTES&&a->ns20==512&&a->nb21==1024&&
        a->nb22==KV_QUERY_BYTES&&a->nb23==KV_QUERY_BYTES&&a->ne31==(int32_t)rows&&a->ne32==1&&a->ne33==1&&
        mask>=1280&&mask<=UINT32_MAX&&mask%2==0&&a->nb32==mask*rows&&a->nb33==mask*rows&&
        a->ne1==64&&a->ne2==(int32_t)rows&&a->ne3==1&&a->scale==1.0f/sqrtf(512.f)&&
        a->max_bias==0&&a->m0==0&&a->m1==0&&a->n_head_log2==0&&a->logit_softcap==0&&a->nb_kvq==KV_QUERY_BYTES;
}
static NSString *expanded_sha(NSData *unique,NSData *ids,unsigned rows,void *destination) {
    need(unique.length&&unique.length%KV_ROW_BYTES==0,"F16 KV row dictionary shape");
    need(ids.length==(size_t)rows*KEYS*4,"KV row-ID shape");
    const uint32_t *map=ids.bytes;const uint8_t *data=unique.bytes;size_t count=(size_t)rows*KEYS;
    size_t unique_count=unique.length/KV_ROW_BYTES;
    CC_SHA256_CTX ctx;CC_SHA256_Init(&ctx);
    for(size_t i=0;i<count;i++){
        need(map[i]<unique_count,"KV row ID outside dictionary");const uint8_t *row=data+(size_t)map[i]*KV_ROW_BYTES;
        CC_SHA256_Update(&ctx,row,KV_ROW_BYTES);if(destination)memcpy((uint8_t*)destination+i*KV_ROW_BYTES,row,KV_ROW_BYTES);
    }
    uint8_t d[32];CC_SHA256_Final(d,&ctx);char h[65];for(unsigned i=0;i<32;i++)snprintf(h+2*i,3,"%02x",d[i]);
    return[NSString stringWithUTF8String:h];
}
static NSDictionary *mask_coverage(NSData *mask,unsigned rows) {
    need(mask.length==(size_t)rows*KEYS*2,"compact mask shape");const uint16_t *p=mask.bytes;
    unsigned min_picks=512,max_picks=0,full_picks=0,min_window=128,max_window=0;
    for(unsigned r=0;r<rows;r++){
        unsigned window=0,picks=0;
        for(unsigned k=0;k<KEYS;k++){
            uint16_t v=p[(size_t)r*KEYS+k];need(v==0||v==0xfc00u,"gathered mask must be zero or negative infinity");
            if(v==0){if(k<128)window++;else picks++;}
        }
        min_picks=picks<min_picks?picks:min_picks;max_picks=picks>max_picks?picks:max_picks;
        min_window=window<min_window?window:min_window;max_window=window>max_window?window:max_window;
        full_picks+=picks==512;
    }
    return@{@"minimum_valid_picks":@(min_picks),@"maximum_valid_picks":@(max_picks),@"full_512_pick_queries":@(full_picks),
        @"minimum_valid_window":@(min_window),@"maximum_valid_window":@(max_window)};
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,NSString *name,unsigned threads,unsigned smem) {
    const unsigned indices[10]={400,401,402,403,404,405,420,421,422,423};
    const int32_t values[10]={1,1,0,0,0,0,512,512,1,1};
    MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
    for(unsigned i=0;i<10;i++){bool b=values[i]!=0;int32_t v=values[i];
        [constants setConstantValue:i<6?(const void*)&b:(const void*)&v type:i<6?MTLDataTypeBool:MTLDataTypeInt atIndex:indices[i]];}
    NSError *error=nil;id<MTLFunction> f=[l newFunctionWithName:name constantValues:constants error:&error];
    if(!f)fprintf(stderr,"function %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(f!=nil,"selected attention function unavailable; no fallback");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];
    if(!p)fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=threads,"attention pipeline SIMD/threads");
    need(p.staticThreadgroupMemoryLength+smem<=d.maxThreadgroupMemoryLength,"attention pipeline scratch limit");return p;
}
typedef struct {
    unsigned index,layer,rows;struct flash_args args;
    id<MTLBuffer> q,kv,unique,ids,mask,sinks,pad,out[3];
    size_t unique_bytes;
    NSData *oracle;
    NSString *input_sha[7];
    NSString *args_sha;
} fixture;
static BOOL overlap(uintptr_t a,size_t an,uintptr_t b,size_t bn){return a<=b?b-a<an:a-b<bn;}
static BOOL bindings_safe(fixture *f,unsigned provider) {
    size_t out_bytes=(size_t)f->rows*QUERY_BYTES;
    if(provider>=3||!valid_args(&f->args,f->rows)||!f->out[provider]||
       !f->unique_bytes||f->unique_bytes%KV_ROW_BYTES)return NO;
    uintptr_t out=(uintptr_t)body(f->out[provider]);
    id<MTLBuffer> readers[7]={f->q,f->kv,f->unique,f->ids,f->mask,f->sinks,f->pad};
    const size_t bytes[7]={out_bytes,(size_t)f->rows*KV_QUERY_BYTES,f->unique_bytes,
        (size_t)f->rows*KEYS*4,(size_t)f->rows*f->args.nb31,SINK_BYTES,16};
    for(unsigned i=0;i<7;i++)if(!readers[i]||readers[i].length<bytes[i]+2*GUARD||
        overlap(out,out_bytes,(uintptr_t)body(readers[i]),bytes[i]))return NO;
    return f->out[provider].length>=out_bytes+2*GUARD&&out%16u==0;
}
static BOOL inputs_unchanged(fixture *f,unsigned provider) {
    id<MTLBuffer> readers[7]={f->q,f->kv,f->unique,f->ids,f->mask,f->sinks,f->pad};
    const size_t bytes[7]={(size_t)f->rows*QUERY_BYTES,(size_t)f->rows*KV_QUERY_BYTES,f->unique_bytes,
        (size_t)f->rows*KEYS*4,(size_t)f->rows*f->args.nb31,SINK_BYTES,16};
    BOOL ok=[sha(&f->args,sizeof(f->args))isEqual:f->args_sha];NSMutableArray *details=[NSMutableArray new];
    NSArray *names=@[@"q_f32",@"expanded_kv_f16",@"kv_unique_f16",@"kv_row_ids_u32",@"mask_original_stride_f16",@"sinks_f32",@"unused_pad"];
    for(unsigned i=0;i<7;i++){
        BOOL g=guards(readers[i],bytes[i]);NSString *actual=sha(body(readers[i]),bytes[i]);BOOL same=[actual isEqual:f->input_sha[i]];
        [details addObject:@{@"name":names[i],@"bytes":@(bytes[i]),@"expected_sha256":f->input_sha[i],@"actual_sha256":actual,@"guards_intact":@(g),@"unchanged":@(same)}];
        ok=g&&same&&ok;
    }
    [input_records addObject:@{@"case":@(f->index),@"provider":@(provider),@"command_ordinal":@(command_ordinal),
        @"all_seven_gpu_inputs_sha_and_guards_unchanged":@(ok),@"args200_unchanged":@([sha(&f->args,sizeof(f->args))isEqual:f->args_sha]),@"buffers":details}];
    return ok;
}
static void save_input_guards(fixture *f) {
    const size_t bytes[2]={f->unique_bytes,(size_t)f->rows*KEYS*4};
    id<MTLBuffer> buffers[2]={f->unique,f->ids};NSArray *names=@[@"kv-unique",@"kv-row-ids"];
    for(unsigned i=0;i<2;i++){
        uint8_t g[2*GUARD];memcpy(g,buffers[i].contents,GUARD);
        memcpy(g+GUARD,(uint8_t*)buffers[i].contents+GUARD+bytes[i],GUARD);
        raw([NSString stringWithFormat:@"input-case-%03u-%@.guards",f->index,names[i]],g,sizeof(g));
    }
}
static id<MTLCommandBuffer> run(id<MTLCommandQueue> queue,fixture *f,unsigned provider,unsigned reps,
        id<MTLComputePipelineState> p,unsigned head_batch,unsigned smem) {
    need(bindings_safe(f,provider),"invalid/aliased attention bindings before command creation");
    /* The previous CB is complete. Re-poison before every measured command so
     * a skipped store cannot inherit a valid parity result. This CPU write is
     * outside GPU timestamps and applies identically to all three providers. */
    memset(body(f->out[provider]),0xa5,(size_t)f->rows*QUERY_BYTES);
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"attention CB");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"concurrent attention encoder");
    for(unsigned rep=0;rep<reps;rep++){
        [e setComputePipelineState:p];[e setBytes:&f->args length:sizeof(f->args) atIndex:0];
        id<MTLBuffer> kv=provider==2?f->unique:f->kv;
        id<MTLBuffer> b[7]={f->q,kv,kv,f->mask,f->sinks,f->pad,f->out[provider]};
        for(unsigned i=0;i<7;i++)[e setBuffer:b[i] offset:GUARD atIndex:i+1];
        if(provider==2)[e setBuffer:f->ids offset:GUARD atIndex:8];
        [e setThreadgroupMemoryLength:smem atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(HEADS/head_batch,f->rows,1)
            threadsPerThreadgroup:MTLSizeMake(32,head_batch,1)];
        id<MTLResource> out=f->out[provider];[e memoryBarrierWithResources:&out count:1];
    }
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted){
        fprintf(stderr,"GPU error: %s\n",cb.error.localizedDescription.UTF8String);
        dump([NSString stringWithFormat:@"gpu-failure-case-%03u-provider-%u-output",f->index,provider],f->out[provider],(size_t)f->rows*QUERY_BYTES);
        save_input_guards(f);emit(@{@"kind":@"gpu_failure",@"case":@(f->index),@"provider":@(provider),@"completed_cb":@NO});
    }
    need(cb.status==MTLCommandBufferStatusCompleted,"attention command completion");command_ordinal++;return cb;
}
static void validate(fixture *f,unsigned provider,BOOL save_output) {
    size_t bytes=(size_t)f->rows*QUERY_BYTES,first=SIZE_MAX;
    const void *oracle=f->oracle?f->oracle.bytes:body(f->out[0]);
    size_t bad=mismatches(oracle,body(f->out[provider]),bytes,&first);
    BOOL finite=finite_f32(body(f->out[provider]),bytes),all_written=written(body(f->out[provider]),bytes),
        guard_ok=guards(f->out[provider],bytes),inputs_ok=inputs_unchanged(f,provider);
    BOOL ok=!bad&&finite&&all_written&&guard_ok&&inputs_ok;
    if(save_output||!ok){
        NSString *name=save_output?[NSString stringWithFormat:@"parity-case-%03u-provider-%u-output",f->index,provider]:
            [NSString stringWithFormat:@"failure-case-%03u-provider-%u-cb-%u-output",f->index,provider,command_ordinal];
        dump(name,f->out[provider],bytes);
        NSDictionary *record=@{@"kind":save_output?@"parity":@"parity_failure",@"phase":phase,@"mode":@"concurrent",@"case":@(f->index),@"layer":@(f->layer),@"provider":@(provider),
            @"rows":@(f->rows),@"keys":@640,@"output_bytes":@(bytes),@"mismatches":@(bad),
            @"first_mismatch":first==SIZE_MAX?(id)[NSNull null]:@(first),@"full_f32_bit_exact":@(ok),
            @"first_expected_bits":first==SIZE_MAX?(id)[NSNull null]:@(((const uint32_t*)oracle)[first]),
            @"first_actual_bits":first==SIZE_MAX?(id)[NSNull null]:@(((const uint32_t*)body(f->out[provider]))[first]),
            @"finite":@(finite),@"written":@(all_written),@"guards_intact":@(guard_ok),@"all_gpu_inputs_unchanged":@(inputs_ok),
            @"finite_guards_written":@(finite&&all_written&&guard_ok),@"completed_cb":@YES,@"command_ordinal":@(command_ordinal),
            @"output_sha256":sha(body(f->out[provider]),bytes)};
        [parity_records addObject:record];manifest[@"parity"]=parity_records;manifest[@"input_integrity"]=input_records;
        emit(record);
    }
    if(!ok)save_input_guards(f);
    need(ok,"complete attention output bits/finite/written/guards");
}
static unsigned args_self_test(void) {
    struct flash_args a={.ne01=1,.ne02=64,.ne03=1,.nb01=QUERY_BYTES,.nb02=2048,.nb03=QUERY_BYTES,
        .ne11=KEYS,.ne_12_2=1,.ne_12_3=1,.ns10=512,.nb11=1024,.nb12=KV_QUERY_BYTES,.nb13=KV_QUERY_BYTES,
        .ns20=512,.nb21=1024,.nb22=KV_QUERY_BYTES,.nb23=KV_QUERY_BYTES,.ne31=1,.ne32=1,.ne33=1,
        .nb31=1280,.nb32=1280,.nb33=1280,.ne1=64,.ne2=1,.ne3=1,.scale=1.f/sqrtf(512.f),.nb_kvq=KV_QUERY_BYTES};
    need(valid_args(&a,1),"valid gathered prefill ABI");unsigned rejected=0;struct flash_args bad=a;
    bad.ne02=63;rejected+=!valid_args(&bad,1);bad=a;bad.nb_kvq=0;rejected+=!valid_args(&bad,1);
    bad=a;bad.nb31=1278;rejected+=!valid_args(&bad,1);bad=a;bad.scale=0;rejected+=!valid_args(&bad,1);
    bad=a;bad.ne11=512;rejected+=!valid_args(&bad,1);bad=a;bad.ne01=0;rejected+=!valid_args(&bad,1);
    need(rejected==6,"prefill ABI negative controls");
    need(overlap(100,100,199,3)&&!overlap(100,100,200,3)&&overlap(100,100,100,100),"buffer overlap boundaries");
    return rejected;
}
static NSDictionary *binding_contract(void) {
    return @{@"mode":@"concurrent",@"providers":@3,@"threads":@[@32,@1,@1],@"groups_x":@64,
        @"groups_y":@"captured_rows",@"dynamic_threadgroup_bytes":@3328,@"args_bytes":@200,
        @"native_symbol":@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",
        @"candidate_symbol":@"kernel_myds_prefill_attention_indirect_f16_dk512_dv512",
        @"native_kv":@"complete_expanded_f16",@"candidate_kv":@"exact_f16_row_dictionary",
        @"candidate_row_ids_slot":@8,@"production_builder_verified":@NO};
}
static NSString *evidence_file(NSString *base,NSString *name) {
    need([name isKindOfClass:[NSString class]]&&name.length&&!name.isAbsolutePath,"relative correctness artifact path");
    NSString *p=canonical([base stringByAppendingPathComponent:name]);
    need([p hasPrefix:[base stringByAppendingString:@"/"]],"correctness artifact path escapes directory");
    struct stat st;need(!lstat([base stringByAppendingPathComponent:name].fileSystemRepresentation,&st)&&S_ISREG(st.st_mode),
        "correctness artifact must be a regular file, not a symlink");return p;
}
/* This gate runs entirely on the CPU before creating a Metal device. It binds a
 * separately completed all-layer correctness phase to this exact executable,
 * libraries, model identity and payloads. Every archived output is SHA-checked,
 * then compared in full against its native oracle, including finite/poison and
 * guards. A host-only manifest or a partly passed layer cannot unlock timing. */
static NSArray *correctness_oracles(NSString *path,NSString *expected) {
    NSData *bytes=read_file(path);need([sha(bytes.bytes,bytes.length)isEqual:expected],"correctness manifest SHA mismatch");
    NSDictionary *gate=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];
    need([gate isKindOfClass:[NSDictionary class]]&&[gate[@"status"]isEqual:@"complete"]&&
        [gate[@"phase"]isEqual:@"correctness"]&&[gate[@"probe"]isEqual:@"prefill_attention_indirect_real_capture"]&&
        [gate[@"full_f32_bit_exact"]isEqual:@YES]&&[gate[@"attention_output_checked"]isEqual:@YES]&&
        [gate[@"all_gpu_inputs_unchanged"]isEqual:@YES]&&[gate[@"providers"]isEqual:@3]&&
        [gate[@"timing_command_buffers"]isEqual:@0],"completed all-provider correctness manifest required");
    for(NSString *key in @[@"binary_sha256",@"capture_json_sha256",@"probe_metallib_sha256",@"production_metallib_sha256",
        @"model",@"cases",@"total_cases",@"samples",@"repetitions_per_cb",@"binding_contract"])
        need([gate[key]isEqual:manifest[key]],"correctness identity differs from current replay");
    NSString *base=canonical([path stringByDeletingLastPathComponent]);NSArray *records=gate[@"files"];
    need([records isKindOfClass:[NSArray class]]&&records.count,"correctness artifact list required");
    NSMutableDictionary *by_name=[NSMutableDictionary new],*inode_sha=[NSMutableDictionary new];
    for(NSDictionary *r in records){
        need([r isKindOfClass:[NSDictionary class]],"correctness artifact descriptor");NSString *name=r[@"path"];
        NSString *p=evidence_file(base,name);need(!by_name[name],"duplicate correctness artifact path");
        struct stat st;need(!stat(p.fileSystemRepresentation,&st)&&st.st_size>=0,"correctness artifact stat/nonnegative length");
        need(json_uint(r[@"bytes"],SIZE_MAX,"correctness artifact byte count")==(uint64_t)st.st_size,"correctness artifact file length");
        NSString *key=[NSString stringWithFormat:@"%llu:%llu:%llu",(unsigned long long)st.st_dev,
            (unsigned long long)st.st_ino,(unsigned long long)st.st_size];
        NSString *digest=inode_sha[key];if(!digest){digest=file_sha(p);inode_sha[key]=digest;}
        need([digest isEqual:r[@"sha256"]],"correctness archived artifact SHA mismatch");by_name[name]=r;
    }
    NSArray *rows=gate[@"parity"],*inputs=gate[@"input_integrity"];unsigned count=[manifest[@"total_cases"]unsignedIntValue];
    need([rows isKindOfClass:[NSArray class]]&&rows.count==count*3&&
        [inputs isKindOfClass:[NSArray class]]&&inputs.count==count*3,"complete nine-provider correctness evidence required");
    NSMutableSet *seen=[NSMutableSet new];
    for(NSDictionary *r in rows){
        unsigned c=(unsigned)json_uint(r[@"case"],count-1,"correctness case"),p=(unsigned)json_uint(r[@"provider"],2,"correctness provider");
        NSString *key=[NSString stringWithFormat:@"%u:%u",c,p];need(![seen containsObject:key],"duplicate correctness case/provider");[seen addObject:key];
        need([r[@"kind"]isEqual:@"parity"]&&[r[@"mode"]isEqual:@"concurrent"]&&[r[@"phase"]isEqual:@"correctness"]&&
            [r[@"full_f32_bit_exact"]isEqual:@YES]&&[r[@"finite_guards_written"]isEqual:@YES]&&
            [r[@"all_gpu_inputs_unchanged"]isEqual:@YES]&&[r[@"completed_cb"]isEqual:@YES]&&
            json_uint(r[@"mismatches"],SIZE_MAX,"correctness mismatch count")==0,"correctness provider did not pass");
        NSDictionary *ci=manifest[@"cases"][c];need([r[@"layer"]isEqual:ci[@"layer"]]&&[r[@"rows"]isEqual:ci[@"rows"]]&&
            [r[@"output_bytes"]isEqual:ci[@"q_bytes"]],"correctness parity shape/layer differs");
    }
    for(NSDictionary *r in inputs)need([r[@"all_seven_gpu_inputs_sha_and_guards_unchanged"]isEqual:@YES]&&
        [r[@"args200_unchanged"]isEqual:@YES],"correctness GPU input integrity failed");
    NSMutableArray *oracles=[NSMutableArray new];uint8_t guard_bytes[2*GUARD];memset(guard_bytes,0xa5,sizeof(guard_bytes));
    for(unsigned i=0;i<count;i++){
        NSDictionary *ci=manifest[@"cases"][i];size_t n=[ci[@"q_bytes"]unsignedLongLongValue];NSData *native=nil;
        for(unsigned p=0;p<3;p++){
            NSString *name=[NSString stringWithFormat:@"parity-case-%03u-provider-%u-output.f32",i,p];
            need(by_name[name]!=nil,"correctness provider output missing from artifact manifest");
            NSString *file=evidence_file(base,name);NSData *d=read_file(file);need(d.length==n,"correctness complete output byte length");
            if(!p){native=d;[oracles addObject:@{@"path":file,@"sha256":by_name[name][@"sha256"]}];need(finite_f32(d.bytes,n)&&written(d.bytes,n),"native correctness oracle finite/full written");}
            else need(!memcmp(native.bytes,d.bytes,n),"correctness outputs not full bit-exact");
            NSString *gn=[name.stringByDeletingPathExtension stringByAppendingString:@".guards"];
            need(by_name[gn]!=nil,"correctness provider guard missing");NSData *g=read_file(evidence_file(base,gn));
            need(g.length==sizeof(guard_bytes)&&!memcmp(g.bytes,guard_bytes,sizeof(guard_bytes)),"correctness output guards changed");
        }
        for(NSString *kind in @[@"kv-unique",@"kv-row-ids"]){
            NSString *name=[NSString stringWithFormat:@"input-case-%03u-%@.guards",i,kind];need(by_name[name]!=nil,"dictionary/row-ID guards missing");
            NSData *g=read_file(evidence_file(base,name));need(g.length==sizeof(guard_bytes)&&!memcmp(g.bytes,guard_bytes,sizeof(guard_bytes)),"dictionary/row-ID guards changed");
        }
    }
    manifest[@"correctness_manifest_path"]=path;manifest[@"correctness_manifest_sha256"]=expected;
    manifest[@"prior_correctness_providers_verified"]=@(count*3);raw(@"correctness-manifest.json",bytes.bytes,bytes.length);
    return oracles;
}
int main(int argc,char **argv){@autoreleasepool {
    if(argc==2&&!strcmp(argv[1],"--cpu-self-test")) {
        emit(@{@"kind":@"cpu_self_test",@"invalid_args_rejected":@(args_self_test()),@"passed":@YES,@"gpu_executed":@NO});return 0;
    }
    NSString *model=nil,*capture_path=nil,*probe_path=nil,*gold_path=nil,*gate_path=nil,*gate_sha=nil;
    unsigned samples=9,reps=1;BOOL host=NO,correctness_only=NO;
    for(int i=1;i<argc;i++) {
        const char *k=argv[i];if(!strcmp(k,"--host-only")){host=YES;continue;}
        if(!strcmp(k,"--correctness-only")){need(!correctness_only,"duplicate correctness-only");correctness_only=YES;continue;}
        need(i+1<argc,"option value");const char *v=argv[++i];
        if(!strcmp(k,"--model"))model=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--capture-json"))capture_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--probe-metallib"))probe_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--production-metallib"))gold_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--out-dir"))output=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--correctness-manifest")){need(!gate_path,"duplicate correctness manifest");gate_path=canonical([NSString stringWithUTF8String:v]);}
        else if(!strcmp(k,"--correctness-manifest-sha256")){need(!gate_sha,"duplicate correctness SHA");gate_sha=[NSString stringWithUTF8String:v];}
        else if(!strcmp(k,"--samples"))samples=number(v,9,9);
        else if(!strcmp(k,"--reps"))reps=number(v,1,1);
        else need(NO,"unknown option");
    }
    need(model&&capture_path&&probe_path&&gold_path&&output,"required model/capture/libraries/new output");
    need((gate_path!=nil)==(gate_sha!=nil),"correctness manifest and SHA must be supplied together");
    need(!correctness_only||!gate_path,"correctness-only and performance gate are mutually exclusive");
    need(host||correctness_only||gate_path,"explicit correctness-only or SHA-bound correctness manifest required before GPU");
    need(!gate_path||(gate_path.isAbsolutePath&&gate_sha.length==64),"absolute correctness manifest and SHA256 required");
    phase=correctness_only?@"correctness":gate_path?@"performance":@"host_validation";
    need(model.isAbsolutePath&&capture_path.isAbsolutePath&&probe_path.isAbsolutePath&&gold_path.isAbsolutePath&&output.isAbsolutePath,"absolute input/output paths");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new output directory required");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output directory creation");
    files=[NSMutableArray new];stored_payloads=[NSMutableDictionary new];manifest=[NSMutableDictionary new];
    parity_records=[NSMutableArray new];input_records=[NSMutableArray new];
    manifest[@"schema_version"]=@1;manifest[@"probe"]=@"prefill_attention_indirect_real_capture";manifest[@"phase"]=phase;
    manifest[@"fullmodel"]=@NO;manifest[@"cold_dram_proven"]=@NO;manifest[@"mode"]=@"concurrent";
    manifest[@"scope"]=@"layout_only_cpu_dictionary_replay";manifest[@"production_builder_verified"]=@NO;
    manifest[@"providers"]=@3;manifest[@"binding_contract"]=binding_contract();manifest[@"timing_command_buffers"]=@0;
    manifest[@"capture_provenance_externally_required"]=@YES;manifest[@"deployment_identity_externally_required"]=@YES;
    manifest[@"logical_output_storage"]=@"identical_payload_hardlinks_after_complete_output_memcmp";
    manifest[@"invalid_args_rejected"]=@(args_self_test());manifest[@"samples"]=@(samples);manifest[@"repetitions_per_cb"]=@(reps);
    manifest[@"binary_path"]=binary_path();manifest[@"binary_sha256"]=file_sha(manifest[@"binary_path"]);
    manifest[@"probe_metallib_path"]=probe_path;manifest[@"probe_metallib_sha256"]=file_sha(probe_path);
    manifest[@"production_metallib_path"]=gold_path;manifest[@"production_metallib_sha256"]=file_sha(gold_path);
    NSData *capture_data=read_file(capture_path);NSDictionary *capture=[NSJSONSerialization JSONObjectWithData:capture_data options:0 error:nil];
    need([capture isKindOfClass:[NSDictionary class]],"capture JSON object");
    need(json_uint(capture[@"schema_version"],1,"schema version")==1&&
        [capture[@"capture_kind"]isEqual:@"myds_prefill_attention_multihead"]&&
        [capture[@"source"]isKindOfClass:[NSDictionary class]]&&[capture[@"source"]count]>0,"prefill capture kind/source");
    manifest[@"capture_json_sha256"]=sha(capture_data.bytes,capture_data.length);manifest[@"capture_source"]=capture[@"source"];
    manifest[@"capture_json_path"]=capture_path;raw(@"capture.json",capture_data.bytes,capture_data.length);
    struct stat st;need(!stat(model.fileSystemRepresentation,&st)&&st.st_size>0,"model stat");
    gguf_file_t *gf=gguf_open(model.fileSystemRepresentation);need(gf!=NULL,"GGUF open");
    need(json_uint(capture[@"model_bytes"],UINT64_MAX,"model bytes")==(uint64_t)st.st_size,"capture model actual file bytes");
    size_t header=(size_t)json_uint(capture[@"model_header_bytes"],st.st_size,"model prefix bytes");
    need(header>0&&[capture[@"model_header_sha256"]isEqual:sha(gf->mmap_addr,header)],"capture model prefix SHA");
    manifest[@"model"]=@{@"path":model,@"bytes":@(st.st_size),@"header_bytes":@(header),
        @"header_sha256":sha(gf->mmap_addr,header),@"tensor_data_offset":@(gf->tensor_data_offset),
        @"device":@(st.st_dev),@"inode":@(st.st_ino),@"mtime_sec":@(st.st_mtimespec.tv_sec),@"mtime_nsec":@(st.st_mtimespec.tv_nsec)};
    NSArray *cases=capture[@"cases"];need([cases isKindOfClass:[NSArray class]]&&cases.count&&cases.count<=16,"1..16 complete captured chunks");
    NSMutableArray *validated=[NSMutableArray new],*case_identity=[NSMutableArray new];
    NSString *base=[capture_path stringByDeletingLastPathComponent];
    for(unsigned i=0;i<cases.count;i++){@autoreleasepool {
        NSDictionary *c=cases[i];need([c isKindOfClass:[NSDictionary class]],"case object");
        unsigned layer=(unsigned)json_uint(c[@"layer"],39,"layer"),rows=(unsigned)json_uint(c[@"rows"],2048,"rows");need(rows>0,"positive captured query count");
        need(json_uint(c[@"keys"],KEYS,"keys")==KEYS,"gathered 640 key shape");
        uint64_t base_pos=json_uint(c[@"base_pos"],UINT32_MAX,"base position"),c_start=json_uint(c[@"c_start"],UINT32_MAX,"chunk start");
        need(base_pos+c_start+rows<=UINT32_MAX,"captured query position range");
        NSData *a_data=payload(c,@"args200",base,200);struct flash_args a;memcpy(&a,a_data.bytes,sizeof(a));
        need(valid_args(&a,rows),"captured gathered prefill ABI/strides/scale");
        NSData *q=payload(c,@"q_f32",base,(size_t)rows*QUERY_BYTES),
            *ids=payload(c,@"kv_row_ids_u32",base,(size_t)rows*KEYS*4),
            *mask=payload(c,@"mask_f16",base,(size_t)rows*KEYS*2),*sinks=payload(c,@"sinks_f32",base,SINK_BYTES);
        size_t unique_bytes=(size_t)json_uint(c[@"kv_unique_f16"][@"bytes"],(uint64_t)rows*KV_QUERY_BYTES,"dictionary bytes");
        need(unique_bytes>0&&unique_bytes%KV_ROW_BYTES==0,"dictionary row alignment");
        NSData *unique=payload(c,@"kv_unique_f16",base,unique_bytes);
        need(finite_f32(q.bytes,q.length)&&finite_f32(sinks.bytes,sinks.length)&&finite_f16(unique.bytes,unique.length/2),"finite real Q/KV/sinks");
        NSString *expanded=c[@"kv_expanded_sha256"];need([expanded isKindOfClass:[NSString class]]&&expanded.length==64,"expanded KV SHA");
        need([expanded_sha(unique,ids,rows,NULL)isEqual:expanded],"lossless expanded KV SHA mismatch");
        NSDictionary *coverage=mask_coverage(mask,rows);
        NSString *sink_name=[NSString stringWithFormat:@"blk.%u.attn_sinks.weight",layer];
        const gguf_tensor_t *sink_tensor=gguf_find_tensor(gf,sink_name.UTF8String);
        need(sink_tensor&&sink_tensor->type==GGML_TYPE_F32&&sink_tensor->n_dims==1&&sink_tensor->dims[0]==64&&sink_tensor->size_bytes==SINK_BYTES,"real model sink tensor descriptor");
        need(gf->tensor_data_offset<=(uint64_t)st.st_size&&sink_tensor->offset<=(uint64_t)st.st_size-gf->tensor_data_offset&&
            SINK_BYTES<=(uint64_t)st.st_size-gf->tensor_data_offset-sink_tensor->offset,"sink tensor inside real file");
        need(!memcmp(sink_tensor->data,sinks.bytes,SINK_BYTES),"captured sinks differ from model weights");
        [validated addObject:@{@"metadata":c,@"args":a_data,@"q":q,@"ids":ids,@"mask":mask,@"sinks":sinks,@"unique":unique}];
        [case_identity addObject:@{@"case":@(i),@"layer":@(layer),@"rows":@(rows),@"keys":@640,
            @"base_pos":@(base_pos),@"c_start":@(c_start),@"q_bytes":@(q.length),@"kv_expanded_bytes":@((uint64_t)rows*KV_QUERY_BYTES),
            @"kv_expanded_sha256":expanded,@"unique_kv_rows":@(unique.length/KV_ROW_BYTES),@"original_mask_stride":@(a.nb31),
            @"mask_padding_reconstructed_not_captured":@YES,@"mask_coverage":coverage,
            @"sink_tensor":sink_name,@"sink_sha256":sha(sink_tensor->data,SINK_BYTES),
            @"sink_file_offset":@(gf->tensor_data_offset+sink_tensor->offset)}];
        NSDictionary *objects=@{@"args200":a_data,@"q.f32":q,@"kv-unique.f16":unique,@"kv-rows.u32":ids,@"mask-compact.f16":mask,@"sinks.f32":sinks};
        for(NSString *kind in objects){NSData *data=objects[kind];raw([NSString stringWithFormat:@"capture-case-%03u-%@",i,kind],data.bytes,data.length);}
    }}
    manifest[@"cases"]=case_identity;manifest[@"total_cases"]=@(cases.count);manifest[@"argv"]=[NSProcessInfo processInfo].arguments;
    NSArray *oracles=gate_path?correctness_oracles(gate_path,gate_sha):nil;
    emit(@{@"kind":@"identity",@"manifest":manifest});
    if(host){finish(@"host_only");gguf_close(gf);emit(@{@"kind":@"complete",@"host_only":@YES,@"metal_device_created":@NO});return 0;}
    need(cases.count==3,"GPU replay requires all three real full chunks; no small fixture");
    const unsigned required_layers[3]={2,8,19};
    for(unsigned i=0;i<cases.count;i++){
        NSDictionary *c=cases[i],*ci=case_identity[i];
        need([c[@"layer"]unsignedIntValue]==required_layers[i]&&[c[@"rows"]unsignedIntValue]==2048&&
            [c[@"base_pos"]unsignedIntValue]==0&&[c[@"c_start"]unsignedIntValue]==2048&&
            [c[@"ratio"]unsignedIntValue]==2&&[c[@"n_comp"]unsignedIntValue]==2048&&[c[@"top_k"]unsignedIntValue]==512,
            "GPU replay requires the bound R2 layer/position/2048-query capture");
        need([ci[@"mask_coverage"][@"minimum_valid_picks"]unsignedIntValue]==512&&
            [ci[@"mask_coverage"][@"full_512_pick_queries"]unsignedIntValue]==2048&&
            [ci[@"mask_coverage"][@"minimum_valid_window"]unsignedIntValue]==128,"complete 640-active-key queries required");
    }
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");NSError *error=nil;
    id<MTLLibrary> probe=[device newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];
    if(!probe)fprintf(stderr,"probe library: %s\n",error.localizedDescription.UTF8String);need(probe!=nil,"probe library");
    id<MTLLibrary> golden=[device newLibraryWithURL:[NSURL fileURLWithPath:gold_path]error:&error];need(golden!=nil,"production library");
    NSArray *names=@[@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",
        @"kernel_myds_prefill_attention_indirect_f16_dk512_dv512"];
    const unsigned batches[3]={1,1,1},smem[3]={3328,3328,3328};
    id<MTLComputePipelineState> psos[3];NSMutableArray *bindings=[NSMutableArray new];
    for(unsigned p=0;p<3;p++) {
        psos[p]=pipeline(device,p?probe:golden,names[p],32*batches[p],smem[p]);
        NSMutableArray *ordered_constants=[NSMutableArray new];
        const unsigned fc_indices[10]={400,401,402,403,404,405,420,421,422,423};
        const int32_t fc_values[10]={1,1,0,0,0,0,512,512,1,1};
        for(unsigned i=0;i<10;i++)[ordered_constants addObject:@{@"index":@(fc_indices[i]),@"is_bool":@(i<6),@"value":@(fc_values[i])}];
        [bindings addObject:@{@"provider":@(p),@"symbol":names[p],@"library":p?@"probe":@"production",
            @"head_batch":@(batches[p]),@"groups_x":@(64/batches[p]),@"groups_y":@"captured_rows",
            @"threads":@[@32,@(batches[p]),@1],@"args_bytes":@200,@"dynamic_threadgroup_bytes":@(smem[p]),
            @"static_threadgroup_bytes":@(psos[p].staticThreadgroupMemoryLength),@"max_threads":@(psos[p].maxTotalThreadsPerThreadgroup),
            @"thread_execution_width":@(psos[p].threadExecutionWidth),@"logical_nsg":@1,@"logical_nwg":@1,
            @"ordered_constants":ordered_constants,@"buffer_slots":p==2?
                @[@"args200",@"q_f32",@"kv_unique_f16",@"kv_unique_f16",@"mask_original_stride_f16",@"sinks_f32",@"unused_pad",@"full_output_f32",@"kv_row_ids_u32"]:
                @[@"args200",@"q_f32",@"expanded_kv_f16",@"expanded_kv_f16",@"mask_original_stride_f16",@"sinks_f32",@"unused_pad",@"full_output_f32"]}];
    }
    manifest[@"device"]=device.name;manifest[@"bindings"]=bindings;emit(@{@"kind":@"bindings",@"bindings":bindings,@"device":device.name});
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"command queue");
    for(unsigned index=0;index<validated.count;index++){@autoreleasepool {
        NSDictionary *v=validated[index],*c=v[@"metadata"];fixture f={0};f.index=index;f.layer=[c[@"layer"]unsignedIntValue];f.rows=[c[@"rows"]unsignedIntValue];
        NSData *args=v[@"args"],*q=v[@"q"],*ids=v[@"ids"],*unique=v[@"unique"],*mask=v[@"mask"],*sinks=v[@"sinks"];
        memcpy(&f.args,args.bytes,200);size_t qbytes=(size_t)f.rows*QUERY_BYTES,kvbytes=(size_t)f.rows*KV_QUERY_BYTES,maskbytes=(size_t)f.rows*f.args.nb31;
        f.unique_bytes=unique.length;
        f.q=alloc(device,qbytes,NO);f.kv=alloc(device,kvbytes,NO);f.unique=alloc(device,unique.length,NO);
        f.ids=alloc(device,ids.length,NO);f.mask=alloc(device,maskbytes,NO);f.sinks=alloc(device,SINK_BYTES,NO);f.pad=alloc(device,16,YES);
        memcpy(body(f.q),q.bytes,qbytes);memcpy(body(f.sinks),sinks.bytes,SINK_BYTES);
        memcpy(body(f.unique),unique.bytes,unique.length);memcpy(body(f.ids),ids.bytes,ids.length);
        need([expanded_sha(unique,ids,f.rows,body(f.kv))isEqual:c[@"kv_expanded_sha256"]],"GPU-bound complete expanded KV identity");
        uint16_t *mask_body=body(f.mask);for(size_t i=0;i<maskbytes/2;i++)mask_body[i]=0xfc00u;
        for(unsigned r=0;r<f.rows;r++)memcpy((uint8_t*)body(f.mask)+(size_t)r*f.args.nb31,(const uint8_t*)mask.bytes+(size_t)r*1280,1280);
        id<MTLBuffer> inputs[7]={f.q,f.kv,f.unique,f.ids,f.mask,f.sinks,f.pad};
        const size_t input_bytes[7]={qbytes,kvbytes,unique.length,ids.length,maskbytes,SINK_BYTES,16};
        for(unsigned i=0;i<7;i++)f.input_sha[i]=sha(body(inputs[i]),input_bytes[i]);f.args_sha=sha(&f.args,sizeof(f.args));
        for(unsigned p=0;p<3;p++)f.out[p]=alloc(device,qbytes,YES);
        if(correctness_only){
            for(unsigned p=0;p<3;p++){(void)run(queue,&f,p,1,psos[p],batches[p],smem[p]);validate(&f,p,YES);}
        }else{
            NSDictionary *oracle=oracles[index];f.oracle=read_file(oracle[@"path"]);
            need(f.oracle.length==qbytes&&[sha(f.oracle.bytes,f.oracle.length)isEqual:oracle[@"sha256"]],"bound complete native oracle changed before timing");
            const unsigned candidate=2;
            for(unsigned control=0;control<2;control++) {
            unsigned providers[3]={control,candidate,control};double times[3][64];
            for(unsigned sample=0;sample<=samples;sample++)for(unsigned arm=0;arm<3;arm++) {
                unsigned p=providers[arm];double begin=wall();id<MTLCommandBuffer> cb=run(queue,&f,p,reps,psos[p],batches[p],smem[p]);double end=wall();validate(&f,p,NO);
                need(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime&&cb.GPUStartTime>0,"valid GPU timeline span");
                double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/reps;if(sample)times[arm][sample-1]=us;
                emit(@{@"kind":@"timing",@"mode":@"concurrent",@"case":@(index),@"layer":@(f.layer),@"rows":@(f.rows),@"keys":@640,
                    @"variant":@"indirect",@"control_kind":control?@"probe":@"production",@"provider":@(p),@"symbol":names[p],
                    @"arm":@[@"A1",@"B",@"A2"][arm],@"sample":@(sample),@"warmup":@(sample==0),@"dispatches":@(reps),@"commits":@1,
                    @"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"gpu_us_per_chunk":@(us),
                    @"wall_us_per_chunk":@((end-begin)*1e6/reps),@"full_f32_bit_exact":@YES,@"all_gpu_inputs_unchanged":@YES,@"component_only":@YES,@"completed_cb":@YES,
                    @"output_sha256":sha(body(f.out[p]),qbytes),@"scope":@"layout_only_cpu_dictionary_replay"});
                manifest[@"timing_command_buffers"]=@([manifest[@"timing_command_buffers"]unsignedIntValue]+1);
            }
            double a1=median(times[0],samples),b=median(times[1],samples),a2=median(times[2],samples),a=(a1+a2)/2;
            emit(@{@"kind":@"summary",@"mode":@"concurrent",@"case":@(index),@"layer":@(f.layer),@"rows":@(f.rows),@"keys":@640,
                @"variant":@"indirect",@"control_kind":control?@"probe":@"production",@"control_symbol":names[control],@"candidate_symbol":names[candidate],
                @"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),@"control_drift_percent":@((a2/a1-1)*100),@"time_saved_percent":@((1-b/a)*100),
                @"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),@"fullmodel_speedup_proven":@NO,@"cold_dram_proven":@NO,
                @"scope":@"layout_only_cpu_dictionary_replay",@"production_builder_verified":@NO});
            }
        }
        need(guards(f.q,qbytes)&&guards(f.kv,kvbytes)&&guards(f.mask,maskbytes)&&guards(f.sinks,SINK_BYTES)&&guards(f.pad,16),"all input guards");
        need(!memcmp(body(f.q),q.bytes,qbytes)&&!memcmp(body(f.sinks),sinks.bytes,SINK_BYTES)&&
            [sha(body(f.kv),kvbytes)isEqual:c[@"kv_expanded_sha256"]],"complete real Q/KV/sinks unchanged");
        for(unsigned r=0;r<f.rows;r++)need(!memcmp((uint8_t*)body(f.mask)+(size_t)r*f.args.nb31,(const uint8_t*)mask.bytes+(size_t)r*1280,1280),"consumed mask bytes unchanged");
        need(guards(f.unique,unique.length)&&guards(f.ids,ids.length)&&!memcmp(body(f.unique),unique.bytes,unique.length)&&
            !memcmp(body(f.ids),ids.bytes,ids.length),"complete dictionary/row-ID bytes and guards unchanged");
        save_input_guards(&f);
        f.q=nil;f.kv=nil;f.unique=nil;f.ids=nil;f.mask=nil;f.sinks=nil;f.pad=nil;f.oracle=nil;for(unsigned p=0;p<3;p++)f.out[p]=nil;
    }}
    need([file_sha(probe_path)isEqual:manifest[@"probe_metallib_sha256"]]&&[file_sha(gold_path)isEqual:manifest[@"production_metallib_sha256"]]&&
        [file_sha(capture_path)isEqual:manifest[@"capture_json_sha256"]],"library/capture identity changed during run");
    need([file_sha(binary_path())isEqual:manifest[@"binary_sha256"]],"replay executable changed during run");
    manifest[@"full_f32_bit_exact"]=@YES;manifest[@"attention_output_checked"]=@YES;manifest[@"all_gpu_inputs_unchanged"]=@YES;
    need(correctness_only?parity_records.count==cases.count*3:[manifest[@"timing_command_buffers"]unsignedIntValue]==180,"complete phase command counts");
    finish(@"complete");gguf_close(gf);emit(@{@"kind":@"complete",@"phase":phase,@"cases":@(cases.count),@"providers":@3,
        @"attention_output_checked":@YES,@"full_f32_bit_exact":@YES,@"all_gpu_inputs_unchanged":@YES,@"mode":@"concurrent",@"fullmodel":@NO,@"cold_dram_proven":@NO,
        @"timing_command_buffers":manifest[@"timing_command_buffers"],@"scope":@"layout_only_cpu_dictionary_replay",@"production_builder_verified":@NO});
}return 0;}
