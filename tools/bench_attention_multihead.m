/* Captured prefill attention with full gathered-KV working set. No runtime init.
 * --model GGUF --capture-json JSON --probe-metallib PATH --production-metallib PATH
 * --out-dir NEW [--samples 9 --reps 1 --host-only]
 * Schema 1, capture_kind=myds_prefill_attention_multihead. Each case stores
 * args200, Q F32, exact F16 KV row dictionary + u32 IDs + expanded SHA256,
 * compact 640-column F16 mask, and model F32 sinks. Payloads require path/SHA/bytes.
 * One full captured query chunk is processed at a time; no small hot fixture is
 * substituted for the captured working set. Timings use concurrent encoders only.
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
static void finish(NSString *status) {
    if(!manifest||!output)return;
    manifest[@"status"]=status; manifest[@"files"]=files;
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
    id<MTLBuffer> q,kv,mask,sinks,pad,out[4];
} fixture;
static BOOL overlap(uintptr_t a,size_t an,uintptr_t b,size_t bn){return a<=b?b-a<an:a-b<bn;}
static BOOL bindings_safe(fixture *f,unsigned provider) {
    size_t out_bytes=(size_t)f->rows*QUERY_BYTES;
    if(provider>=4||!valid_args(&f->args,f->rows)||!f->out[provider])return NO;
    uintptr_t out=(uintptr_t)body(f->out[provider]);
    id<MTLBuffer> readers[5]={f->q,f->kv,f->mask,f->sinks,f->pad};
    const size_t bytes[5]={out_bytes,(size_t)f->rows*KV_QUERY_BYTES,(size_t)f->rows*f->args.nb31,SINK_BYTES,16};
    for(unsigned i=0;i<5;i++)if(!readers[i]||readers[i].length<bytes[i]+2*GUARD||
        overlap(out,out_bytes,(uintptr_t)body(readers[i]),bytes[i]))return NO;
    return f->out[provider].length>=out_bytes+2*GUARD&&out%16u==0;
}
static id<MTLCommandBuffer> run(id<MTLCommandQueue> queue,fixture *f,unsigned provider,unsigned reps,
        id<MTLComputePipelineState> p,unsigned head_batch,unsigned smem) {
    need(bindings_safe(f,provider),"invalid/aliased attention bindings before command creation");
    /* The previous CB is complete. Re-poison before every measured command so
     * a skipped store cannot inherit a valid parity result. This CPU write is
     * outside GPU timestamps and applies identically to all four providers. */
    memset(body(f->out[provider]),0xa5,(size_t)f->rows*QUERY_BYTES);
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"attention CB");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"concurrent attention encoder");
    for(unsigned rep=0;rep<reps;rep++){
        [e setComputePipelineState:p];[e setBytes:&f->args length:sizeof(f->args) atIndex:0];
        id<MTLBuffer> b[7]={f->q,f->kv,f->kv,f->mask,f->sinks,f->pad,f->out[provider]};
        for(unsigned i=0;i<7;i++)[e setBuffer:b[i] offset:GUARD atIndex:i+1];
        [e setThreadgroupMemoryLength:smem atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(HEADS/head_batch,f->rows,1)
            threadsPerThreadgroup:MTLSizeMake(32,head_batch,1)];
        id<MTLResource> out=f->out[provider];[e memoryBarrierWithResources:&out count:1];
    }
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU error: %s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"attention command completion");return cb;
}
static void validate(fixture *f,unsigned provider,BOOL save_output) {
    size_t bytes=(size_t)f->rows*QUERY_BYTES,first=SIZE_MAX;
    size_t bad=mismatches(body(f->out[0]),body(f->out[provider]),bytes,&first);
    BOOL ok=!bad&&finite_f32(body(f->out[provider]),bytes)&&written(body(f->out[provider]),bytes)&&guards(f->out[provider],bytes);
    if(save_output){
        dump([NSString stringWithFormat:@"parity-case-%03u-provider-%u-output",f->index,provider],f->out[provider],bytes);
        emit(@{@"kind":@"parity",@"mode":@"concurrent",@"case":@(f->index),@"layer":@(f->layer),@"provider":@(provider),
            @"rows":@(f->rows),@"keys":@640,@"output_bytes":@(bytes),@"mismatches":@(bad),
            @"first_mismatch":first==SIZE_MAX?(id)[NSNull null]:@(first),@"full_f32_bit_exact":@(ok),
            @"finite_guards_written":@(ok),@"completed_cb":@YES});
    }
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
int main(int argc,char **argv){@autoreleasepool {
    if(argc==2&&!strcmp(argv[1],"--cpu-self-test")) {
        emit(@{@"kind":@"cpu_self_test",@"invalid_args_rejected":@(args_self_test()),@"passed":@YES,@"gpu_executed":@NO});return 0;
    }
    NSString *model=nil,*capture_path=nil,*probe_path=nil,*gold_path=nil;unsigned samples=9,reps=1;BOOL host=NO;
    for(int i=1;i<argc;i++) {
        const char *k=argv[i];if(!strcmp(k,"--host-only")){host=YES;continue;}
        need(i+1<argc,"option value");const char *v=argv[++i];
        if(!strcmp(k,"--model"))model=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--capture-json"))capture_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--probe-metallib"))probe_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--production-metallib"))gold_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--out-dir"))output=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--samples"))samples=number(v,3,64);
        else if(!strcmp(k,"--reps"))reps=number(v,1,16);
        else need(NO,"unknown option");
    }
    need(model&&capture_path&&probe_path&&gold_path&&output,"required model/capture/libraries/new output");
    need(model.isAbsolutePath&&capture_path.isAbsolutePath&&probe_path.isAbsolutePath&&gold_path.isAbsolutePath&&output.isAbsolutePath,"absolute input/output paths");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new output directory required");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output directory creation");
    files=[NSMutableArray new];stored_payloads=[NSMutableDictionary new];manifest=[NSMutableDictionary new];
    manifest[@"schema_version"]=@1;manifest[@"probe"]=@"prefill_attention_multihead_real_capture";
    manifest[@"fullmodel"]=@NO;manifest[@"cold_dram_proven"]=@NO;manifest[@"mode"]=@"concurrent";
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
        @"header_sha256":sha(gf->mmap_addr,header),@"tensor_data_offset":@(gf->tensor_data_offset)};
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
    emit(@{@"kind":@"identity",@"manifest":manifest});
    if(host){finish(@"host_only");gguf_close(gf);emit(@{@"kind":@"complete",@"host_only":@YES,@"metal_device_created":@NO});return 0;}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");NSError *error=nil;
    id<MTLLibrary> probe=[device newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];
    if(!probe)fprintf(stderr,"probe library: %s\n",error.localizedDescription.UTF8String);need(probe!=nil,"probe library");
    id<MTLLibrary> golden=[device newLibraryWithURL:[NSURL fileURLWithPath:gold_path]error:&error];need(golden!=nil,"production library");
    NSArray *names=@[@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",
        @"kernel_myds_prefill_attention_head2_f16_dk512_dv512",@"kernel_myds_prefill_attention_head4_f16_dk512_dv512"];
    const unsigned batches[4]={1,1,2,4},smem[4]={3328,3328,23040,29696};
    id<MTLComputePipelineState> psos[4];NSMutableArray *bindings=[NSMutableArray new];
    for(unsigned p=0;p<4;p++) {
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
            @"ordered_constants":ordered_constants,@"buffer_slots":@[@"args200",@"q_f32",@"expanded_kv_f16",@"expanded_kv_f16",@"mask_original_stride_f16",@"sinks_f32",@"unused_pad",@"full_output_f32"]}];
    }
    manifest[@"device"]=device.name;manifest[@"bindings"]=bindings;emit(@{@"kind":@"bindings",@"bindings":bindings,@"device":device.name});
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"command queue");
    for(unsigned index=0;index<validated.count;index++){@autoreleasepool {
        NSDictionary *v=validated[index],*c=v[@"metadata"];fixture f={0};f.index=index;f.layer=[c[@"layer"]unsignedIntValue];f.rows=[c[@"rows"]unsignedIntValue];
        NSData *args=v[@"args"],*q=v[@"q"],*ids=v[@"ids"],*unique=v[@"unique"],*mask=v[@"mask"],*sinks=v[@"sinks"];
        memcpy(&f.args,args.bytes,200);size_t qbytes=(size_t)f.rows*QUERY_BYTES,kvbytes=(size_t)f.rows*KV_QUERY_BYTES,maskbytes=(size_t)f.rows*f.args.nb31;
        f.q=alloc(device,qbytes,NO);f.kv=alloc(device,kvbytes,NO);f.mask=alloc(device,maskbytes,NO);f.sinks=alloc(device,SINK_BYTES,NO);f.pad=alloc(device,16,YES);
        memcpy(body(f.q),q.bytes,qbytes);memcpy(body(f.sinks),sinks.bytes,SINK_BYTES);
        need([expanded_sha(unique,ids,f.rows,body(f.kv))isEqual:c[@"kv_expanded_sha256"]],"GPU-bound complete expanded KV identity");
        uint16_t *mask_body=body(f.mask);for(size_t i=0;i<maskbytes/2;i++)mask_body[i]=0xfc00u;
        for(unsigned r=0;r<f.rows;r++)memcpy((uint8_t*)body(f.mask)+(size_t)r*f.args.nb31,(const uint8_t*)mask.bytes+(size_t)r*1280,1280);
        for(unsigned p=0;p<4;p++){f.out[p]=alloc(device,qbytes,YES);(void)run(queue,&f,p,1,psos[p],batches[p],smem[p]);validate(&f,p,YES);}
        for(unsigned candidate=2;candidate<4;candidate++)for(unsigned control=0;control<2;control++) {
            unsigned providers[3]={control,candidate,control};double times[3][64];
            for(unsigned sample=0;sample<=samples;sample++)for(unsigned arm=0;arm<3;arm++) {
                unsigned p=providers[arm];double begin=wall();id<MTLCommandBuffer> cb=run(queue,&f,p,reps,psos[p],batches[p],smem[p]);double end=wall();validate(&f,p,NO);
                need(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime&&cb.GPUStartTime>0,"valid GPU timeline span");
                double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/reps;if(sample)times[arm][sample-1]=us;
                emit(@{@"kind":@"timing",@"mode":@"concurrent",@"case":@(index),@"layer":@(f.layer),@"rows":@(f.rows),@"keys":@640,
                    @"variant":@(batches[candidate]),@"control_kind":control?@"probe":@"production",@"provider":@(p),@"symbol":names[p],
                    @"arm":@[@"A1",@"B",@"A2"][arm],@"sample":@(sample),@"warmup":@(sample==0),@"dispatches":@(reps),@"commits":@1,
                    @"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"gpu_us_per_chunk":@(us),
                    @"wall_us_per_chunk":@((end-begin)*1e6/reps),@"full_f32_bit_exact":@YES,@"component_only":@YES,@"completed_cb":@YES});
            }
            double a1=median(times[0],samples),b=median(times[1],samples),a2=median(times[2],samples),a=(a1+a2)/2;
            emit(@{@"kind":@"summary",@"mode":@"concurrent",@"case":@(index),@"layer":@(f.layer),@"rows":@(f.rows),@"keys":@640,
                @"variant":@(batches[candidate]),@"control_kind":control?@"probe":@"production",@"control_symbol":names[control],@"candidate_symbol":names[candidate],
                @"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),@"control_drift_percent":@((a2/a1-1)*100),@"time_saved_percent":@((1-b/a)*100),
                @"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),@"fullmodel_speedup_proven":@NO,@"cold_dram_proven":@NO});
        }
        need(guards(f.q,qbytes)&&guards(f.kv,kvbytes)&&guards(f.mask,maskbytes)&&guards(f.sinks,SINK_BYTES)&&guards(f.pad,16),"all input guards");
        need(!memcmp(body(f.q),q.bytes,qbytes)&&!memcmp(body(f.sinks),sinks.bytes,SINK_BYTES)&&
            [sha(body(f.kv),kvbytes)isEqual:c[@"kv_expanded_sha256"]],"complete real Q/KV/sinks unchanged");
        for(unsigned r=0;r<f.rows;r++)need(!memcmp((uint8_t*)body(f.mask)+(size_t)r*f.args.nb31,(const uint8_t*)mask.bytes+(size_t)r*1280,1280),"consumed mask bytes unchanged");
        f.q=nil;f.kv=nil;f.mask=nil;f.sinks=nil;f.pad=nil;for(unsigned p=0;p<4;p++)f.out[p]=nil;
    }}
    need([file_sha(probe_path)isEqual:manifest[@"probe_metallib_sha256"]]&&[file_sha(gold_path)isEqual:manifest[@"production_metallib_sha256"]]&&
        [file_sha(capture_path)isEqual:manifest[@"capture_json_sha256"]],"library/capture identity changed during run");
    finish(@"complete");gguf_close(gf);emit(@{@"kind":@"complete",@"cases":@(cases.count),@"providers":@4,
        @"attention_output_checked":@YES,@"full_f32_bit_exact":@YES,@"mode":@"concurrent",@"fullmodel":@NO,@"cold_dram_proven":@NO});
}return 0;}
