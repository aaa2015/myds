/* Real captured Qa/KV Q4_K BF16 dual-output component. No inference/runtime init.
 * make bin/bench_qkv_dual
 * --model GGUF --capture-json JSON --probe-metallib PATH
 * --production-metallib PATH --out-dir NEW [--samples 9 --reps 8 --host-only]
 * Capture schema 1: capture_kind=myds_decode_qkv, model_bytes,
 * model_header_bytes, model_header_sha256, source(object), cases(array).
 * Each case: layer, position, input_f32:{path,sha256}; 5120 raw little-endian F32.
 * Fixed production Q4 NR2, NSG4, fused BF16 rounding. Tests serial and concurrent
 * encoders. A component result is not cold DRAM, full-model speedup or adoption.
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

enum { DIM=5120, QA=1280, KV=512, GUARD=64, ROW_BYTES=2880,
       QA_WEIGHT_BYTES=3686400, KV_WEIGHT_BYTES=1474560, INPUT_BYTES=20480,
       QA_BYTES=5120, KV_BYTES=2048, NSG=4, NR=2, TG_BYTES=256 };
struct mv_args {
    int32_t ne00,ne01,ne02; uint64_t nb00,nb01,nb02,nb03;
    int32_t ne10,ne11,ne12; uint64_t nb10,nb11,nb12,nb13;
    int32_t ne0,ne1,nr0; int16_t r2,r3;
};
_Static_assert(sizeof(struct mv_args)==112,"Q4 production ABI");
static NSMutableDictionary *manifest;
static NSMutableArray *files;
static NSString *output;
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
    NSData *d=[NSData dataWithBytes:p length:bytes];
    need([d writeToFile:[output stringByAppendingPathComponent:n]
        options:NSDataWritingWithoutOverwriting error:nil],"new raw artifact write");
    [files addObject:@{@"path":n,@"bytes":@(bytes),@"sha256":sha(p,bytes)}];
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
static BOOL bf16(const void *p,size_t n) {
    const uint32_t *v=p;for(size_t i=0;i<n/4;i++)if(v[i]&0xffffu)return NO;return YES;
}
/* These are the component encoder's own binding checks. Production host-entry
 * rejection and provider adoption are validated separately by the outer gate. */
struct span { uintptr_t base; size_t bytes; };
static BOOL overlap(struct span a,struct span b) {
    if(!a.base||!b.base||!a.bytes||!b.bytes)return NO;
    return a.base<=b.base?b.base-a.base<a.bytes:a.base-b.base<b.bytes;
}
static BOOL bindings_safe(struct span qa,struct span kv,struct span qw,struct span kw,struct span x) {
    struct span read[3]={qw,kw,x};
    if(qa.bytes!=QA_BYTES||kv.bytes!=KV_BYTES||qw.bytes!=QA_WEIGHT_BYTES||kw.bytes!=KV_WEIGHT_BYTES||
        x.bytes!=INPUT_BYTES||!qa.base||!kv.base||!qw.base||!kw.base||!x.base||
        qa.base%16u||kv.base%16u||qw.base%16u||kw.base%16u||x.base%16u||overlap(qa,kv))return NO;
    for(unsigned i=0;i<3;i++)if(overlap(qa,read[i])||overlap(kv,read[i]))return NO;
    return YES;
}
static unsigned alias_self_test(void) {
    struct span qa={0x10000,QA_BYTES},kv={0x20000,KV_BYTES},qw={0x100000,QA_WEIGHT_BYTES},
        kw={0x1000000,KV_WEIGHT_BYTES},x={0x2000000,INPUT_BYTES};
    need(bindings_safe(qa,kv,qw,kw,x),"valid component buffer spans");unsigned bad=0;
    struct span v=kv;v.base=qa.base;bad+=!bindings_safe(qa,v,qw,kw,x);
    v=qa;v.base=x.base;bad+=!bindings_safe(v,kv,qw,kw,x);
    v=kv;v.base=x.base+16;bad+=!bindings_safe(qa,v,qw,kw,x);
    v=qa;v.base=qw.base+16;bad+=!bindings_safe(v,kv,qw,kw,x);
    v=kv;v.base=kw.base+16;bad+=!bindings_safe(qa,v,qw,kw,x);
    v=qa;v.base++;bad+=!bindings_safe(v,kv,qw,kw,x);
    v=x;v.bytes-=4;bad+=!bindings_safe(qa,kv,qw,kw,v);
    v=qa;v.base=0;bad+=!bindings_safe(v,kv,qw,kw,x);
    need(bad==8,"component alias/alignment/size rejection controls");return bad;
}
static struct span span(id<MTLBuffer> b,size_t bytes) {
    return(struct span){(uintptr_t)body(b),bytes};
}
typedef struct { unsigned layer,index;uint64_t position;id<MTLBuffer> input;
    id<MTLBuffer> qa[3],kv[3]; } fixture;
static struct mv_args fixed_args(unsigned rows) {
    return(struct mv_args){.ne00=DIM,.ne01=(int32_t)rows,.ne02=1,.nb00=144,.nb01=ROW_BYTES,
        .nb02=(uint64_t)ROW_BYTES*rows,.nb03=(uint64_t)ROW_BYTES*rows,.ne10=DIM,.ne11=1,.ne12=1,
        .nb10=4,.nb11=INPUT_BYTES,.nb12=INPUT_BYTES,.nb13=INPUT_BYTES,
        .ne0=(int32_t)rows,.ne1=1,.nr0=NR,.r2=1,.r3=1};
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,NSString *name) {
    NSError *error=nil;short nsg=NSG;MTLFunctionConstantValues *c=[MTLFunctionConstantValues new];
    [c setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> f=[l newFunctionWithName:name constantValues:c error:&error];
    if(!f)fprintf(stderr,"function %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(f!=nil,"explicit function absent; fallback forbidden");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];
    if(!p)fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"pipeline SIMD/threads");
    need(p.staticThreadgroupMemoryLength+TG_BYTES<=d.maxThreadgroupMemoryLength,"pipeline scratch limit");return p;
}
static void bind_native(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,
        id<MTLBuffer> w,id<MTLBuffer> x,id<MTLBuffer> y,unsigned rows) {
    struct mv_args a=fixed_args(rows);[e setComputePipelineState:p];
    [e setBytes:&a length:sizeof(a) atIndex:0];[e setBuffer:w offset:GUARD atIndex:1];
    [e setBuffer:x offset:GUARD atIndex:2];[e setBuffer:y offset:GUARD atIndex:3];
    [e setThreadgroupMemoryLength:TG_BYTES atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(rows/(NR*NSG),1,1) threadsPerThreadgroup:MTLSizeMake(32,NSG,1)];
}
static id<MTLCommandBuffer> run(id<MTLCommandQueue> q,fixture *f,unsigned count,unsigned provider,
        unsigned reps,BOOL concurrent,id<MTLBuffer> qw,id<MTLBuffer> kw,id<MTLComputePipelineState> p) {
    for(unsigned i=0;i<count;i++)need(bindings_safe(span(f[i].qa[provider],QA_BYTES),span(f[i].kv[provider],KV_BYTES),
        span(qw,QA_WEIGHT_BYTES),span(kw,KV_WEIGHT_BYTES),span(f[i].input,INPUT_BYTES)),"reject alias/invalid binding before CB creation");
    id<MTLCommandBuffer> cb=[q commandBuffer];need(cb!=nil,"command buffer");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];
    pass.dispatchType=concurrent?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"compute encoder");
    for(unsigned rep=0;rep<reps;rep++)for(unsigned i=0;i<count;i++) {
        if(provider<2) {
            /* Preserve the two independent production projections, including
             * their opportunity to overlap in the concurrent encoder. */
            bind_native(e,p,qw,f[i].input,f[i].qa[provider],QA);
            bind_native(e,p,kw,f[i].input,f[i].kv[provider],KV);
        } else {
            struct mv_args a=fixed_args(QA),b=fixed_args(KV);[e setComputePipelineState:p];
            [e setBytes:&a length:sizeof(a) atIndex:0];[e setBytes:&b length:sizeof(b) atIndex:1];
            [e setBuffer:qw offset:GUARD atIndex:2];[e setBuffer:kw offset:GUARD atIndex:3];
            [e setBuffer:f[i].input offset:GUARD atIndex:4];
            [e setBuffer:f[i].qa[provider] offset:GUARD atIndex:5];[e setBuffer:f[i].kv[provider] offset:GUARD atIndex:6];
            [e setThreadgroupMemoryLength:TG_BYTES atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((QA+KV)/(NR*NSG),1,1) threadsPerThreadgroup:MTLSizeMake(32,NSG,1)];
        }
        /* Both arms join the same two outputs once per pair. This prevents
         * repeated-fixture WAW races without serializing Qa versus KV. The
         * downstream RMSNorm/HC/indexer graph is outside this component. */
        id<MTLResource> joined[2]={f[i].qa[provider],f[i].kv[provider]};
        [e memoryBarrierWithResources:joined count:2];
    }
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)
        fprintf(stderr,"GPU failure status=%lu: %s\n",(unsigned long)cb.status,cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"command buffer completion");return cb;
}
static void validate(fixture *f,unsigned count,unsigned provider,NSString *mode,BOOL save_outputs) {
    for(unsigned i=0;i<count;i++) {
        size_t first=SIZE_MAX;size_t qb=mismatches(body(f[i].qa[0]),body(f[i].qa[provider]),QA_BYTES,&first),
            kb=mismatches(body(f[i].kv[0]),body(f[i].kv[provider]),KV_BYTES,&first);
        BOOL ok=!qb&&!kb;
        id<MTLBuffer> out[2]={f[i].qa[provider],f[i].kv[provider]};size_t bytes[2]={QA_BYTES,KV_BYTES};
        for(unsigned k=0;k<2;k++)ok&=guards(out[k],bytes[k])&&finite_f32(body(out[k]),bytes[k])&&
            written(body(out[k]),bytes[k])&&bf16(body(out[k]),bytes[k]);
        ok&=guards(f[i].input,INPUT_BYTES);
        if(save_outputs) {
            NSString *prefix=[NSString stringWithFormat:@"parity-%@-case-%03u-provider-%u",mode,f[i].index,provider];
            dump([prefix stringByAppendingString:@"-qa"],out[0],QA_BYTES);
            dump([prefix stringByAppendingString:@"-kv"],out[1],KV_BYTES);
            emit(@{@"kind":@"parity",@"mode":mode,@"case":@(f[i].index),@"layer":@(f[i].layer),
                @"position":@(f[i].position),@"provider":@(provider),@"qa_mismatches":@(qb),
                @"kv_mismatches":@(kb),@"full_f32_bit_exact":@(ok),@"finite_guards_written":@(ok),
                @"bf16_postround":@(ok),@"completed_cb":@YES});
        }
        need(ok,"Qa/KV full bits/BF16/finite/guards/writes");
    }
}
int main(int argc,char **argv){@autoreleasepool {
    if(argc==2&&!strcmp(argv[1],"--cpu-self-test")) {
        emit(@{@"kind":@"cpu_self_test",@"rejected_invalid_bindings":@(alias_self_test()),@"passed":@YES,@"gpu_executed":@NO});return 0;
    }
    NSString *model=nil,*capture_path=nil,*probe_path=nil,*gold_path=nil;unsigned samples=9,reps=8;BOOL host=NO;
    for(int i=1;i<argc;i++) {
        const char *k=argv[i];if(!strcmp(k,"--host-only")){host=YES;continue;}
        need(i+1<argc,"option value");const char *v=argv[++i];
        if(!strcmp(k,"--model"))model=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--capture-json"))capture_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--probe-metallib"))probe_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--production-metallib"))gold_path=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--out-dir"))output=canonical([NSString stringWithUTF8String:v]);
        else if(!strcmp(k,"--samples"))samples=number(v,3,64);
        else if(!strcmp(k,"--reps"))reps=number(v,1,256);
        else need(NO,"unknown option");
    }
    need(model&&capture_path&&probe_path&&gold_path&&output,"required model/capture/libraries/new output directory");
    need(model.isAbsolutePath&&capture_path.isAbsolutePath&&probe_path.isAbsolutePath&&gold_path.isAbsolutePath&&output.isAbsolutePath,"absolute paths required");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"output must not exist");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output directory");
    files=[NSMutableArray new];manifest=[NSMutableDictionary new];manifest[@"schema_version"]=@1;
    manifest[@"probe"]=@"qkv_dual_real_capture";manifest[@"fullmodel"]=@NO;manifest[@"cold_dram_proven"]=@NO;
    manifest[@"repeated_fixture_cache_reuse"]=@YES;manifest[@"capture_provenance_externally_required"]=@YES;
    manifest[@"deployment_identity_externally_required"]=@YES;manifest[@"nsg"]=@(NSG);manifest[@"nr0"]=@(NR);
    manifest[@"samples"]=@(samples);manifest[@"repetitions_per_cb"]=@(reps);
    manifest[@"modes"]=@[@"serial",@"concurrent"];manifest[@"host_invalid_bindings_rejected"]=@(alias_self_test());
    manifest[@"production_host_rejection_proven"]=@NO;manifest[@"bf16_postround"]=@YES;
    manifest[@"binary_path"]=binary_path();manifest[@"binary_sha256"]=file_sha(manifest[@"binary_path"]);
    manifest[@"probe_metallib_path"]=probe_path;manifest[@"probe_metallib_sha256"]=file_sha(probe_path);
    manifest[@"production_metallib_path"]=gold_path;manifest[@"production_metallib_sha256"]=file_sha(gold_path);
    NSData *capture_data=read_file(capture_path);
    NSDictionary *capture=[NSJSONSerialization JSONObjectWithData:capture_data options:0 error:nil];
    need([capture isKindOfClass:[NSDictionary class]],"capture JSON object");
    need(json_uint(capture[@"schema_version"],1,"capture schema")==1&&
        [capture[@"capture_kind"]isEqual:@"myds_decode_qkv"]&&
        [capture[@"source"]isKindOfClass:[NSDictionary class]]&&[capture[@"source"]count]>0,"capture kind/source");
    manifest[@"capture_json_sha256"]=sha(capture_data.bytes,capture_data.length);
    manifest[@"capture_json_path"]=capture_path;manifest[@"capture_source"]=capture[@"source"];
    raw(@"capture.json",capture_data.bytes,capture_data.length);
    struct stat st;need(!stat(model.fileSystemRepresentation,&st)&&st.st_size>0,"model stat");
    gguf_file_t *gf=gguf_open(model.fileSystemRepresentation);need(gf!=NULL,"GGUF open");
    need(json_uint(capture[@"model_bytes"],UINT64_MAX,"model byte count")==(uint64_t)st.st_size,"captured model size differs");
    size_t header=(size_t)json_uint(capture[@"model_header_bytes"],(uint64_t)st.st_size,"model header byte count");
    need(header>0&&[capture[@"model_header_sha256"]isKindOfClass:[NSString class]]&&
        [sha(gf->mmap_addr,header)isEqual:capture[@"model_header_sha256"]],"captured model header SHA differs");
    manifest[@"model"]=@{@"path":model,@"bytes":@(st.st_size),@"mapped_bytes":@(gf->mmap_size),
        @"header_bytes":@(header),@"header_sha256":sha(gf->mmap_addr,header),@"tensor_data_offset":@(gf->tensor_data_offset),
        @"mtime_ns":@((uint64_t)st.st_mtimespec.tv_sec*1000000000ull+st.st_mtimespec.tv_nsec)};
    NSArray *cases=capture[@"cases"];need([cases isKindOfClass:[NSArray class]]&&cases.count&&cases.count<=64,"1..64 captured cases");
    NSMutableArray *validated=[NSMutableArray new],*tensor_identity=[NSMutableArray new];
    NSMutableDictionary *layers=[NSMutableDictionary new];NSString *base=[capture_path stringByDeletingLastPathComponent];
    for(unsigned i=0;i<cases.count;i++) {
        NSDictionary *c=cases[i];need([c isKindOfClass:[NSDictionary class]],"case object");
        unsigned layer=(unsigned)json_uint(c[@"layer"],39,"layer 0..39");
        (void)json_uint(c[@"position"],UINT32_MAX,"position");
        NSData *x=payload(c,@"input_f32",base,INPUT_BYTES);need(finite_f32(x.bytes,x.length),"finite captured attention norm input");
        [validated addObject:@{@"metadata":c,@"input":x,@"index":@(i)}];
        NSString *key=[NSString stringWithFormat:@"%u",layer];if(!layers[key])layers[key]=[NSMutableArray new];
        [layers[key]addObject:@(i)];raw([NSString stringWithFormat:@"capture-case-%03u-input.f32",i],x.bytes,x.length);
    }
    NSArray *layer_keys=[[layers allKeys]sortedArrayUsingComparator:^NSComparisonResult(NSString *a,NSString *b){return[@([a intValue])compare:@([b intValue])];}];
    for(NSString *key in layer_keys)for(unsigned kind=0;kind<2;kind++) {
        NSString *name=[NSString stringWithFormat:@"blk.%@.%@.weight",key,kind?@"attn_kv":@"attn_q_a"];
        const gguf_tensor_t *t=gguf_find_tensor(gf,name.UTF8String);unsigned rows=kind?KV:QA;size_t bytes=kind?KV_WEIGHT_BYTES:QA_WEIGHT_BYTES;
        need(t&&t->type==GGML_TYPE_Q4_K&&t->n_dims==2&&t->dims[0]==DIM&&t->dims[1]==rows&&t->size_bytes==bytes,"exact real Qa/KV Q4 tensor descriptor");
        need(gf->tensor_data_offset<=(uint64_t)st.st_size&&t->offset<=(uint64_t)st.st_size-gf->tensor_data_offset&&
            t->size_bytes<=(uint64_t)st.st_size-gf->tensor_data_offset-t->offset,"Qa/KV tensor inside actual file bytes");
        [tensor_identity addObject:@{@"name":name,@"type":@(t->type),@"dimensions":@[@(DIM),@(rows)],@"bytes":@(bytes),
            @"file_offset":@(gf->tensor_data_offset+t->offset),@"sha256":sha(t->data,bytes)}];
    }
    manifest[@"tensors"]=tensor_identity;manifest[@"total_cases"]=@(cases.count);
    NSMutableArray *args=[NSMutableArray new];for(int i=0;i<argc;i++)[args addObject:[NSString stringWithUTF8String:argv[i]]];
    manifest[@"argv"]=args;emit(@{@"kind":@"identity",@"manifest":manifest});
    if(host){finish(@"host_only");gguf_close(gf);emit(@{@"kind":@"complete",@"host_only":@YES,@"metal_device_created":@NO});return 0;}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");
    NSError *error=nil;id<MTLLibrary> probe=[device newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];
    if(!probe)fprintf(stderr,"probe library: %s\n",error.localizedDescription.UTF8String);need(probe!=nil,"probe library");
    id<MTLLibrary> golden=[device newLibraryWithURL:[NSURL fileURLWithPath:gold_path]error:&error];need(golden!=nil,"production library");
    NSArray *names=@[@"kernel_mul_mv_q4_K_dense_f32_bf16",@"kernel_mul_mv_q4_K_dense_f32_bf16",@"kernel_myds_qkv_dual_q4_k_bf16_nr2"];
    id<MTLComputePipelineState> psos[3];NSMutableArray *bindings=[NSMutableArray new];
    for(unsigned p=0;p<3;p++) {
        psos[p]=pipeline(device,p?probe:golden,names[p]);
        [bindings addObject:@{@"provider":@(p),@"symbol":names[p],@"library":p?@"probe":@"production",
            @"fc600":@(NSG),@"nr0":@(NR),@"groups":p==2?@[@224]:@[@160,@64],@"threads":@[@32,@4,@1],
            @"args_bytes":@112,@"dynamic_threadgroup_bytes":@(TG_BYTES),@"static_threadgroup_bytes":@(psos[p].staticThreadgroupMemoryLength),
            @"max_threads":@(psos[p].maxTotalThreadsPerThreadgroup),@"thread_execution_width":@(psos[p].threadExecutionWidth)}];
    }
    manifest[@"device"]=device.name;manifest[@"bindings"]=bindings;
    emit(@{@"kind":@"bindings",@"device":device.name,@"bindings":bindings});
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"command queue");
    for(NSString *key in layer_keys){@autoreleasepool {
        NSArray *indices=layers[key];unsigned count=(unsigned)indices.count;
        fixture *f=calloc(count,sizeof(fixture));need(f!=NULL,"fixture allocation");
        id<MTLBuffer> qw=alloc(device,QA_WEIGHT_BYTES,YES),kw=alloc(device,KV_WEIGHT_BYTES,YES);
        const gguf_tensor_t *qt=gguf_find_tensor(gf,[[NSString stringWithFormat:@"blk.%@.attn_q_a.weight",key]UTF8String]),
            *kt=gguf_find_tensor(gf,[[NSString stringWithFormat:@"blk.%@.attn_kv.weight",key]UTF8String]);
        memcpy(body(qw),qt->data,QA_WEIGHT_BYTES);memcpy(body(kw),kt->data,KV_WEIGHT_BYTES);
        raw([NSString stringWithFormat:@"weight-layer-%@-qa.q4k",key],body(qw),QA_WEIGHT_BYTES);
        raw([NSString stringWithFormat:@"weight-layer-%@-kv.q4k",key],body(kw),KV_WEIGHT_BYTES);
        for(unsigned i=0;i<count;i++) {
            unsigned idx=[indices[i]unsignedIntValue];NSDictionary *v=validated[idx],*c=v[@"metadata"];
            f[i].index=idx;f[i].layer=key.intValue;f[i].position=[c[@"position"]unsignedLongLongValue];
            f[i].input=alloc(device,INPUT_BYTES,YES);NSData *x=v[@"input"];memcpy(body(f[i].input),x.bytes,x.length);
            for(unsigned p=0;p<3;p++){f[i].qa[p]=alloc(device,QA_BYTES,YES);f[i].kv[p]=alloc(device,KV_BYTES,YES);}
        }
        for(unsigned mode_index=0;mode_index<2;mode_index++) {
            BOOL concurrent=mode_index!=0;NSString *mode=concurrent?@"concurrent":@"serial";
            for(unsigned p=0;p<3;p++) {
                /* Reset before each mode so a skipped write cannot be hidden by
                 * a previous mode's valid data. No reset is inside timed CBs. */
                for(unsigned i=0;i<count;i++){memset(f[i].qa[p].contents,0xa5,f[i].qa[p].length);memset(f[i].kv[p].contents,0xa5,f[i].kv[p].length);}
                (void)run(queue,f,count,p,1,concurrent,qw,kw,psos[p]);validate(f,count,p,mode,YES);
            }
            for(unsigned control=0;control<2;control++) {
                unsigned providers[3]={control,2,control};double times[3][64];
                for(unsigned sample=0;sample<=samples;sample++)for(unsigned arm=0;arm<3;arm++) {
                    unsigned p=providers[arm];double begin=wall();
                    id<MTLCommandBuffer> cb=run(queue,f,count,p,reps,concurrent,qw,kw,psos[p]);double end=wall();
                    validate(f,count,p,mode,NO);need(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime&&cb.GPUStartTime>0,"valid GPU timestamps");
                    double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/(count*reps);if(sample)times[arm][sample-1]=us;
                    emit(@{@"kind":@"timing",@"mode":mode,@"layer":@(key.intValue),@"control_kind":control?@"probe":@"production",
                        @"provider":@(p),@"symbol":names[p],@"arm":@[@"A1",@"B",@"A2"][arm],@"sample":@(sample),
                        @"warmup":@(sample==0),@"pairs":@(count*reps),@"dispatches":@(count*reps*(p==2?1:2)),@"commits":@1,
                        @"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"gpu_us_per_pair":@(us),
                        @"wall_us_per_pair":@((end-begin)*1e6/(count*reps)),@"full_f32_bit_exact":@YES,@"component_only":@YES,@"completed_cb":@YES});
                }
                double a1=median(times[0],samples),b=median(times[1],samples),a2=median(times[2],samples),a=(a1+a2)/2;
                emit(@{@"kind":@"summary",@"mode":mode,@"layer":@(key.intValue),@"control_kind":control?@"probe":@"production",
                    @"control_symbol":names[control],@"candidate_symbol":names[2],@"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),
                    @"control_drift_percent":@((a2/a1-1)*100),@"time_saved_percent":@((1-b/a)*100),
                    @"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),@"fullmodel_speedup_proven":@NO,@"cold_dram_proven":@NO});
            }
        }
        need(guards(qw,QA_WEIGHT_BYTES)&&guards(kw,KV_WEIGHT_BYTES)&&!memcmp(body(qw),qt->data,QA_WEIGHT_BYTES)&&
            !memcmp(body(kw),kt->data,KV_WEIGHT_BYTES),"weight guards and complete bytes unchanged");
        for(unsigned i=0;i<count;i++) {
            NSData *x=validated[f[i].index][@"input"];need(guards(f[i].input,INPUT_BYTES)&&!memcmp(body(f[i].input),x.bytes,x.length),"captured input bytes unchanged");
            f[i].input=nil;for(unsigned p=0;p<3;p++){f[i].qa[p]=nil;f[i].kv[p]=nil;}
        }
        free(f);
    }}
    need([file_sha(probe_path)isEqual:manifest[@"probe_metallib_sha256"]]&&[file_sha(gold_path)isEqual:manifest[@"production_metallib_sha256"]]&&
        [file_sha(capture_path)isEqual:manifest[@"capture_json_sha256"]],"input library/capture changed during benchmark");
    finish(@"complete");gguf_close(gf);emit(@{@"kind":@"complete",@"full_f32_bit_exact":@YES,@"qa_kv_checked":@YES,
        @"cases":@(cases.count),@"modes":@[@"serial",@"concurrent"],@"fullmodel":@NO,@"cold_dram_proven":@NO});
}return 0;}
