/* Real captured IQ2 gate/up component validation; never initializes myds/server.
 * make bin/bench_iq2_signed4
 * --model GGUF --capture-json JSON --probe-metallib PATH
 * --production-metallib PATH --out-dir NEW [--nsg 2|4 --samples 9 --reps 8]
 * --host-only validates captures, descriptors and selected expert hashes without Metal.
 * Capture schema 1: capture_kind=myds_decode_routed_moe, model_bytes,
 * model_header_bytes, model_header_sha256, source(object), cases(array).
 * Each case: layer, position, clamp, input_f32, ids_i32, route_weights_f32.
 * Each payload: {path,sha256}; relative paths resolve beside capture JSON.
 * A full-model speedup, cold DRAM, and deployment adoption are separate gates.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "myds_gguf.h"
#include <mach-o/dyld.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum { DIM=5120, WIDTH=2304, EXPERTS=384, ACTIVE=6, GUARD=64,
       EXPERT_BYTES=3041280, TENSOR_BYTES=1167851520, MID_BYTES=55296 };
struct mv_args {
    int32_t nei0,nei1; uint64_t nbi1; int32_t ne00,ne01,ne02;
    uint64_t nb00,nb01,nb02; int32_t ne10,ne11,ne12,ne13;
    uint64_t nb10,nb11,nb12; int32_t ne0,ne1; uint64_t nb1;
    int32_t nr0,tp_rank,tp_world,tp_addend,tp_expert_base;
};
struct act_args {
    uint32_t width,rows; uint64_t gate_row_stride,up_row_stride,mid_row_stride,weight_stride;
    uint32_t write_clamped; float clamp_value;
};
_Static_assert(sizeof(struct mv_args)==136,"IQ2 ABI");
_Static_assert(sizeof(struct act_args)==48,"SwiGLU ABI");
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
static BOOL poison(id<MTLBuffer> b){const uint8_t *p=b.contents;for(size_t i=0;i<b.length;i++)if(p[i]!=0xa5)return NO;return YES;}
static void dump(NSString *n,id<MTLBuffer> b,size_t bytes) {
    raw([n stringByAppendingString:@".f32"],body(b),bytes);
    uint8_t g[2*GUARD];memcpy(g,b.contents,GUARD);memcpy(g+GUARD,(uint8_t*)b.contents+GUARD+bytes,GUARD);
    raw([n stringByAppendingString:@".guards"],g,sizeof(g));
}
typedef struct {
    unsigned layer,index;uint64_t position;float clamp;
    id<MTLBuffer> input,ids,weights;
    id<MTLBuffer> gate[5],up[5],mid[5];
} fixture;
static struct mv_args fixed_args(void) {
    return(struct mv_args){.nei0=6,.nei1=1,.nbi1=24,.ne00=5120,.ne01=2304,.ne02=384,
        .nb00=66,.nb01=1320,.nb02=3041280,.ne10=5120,.ne11=1,.ne12=1,.ne13=1,
        .nb10=4,.nb11=20480,.nb12=20480,.ne0=2304,.ne1=6,.nb1=9216,.nr0=4};
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,
        NSString *name,short nsg,unsigned smem) {
    NSError *error=nil;MTLFunctionConstantValues *c=[MTLFunctionConstantValues new];
    [c setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> f=[l newFunctionWithName:name constantValues:c error:&error];
    if(!f)fprintf(stderr,"function %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(f!=nil,"explicit function absent; fallback forbidden");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];
    if(!p)fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=32u*(unsigned)nsg,"pipeline SIMD/threads");
    need(p.staticThreadgroupMemoryLength+smem<=d.maxThreadgroupMemoryLength,"pipeline scratch limit");return p;
}
static id<MTLCommandBuffer> run(id<MTLCommandQueue> q,fixture *f,unsigned count,unsigned provider,
        unsigned reps,short nsg,id<MTLBuffer> gate,id<MTLBuffer> up,
        id<MTLComputePipelineState> p,unsigned smem) {
    id<MTLCommandBuffer> cb=[q commandBuffer];need(cb!=nil,"command buffer");
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];need(e!=nil,"compute encoder");
    struct mv_args a=fixed_args();
    for(unsigned rep=0;rep<reps;rep++)for(unsigned i=0;i<count;i++) {
        struct act_args act={.width=2304,.rows=6,.gate_row_stride=9216,.up_row_stride=9216,
            .mid_row_stride=9216,.weight_stride=4,.write_clamped=provider>=3,.clamp_value=f[i].clamp};
        [e setComputePipelineState:p];[e setBytes:&a length:sizeof(a) atIndex:0];
        [e setBytes:&act length:sizeof(act) atIndex:1];
        [e setBuffer:gate offset:GUARD atIndex:2];[e setBuffer:up offset:GUARD atIndex:3];
        [e setBuffer:f[i].input offset:GUARD atIndex:4];
        [e setBuffer:f[i].gate[provider] offset:GUARD atIndex:5];
        [e setBuffer:f[i].up[provider] offset:GUARD atIndex:6];
        [e setBuffer:f[i].mid[provider] offset:GUARD atIndex:7];
        [e setBuffer:f[i].ids offset:GUARD atIndex:8];[e setBuffer:f[i].weights offset:GUARD atIndex:9];
        [e setThreadgroupMemoryLength:smem atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(WIDTH/(4*nsg),1,ACTIVE)
            threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
        id<MTLResource> r=f[i].mid[provider];[e memoryBarrierWithResources:&r count:1];
    }
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)
        fprintf(stderr,"GPU failure status=%lu: %s\n",(unsigned long)cb.status,cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"command buffer completion");return cb;
}
static void validate(fixture *f,unsigned count,unsigned provider,BOOL save_outputs) {
    for(unsigned i=0;i<count;i++) {
        id<MTLBuffer> mid=f[i].mid[provider];size_t first=SIZE_MAX;
        size_t bad=mismatches(body(f[i].mid[0]),body(mid),MID_BYTES,&first);
        BOOL ok=!bad&&guards(mid,MID_BYTES)&&finite_f32(body(mid),MID_BYTES)&&written(body(mid),MID_BYTES);
        size_t gate_bad=0,up_bad=0;
        if(provider>=3) {
            gate_bad=mismatches(body(f[i].gate[3]),body(f[i].gate[provider]),MID_BYTES,&first);
            up_bad=mismatches(body(f[i].up[3]),body(f[i].up[provider]),MID_BYTES,&first);
            ok&=!gate_bad&&!up_bad&&guards(f[i].gate[provider],MID_BYTES)&&guards(f[i].up[provider],MID_BYTES)&&
                finite_f32(body(f[i].gate[provider]),MID_BYTES)&&finite_f32(body(f[i].up[provider]),MID_BYTES)&&
                written(body(f[i].gate[provider]),MID_BYTES)&&written(body(f[i].up[provider]),MID_BYTES);
        } else ok&=poison(f[i].gate[provider])&&poison(f[i].up[provider]);
        ok&=guards(f[i].input,DIM*4)&&guards(f[i].ids,24)&&guards(f[i].weights,24);
        if(save_outputs) {
            NSString *prefix=[NSString stringWithFormat:@"parity-case-%03u-provider-%u",f[i].index,provider];
            dump([prefix stringByAppendingString:@"-mid"],mid,MID_BYTES);
            if(provider>=3){dump([prefix stringByAppendingString:@"-gate"],f[i].gate[provider],MID_BYTES);
                dump([prefix stringByAppendingString:@"-up"],f[i].up[provider],MID_BYTES);}
            emit(@{@"kind":@"parity",@"case":@(f[i].index),@"layer":@(f[i].layer),
                @"position":@(f[i].position),@"provider":@(provider),@"mid_mismatches":@(bad),
                @"gate_mismatches":provider>=3?(id)@(gate_bad):[NSNull null],
                @"up_mismatches":provider>=3?(id)@(up_bad):[NSNull null],
                @"gate_up_capture":@(provider>=3),@"full_f32_bit_exact":@(ok),
                @"finite_guards_written":@(ok),@"completed_cb":@YES});
        }
        need(ok,"output bits/finite/guards/writes/release poison");
    }
}
int main(int argc,char **argv){@autoreleasepool {
    NSString *model=nil,*capture_path=nil,*probe_path=nil,*gold_path=nil;unsigned samples=9,reps=8;
    short nsg=2;BOOL host=NO;
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
        else if(!strcmp(k,"--nsg")){nsg=(short)number(v,2,4);need(nsg==2||nsg==4,"NSG 2 or 4");}
        else need(NO,"unknown option");
    }
    need(model&&capture_path&&probe_path&&gold_path&&output,"required model/capture/libraries/new output directory");
    need(model.isAbsolutePath&&capture_path.isAbsolutePath&&probe_path.isAbsolutePath&&gold_path.isAbsolutePath&&output.isAbsolutePath,"absolute paths required");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"output must not exist");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output directory");
    files=[NSMutableArray new];manifest=[NSMutableDictionary new];
    manifest[@"schema_version"]=@1;manifest[@"probe"]=@"iq2_signed4_real_capture";
    manifest[@"fullmodel"]=@NO;manifest[@"cold_dram_proven"]=@NO;
    manifest[@"repeated_fixture_cache_reuse"]=@YES;manifest[@"capture_provenance_externally_required"]=@YES;
    manifest[@"deployment_identity_externally_required"]=@YES;manifest[@"nsg"]=@(nsg);
    manifest[@"samples"]=@(samples);manifest[@"repetitions_per_cb"]=@(reps);
    manifest[@"binary_path"]=binary_path();manifest[@"binary_sha256"]=file_sha(manifest[@"binary_path"]);
    manifest[@"probe_metallib_path"]=probe_path;manifest[@"probe_metallib_sha256"]=file_sha(probe_path);
    manifest[@"production_metallib_path"]=gold_path;manifest[@"production_metallib_sha256"]=file_sha(gold_path);
    NSData *capture_data=read_file(capture_path);
    NSDictionary *capture=[NSJSONSerialization JSONObjectWithData:capture_data options:0 error:nil];
    need([capture isKindOfClass:[NSDictionary class]],"capture JSON object");
    need(json_uint(capture[@"schema_version"],1,"capture schema")==1&&
        [capture[@"capture_kind"]isEqual:@"myds_decode_routed_moe"]&&
        [capture[@"source"]isKindOfClass:[NSDictionary class]]&&[capture[@"source"]count]>0,"capture kind/source");
    manifest[@"capture_json_sha256"]=sha(capture_data.bytes,capture_data.length);
    manifest[@"capture_json_path"]=capture_path;manifest[@"capture_source"]=capture[@"source"];
    raw(@"capture.json",capture_data.bytes,capture_data.length);
    struct stat st;need(!stat(model.fileSystemRepresentation,&st),"model stat");
    gguf_file_t *gf=gguf_open(model.fileSystemRepresentation);need(gf!=NULL,"GGUF open");
    need(st.st_size>0&&json_uint(capture[@"model_bytes"],UINT64_MAX,"model byte count")==
        (uint64_t)st.st_size,"captured model size differs");
    size_t header=(size_t)json_uint(capture[@"model_header_bytes"],(uint64_t)st.st_size,"model header byte count");
    need(header>0&&[capture[@"model_header_sha256"]isKindOfClass:[NSString class]]&&
        [sha(gf->mmap_addr,header)isEqual:capture[@"model_header_sha256"]],"captured model header SHA differs");
    manifest[@"model"]=@{@"path":model,@"bytes":@(st.st_size),@"mapped_bytes":@(gf->mmap_size),@"header_bytes":@(header),
        @"header_sha256":sha(gf->mmap_addr,header),@"tensor_data_offset":@(gf->tensor_data_offset),
        @"mtime_ns":@((uint64_t)st.st_mtimespec.tv_sec*1000000000ull+st.st_mtimespec.tv_nsec)};
    const char *arch=gguf_get_str(gf,"general.architecture",NULL);need(arch&&*arch,"model architecture metadata");
    char clamp_key[256];int clamp_len=snprintf(clamp_key,sizeof(clamp_key),"%s.swiglu_limit",arch);
    need(clamp_len>0&&(size_t)clamp_len<sizeof(clamp_key),"clamp metadata key");
    float model_clamp=gguf_get_f32(gf,clamp_key,10.f);
    need(isfinite(model_clamp)&&model_clamp>=0,"model finite nonnegative clamp");
    manifest[@"model_clamp"]=@(model_clamp);manifest[@"model_clamp_key"]=[NSString stringWithUTF8String:clamp_key];
    manifest[@"model_clamp_uses_engine_default"]=@(gguf_find_key(gf,clamp_key)==NULL);
    NSArray *cases=capture[@"cases"];need([cases isKindOfClass:[NSArray class]]&&cases.count&&cases.count<=64,"1..64 captured cases");
    NSMutableArray *validated=[NSMutableArray new],*tensor_identity=[NSMutableArray new];
    NSMutableDictionary *layers=[NSMutableDictionary new];NSString *base=[capture_path stringByDeletingLastPathComponent];
    for(unsigned i=0;i<cases.count;i++) {
        NSDictionary *c=cases[i];need([c isKindOfClass:[NSDictionary class]],"case object");
        unsigned layer=(unsigned)json_uint(c[@"layer"],39,"layer 0..39");
        (void)json_uint(c[@"position"],UINT32_MAX,"position");
        need([c[@"clamp"]isKindOfClass:[NSNumber class]]&&isfinite([c[@"clamp"]doubleValue])&&
            [c[@"clamp"]doubleValue]>=0&&[c[@"clamp"]doubleValue]<=FLT_MAX,"finite nonnegative clamp");
        float capture_clamp=[c[@"clamp"]floatValue];
        need(!memcmp(&capture_clamp,&model_clamp,sizeof(float)),"capture clamp differs from model/runtime metadata");
        NSData *x=payload(c,@"input_f32",base,DIM*4),*ids=payload(c,@"ids_i32",base,24),
            *weights=payload(c,@"route_weights_f32",base,24);
        need(finite_f32(x.bytes,x.length)&&finite_f32(weights.bytes,weights.length),"finite captured input/route weights");
        const int32_t *ip=ids.bytes;for(unsigned j=0;j<6;j++) {
            need(ip[j]>=0&&ip[j]<384,"captured expert ID range");
            for(unsigned k=0;k<j;k++)need(ip[j]!=ip[k],"captured Top6 IDs unique");
        }
        [validated addObject:@{@"metadata":c,@"input":x,@"ids":ids,@"weights":weights,@"index":@(i)}];
        NSString *key=[NSString stringWithFormat:@"%u",layer];if(!layers[key])layers[key]=[NSMutableArray new];
        [layers[key]addObject:@(i)];
        raw([NSString stringWithFormat:@"capture-case-%03u-input.f32",i],x.bytes,x.length);
        raw([NSString stringWithFormat:@"capture-case-%03u-ids.i32",i],ids.bytes,ids.length);
        raw([NSString stringWithFormat:@"capture-case-%03u-weights.f32",i],weights.bytes,weights.length);
    }
    NSArray *layer_keys=[[layers allKeys]sortedArrayUsingComparator:^NSComparisonResult(NSString *a,NSString *b){return [@([a intValue])compare:@([b intValue])];}];
    /* Only metadata and selected experts are read; no 362 GB GPU resource is created. */
    for(NSString *key in layer_keys)for(NSString *part in @[@"ffn_gate_exps.weight",@"ffn_up_exps.weight"]) {
        NSString *name=[NSString stringWithFormat:@"blk.%@.%@",key,part];
        const gguf_tensor_t *t=gguf_find_tensor(gf,name.UTF8String);
        need(t&&t->type==GGML_TYPE_IQ2_XXS&&t->n_dims==3&&t->dims[0]==DIM&&t->dims[1]==WIDTH&&
            t->dims[2]==EXPERTS&&t->size_bytes==TENSOR_BYTES,"exact real IQ2 expert tensor descriptor");
        need(gf->tensor_data_offset<=(uint64_t)st.st_size&&t->offset<=(uint64_t)st.st_size-gf->tensor_data_offset&&
            t->size_bytes<=(uint64_t)st.st_size-gf->tensor_data_offset-t->offset,"expert tensor inside actual file bytes");
        NSMutableSet *selected=[NSMutableSet new];for(NSNumber *idx in layers[key]) {
            NSData *ids=validated[idx.unsignedIntValue][@"ids"];const int32_t *p=ids.bytes;
            for(unsigned j=0;j<6;j++)[selected addObject:@(p[j])];
        }
        NSMutableArray *es=[NSMutableArray new];for(NSNumber *eid in [[selected allObjects]sortedArrayUsingSelector:@selector(compare:)]) {
            const uint8_t *p=(const uint8_t*)t->data+eid.unsignedIntValue*(size_t)EXPERT_BYTES;
            [es addObject:@{@"id":eid,@"sha256":sha(p,EXPERT_BYTES),@"bytes":@(EXPERT_BYTES)}];
        }
        [tensor_identity addObject:@{@"name":name,@"type":@(t->type),@"dimensions":@[@(DIM),@(WIDTH),@(EXPERTS)],
            @"bytes":@(t->size_bytes),@"file_offset":@(gf->tensor_data_offset+t->offset),@"selected_experts":es}];
    }
    manifest[@"tensors"]=tensor_identity;manifest[@"total_cases"]=@(cases.count);
    NSMutableArray *args=[NSMutableArray new];for(int i=0;i<argc;i++)[args addObject:[NSString stringWithUTF8String:argv[i]]];
    manifest[@"argv"]=args;emit(@{@"kind":@"identity",@"manifest":manifest});
    if(host){finish(@"host_only");gguf_close(gf);emit(@{@"kind":@"complete",@"host_only":@YES,@"metal_device_created":@NO});return 0;}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");
    NSError *error=nil;id<MTLLibrary> probe=[device newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];
    if(!probe)fprintf(stderr,"probe library: %s\n",error.localizedDescription.UTF8String);
    need(probe!=nil,"probe library");
    id<MTLLibrary> golden=[device newLibraryWithURL:[NSURL fileURLWithPath:gold_path]error:&error];
    need(golden!=nil,"production library");
    NSArray *names=@[@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32",@"kernel_myds_iq2_v41_shape64_f32",
        @"kernel_myds_iq2_v41_signed4_f32",@"probe_iq2_v41_shape64_capture_f32",@"probe_iq2_v41_signed4_capture_f32"];
    const unsigned smem[5]={2176,2176,5248,2176,5248};
    id<MTLComputePipelineState> psos[5];NSMutableArray *bindings=[NSMutableArray new];
    for(unsigned p=0;p<5;p++) {
        psos[p]=pipeline(device,p?probe:golden,names[p],nsg,smem[p]);
        [bindings addObject:@{@"provider":@(p),@"symbol":names[p],@"library":p?@"probe":@"production",
            @"fc600":@(nsg),@"nr0":@4,@"groups":@[@(WIDTH/(4*nsg)),@1,@6],
            @"threads":@[@32,@(nsg),@1],@"dynamic_threadgroup_bytes":@(smem[p]),
            @"static_threadgroup_bytes":@(psos[p].staticThreadgroupMemoryLength),
            @"max_threads":@(psos[p].maxTotalThreadsPerThreadgroup),@"thread_execution_width":@(psos[p].threadExecutionWidth)}];
    }
    manifest[@"device"]=device.name;manifest[@"bindings"]=bindings;
    emit(@{@"kind":@"bindings",@"device":device.name,@"bindings":bindings});
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"command queue");
    for(NSString *key in layer_keys){@autoreleasepool {
        NSArray *indices=layers[key];unsigned count=(unsigned)indices.count;
        fixture *f=calloc(count,sizeof(fixture));need(f!=NULL,"fixture allocation");
        id<MTLBuffer> gate=alloc(device,TENSOR_BYTES,NO),up=alloc(device,TENSOR_BYTES,NO);
        NSMutableSet *selected=[NSMutableSet new];
        for(unsigned i=0;i<count;i++) {
            unsigned idx=[indices[i]unsignedIntValue];NSDictionary *v=validated[idx],*c=v[@"metadata"];
            f[i].index=idx;f[i].layer=key.intValue;f[i].position=[c[@"position"]unsignedLongLongValue];f[i].clamp=[c[@"clamp"]floatValue];
            f[i].input=alloc(device,DIM*4,YES);f[i].ids=alloc(device,24,YES);f[i].weights=alloc(device,24,YES);
            NSData *x=v[@"input"],*ids=v[@"ids"],*w=v[@"weights"];
            memcpy(body(f[i].input),x.bytes,x.length);memcpy(body(f[i].ids),ids.bytes,ids.length);memcpy(body(f[i].weights),w.bytes,w.length);
            const int32_t *ip=ids.bytes;for(unsigned j=0;j<6;j++)[selected addObject:@(ip[j])];
            for(unsigned p=0;p<5;p++){f[i].gate[p]=alloc(device,MID_BYTES,YES);f[i].up[p]=alloc(device,MID_BYTES,YES);f[i].mid[p]=alloc(device,MID_BYTES,YES);}
        }
        for(unsigned kind=0;kind<2;kind++) {
            NSString *name=[NSString stringWithFormat:@"blk.%@.%@",key,kind?@"ffn_up_exps.weight":@"ffn_gate_exps.weight"];
            const gguf_tensor_t *t=gguf_find_tensor(gf,name.UTF8String);id<MTLBuffer> b=kind?up:gate;
            for(NSNumber *eid in selected)memcpy((uint8_t*)body(b)+eid.unsignedIntValue*(size_t)EXPERT_BYTES,
                (const uint8_t*)t->data+eid.unsignedIntValue*(size_t)EXPERT_BYTES,EXPERT_BYTES);
        }
        for(unsigned p=0;p<5;p++){(void)run(queue,f,count,p,1,nsg,gate,up,psos[p],smem[p]);validate(f,count,p,YES);}
        /* Release PSOs only in timing. One serial CB per arm, never one commit per dispatch.
         * Both controls surround every candidate sample; formal order is A1,B,A2. */
        for(unsigned control=0;control<2;control++) {
            unsigned provider[3]={control,2,control};double times[3][64];
            for(unsigned sample=0;sample<=samples;sample++)for(unsigned arm=0;arm<3;arm++) {
                unsigned p=provider[arm];double begin=wall();
                id<MTLCommandBuffer> cb=run(queue,f,count,p,reps,nsg,gate,up,psos[p],smem[p]);double end=wall();
                validate(f,count,p,NO);
                need(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime&&cb.GPUStartTime>0,"valid GPU timestamps");
                double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/(count*reps);
                if(sample)times[arm][sample-1]=us;
                emit(@{@"kind":@"timing",@"layer":@(key.intValue),@"control_provider":@(control),
                    @"provider":@(p),@"symbol":names[p],@"arm":@[@"A1",@"B",@"A2"][arm],
                    @"sample":@(sample),@"warmup":@(sample==0),@"dispatches":@(count*reps),
                    @"commits":@1,@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),
                    @"gpu_us_per_dispatch":@(us),@"wall_us_per_dispatch":@((end-begin)*1e6/(count*reps)),
                    @"full_f32_bit_exact":@YES,@"component_only":@YES,@"completed_cb":@YES});
            }
            double a1=median(times[0],samples),b=median(times[1],samples),a2=median(times[2],samples),a=(a1+a2)/2;
            emit(@{@"kind":@"summary",@"layer":@(key.intValue),@"control_symbol":names[control],
                @"candidate_symbol":names[2],@"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),
                @"control_drift_percent":@((a2/a1-1)*100),@"time_saved_percent":@((1-b/a)*100),
                @"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),
                @"fullmodel_speedup_proven":@NO,@"cold_dram_proven":@NO});
        }
        need(guards(gate,TENSOR_BYTES)&&guards(up,TENSOR_BYTES),"weight allocation guards");
        for(unsigned kind=0;kind<2;kind++) {
            NSString *name=[NSString stringWithFormat:@"blk.%@.%@",key,kind?@"ffn_up_exps.weight":@"ffn_gate_exps.weight"];
            const gguf_tensor_t *t=gguf_find_tensor(gf,name.UTF8String);id<MTLBuffer> b=kind?up:gate;
            for(NSNumber *eid in selected)need(!memcmp((uint8_t*)body(b)+eid.unsignedIntValue*(size_t)EXPERT_BYTES,
                (const uint8_t*)t->data+eid.unsignedIntValue*(size_t)EXPERT_BYTES,EXPERT_BYTES),"selected real weight bytes changed");
        }
        for(unsigned i=0;i<count;i++) {
            NSDictionary *v=validated[f[i].index];NSData *x=v[@"input"],*ids=v[@"ids"],*w=v[@"weights"];
            need(!memcmp(body(f[i].input),x.bytes,x.length)&&!memcmp(body(f[i].ids),ids.bytes,ids.length)&&
                !memcmp(body(f[i].weights),w.bytes,w.length),"captured input bytes changed");
            f[i].input=nil;f[i].ids=nil;f[i].weights=nil;
            for(unsigned p=0;p<5;p++){f[i].gate[p]=nil;f[i].up[p]=nil;f[i].mid[p]=nil;}
        }
        free(f);
    }}
    need([file_sha(probe_path)isEqual:manifest[@"probe_metallib_sha256"]]&&
        [file_sha(gold_path)isEqual:manifest[@"production_metallib_sha256"]]&&
        [file_sha(capture_path)isEqual:manifest[@"capture_json_sha256"]],"input library/capture changed during benchmark");
    finish(@"complete");gguf_close(gf);
    emit(@{@"kind":@"complete",@"full_f32_bit_exact":@YES,@"gate_up_mid_checked":@YES,
        @"cases":@(cases.count),@"fullmodel":@NO,@"cold_dram_proven":@NO});
}return 0;}
