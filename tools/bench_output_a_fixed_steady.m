/* Real ordinary-decode Output_A fixed-wrapper component. No inference init.
 * Compile with the project Objective-C flags, gguf.o, Foundation and Metal.
 * --model GGUF --capture-json JSON --probe-metallib PATH
 * --production-metallib PATH --out-dir NEW
 *   --correctness-only
 * or --correctness-manifest ABS --correctness-manifest-sha256 HEX
 * Optional --host-only validates all CPU contracts without creating a device.
 * Separate steady measurement: 15 formal samples, 1 warmup, 256 repetitions
 * x 4 real inputs = 1024 projections per CB; original arithmetic/IO unchanged.
 * Correctness must finish all 12 cases x 3 providers before a separate timing
 * phase can submit any GPU work. This repeated fixture is not cold DRAM or a
 * full-model/adoption result. The ordinary concurrent encoder is preserved.
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

enum { GROUPS=8, WIDTH=4096, RANK=1024, OUT_FLOATS=8192, GUARD=64,
       WEIGHT_BYTES=18874368, INPUT_BYTES=131072, OUTPUT_BYTES=32768,
       NSG=4, NR=2, TG_BYTES=256 };
struct id_args {
    int32_t nei0,nei1; uint64_t nbi1; int32_t ne00,ne01,ne02;
    uint64_t nb00,nb01,nb02; int32_t ne10,ne11,ne12,ne13;
    uint64_t nb10,nb11,nb12; int32_t ne0,ne1; uint64_t nb1;
    int32_t nr0,tp_rank,tp_world,tp_addend,tp_expert_base;
};
_Static_assert(sizeof(struct id_args)==136,"Output_A original id ABI");
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
    need(json_uint(v[@"bytes"],SIZE_MAX,"capture payload descriptor bytes")==bytes,"payload descriptor exact bytes");
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
static NSString *phase;
static NSMutableArray *parity_records,*input_records;
static unsigned command_ordinal;
struct span { uintptr_t base;size_t bytes; };
static BOOL overlap(struct span a,struct span b) {
    if(!a.base||!b.base||!a.bytes||!b.bytes)return NO;
    return a.base<=b.base?b.base-a.base<a.bytes:a.base-b.base<b.bytes;
}
static BOOL bindings_safe(struct span out,struct span w,struct span x) {
    return out.bytes==OUTPUT_BYTES&&w.bytes==WEIGHT_BYTES&&x.bytes==INPUT_BYTES&&
        out.base&&w.base&&x.base&&!(out.base%16u)&&!(w.base%16u)&&!(x.base%16u)&&
        !overlap(out,w)&&!overlap(out,x);
}
static struct span span(id<MTLBuffer> b,size_t bytes){return(struct span){(uintptr_t)body(b),bytes};}
static unsigned alias_self_test(void) {
    struct span y={0x10000,OUTPUT_BYTES},w={0x100000,WEIGHT_BYTES},x={0x2000000,INPUT_BYTES};
    need(bindings_safe(y,w,x),"valid Output_A component bindings");unsigned bad=0;
    struct span v=y;v.base=w.base;bad+=!bindings_safe(v,w,x);
    v=y;v.base=x.base;bad+=!bindings_safe(v,w,x);
    v=y;v.base=x.base+16;bad+=!bindings_safe(v,w,x);
    v=y;v.base++;bad+=!bindings_safe(v,w,x);
    v=x;v.bytes-=4;bad+=!bindings_safe(y,w,v);
    v=w;v.bytes-=144;bad+=!bindings_safe(y,v,x);
    v=y;v.base=0;bad+=!bindings_safe(v,w,x);
    v=y;v.bytes-=4;bad+=!bindings_safe(v,w,x);
    need(bad==8,"Output_A alias/alignment/length rejection controls");return bad;
}
static struct id_args fixed_args(void) {
    return(struct id_args){.nei0=8,.nei1=1,.nbi1=32,.ne00=4096,.ne01=1024,.ne02=8,
        .nb00=144,.nb01=2304,.nb02=2359296,.ne10=4096,.ne11=8,.ne12=1,.ne13=1,
        .nb10=4,.nb11=16384,.nb12=131072,.ne0=1024,.ne1=8,.nb1=4096,.nr0=2};
}
static NSDictionary *binding_contract(void) {
    return @{@"args_bytes":@136,@"buffer_slots":@[@"args136",@"q4k_weights",@"heads_f32",@"low_f32"],
        @"groups":@[@128,@1,@8],@"threads":@[@32,@4,@1],@"fc600_short":@4,@"nr0":@2,
        @"dynamic_threadgroup_bytes":@256,@"input_bytes":@(INPUT_BYTES),
        @"weight_bytes":@(WEIGHT_BYTES),@"output_bytes":@(OUTPUT_BYTES),@"bf16_postround":@YES};
}
typedef struct { unsigned layer,index,position;id<MTLBuffer> input,out[3];
    NSData *oracle;NSString *input_sha; } fixture;
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,NSString *name) {
    short nsg=NSG;MTLFunctionConstantValues *c=[MTLFunctionConstantValues new];
    [c setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];NSError *error=nil;
    id<MTLFunction> f=[l newFunctionWithName:name constantValues:c error:&error];
    if(!f)fprintf(stderr,"function %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(f!=nil,"explicit Output_A symbol absent; fallback forbidden");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];
    if(!p)fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"Output_A pipeline geometry");
    need(p.staticThreadgroupMemoryLength+TG_BYTES<=d.maxThreadgroupMemoryLength,"Output_A TG limit");return p;
}
static id<MTLCommandBuffer> run(id<MTLCommandQueue> q,fixture *f,unsigned count,
        unsigned provider,unsigned reps,id<MTLBuffer> w,id<MTLComputePipelineState> p) {
    for(unsigned i=0;i<count;i++) {
        need(bindings_safe(span(f[i].out[provider],OUTPUT_BYTES),span(w,WEIGHT_BYTES),
            span(f[i].input,INPUT_BYTES)),"reject invalid/aliased OutA bindings before CB");
        for(unsigned j=0;j<count;j++) {
            need(!overlap(span(f[i].out[provider],OUTPUT_BYTES),span(f[j].input,INPUT_BYTES)),"reject cross-case reader alias");
            if(i!=j)need(!overlap(span(f[i].out[provider],OUTPUT_BYTES),span(f[j].out[provider],OUTPUT_BYTES)),"reject cross-case writer alias");
        }
        /* The previous CB is complete. Every new CB starts with poison, so
         * prior valid results cannot hide a missing store. CPU writes are
         * outside GPU timestamps and identical for all three providers. */
        memset(body(f[i].out[provider]),0xa5,OUTPUT_BYTES);
    }
    id<MTLCommandBuffer> cb=[q commandBuffer];need(cb!=nil,"Output_A CB");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];
    pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"Output_A concurrent encoder");
    struct id_args args=fixed_args();
    for(unsigned rep=0;rep<reps;rep++)for(unsigned i=0;i<count;i++) {
        [e setComputePipelineState:p];[e setBytes:&args length:sizeof(args) atIndex:0];
        [e setBuffer:w offset:GUARD atIndex:1];[e setBuffer:f[i].input offset:GUARD atIndex:2];
        [e setBuffer:f[i].out[provider] offset:GUARD atIndex:3];
        [e setThreadgroupMemoryLength:TG_BYTES atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(128,1,8) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        /* Same output join for all providers. Independent fixture buffers
         * remain independent; repeated writes to each output are ordered. */
        id<MTLResource> out=f[i].out[provider];[e memoryBarrierWithResources:&out count:1];
    }
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU status=%lu: %s\n",
        (unsigned long)cb.status,cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"Output_A CB completed");command_ordinal++;return cb;
}
static void validate(fixture *f,unsigned count,unsigned provider,id<MTLBuffer> w,
        NSString *weight_sha,BOOL save) {
    BOOL wg=guards(w,WEIGHT_BYTES);NSString *actual_w=sha(body(w),WEIGHT_BYTES);
    BOOL weights_ok=wg&&[actual_w isEqual:weight_sha];
    for(unsigned i=0;i<count;i++) {
        const void *oracle=f[i].oracle?f[i].oracle.bytes:body(f[i].out[0]);
        size_t first=SIZE_MAX,bad=mismatches(oracle,body(f[i].out[provider]),OUTPUT_BYTES,&first);
        BOOL fin=finite_f32(body(f[i].out[provider]),OUTPUT_BYTES),wr=written(body(f[i].out[provider]),OUTPUT_BYTES),
            rounded=bf16(body(f[i].out[provider]),OUTPUT_BYTES),og=guards(f[i].out[provider],OUTPUT_BYTES),
            ig=guards(f[i].input,INPUT_BYTES);
        NSString *actual_x=sha(body(f[i].input),INPUT_BYTES);
        BOOL input_ok=ig&&[actual_x isEqual:f[i].input_sha];
        BOOL ok=!bad&&fin&&wr&&rounded&&og&&input_ok&&weights_ok;
        [input_records addObject:@{@"command_ordinal":@(command_ordinal),@"case":@(f[i].index),@"provider":@(provider),
            @"input_sha256":actual_x,@"expected_input_sha256":f[i].input_sha,@"input_guards_intact":@(ig),
            @"weight_sha256":actual_w,@"expected_weight_sha256":weight_sha,@"weight_guards_intact":@(wg),
            @"all_gpu_inputs_unchanged":@(input_ok&&weights_ok)}];
        if(save||!ok) {
            NSString *prefix=save?[NSString stringWithFormat:@"parity-case-%03u-provider-%u-output",f[i].index,provider]:
                [NSString stringWithFormat:@"failure-case-%03u-provider-%u-cb-%u-output",f[i].index,provider,command_ordinal];
            dump(prefix,f[i].out[provider],OUTPUT_BYTES);
            NSDictionary *record=@{@"kind":save?@"parity":@"parity_failure",@"phase":phase,@"mode":@"concurrent",
                @"case":@(f[i].index),@"layer":@(f[i].layer),@"position":@(f[i].position),@"provider":@(provider),
                @"output_bytes":@(OUTPUT_BYTES),@"mismatches":@(bad),@"first_mismatch":first==SIZE_MAX?(id)[NSNull null]:@(first),
                @"full_f32_bit_exact":@(ok),@"finite":@(fin),@"written":@(wr),@"guards_intact":@(og),
                @"bf16_postround":@(rounded),@"finite_guards_written":@(fin&&wr&&og),
                @"all_gpu_inputs_unchanged":@(input_ok&&weights_ok),@"completed_cb":@YES,
                @"command_ordinal":@(command_ordinal),@"output_sha256":sha(body(f[i].out[provider]),OUTPUT_BYTES)};
            [parity_records addObject:record];emit(record);
        }
        manifest[@"parity"]=parity_records;manifest[@"input_integrity"]=input_records;
        need(ok,"Output_A full F32 bits/BF16/finite/poison/guards and input identity");
    }
}
static NSString *evidence_file(NSString *base,NSString *name) {
    need([name isKindOfClass:[NSString class]]&&name.length&&!name.isAbsolutePath,"relative correctness artifact path");
    NSString *p=[[base stringByAppendingPathComponent:name]stringByStandardizingPath];
    need([p hasPrefix:[base stringByAppendingString:@"/"]],"correctness artifact path escapes directory");
    struct stat st;need(!lstat(p.fileSystemRepresentation,&st)&&S_ISREG(st.st_mode),"regular correctness artifact required; symlink forbidden");
    return p;
}
static NSArray *correctness_oracles(NSString *path,NSString *expected) {
    NSData *bytes=read_file(path);need([sha(bytes.bytes,bytes.length)isEqual:expected],"correctness manifest SHA mismatch");
    NSDictionary *gate=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];
    need([gate isKindOfClass:[NSDictionary class]]&&[gate[@"status"]isEqual:@"complete"]&&
        [gate[@"phase"]isEqual:@"correctness"]&&[gate[@"probe"]isEqual:@"decode_output_a_fixed_real_capture"]&&
        [gate[@"full_f32_bit_exact"]isEqual:@YES]&&[gate[@"output_a_checked"]isEqual:@YES]&&
        [gate[@"timing_command_buffers"]isEqual:@0],"completed all-case correctness gate required");
    for(NSString *key in @[@"binary_sha256",@"capture_json_sha256",@"capture_source",@"probe_metallib_sha256",
        @"production_metallib_sha256",@"model",@"cases",@"tensors",@"total_cases",@"samples",@"repetitions_per_cb",@"binding_contract"])
        need([gate[key]isEqual:manifest[key]],"correctness identity differs from current replay");
    NSString *base=[path stringByDeletingLastPathComponent];NSArray *records=gate[@"files"];
    need([records isKindOfClass:[NSArray class]]&&records.count,"correctness artifact manifest");
    NSMutableDictionary *by_name=[NSMutableDictionary new];
    for(NSDictionary *r in records) {
        need([r isKindOfClass:[NSDictionary class]],"correctness file descriptor");NSString *name=r[@"path"];
        need(!by_name[name],"duplicate correctness artifact path");NSString *p=evidence_file(base,name);NSData *d=read_file(p);
        need(json_uint(r[@"bytes"],SIZE_MAX,"correctness bytes")==d.length&&[sha(d.bytes,d.length)isEqual:r[@"sha256"]],"correctness file bytes/SHA");by_name[name]=r;
    }
    NSArray *parity=gate[@"parity"],*inputs=gate[@"input_integrity"];
    unsigned count=[manifest[@"total_cases"]unsignedIntValue];
    need([parity isKindOfClass:[NSArray class]]&&parity.count==count*3&&
        [inputs isKindOfClass:[NSArray class]]&&inputs.count==count*3,"all correctness output/input records required");
    NSMutableSet *seen=[NSMutableSet new];
    for(NSDictionary *r in parity) {
        unsigned c=(unsigned)json_uint(r[@"case"],count-1,"correctness case"),p=(unsigned)json_uint(r[@"provider"],2,"correctness provider");
        NSString *key=[NSString stringWithFormat:@"%u:%u",c,p];need(![seen containsObject:key],"duplicate correctness case/provider");[seen addObject:key];
        NSDictionary *ci=manifest[@"cases"][c];
        need([r[@"kind"]isEqual:@"parity"]&&[r[@"phase"]isEqual:@"correctness"]&&[r[@"mode"]isEqual:@"concurrent"]&&
            [r[@"layer"]isEqual:ci[@"layer"]]&&[r[@"position"]isEqual:ci[@"position"]]&&[r[@"output_bytes"]isEqual:@(OUTPUT_BYTES)]&&
            [r[@"full_f32_bit_exact"]isEqual:@YES]&&[r[@"finite_guards_written"]isEqual:@YES]&&
            [r[@"bf16_postround"]isEqual:@YES]&&[r[@"all_gpu_inputs_unchanged"]isEqual:@YES]&&
            [r[@"completed_cb"]isEqual:@YES]&&json_uint(r[@"mismatches"],SIZE_MAX,"mismatch count")==0,"correctness parity did not pass");
    }
    [seen removeAllObjects];
    for(NSDictionary *r in inputs) {
        unsigned c=(unsigned)json_uint(r[@"case"],count-1,"input integrity case"),p=(unsigned)json_uint(r[@"provider"],2,"input integrity provider");
        NSString *key=[NSString stringWithFormat:@"%u:%u",c,p];need(![seen containsObject:key],"duplicate correctness input integrity record");[seen addObject:key];
        NSDictionary *ci=manifest[@"cases"][c],*tensor=nil;
        for(NSDictionary *t in manifest[@"tensors"])if([t[@"layer"]isEqual:ci[@"layer"]])tensor=t;
        need(tensor&&[r[@"all_gpu_inputs_unchanged"]isEqual:@YES]&&
            [r[@"input_sha256"]isEqual:ci[@"input_sha256"]]&&[r[@"expected_input_sha256"]isEqual:ci[@"input_sha256"]]&&
            [r[@"weight_sha256"]isEqual:tensor[@"sha256"]]&&[r[@"expected_weight_sha256"]isEqual:tensor[@"sha256"]]&&
            [r[@"input_guards_intact"]isEqual:@YES]&&[r[@"weight_guards_intact"]isEqual:@YES],"correctness input integrity did not pass");
    }
    NSMutableArray *oracles=[NSMutableArray new];uint8_t guard_bytes[2*GUARD];memset(guard_bytes,0xa5,sizeof(guard_bytes));
    for(unsigned c=0;c<count;c++) {
        NSData *native=nil;
        for(unsigned p=0;p<3;p++) {
            NSString *name=[NSString stringWithFormat:@"parity-case-%03u-provider-%u-output.f32",c,p];
            need(by_name[name]!=nil,"complete correctness output missing");NSString *file=evidence_file(base,name);NSData *d=read_file(file);
            need(d.length==OUTPUT_BYTES&&finite_f32(d.bytes,d.length)&&written(d.bytes,d.length)&&bf16(d.bytes,d.length),"full correctness output finite/BF16/written");
            if(!p){native=d;[oracles addObject:@{@"path":file,@"sha256":by_name[name][@"sha256"]}];}
            else need(!memcmp(native.bytes,d.bytes,d.length),"complete correctness outputs differ");
            NSString *gn=[NSString stringWithFormat:@"parity-case-%03u-provider-%u-output.guards",c,p];
            need(by_name[gn]!=nil,"correctness guard file missing");NSData *g=read_file(evidence_file(base,gn));
            need(g.length==sizeof(guard_bytes)&&!memcmp(g.bytes,guard_bytes,sizeof(guard_bytes)),"correctness output guards damaged");
        }
        NSString *xn=[NSString stringWithFormat:@"capture-case-%03u-input.f32",c];
        need([by_name[xn][@"sha256"]isEqual:manifest[@"cases"][c][@"input_sha256"]]&&
            [by_name[xn][@"bytes"]isEqual:@(INPUT_BYTES)],"correctness captured input artifact identity");
        NSString *gn=[NSString stringWithFormat:@"input-case-%03u.guards",c];
        need(by_name[gn]!=nil,"correctness input guard artifact required");NSData *g=read_file(evidence_file(base,gn));
        need(g.length==sizeof(guard_bytes)&&!memcmp(g.bytes,guard_bytes,sizeof(guard_bytes)),"correctness input guards damaged");
    }
    for(NSDictionary *t in manifest[@"tensors"]) {
        NSString *n=[NSString stringWithFormat:@"weight-layer-%@-output-a.q4k",t[@"layer"]];
        need([by_name[n][@"sha256"]isEqual:t[@"sha256"]]&&[by_name[n][@"bytes"]isEqual:@(WEIGHT_BYTES)],"complete actual weight payload missing");
        NSString *gn=[NSString stringWithFormat:@"weight-layer-%@-output-a.guards",t[@"layer"]];
        need(by_name[gn]!=nil,"correctness weight guard artifact required");NSData *g=read_file(evidence_file(base,gn));
        need(g.length==sizeof(guard_bytes)&&!memcmp(g.bytes,guard_bytes,sizeof(guard_bytes)),"correctness weight guards damaged");
    }
    need(by_name[@"args136.bin"]!=nil,"original argument artifact required");
    NSData *args_data=read_file(evidence_file(base,@"args136.bin"));struct id_args args=fixed_args();
    need(args_data.length==sizeof(args)&&!memcmp(args_data.bytes,&args,sizeof(args)),"original argument artifact differs");
    manifest[@"correctness_manifest_path"]=path;manifest[@"correctness_manifest_sha256"]=expected;
    manifest[@"prior_correctness_providers_verified"]=@(count*3);raw(@"correctness-manifest.json",bytes.bytes,bytes.length);return oracles;
}
int main(int argc,char **argv){@autoreleasepool {
    if(argc==2&&!strcmp(argv[1],"--cpu-self-test")) {
        emit(@{@"kind":@"cpu_self_test",@"rejected_invalid_bindings":@(alias_self_test()),@"passed":@YES,@"gpu_executed":@NO});return 0;
    }
    NSString *model=nil,*capture_path=nil,*probe_path=nil,*gold_path=nil,*gate_path=nil,*gate_sha=nil;
    unsigned samples=15,reps=256;BOOL host=NO,correctness_only=NO;
    for(int i=1;i<argc;i++) {
        const char *k=argv[i];if(!strcmp(k,"--host-only")){need(!host,"duplicate host-only");host=YES;continue;}
        if(!strcmp(k,"--correctness-only")){need(!correctness_only,"duplicate correctness-only");correctness_only=YES;continue;}
        need(i+1<argc,"option value");const char *v=argv[++i];
        if(!strcmp(k,"--model")){need(!model,"duplicate model");model=canonical([NSString stringWithUTF8String:v]);}
        else if(!strcmp(k,"--capture-json")){need(!capture_path,"duplicate capture");capture_path=canonical([NSString stringWithUTF8String:v]);}
        else if(!strcmp(k,"--probe-metallib")){need(!probe_path,"duplicate probe library");probe_path=canonical([NSString stringWithUTF8String:v]);}
        else if(!strcmp(k,"--production-metallib")){need(!gold_path,"duplicate production library");gold_path=canonical([NSString stringWithUTF8String:v]);}
        else if(!strcmp(k,"--out-dir")){need(!output,"duplicate output");output=canonical([NSString stringWithUTF8String:v]);}
        else if(!strcmp(k,"--correctness-manifest")){need(!gate_path,"duplicate correctness manifest");gate_path=canonical([NSString stringWithUTF8String:v]);}
        else if(!strcmp(k,"--correctness-manifest-sha256")){need(!gate_sha,"duplicate correctness SHA");gate_sha=[NSString stringWithUTF8String:v];}
        else if(!strcmp(k,"--samples"))samples=number(v,15,15);
        else if(!strcmp(k,"--reps"))reps=number(v,256,256);
        else need(NO,"unknown option");
    }
    need(model&&capture_path&&probe_path&&gold_path&&output,"required model/capture/libraries/new output");
    need((gate_path!=nil)==(gate_sha!=nil),"correctness manifest and SHA must be supplied together");
    need(!correctness_only||!gate_path,"correctness-only and performance gate are mutually exclusive");
    need(host||correctness_only||gate_path,"explicit correctness-only or bound complete correctness manifest required before GPU");
    need(!gate_path||(gate_path.isAbsolutePath&&gate_sha.length==64),"absolute correctness manifest and SHA required");
    phase=correctness_only?@"correctness":gate_path?@"performance":@"host_validation";
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"output must not exist");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output creation");
    files=[NSMutableArray new];manifest=[NSMutableDictionary new];parity_records=[NSMutableArray new];input_records=[NSMutableArray new];
    manifest[@"schema_version"]=@1;manifest[@"probe"]=@"decode_output_a_fixed_real_capture";manifest[@"phase"]=phase;
    manifest[@"fullmodel"]=@NO;manifest[@"cold_dram_proven"]=@NO;manifest[@"mode"]=@"concurrent";
    manifest[@"repeated_fixture_cache_reuse"]=@YES;manifest[@"capture_provenance_externally_required"]=@YES;
    manifest[@"deployment_identity_externally_required"]=@YES;manifest[@"binding_contract"]=binding_contract();
    manifest[@"samples"]=@(samples);manifest[@"repetitions_per_cb"]=@(reps);manifest[@"timing_command_buffers"]=@0;
    manifest[@"host_invalid_bindings_rejected"]=@(alias_self_test());manifest[@"production_host_rejection_proven"]=@NO;
    manifest[@"binary_path"]=binary_path();manifest[@"binary_sha256"]=file_sha(manifest[@"binary_path"]);
    manifest[@"probe_metallib_path"]=probe_path;manifest[@"probe_metallib_sha256"]=file_sha(probe_path);
    manifest[@"production_metallib_path"]=gold_path;manifest[@"production_metallib_sha256"]=file_sha(gold_path);
    NSData *capture_data=read_file(capture_path);NSDictionary *capture=[NSJSONSerialization JSONObjectWithData:capture_data options:0 error:nil];
    need([capture isKindOfClass:[NSDictionary class]],"capture JSON object");
    need(json_uint(capture[@"schema_version"],1,"schema version")==1&&[capture[@"capture_kind"]isEqual:@"myds_decode_output_a"]&&
        [capture[@"source"]isKindOfClass:[NSDictionary class]]&&[capture[@"source"]count]>0,"ordinary Output_A capture kind/source");
    manifest[@"capture_json_path"]=capture_path;manifest[@"capture_json_sha256"]=sha(capture_data.bytes,capture_data.length);
    manifest[@"capture_source"]=capture[@"source"];raw(@"capture.json",capture_data.bytes,capture_data.length);
    struct stat st;need(!stat(model.fileSystemRepresentation,&st)&&st.st_size>0,"model stat");
    gguf_file_t *gf=gguf_open(model.fileSystemRepresentation);need(gf!=NULL,"GGUF open");
    need(json_uint(capture[@"model_bytes"],UINT64_MAX,"model bytes")==(uint64_t)st.st_size,"captured model size differs");
    size_t header=(size_t)json_uint(capture[@"model_header_bytes"],st.st_size,"model prefix bytes");
    need(header>0&&[capture[@"model_header_sha256"]isEqual:sha(gf->mmap_addr,header)],"captured model prefix SHA differs");
    manifest[@"model"]=@{@"path":model,@"bytes":@(st.st_size),@"header_bytes":@(header),@"header_sha256":sha(gf->mmap_addr,header),
        @"tensor_data_offset":@(gf->tensor_data_offset),@"device":@(st.st_dev),@"inode":@(st.st_ino),
        @"mtime_sec":@(st.st_mtimespec.tv_sec),@"mtime_nsec":@(st.st_mtimespec.tv_nsec)};
    NSArray *cases=capture[@"cases"];need([cases isKindOfClass:[NSArray class]]&&cases.count&&cases.count<=12,"1..12 captured inputs");
    NSMutableArray *validated=[NSMutableArray new],*case_identity=[NSMutableArray new],*tensor_identity=[NSMutableArray new];
    NSMutableDictionary *layers=[NSMutableDictionary new];NSMutableSet *unique=[NSMutableSet new];
    NSString *base=[capture_path stringByDeletingLastPathComponent];
    for(unsigned i=0;i<cases.count;i++) {
        NSDictionary *c=cases[i];need([c isKindOfClass:[NSDictionary class]],"case object");
        unsigned layer=(unsigned)json_uint(c[@"layer"],39,"layer"),pos=(unsigned)json_uint(c[@"position"],15,"position");
        need((layer==0||layer==19||layer==39)&&(pos==0||pos==1||pos==7||pos==15),"selected real ordinary layer/position");
        NSString *pair=[NSString stringWithFormat:@"%u:%u",layer,pos];need(![unique containsObject:pair],"duplicate captured layer/position");[unique addObject:pair];
        need([c[@"need_logits"]isEqual:@YES]&&[c[@"ordinary_decode_allowed"]isEqual:@YES]&&[c[@"trace_on"]isEqual:@NO]&&
            [c[@"boundary"]isEqual:@"post_inverse_rope_bf16_before_output_a"]&&[c[@"observer_commit_wait"]isEqual:@YES]&&
            [c[@"observer_batch_restored"]isEqual:@YES]&&[c[@"performance_from_capture"]isEqual:@NO]&&
            json_uint(c[@"weight_type"],UINT32_MAX,"captured weight type")==GGML_TYPE_Q4_K&&
            json_uint(c[@"attention_rope_consumed"],1,"actual RF consumed flag")<=1&&
            json_uint(c[@"compressed"],1,"actual compressed flag")<=1&&
            json_uint(c[@"call_ordinal"],640,"actual call ordinal")==pos*40u+layer+1u,"actual ordinary post-inverse-RoPE metadata required");
        NSData *x=payload(c,@"input_f32",base,INPUT_BYTES);
        need(finite_f32(x.bytes,x.length)&&bf16(x.bytes,x.length),"captured heads full finite BF16 bits");
        [validated addObject:@{@"metadata":c,@"input":x}];
        [case_identity addObject:@{@"case":@(i),@"layer":@(layer),@"position":@(pos),@"input_bytes":@(INPUT_BYTES),
            @"input_sha256":sha(x.bytes,x.length),@"attention_rope_consumed":c[@"attention_rope_consumed"],@"compressed":c[@"compressed"],
            @"boundary":c[@"boundary"]}];
        NSString *key=[NSString stringWithFormat:@"%u",layer];if(!layers[key])layers[key]=[NSMutableArray new];[layers[key]addObject:@(i)];
        raw([NSString stringWithFormat:@"capture-case-%03u-input.f32",i],x.bytes,x.length);
    }
    NSArray *layer_keys=[[layers allKeys]sortedArrayUsingComparator:^NSComparisonResult(NSString *a,NSString *b){return[@(a.intValue)compare:@(b.intValue)];}];
    for(NSString *key in layer_keys) {
        NSString *name=[NSString stringWithFormat:@"blk.%@.attn_output_a.weight",key];const gguf_tensor_t *t=gguf_find_tensor(gf,name.UTF8String);
        need(t&&t->type==GGML_TYPE_Q4_K&&t->n_dims==2&&t->dims[0]==4096&&t->dims[1]==8192&&t->size_bytes==WEIGHT_BYTES,"exact actual Output_A Q4 tensor shape/type");
        need(gf->tensor_data_offset<=(uint64_t)st.st_size&&t->offset<=(uint64_t)st.st_size-gf->tensor_data_offset&&
            WEIGHT_BYTES<=(uint64_t)st.st_size-gf->tensor_data_offset-t->offset,"Output_A tensor inside actual file bytes");
        [tensor_identity addObject:@{@"layer":@(key.intValue),@"name":name,@"type":@(t->type),@"dimensions":@[@4096,@8192],
            @"bytes":@(WEIGHT_BYTES),@"file_offset":@(gf->tensor_data_offset+t->offset),@"sha256":sha(t->data,WEIGHT_BYTES)}];
    }
    manifest[@"tensors"]=tensor_identity;manifest[@"cases"]=case_identity;manifest[@"total_cases"]=@(cases.count);manifest[@"argv"]=[NSProcessInfo processInfo].arguments;
    struct id_args args=fixed_args();raw(@"args136.bin",&args,sizeof(args));
    NSArray *oracles=gate_path?correctness_oracles(gate_path,gate_sha):nil;
    emit(@{@"kind":@"identity",@"manifest":manifest});
    if(host){finish(@"host_only");gguf_close(gf);emit(@{@"kind":@"complete",@"host_only":@YES,@"metal_device_created":@NO});return 0;}
    need(cases.count==12&&layer_keys.count==3,"GPU requires all twelve real ordinary inputs");
    for(NSString *key in layer_keys)need([layers[key]count]==4,"GPU requires four positions per layer");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");NSError *error=nil;
    id<MTLLibrary> probe=[device newLibraryWithURL:[NSURL fileURLWithPath:probe_path]error:&error];
    if(!probe)fprintf(stderr,"probe library: %s\n",error.localizedDescription.UTF8String);need(probe!=nil,"probe library");
    id<MTLLibrary> golden=[device newLibraryWithURL:[NSURL fileURLWithPath:gold_path]error:&error];need(golden!=nil,"production library");
    NSArray *names=@[@"kernel_dsv4_attn_out_low_q4_K_bf16",@"kernel_dsv4_attn_out_low_q4_K_bf16",@"kernel_myds_outa_q4_v41_fixed_bf16"];
    id<MTLComputePipelineState> psos[3];NSMutableArray *bindings=[NSMutableArray new];
    for(unsigned p=0;p<3;p++) {
        psos[p]=pipeline(device,p?probe:golden,names[p]);
        [bindings addObject:@{@"provider":@(p),@"symbol":names[p],@"library":p?@"probe":@"production",
            @"contract":binding_contract(),@"static_threadgroup_bytes":@(psos[p].staticThreadgroupMemoryLength),
            @"max_threads":@(psos[p].maxTotalThreadsPerThreadgroup),@"thread_execution_width":@(psos[p].threadExecutionWidth)}];
    }
    manifest[@"device"]=device.name;manifest[@"bindings"]=bindings;emit(@{@"kind":@"bindings",@"bindings":bindings,@"device":device.name});
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"command queue");
    for(unsigned li=0;li<layer_keys.count;li++){@autoreleasepool {
        NSString *key=layer_keys[li];NSArray *indices=layers[key];unsigned count=(unsigned)indices.count;
        const gguf_tensor_t *t=gguf_find_tensor(gf,[[NSString stringWithFormat:@"blk.%@.attn_output_a.weight",key]UTF8String]);
        NSString *weight_sha=tensor_identity[li][@"sha256"];
        id<MTLBuffer> w=alloc(device,WEIGHT_BYTES,YES);memcpy(body(w),t->data,WEIGHT_BYTES);
        need([sha(body(w),WEIGHT_BYTES)isEqual:weight_sha],"GPU-bound actual Output_A weight SHA");
        raw([NSString stringWithFormat:@"weight-layer-%@-output-a.q4k",key],body(w),WEIGHT_BYTES);
        fixture *f=calloc(count,sizeof(fixture));need(f!=NULL,"Output_A fixtures");
        for(unsigned i=0;i<count;i++) {
            unsigned index=[indices[i]unsignedIntValue];NSDictionary *v=validated[index],*c=v[@"metadata"];NSData *x=v[@"input"];
            f[i].index=index;f[i].layer=key.intValue;f[i].position=[c[@"position"]unsignedIntValue];
            f[i].input=alloc(device,INPUT_BYTES,YES);memcpy(body(f[i].input),x.bytes,x.length);f[i].input_sha=case_identity[index][@"input_sha256"];
            for(unsigned p=0;p<3;p++)f[i].out[p]=alloc(device,OUTPUT_BYTES,YES);
            if(oracles){f[i].oracle=read_file(oracles[index][@"path"]);need([sha(f[i].oracle.bytes,f[i].oracle.length)isEqual:oracles[index][@"sha256"]],"native oracle changed before GPU replay");}
        }
        if(correctness_only) {
            for(unsigned p=0;p<3;p++){(void)run(queue,f,count,p,1,w,psos[p]);validate(f,count,p,w,weight_sha,YES);}
        } else {
            for(unsigned control=0;control<2;control++) {
                unsigned providers[3]={control,2,control};double times[3][64];
                for(unsigned sample=0;sample<=samples;sample++)for(unsigned arm=0;arm<3;arm++) {
                    unsigned p=providers[arm];double begin=wall();id<MTLCommandBuffer> cb=run(queue,f,count,p,reps,w,psos[p]);double end=wall();
                    validate(f,count,p,w,weight_sha,NO);
                    need(isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime&&cb.GPUStartTime>0,"valid complete CB GPU timeline");
                    double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/(count*reps);if(sample)times[arm][sample-1]=us;
                    emit(@{@"kind":@"timing",@"mode":@"concurrent",@"layer":@(key.intValue),@"control_kind":control?@"probe":@"production",
                        @"provider":@(p),@"symbol":names[p],@"arm":@[@"A1",@"B",@"A2"][arm],@"sample":@(sample),@"warmup":@(sample==0),
                        @"cases_in_cb":@(count),@"repetitions":@(reps),@"dispatches":@(count*reps),@"projections_per_cb":@(count*reps),@"commits":@1,
                        @"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"gpu_us_per_projection":@(us),
                        @"wall_us_per_projection":@((end-begin)*1e6/(count*reps)),@"full_f32_bit_exact":@YES,
                        @"all_gpu_inputs_unchanged":@YES,@"component_only":@YES,@"completed_cb":@YES,@"command_ordinal":@(command_ordinal)});
                    manifest[@"timing_command_buffers"]=@([manifest[@"timing_command_buffers"]unsignedIntValue]+1);
                }
                double a1=median(times[0],samples),b=median(times[1],samples),a2=median(times[2],samples),a=(a1+a2)/2;
                emit(@{@"kind":@"summary",@"mode":@"concurrent",@"layer":@(key.intValue),@"control_kind":control?@"probe":@"production",
                    @"control_symbol":names[control],@"candidate_symbol":names[2],@"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),
                    @"control_drift_percent":@((a2/a1-1)*100),@"time_saved_percent":@((1-b/a)*100),
                    @"conservative_time_saved_percent":@((1-b/fmin(a1,a2))*100),@"fullmodel_speedup_proven":@NO,@"cold_dram_proven":@NO});
            }
        }
        uint8_t g[2*GUARD];memcpy(g,w.contents,GUARD);memcpy(g+GUARD,(uint8_t*)w.contents+GUARD+WEIGHT_BYTES,GUARD);
        raw([NSString stringWithFormat:@"weight-layer-%@-output-a.guards",key],g,sizeof(g));
        need(guards(w,WEIGHT_BYTES)&&[sha(body(w),WEIGHT_BYTES)isEqual:weight_sha]&&[sha(t->data,WEIGHT_BYTES)isEqual:weight_sha],"real mapped/Metal weights unchanged");
        for(unsigned i=0;i<count;i++) {
            memcpy(g,f[i].input.contents,GUARD);memcpy(g+GUARD,(uint8_t*)f[i].input.contents+GUARD+INPUT_BYTES,GUARD);
            raw([NSString stringWithFormat:@"input-case-%03u.guards",f[i].index],g,sizeof(g));
            need(guards(f[i].input,INPUT_BYTES)&&[sha(body(f[i].input),INPUT_BYTES)isEqual:f[i].input_sha],"complete captured input unchanged");
            f[i].input=nil;f[i].oracle=nil;f[i].input_sha=nil;for(unsigned p=0;p<3;p++)f[i].out[p]=nil;
        }
        free(f);
    }}
    struct stat final_st;need(!stat(model.fileSystemRepresentation,&final_st)&&st.st_dev==final_st.st_dev&&st.st_ino==final_st.st_ino&&
        st.st_size==final_st.st_size&&st.st_mtimespec.tv_sec==final_st.st_mtimespec.tv_sec&&st.st_mtimespec.tv_nsec==final_st.st_mtimespec.tv_nsec,"actual model identity changed");
    need([file_sha(probe_path)isEqual:manifest[@"probe_metallib_sha256"]]&&[file_sha(gold_path)isEqual:manifest[@"production_metallib_sha256"]]&&
        [file_sha(capture_path)isEqual:manifest[@"capture_json_sha256"]]&&[file_sha(binary_path())isEqual:manifest[@"binary_sha256"]],"library/capture/binary changed during replay");
    need(correctness_only?parity_records.count==36:[manifest[@"timing_command_buffers"]unsignedIntValue]==288,"complete all-case phase command counts");
    manifest[@"full_f32_bit_exact"]=@YES;manifest[@"output_a_checked"]=@YES;manifest[@"all_gpu_inputs_unchanged"]=@YES;
    finish(@"complete");gguf_close(gf);emit(@{@"kind":@"complete",@"phase":phase,@"cases":@12,@"providers":@3,
        @"output_a_checked":@YES,@"full_f32_bit_exact":@YES,@"all_gpu_inputs_unchanged":@YES,@"mode":@"concurrent",
        @"timing_command_buffers":manifest[@"timing_command_buffers"],@"fullmodel":@NO,@"cold_dram_proven":@NO});
}return 0;}
