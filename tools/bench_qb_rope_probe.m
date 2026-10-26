/* Private synthetic/component Q_B + forward-RoPE exact-output gate.
 * Build only with build_qb_rope_probe.py; frozen_rope_init.inc is copied from
 * the production C source, with its original precise host frequency order.
 * --metallib PROBE --golden-metallib ACTUAL_PRODUCTION --build-manifest JSON
 * --out-dir NEW --synthetic [--host-only] [--timing]
 * Default: correctness only. --timing is an explicitly synthetic component
 * screen after every full-output gate, never a full-model/adoption result.
 * --host-only returns before MTLCreateSystemDefaultDevice or GPU allocation.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <inttypes.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "frozen_rope_init.inc"

enum { GUARD=64, ROWS=32768, WIDTH=1280, HEADS=64, HDIM=512, NSG=4, NR=2,
       WEIGHT_BYTES=23592960, INPUT_BYTES=5120, OUTPUT_BYTES=131072, MAX_CASES=128 };
struct mv_args {
    int32_t ne00,ne01,ne02; uint64_t nb00,nb01,nb02,nb03;
    int32_t ne10,ne11,ne12; uint64_t nb10,nb11,nb12,nb13;
    int32_t ne0,ne1,nr0; int16_t r2,r3;
};
struct rope_args { uint32_t width,heads,rows,start,inverse,stride; float frequencies[32]; };
struct q4_block { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
_Static_assert(sizeof(struct mv_args)==112,"original Q4 constant-buffer ABI");
_Static_assert(offsetof(struct mv_args,nb00)==16&&offsetof(struct mv_args,ne10)==48&&
               offsetof(struct mv_args,ne0)==96,"original Q4 member offsets");
_Static_assert(sizeof(struct rope_args)==152&&offsetof(struct rope_args,frequencies)==24,"original RoPE ABI");
_Static_assert(sizeof(struct q4_block)==144,"original Q4_K raw block ABI");

static NSString *output;
static NSMutableDictionary *manifest;
static NSMutableArray *files,*parity_records,*timing_records,*input_records;
static unsigned command_ordinal;
static void finish(NSString *status) {
    if(!manifest||!output)return;
    manifest[@"status"]=status;manifest[@"files"]=files;
    manifest[@"parity_records"]=parity_records;manifest[@"input_records"]=input_records;
    manifest[@"timing_records"]=timing_records;manifest[@"gpu_command_buffers_submitted"]=@(command_ordinal);
    NSData *d=[NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    if(d)[d writeToFile:[output stringByAppendingPathComponent:@"artifact-manifest.json"] atomically:YES];
}
static void need(BOOL ok,const char *reason) {
    if(ok)return;
    fprintf(stderr,"qb_rope_probe: %s\n",reason);
    if(manifest)manifest[@"failure"]=[NSString stringWithUTF8String:reason];
    finish(@"failed");exit(1);
}
static NSString *sha(const void *p,size_t n) {
    CC_SHA256_CTX c;CC_SHA256_Init(&c);const uint8_t *b=p;
    while(n){CC_LONG k=(CC_LONG)(n>0x10000000u?0x10000000u:n);CC_SHA256_Update(&c,b,k);b+=k;n-=k;}
    uint8_t d[32];CC_SHA256_Final(d,&c);char s[65];
    for(unsigned i=0;i<32;i++)snprintf(s+2*i,3,"%02x",d[i]);
    return[NSString stringWithUTF8String:s];
}
static NSData *read_file(NSString *p) {
    NSData *d=[NSData dataWithContentsOfFile:p options:NSDataReadingMappedIfSafe error:nil];
    need(d!=nil,"identity file read");return d;
}
static NSString *canonical(NSString *p){return[[p stringByStandardizingPath]stringByResolvingSymlinksInPath];}
static NSString *file_sha(NSString *p){NSData *d=read_file(p);return sha(d.bytes,d.length);}
static NSString *binary_path(void) {
    uint32_t n=0;_NSGetExecutablePath(NULL,&n);char *p=calloc(n,1);need(p!=NULL,"binary path allocation");
    need(!_NSGetExecutablePath(p,&n),"binary path");NSString *s=canonical([NSString stringWithUTF8String:p]);free(p);return s;
}
static void emit(NSDictionary *r) {
    NSData *d=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingSortedKeys error:nil];
    need(d!=nil,"JSON encoding");fwrite(d.bytes,1,d.length,stdout);putchar('\n');fflush(stdout);
}
static void raw(NSString *name,const void *p,size_t bytes) {
    NSData *d=[NSData dataWithBytes:p length:bytes];
    need([d writeToFile:[output stringByAppendingPathComponent:name] options:NSDataWritingWithoutOverwriting error:nil],"new raw artifact write");
    [files addObject:@{@"path":name,@"bytes":@(bytes),@"sha256":sha(p,bytes)}];
}
static BOOL index_source(unsigned l){return l==2||l==8||l==14||l==20||(l>=24&&l%4==0);}
static BOOL eligible_layer(unsigned l){return l<40&&!index_source(l);}
static BOOL compressed_layer(unsigned l){return l>=2;}
static uint64_t rng=UINT64_C(0xd17aa093f53f1b2d);
static uint64_t random64(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static float round_bf16(float x) {
    uint32_t u;memcpy(&u,&x,4);
    if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16u)&1u);
    u&=0xffff0000u;memcpy(&x,&u,4);return x;
}
static NSMutableData *make_weights(void) {
    NSMutableData *d=[NSMutableData dataWithLength:WEIGHT_BYTES];
    struct q4_block *b=d.mutableBytes;
    for(size_t i=0;i<WEIGHT_BYTES/sizeof(*b);i++) {
        b[i].d=(uint16_t)(0x1400u|(random64()&0x3ffu));
        b[i].dmin=(uint16_t)(0x1000u|(random64()&0x3ffu));
        for(unsigned j=0;j<12;j++)b[i].scales[j]=(uint8_t)random64();
        for(unsigned j=0;j<128;j++)b[i].qs[j]=(uint8_t)random64();
    }
    return d;
}
static NSMutableData *make_input(unsigned layer,unsigned pos) {
    NSMutableData *d=[NSMutableData dataWithLength:INPUT_BYTES];float *x=d.mutableBytes;
    for(unsigned i=0;i<WIDTH;i++) {
        int v=(int)(random64()%8193u)-4096;
        float f=(float)v/2048.0f;
        if(layer==0&&pos==0)f=(i&1u)?-1.0f:1.0f;
        if(layer==1&&pos==1)f=(i%7u==0)?0.0f:f/16.0f;
        x[i]=round_bf16(f);
    }
    return d;
}
static BOOL finite_bf16(const void *p,size_t bytes) {
    const uint32_t *v=p;for(size_t i=0;i<bytes/4;i++)if((v[i]&0xffffu)||((v[i]&0x7f800000u)==0x7f800000u))return NO;
    return YES;
}
static size_t differences(const void *a,const void *b,size_t *first,size_t *tail) {
    const uint32_t *x=a,*y=b;size_t n=0;*first=SIZE_MAX;*tail=0;
    for(size_t i=0;i<ROWS;i++)if(x[i]!=y[i]){if(!n)*first=i;n++;if(i%HDIM>=448u)(*tail)++;}
    return n;
}
static void *body_ptr(id<MTLBuffer> b){return(uint8_t*)b.contents+GUARD;}
static id<MTLBuffer> allocate(id<MTLDevice> d,const void *p,size_t bytes) {
    need(bytes+2u*GUARD<=d.maxBufferLength,"buffer within maxBufferLength");
    id<MTLBuffer> b=[d newBufferWithLength:bytes+2u*GUARD options:MTLResourceStorageModeShared];
    need(b!=nil,"Metal buffer allocation");memset(b.contents,0xa5,b.length);
    if(p)memcpy(body_ptr(b),p,bytes);return b;
}
static BOOL guards(id<MTLBuffer> b,size_t bytes) {
    const uint8_t *p=b.contents;for(unsigned i=0;i<GUARD;i++)if(p[i]!=0xa5||p[GUARD+bytes+i]!=0xa5)return NO;return YES;
}
static void dump(NSString *name,id<MTLBuffer> b,size_t bytes) {
    raw([name stringByAppendingString:@".raw"],body_ptr(b),bytes);
    raw([name stringByAppendingString:@".prefix-guard.raw"],b.contents,GUARD);
    raw([name stringByAppendingString:@".suffix-guard.raw"],(uint8_t*)b.contents+GUARD+bytes,GUARD);
}
static struct mv_args q_args(void) {
    return(struct mv_args){.ne00=WIDTH,.ne01=ROWS,.ne02=1,.nb00=144,.nb01=720,
        .nb02=WEIGHT_BYTES,.nb03=WEIGHT_BYTES,.ne10=WIDTH,.ne11=1,.ne12=1,
        .nb10=4,.nb11=INPUT_BYTES,.nb12=INPUT_BYTES,.nb13=INPUT_BYTES,
        .ne0=ROWS,.ne1=1,.nr0=NR,.r2=1,.r3=1};
}
static struct rope_args r_args(unsigned pos,BOOL compressed) {
    struct rope_args a={.width=HDIM,.heads=HEADS,.rows=1,.start=pos,.inverse=0,.stride=1};
    pthread_once(&rope_once,init_rope);memcpy(a.frequencies,rope_frequencies[compressed?1:0],sizeof(a.frequencies));return a;
}
static BOOL bindings_safe(id<MTLBuffer> y,id<MTLBuffer> w,id<MTLBuffer> x) {
    return y&&w&&x&&y!=w&&y!=x&&w!=x&&y.length==OUTPUT_BYTES+2u*GUARD&&
        w.length==WEIGHT_BYTES+2u*GUARD&&x.length==INPUT_BYTES+2u*GUARD&&
        !((uintptr_t)body_ptr(y)%16u)&&!((uintptr_t)body_ptr(w)%16u)&&!((uintptr_t)body_ptr(x)%16u);
}
typedef struct { unsigned layer,pos,index;BOOL compressed;NSData *input;NSString *input_sha;
    id<MTLBuffer> x,y[3];NSData *oracle; } fixture;
static fixture cases[MAX_CASES];
static unsigned case_count;
static id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,NSString*n,BOOL q4) {
    NSError *error=nil;id<MTLFunction> f;
    if(q4){short v=NSG;MTLFunctionConstantValues *c=[MTLFunctionConstantValues new];
        [c setConstantValue:&v type:MTLDataTypeShort atIndex:600];f=[l newFunctionWithName:n constantValues:c error:&error];}
    else f=[l newFunctionWithName:n];
    if(!f)fprintf(stderr,"missing %s: %s\n",n.UTF8String,error.localizedDescription.UTF8String);
    need(f!=nil,"explicit native/candidate symbol missing; fallback forbidden");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];
    if(!p)fprintf(stderr,"pipeline %s: %s\n",n.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=(q4?128u:32u),"pipeline execution geometry");
    need(p.staticThreadgroupMemoryLength+(q4?256u:0u)<=d.maxThreadgroupMemoryLength,"pipeline memory geometry");
    emit(@{@"kind":@"pipeline",@"symbol":n,@"fc600_short":q4?@4:[NSNull null],@"thread_execution_width":@(p.threadExecutionWidth),
           @"max_threads":@(p.maxTotalThreadsPerThreadgroup),@"static_tg_bytes":@(p.staticThreadgroupMemoryLength),@"fallback":@NO});
    return p;
}
static id<MTLCommandBuffer> run(id<MTLCommandQueue>q,fixture*f,unsigned provider,unsigned repetitions,
        id<MTLBuffer>w,id<MTLComputePipelineState>qb,id<MTLComputePipelineState>rope) {
    need(provider<3&&bindings_safe(f->y[provider],w,f->x),"complete disjoint Q4 ranges/alignments before CB");
    memset(body_ptr(f->y[provider]),0xa5,OUTPUT_BYTES);
    id<MTLCommandBuffer> cb=[q commandBuffer];need(cb!=nil,"command buffer");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"concurrent compute encoder");
    struct mv_args a=q_args();struct rope_args r=r_args(f->pos,f->compressed);
    for(unsigned rep=0;rep<repetitions;rep++) {
        [e setComputePipelineState:qb];[e setBytes:&a length:sizeof(a) atIndex:0];
        [e setBuffer:w offset:GUARD atIndex:1];[e setBuffer:f->x offset:GUARD atIndex:2];
        [e setBuffer:f->y[provider] offset:GUARD atIndex:3];[e setThreadgroupMemoryLength:256 atIndex:0];
        if(provider==2)[e setBytes:&r length:sizeof(r) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(4096,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        id<MTLResource> out=f->y[provider];[e memoryBarrierWithResources:&out count:1];
        if(provider!=2) {
            [e setComputePipelineState:rope];[e setBytes:&r length:sizeof(r) atIndex:0];
            [e setBuffer:f->y[provider] offset:GUARD atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        // Keep both joins in every arm. The real prestage_window and other
        // cohort resources are absent: this cannot prove runtime overlap.
        [e memoryBarrierWithResources:&out count:1];
    }
    [e endEncoding];command_ordinal++;[cb commit];[cb waitUntilCompleted];return cb;
}
static BOOL save_inputs(id<MTLBuffer>w,NSData*before,NSString*stage) {
    BOOL ok=guards(w,WEIGHT_BYTES)&&!memcmp(body_ptr(w),before.bytes,WEIGHT_BYTES);
    NSString *name=[stage stringByAppendingString:@".weights"];dump(name,w,WEIGHT_BYTES);
    NSDictionary *wr=@{@"kind":@"inputs",@"stage":stage,@"role":@"weights",@"bytes":@(WEIGHT_BYTES),
        @"before_sha256":sha(before.bytes,before.length),@"after_sha256":sha(body_ptr(w),WEIGHT_BYTES),
        @"guards":@(guards(w,WEIGHT_BYTES)),@"bytes_equal_before":@(!memcmp(body_ptr(w),before.bytes,WEIGHT_BYTES)),@"raw_prefix":name};
    [input_records addObject:wr];emit(wr);
    for(unsigned i=0;i<case_count;i++) {
        fixture *f=&cases[i];BOOL same=!memcmp(body_ptr(f->x),f->input.bytes,INPUT_BYTES);BOOL g=guards(f->x,INPUT_BYTES);
        NSString *n=[NSString stringWithFormat:@"%@.case%03u.input",stage,i];dump(n,f->x,INPUT_BYTES);
        NSDictionary *r=@{@"kind":@"inputs",@"stage":stage,@"role":@"qr_norm",@"case":@(i),@"bytes":@(INPUT_BYTES),
            @"before_sha256":f->input_sha,@"after_sha256":sha(body_ptr(f->x),INPUT_BYTES),
            @"guards":@(g),@"bytes_equal_before":@(same),@"raw_prefix":n};
        [input_records addObject:r];emit(r);ok=ok&&same&&g;
    }
    return ok;
}
static BOOL validate_output(fixture*f,unsigned provider,BOOL save,NSString*stage) {
    size_t first=SIZE_MAX,tail=0;BOOL g=guards(f->y[provider],OUTPUT_BYTES);
    if(provider==0&&!f->oracle)f->oracle=[NSData dataWithBytes:body_ptr(f->y[0]) length:OUTPUT_BYTES];
    size_t bad=differences(f->oracle.bytes,body_ptr(f->y[provider]),&first,&tail);
    BOOL format=finite_bf16(body_ptr(f->y[provider]),OUTPUT_BYTES);
    NSDictionary *r=@{@"kind":@"parity",@"stage":stage,@"case":@(f->index),@"layer":@(f->layer),@"position":@(f->pos),
        @"compressed":@(f->compressed),@"provider":@(provider),@"complete_f32_count":@(ROWS),@"mismatches":@(bad),
        @"tail_mismatches":@(tail),@"unrotated_mismatches":@(bad-tail),@"first_mismatch":first==SIZE_MAX?[NSNull null]:@(first),
        @"finite_bf16":@(format),@"guards":@(g),@"sha256":sha(body_ptr(f->y[provider]),OUTPUT_BYTES)};
    [parity_records addObject:r];emit(r);
    if(save||bad||!format||!g)dump([NSString stringWithFormat:@"%@.case%03u.provider%u.output",stage,f->index,provider],f->y[provider],OUTPUT_BYTES);
    return !bad&&format&&g;
}
static void add_case(unsigned layer,unsigned pos) {
    need(case_count<MAX_CASES&&eligible_layer(layer)&&pos<1048576u,"case in ordinary non-index-source shape/position scope");
    fixture *f=&cases[case_count];f->layer=layer;f->pos=pos;f->index=case_count;f->compressed=compressed_layer(layer);
    f->input=make_input(layer,pos);f->input_sha=sha(f->input.bytes,f->input.length);case_count++;
}
static double median(double v[9]) {
    double a[9];memcpy(a,v,sizeof(a));for(unsigned i=0;i<9;i++)for(unsigned j=i+1;j<9;j++)if(a[i]>a[j]){double t=a[i];a[i]=a[j];a[j]=t;}return a[4];
}
int main(int argc,char **argv){@autoreleasepool {
    NSString *probe=nil,*gold=nil,*build_path=nil;BOOL synthetic=NO,host=NO,timing=NO;
    for(int i=1;i<argc;i++) {
        const char *k=argv[i];
        if(!strcmp(k,"--synthetic")){synthetic=YES;continue;}
        if(!strcmp(k,"--host-only")){host=YES;continue;}
        if(!strcmp(k,"--timing")){timing=YES;continue;}
        need(i+1<argc,"argument value required");NSString *v=[NSString stringWithUTF8String:argv[++i]];
        if(!strcmp(k,"--metallib"))probe=canonical(v);
        else if(!strcmp(k,"--golden-metallib"))gold=canonical(v);
        else if(!strcmp(k,"--build-manifest"))build_path=canonical(v);
        else if(!strcmp(k,"--out-dir"))output=canonical(v);
        else need(NO,"unknown argument");
    }
    need(synthetic&&probe&&gold&&build_path&&output,"explicit --synthetic and all paths required");
    need(output.isAbsolutePath&&probe.isAbsolutePath&&gold.isAbsolutePath&&build_path.isAbsolutePath,"absolute artifact paths required");
    need(![probe isEqual:gold],"actual golden library must have a separate path; do not silently substitute the probe");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new output directory required");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:NO attributes:nil error:nil],"create output directory");
    files=[NSMutableArray new];parity_records=[NSMutableArray new];timing_records=[NSMutableArray new];input_records=[NSMutableArray new];
    NSData *build_data=read_file(build_path);NSDictionary *build=[NSJSONSerialization JSONObjectWithData:build_data options:0 error:nil];
    need([build isKindOfClass:[NSDictionary class]],"build identity JSON");
    need([build[@"kind"]isEqual:@"qb_forward_rope_private_full_tu_build"]&&[build[@"complete_native_prefix_byte_exact"]boolValue]&&
        [build[@"proof"][@"reverse_edits_restore_original_body_bytes"]boolValue]&&![build[@"production_registered"]boolValue],"full production TU prefix and reverse edit proof required");
    need([file_sha(probe)isEqual:build[@"metallib_sha256"]],"probe library bound to completed build identity");
    need([file_sha(binary_path())isEqual:build[@"binary_sha256"]],"running binary bound to build identity");
    manifest=[@{@"schema_version":@1,@"kind":@"qb_forward_rope_synthetic_component",@"host_only":@(host),
        @"synthetic":@YES,@"runtime_modified":@NO,@"production_registered":@NO,@"full_model_evidence":@NO,
        @"timing_requested":@(timing),@"build_manifest_path":build_path,@"build_manifest_sha256":sha(build_data.bytes,build_data.length),
        @"binary_path":binary_path(),@"binary_sha256":file_sha(binary_path()),@"probe_metallib_path":probe,@"probe_metallib_sha256":file_sha(probe),
        @"golden_metallib_path":gold,@"golden_metallib_sha256":file_sha(gold),@"abi":build[@"abi"],
        @"eligible_layers":build[@"eligible_decode_non_index_source_layers"],
        @"symbols":@[@"kernel_myds_q4_v41_qb_nb_bf16",@"kernel_dsv41_rope",@"kernel_myds_probe_qb_rope_nb_bf16"],
        @"providers":@[@"actual-golden-original-Q_B-plus-original-RoPE",@"full-TU-probe-original-Q_B-plus-original-RoPE",@"full-TU-probe-fused-Q_B-forward-RoPE"],
        @"rounding_contract":@"original row-order Q4 dot/simd_sum -> BF16 -> original precise forward-RoPE expressions -> BF16",
        @"limitations":@[@"Synthetic full output gate; not real model quality or production performance proof.",
            @"No actual prestage_window or companion cohort kernels; retained joins alone cannot reproduce runtime overlap.",
            @"Sparse lane0 trig can increase Q_B residency and erase all saved dispatch cost."]} mutableCopy];
    raw(@"build-identity.json",build_data.bytes,build_data.length);
    NSMutableArray *eligible=[NSMutableArray new];unsigned rejects=0;
    for(unsigned l=0;l<40;l++)if(eligible_layer(l))[eligible addObject:@(l)];else rejects++;
    need(eligible.count==32&&rejects==8&&[eligible isEqual:build[@"eligible_decode_non_index_source_layers"]],"32 eligible layers, eight index-source rejects");
    need(!eligible_layer(40)&&!eligible_layer(UINT32_MAX),"out-of-range layers rejected");
    unsigned positions[]={0u,1u,327679u};
    for(unsigned l=0;l<40;l++)if(eligible_layer(l))for(unsigned p=0;p<3;p++)add_case(l,positions[p]);
    unsigned edge_layers[]={0,3,39},edge_positions[]={127,128,16383,16384,65535,65536};
    for(unsigned l=0;l<3;l++)for(unsigned p=0;p<6;p++)add_case(edge_layers[l],edge_positions[p]);
    NSMutableData *weight_data=make_weights();raw(@"weights.before.raw",weight_data.bytes,weight_data.length);
    for(unsigned i=0;i<case_count;i++) {
        fixture *f=&cases[i];need(finite_bf16(f->input.bytes,INPUT_BYTES),"complete input finite/BF16 contract");
        raw([NSString stringWithFormat:@"case%03u.input.before.raw",i],f->input.bytes,f->input.length);
        struct mv_args a=q_args();struct rope_args r=r_args(f->pos,f->compressed);
        raw([NSString stringWithFormat:@"case%03u.mv-args112.raw",i],&a,sizeof(a));
        raw([NSString stringWithFormat:@"case%03u.rope-args152.raw",i],&r,sizeof(r));
        emit(@{@"kind":@"case",@"case":@(i),@"layer":@(f->layer),@"position":@(f->pos),@"compressed":@(f->compressed),
               @"input_sha256":f->input_sha,@"input_bytes":@(INPUT_BYTES),@"output_f32_count":@(ROWS),@"rope_args_sha256":sha(&r,sizeof(r))});
    }
    manifest[@"case_count"]=@(case_count);manifest[@"eligibility_host_rejections"]=@(rejects+2u);
    emit(@{@"kind":@"host_contract",@"case_count":@(case_count),@"eligible_layers":eligible,@"rejected_layer_count":@(rejects+2u),
           @"mv_args_bytes":@112,@"rope_args_bytes":@152,@"weights_bytes":@(WEIGHT_BYTES),@"no_gpu_device_created":@(host)});
    if(host){finish(@"host_only_complete_no_gpu");return 0;}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");
    manifest[@"device_name"]=device.name;manifest[@"device_registry_id"]=@(device.registryID);
    NSError *error=nil;id<MTLLibrary> libraries[2];
    libraries[0]=[device newLibraryWithURL:[NSURL fileURLWithPath:gold] error:&error];
    need(libraries[0]!=nil,"load exact actual golden library URL; fallback forbidden");
    libraries[1]=[device newLibraryWithURL:[NSURL fileURLWithPath:probe] error:&error];
    need(libraries[1]!=nil,"load exact probe library URL; fallback forbidden");
    id<MTLComputePipelineState> qb[3],rope[2];
    for(unsigned i=0;i<2;i++) {
        emit(@{@"kind":@"loaded_library",@"provider_library":@(i),@"path":i?probe:gold,@"sha256":file_sha(i?probe:gold),@"direct_url_load":@YES});
        qb[i]=pipeline(device,libraries[i],@"kernel_myds_q4_v41_qb_nb_bf16",YES);
        rope[i]=pipeline(device,libraries[i],@"kernel_dsv41_rope",NO);
    }
    qb[2]=pipeline(device,libraries[1],@"kernel_myds_probe_qb_rope_nb_bf16",YES);
    id<MTLBuffer> weights=allocate(device,weight_data.bytes,WEIGHT_BYTES);
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"Metal command queue");
    for(unsigned i=0;i<case_count;i++) {
        fixture *f=&cases[i];f->x=allocate(device,f->input.bytes,INPUT_BYTES);
        for(unsigned p=0;p<3;p++)f->y[p]=allocate(device,NULL,OUTPUT_BYTES);
    }
    BOOL correctness=YES;
    for(unsigned i=0;i<case_count;i++)for(unsigned p=0;p<3;p++) {
        fixture *f=&cases[i];id<MTLCommandBuffer> cb=run(queue,f,p,1,weights,qb[p],p<2?rope[p]:nil);
        BOOL complete=cb.status==MTLCommandBufferStatusCompleted;
        emit(@{@"kind":@"gpu_completion",@"stage":@"phase1",@"case":@(i),@"provider":@(p),@"ordinal":@(command_ordinal),
               @"completed":@(complete),@"status":@(cb.status),@"error":cb.error.localizedDescription?:[NSNull null]});
        BOOL parity=validate_output(f,p,YES,@"phase1");correctness=correctness&&complete&&parity;
    }
    BOOL input_ok=save_inputs(weights,weight_data,@"phase1_post");correctness=correctness&&input_ok;
    manifest[@"all_complete_outputs_and_inputs_pass"]=@(correctness);
    need(correctness,"all phase1 outputs and input/guard raw saved; exact gate refused");
    if(!timing){finish(@"synthetic_correctness_complete_not_adoption");return 0;}
    // Fixed component ABA screen: three cases (raw pos0, compressed pos1 and
    // compressed long position), two original controls, one warmup + nine
    // formal samples. No samples are dropped or retried within this cohort.
    unsigned selected[]={0,7,95};unsigned repeats=64;BOOL timing_ok=YES;
    for(unsigned c=0;c<3;c++)for(unsigned control=0;control<2;control++) {
        fixture *f=&cases[selected[c]];double durations[3][9]={{0}};
        for(int sample=-1;sample<9;sample++)for(unsigned arm=0;arm<3;arm++) {
            unsigned p=arm==1?2u:control;
            id<MTLCommandBuffer> cb=run(queue,f,p,repeats,weights,qb[p],p<2?rope[p]:nil);
            double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
            BOOL complete=cb.status==MTLCommandBufferStatusCompleted;
            BOOL parity=validate_output(f,p,NO,[NSString stringWithFormat:@"timing-c%u-control%u-s%d-arm%u",c,control,sample,arm]);
            BOOL good=complete&&isfinite(us)&&us>0&&parity;timing_ok=timing_ok&&good;
            NSDictionary *r=@{@"kind":@"timing",@"cohort":@"single_fixed_synthetic_ABA",@"case":@(f->index),@"control":@(control),
                @"sample":@(sample),@"warmup":@(sample<0),@"arm":@(arm),@"provider":@(p),@"ordinal":@(command_ordinal),
                @"repetitions":@(repeats),@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),
                @"us_per_qb_rope":@(us),@"completed":@(complete),@"output_gate":@(parity),@"diagnostic_only":@YES};
            [timing_records addObject:r];emit(r);
            if(sample>=0)durations[arm][sample]=us;
            if(!good) {
                (void)save_inputs(weights,weight_data,@"phase2_failure_visible");
                manifest[@"all_timing_records_pass"]=@NO;need(NO,"timing failure raw outputs and visible inputs saved; no cohort retry");
            }
        }
        double a1=median(durations[0]),b=median(durations[1]),a2=median(durations[2]);
        emit(@{@"kind":@"component_summary",@"case":@(f->index),@"control":@(control),@"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),
            @"conservative_saving_percent":@(100.0*(fmin(a1,a2)-b)/fmin(a1,a2)),@"control_drift_percent":@(100.0*(a2-a1)/a1),
            @"adoption_gate_evaluated":@NO,@"full_model_evidence":@NO});
    }
    BOOL end_inputs=save_inputs(weights,weight_data,@"phase2_post");timing_ok=timing_ok&&end_inputs;
    manifest[@"all_timing_records_pass"]=@(timing_ok);need(timing_ok,"phase2 visible inputs saved before refusal");
    finish(@"synthetic_component_screen_complete_not_adoption");return 0;
}}
