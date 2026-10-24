/* One-query private multihead stage observer. Diagnostic only, never a benchmark.
 * --input-dir FAILED_COMPONENT --case 0 --production-metallib PATH
 * --probe-metallib FORMAL_PATH --debug-metallib DEBUG_PATH --out-dir NEW
 * [--scan-metallib POLICY_PATH --scan-symbols CSV]
 * Five providers: production, formal native, formal H2, debug-native, debug-H2.
 * Observer parity is required before dumped stage differences explain release.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
enum { GUARD=64, QUERY_BYTES=131072, KV_BYTES=655360,
       RECORD_FLOATS=1124, HEAD_FLOATS=22996, DEBUG_BYTES=5886976 };
struct flash_args {
    int32_t ne01,ne02,ne03;uint64_t nb01,nb02,nb03;int32_t ne11,ne_12_2,ne_12_3,ns10;
    uint64_t nb11,nb12,nb13;int32_t ns20;uint64_t nb21,nb22,nb23;
    int32_t ne31,ne32,ne33;uint64_t nb31,nb32,nb33;int32_t ne1,ne2,ne3;
    float scale,max_bias,m0,m1;int32_t n_head_log2;float logit_softcap;uint64_t nb_kvq;
};
_Static_assert(sizeof(struct flash_args)==200,"attention ABI");
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
static NSData *archived(NSDictionary *prior,NSString *base,NSString *name) {
    NSDictionary *record=nil;
    for(NSDictionary *r in prior[@"files"])if([r[@"path"]isEqual:name]){
        need(!record,"unique archived logical payload");record=r;
    }
    need(record!=nil,"archived logical payload descriptor missing");
    NSData *d=read_file([base stringByAppendingPathComponent:name]);
    need(d.length==json_uint(record[@"bytes"],SIZE_MAX,"archived payload bytes"),"archived payload size");
    need([sha(d.bytes,d.length)isEqual:record[@"sha256"]],"archived payload SHA");return d;
}
static BOOL finite_f16(const void *p,size_t count) {
    const uint16_t *v=p;for(size_t i=0;i<count;i++)if((v[i]&0x7c00u)==0x7c00u)return NO;return YES;
}
static BOOL valid_args(const struct flash_args *a,unsigned rows) {
    return rows>=1&&rows<=2048&&a->ne01==(int32_t)rows&&a->ne02==64&&a->ne03==1&&
        a->nb01==QUERY_BYTES&&a->nb02==2048&&a->nb03==QUERY_BYTES&&a->ne11==640&&
        a->ne_12_2==1&&a->ne_12_3==1&&a->ns10==512&&a->nb11==1024&&
        a->nb12==KV_BYTES&&a->nb13==KV_BYTES&&a->ns20==512&&a->nb21==1024&&
        a->nb22==KV_BYTES&&a->nb23==KV_BYTES&&a->ne31==(int32_t)rows&&a->ne32==1&&a->ne33==1&&
        a->nb31>=1280&&a->nb31<=UINT32_MAX&&a->nb31%2==0&&
        a->nb32==a->nb31*rows&&a->nb33==a->nb31*rows&&a->ne1==64&&a->ne2==(int32_t)rows&&a->ne3==1&&
        a->scale==1.f/sqrtf(512.f)&&a->max_bias==0&&a->m0==0&&a->m1==0&&a->n_head_log2==0&&
        a->logit_softcap==0&&a->nb_kvq==KV_BYTES;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> lib,NSString *name,unsigned batch,unsigned smem) {
    const unsigned ix[10]={400,401,402,403,404,405,420,421,422,423};
    const int32_t val[10]={1,1,0,0,0,0,512,512,1,1};
    MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
    for(unsigned i=0;i<10;i++){bool b=val[i]!=0;int32_t v=val[i];
        [constants setConstantValue:i<6?(const void*)&b:(const void*)&v type:i<6?MTLDataTypeBool:MTLDataTypeInt atIndex:ix[i]];}
    NSError *error=nil;id<MTLFunction> f=[lib newFunctionWithName:name constantValues:constants error:&error];
    if(!f)fprintf(stderr,"function %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(f!=nil,"diagnostic selected function unavailable; no fallback");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];
    if(!p)fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=32*batch,"diagnostic pipeline SIMD/threads");
    need(p.staticThreadgroupMemoryLength+smem<=d.maxThreadgroupMemoryLength,"diagnostic scratch limit");
    return p;
}
typedef struct {struct flash_args args;id<MTLBuffer> q,kv,mask,sinks,pad;} inputs;
static void run(id<MTLCommandQueue> queue,inputs *f,id<MTLComputePipelineState> p,unsigned batch,
                unsigned smem,id<MTLBuffer> out,id<MTLBuffer> trace) {
    id<MTLBuffer> readers[5]={f->q,f->kv,f->mask,f->sinks,f->pad};
    need(out&&out.length==QUERY_BYTES+2*GUARD,"diagnostic output shape");
    for(unsigned i=0;i<5;i++)need(readers[i]&&readers[i]!=out&&readers[i]!=trace,"diagnostic buffer alias rejected");
    need(!trace||(trace!=out&&trace.length==DEBUG_BYTES+2*GUARD),"diagnostic trace shape/alias");
    memset(body(out),0xa5,QUERY_BYTES);if(trace)memset(body(trace),0xa5,DEBUG_BYTES);
    id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"diagnostic command buffer");
    MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"diagnostic concurrent encoder");
    [e setComputePipelineState:p];[e setBytes:&f->args length:200 atIndex:0];
    id<MTLBuffer> b[7]={f->q,f->kv,f->kv,f->mask,f->sinks,f->pad,out};
    for(unsigned i=0;i<7;i++)[e setBuffer:b[i] offset:GUARD atIndex:i+1];
    if(trace)[e setBuffer:trace offset:GUARD atIndex:8];
    [e setThreadgroupMemoryLength:smem atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(64/batch,1,1) threadsPerThreadgroup:MTLSizeMake(32,batch,1)];
    id<MTLResource> join[2]={out,trace};[e memoryBarrierWithResources:join count:trace?2:1];
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU error: %s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"diagnostic command completion");
    need(written(body(out),QUERY_BYTES)&&finite_f32(body(out),QUERY_BYTES)&&guards(out,QUERY_BYTES),"diagnostic output finite/written/guards");
    need(guards(f->q,QUERY_BYTES)&&guards(f->kv,KV_BYTES)&&guards(f->mask,f->args.nb31)&&
         guards(f->sinks,256)&&guards(f->pad,16),"diagnostic input guards");
    if(trace)need(written(body(trace),DEBUG_BYTES)&&guards(trace,DEBUG_BYTES),"diagnostic stage coverage/guards");
}
static NSDictionary *comparison(NSString *kind,const void *ref,const void *candidate,size_t bytes) {
    size_t first=SIZE_MAX,bad=mismatches(ref,candidate,bytes,&first);
    NSMutableDictionary *r=[@{@"kind":kind,@"mismatches":@(bad),@"full_f32_bit_exact":@(!bad),@"checked_f32":@(bytes/4)}mutableCopy];
    if(bad){r[@"first_index"]=@(first);r[@"reference_u32"]=@(((const uint32_t*)ref)[first]);r[@"candidate_u32"]=@(((const uint32_t*)candidate)[first]);}
    return r;
}
static void stages(id<MTLBuffer> reference,id<MTLBuffer> candidate,BOOL unperturbed) {
    const uint32_t *r=body(reference),*c=body(candidate);
    const unsigned offsets[6]={0,32,64,96,100,612},lengths[6]={32,32,32,4,512,512};
    NSArray *names=@[@"mqk",@"score",@"softmax",@"stats_M_S_ms_sum",@"PV_lo",@"acc"];
    for(unsigned head=0;head<64;head++){
        NSMutableArray *head_records=[NSMutableArray new];
        for(unsigned stage=0;stage<8;stage++){
            size_t total=0,first_index=SIZE_MAX;unsigned first_block=UINT32_MAX;uint32_t rv=0,cv=0;
            unsigned blocks=stage<6?20:1,len=stage<6?lengths[stage]:(stage==6?512:4);
            for(unsigned block=0;block<blocks;block++){
                size_t offset=(size_t)head*HEAD_FLOATS+(stage<6?block*RECORD_FLOATS+offsets[stage]:(stage==6?22480:22992));
                size_t first,bad=mismatches(r+offset,c+offset,(size_t)len*4,&first);total+=bad;
                if(bad&&first_index==SIZE_MAX){first_index=first;first_block=block;rv=r[offset+first];cv=c[offset+first];}
            }
            NSMutableDictionary *item=[@{@"stage":stage<6?names[stage]:(stage==6?@"sink_acc":@"sink_stats"),@"mismatches":@(total),@"checked_f32":@(blocks*len)}mutableCopy];
            if(total){item[@"first_block"]=@(first_block);item[@"first_index"]=@(first_index);item[@"reference_u32"]=@(rv);item[@"candidate_u32"]=@(cv);}
            [head_records addObject:item];
        }
        emit(@{@"kind":@"stage_comparison",@"head":@(head),@"observer_unperturbed":@(unperturbed),@"stages":head_records});
    }
}
int main(int argc,const char **argv) {@autoreleasepool {
    NSString *input=nil,*production=nil,*probe=nil,*debug=nil,*scan=nil,*scan_csv=nil;unsigned selected=0;BOOL host=NO;
    for(int i=1;i<argc;i++){
        NSString *arg=@(argv[i]);if([arg isEqual:@"--host-only"]){host=YES;continue;}
        need(i+1<argc,"option requires a value");NSString *value=@(argv[++i]);
        if([arg isEqual:@"--input-dir"])input=canonical(value);
        else if([arg isEqual:@"--case"])selected=number(value.UTF8String,0,15);
        else if([arg isEqual:@"--production-metallib"])production=canonical(value);
        else if([arg isEqual:@"--probe-metallib"])probe=canonical(value);
        else if([arg isEqual:@"--debug-metallib"])debug=canonical(value);
        else if([arg isEqual:@"--scan-metallib"])scan=canonical(value);
        else if([arg isEqual:@"--scan-symbols"])scan_csv=value;
        else if([arg isEqual:@"--out-dir"])output=canonical(value);
        else need(NO,"unknown diagnostic option");
    }
    need(input&&production&&probe&&debug&&output,"required input/three libraries/output");
    need((scan!=nil)==(scan_csv!=nil),"scan library and CSV supplied together");
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new diagnostic output directory required");
    need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"diagnostic output mkdir");
    files=[NSMutableArray new];stored_payloads=[NSMutableDictionary new];
    manifest=[@{@"schema_version":@1,@"diagnostic_only":@YES,@"performance_evidence":@NO,@"query_index":@0,
        @"source_component":input,@"source_case":@(selected),@"argv":[NSProcessInfo processInfo].arguments,
        @"binary_sha256":file_sha(binary_path()),@"production_metallib_sha256":file_sha(production),
        @"probe_metallib_sha256":file_sha(probe),@"debug_metallib_sha256":file_sha(debug)}mutableCopy];
    NSData *prior_data=read_file([input stringByAppendingPathComponent:@"artifact-manifest.json"]);
    NSDictionary *prior=[NSJSONSerialization JSONObjectWithData:prior_data options:0 error:nil];
    need([prior isKindOfClass:[NSDictionary class]]&&[prior[@"files"]isKindOfClass:[NSArray class]],"source component manifest");
    manifest[@"source_manifest_sha256"]=sha(prior_data.bytes,prior_data.length);
    raw(@"source-component-manifest.json",prior_data.bytes,prior_data.length);
    NSString *(^name)(NSString*)=^NSString *(NSString *suffix){return[NSString stringWithFormat:@"capture-case-%03u-%@",selected,suffix];};
    NSData *argdata=archived(prior,input,name(@"args200"));need(argdata.length==200,"original args200 length");
    struct flash_args a;memcpy(&a,argdata.bytes,200);unsigned rows=(unsigned)a.ne01;
    need(valid_args(&a,rows),"original real gathered args");
    NSData *q=archived(prior,input,name(@"q.f32")),*unique=archived(prior,input,name(@"kv-unique.f16")),
        *ids=archived(prior,input,name(@"kv-rows.u32")),*mask=archived(prior,input,name(@"mask-compact.f16")),
        *sinks=archived(prior,input,name(@"sinks.f32"));
    need(q.length==(size_t)rows*QUERY_BYTES&&unique.length&&unique.length%1024==0&&
        ids.length==(size_t)rows*640*4&&mask.length==(size_t)rows*1280&&sinks.length==256,"original real payload shapes");
    need(finite_f32(q.bytes,QUERY_BYTES)&&finite_f32(sinks.bytes,256)&&finite_f16(unique.bytes,unique.length/2),"finite sliced real inputs");
    NSMutableData *kv=[NSMutableData dataWithLength:KV_BYTES],*padded_mask=[NSMutableData dataWithLength:a.nb31];
    const uint32_t *map=ids.bytes;unsigned visible=0;const uint16_t *compact=mask.bytes;uint16_t *padded=padded_mask.mutableBytes;
    for(size_t i=0;i<a.nb31/2;i++)padded[i]=0xfc00u;
    for(unsigned k=0;k<640;k++){
        need(map[k]<unique.length/1024,"first-query KV dictionary ID bounds");
        memcpy((uint8_t*)kv.mutableBytes+k*1024,(const uint8_t*)unique.bytes+(size_t)map[k]*1024,1024);
        need(compact[k]==0||compact[k]==0xfc00u,"first-query real mask encoding");padded[k]=compact[k];visible+=compact[k]==0;
    }
    NSData *original_native=archived(prior,input,[NSString stringWithFormat:@"parity-case-%03u-provider-0-output.f32",selected]),
        *original_h2=archived(prior,input,[NSString stringWithFormat:@"parity-case-%03u-provider-2-output.f32",selected]);
    need(original_native.length==(size_t)rows*QUERY_BYTES&&original_h2.length==original_native.length,"original full output shape");
    a.ne01=a.ne31=a.ne2=1;a.nb32=a.nb33=a.nb31;need(valid_args(&a,1),"sliced single-query ABI");
    raw(@"slice-args200",&a,200);raw(@"slice-q.f32",q.bytes,QUERY_BYTES);raw(@"slice-kv.f16",kv.bytes,KV_BYTES);
    raw(@"slice-mask-padded.f16",padded_mask.bytes,padded_mask.length);raw(@"slice-sinks.f32",sinks.bytes,256);
    raw(@"original-native-query0.f32",original_native.bytes,QUERY_BYTES);raw(@"original-h2-query0.f32",original_h2.bytes,QUERY_BYTES);
    manifest[@"slice"]=@{@"original_rows":@(rows),@"rows":@1,@"heads":@64,@"keys":@640,@"visible_keys":@(visible),
        @"original_mask_stride":@(a.nb31),@"working_set_changed":@YES,@"performance_evidence":@NO};
    manifest[@"trace_layout"]=@{@"heads":@64,@"blocks":@20,@"block_floats":@(RECORD_FLOATS),@"head_floats":@(HEAD_FLOATS),
        @"offsets":@{@"mqk":@0,@"score":@32,@"softmax":@64,@"stats":@96,@"lo":@100,@"acc":@612},
        @"stats":@[@"M",@"S",@"ms",@"sum_vs"],@"sink_acc_offset":@22480,@"sink_stats_offset":@22992,@"total_bytes":@(DEBUG_BYTES)};
    if(host){finish(@"host_only");emit(@{@"kind":@"complete",@"host_only":@YES,@"metal_device_created":@NO});return 0;}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();need(device!=nil,"diagnostic Metal device");NSError *error=nil;
    NSString *library_paths[3]={production,probe,debug};id<MTLLibrary> libs[3];
    for(unsigned i=0;i<3;i++){libs[i]=[device newLibraryWithURL:[NSURL fileURLWithPath:library_paths[i]]error:&error];need(libs[i]!=nil,"diagnostic library loading");}
    id<MTLCommandQueue> queue=[device newCommandQueue];need(queue!=nil,"diagnostic queue");
    inputs f={0};f.args=a;f.q=alloc(device,QUERY_BYTES,NO);f.kv=alloc(device,KV_BYTES,NO);f.mask=alloc(device,a.nb31,NO);
    f.sinks=alloc(device,256,NO);f.pad=alloc(device,16,NO);
    memcpy(body(f.q),q.bytes,QUERY_BYTES);memcpy(body(f.kv),kv.bytes,KV_BYTES);memcpy(body(f.mask),padded_mask.bytes,a.nb31);
    memcpy(body(f.sinks),sinks.bytes,256);memset(body(f.pad),0,16);
    NSString *symbols[5]={@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",@"kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512",
        @"kernel_myds_prefill_attention_head2_f16_dk512_dv512",@"probe_mh_ref_debug",@"probe_mh_h2_debug"};
    const unsigned library_id[5]={0,1,1,2,2},batch[5]={1,1,2,1,2},smem[5]={3328,3328,23040,3328,23040};
    id<MTLBuffer> outs[5],traces[2];traces[0]=alloc(device,DEBUG_BYTES,YES);traces[1]=alloc(device,DEBUG_BYTES,YES);
    NSMutableArray *provider_identity=[NSMutableArray new];
    for(unsigned p=0;p<5;p++){
        id<MTLComputePipelineState> state=pipeline(device,libs[library_id[p]],symbols[p],batch[p],smem[p]);outs[p]=alloc(device,QUERY_BYTES,YES);
        run(queue,&f,state,batch[p],smem[p],outs[p],p>=3?traces[p-3]:nil);
        dump([NSString stringWithFormat:@"provider-%u-output",p],outs[p],QUERY_BYTES);
        if(p>=3)dump([NSString stringWithFormat:@"provider-%u-stages",p],traces[p-3],DEBUG_BYTES);
        NSDictionary *ident=@{@"provider":@(p),@"symbol":symbols[p],@"library_sha256":manifest[@[@"production_metallib_sha256",@"probe_metallib_sha256",@"debug_metallib_sha256"][library_id[p]]],
            @"threads":@[@32,@(batch[p]),@1],@"groups":@[@(64/batch[p]),@1,@1],@"threadgroup_memory":@(smem[p]),
            @"static_threadgroup_memory":@(state.staticThreadgroupMemoryLength),@"max_threads":@(state.maxTotalThreadsPerThreadgroup),@"finite_written_guards":@YES};
        [provider_identity addObject:ident];emit(@{@"kind":@"provider",@"identity":ident});
    }
    manifest[@"providers"]=provider_identity;
    NSDictionary *same=comparison(@"sameprobe_vs_production",body(outs[0]),body(outs[1]),QUERY_BYTES),
        *slice_native=comparison(@"slice_native_vs_full_working_set_query0",original_native.bytes,body(outs[0]),QUERY_BYTES),
        *slice_h2=comparison(@"slice_h2_vs_full_working_set_query0",original_h2.bytes,body(outs[2]),QUERY_BYTES),
        *obs_native=comparison(@"debug_native_vs_native",body(outs[0]),body(outs[3]),QUERY_BYTES),
        *obs_h2=comparison(@"debug_h2_vs_release_h2",body(outs[2]),body(outs[4]),QUERY_BYTES),
        *candidate=comparison(@"release_h2_vs_native",body(outs[0]),body(outs[2]),QUERY_BYTES);
    NSArray *comparisons=@[same,slice_native,slice_h2,obs_native,obs_h2,candidate];for(NSDictionary *r in comparisons)emit(r);
    BOOL sliced=[same[@"full_f32_bit_exact"]boolValue]&&[slice_native[@"full_f32_bit_exact"]boolValue]&&[slice_h2[@"full_f32_bit_exact"]boolValue];
    BOOL unperturbed=sliced&&[obs_native[@"full_f32_bit_exact"]boolValue]&&[obs_h2[@"full_f32_bit_exact"]boolValue];
    manifest[@"comparisons"]=comparisons;manifest[@"slice_matches_full_working_set"]=@(sliced);manifest[@"observer_unperturbed"]=@(unperturbed);
    stages(traces[0],traces[1],unperturbed);
    if(scan){
        manifest[@"scan_metallib_sha256"]=file_sha(scan);id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:scan]error:&error];need(lib!=nil,"policy scan library");
        NSArray *names=[scan_csv componentsSeparatedByString:@","];NSMutableArray *results=[NSMutableArray new];NSMutableSet *seen=[NSMutableSet new];
        need(names.count>0&&names.count<=32,"policy scan symbol count");
        for(unsigned i=0;i<names.count;i++){
            NSString *symbol=names[i];need(symbol.length&&![seen containsObject:symbol],"unique nonempty policy scan symbol");[seen addObject:symbol];
            id<MTLComputePipelineState> state=pipeline(device,lib,symbol,2,23040);id<MTLBuffer> out=alloc(device,QUERY_BYTES,YES);
            run(queue,&f,state,2,23040,out,nil);dump([NSString stringWithFormat:@"scan-%02u-output",i],out,QUERY_BYTES);
            NSMutableDictionary *r=[comparison(@"policy_scan",body(outs[0]),body(out),QUERY_BYTES)mutableCopy];
            r[@"index"]=@(i);r[@"symbol"]=symbol;r[@"slice_matches_full_working_set"]=@(sliced);r[@"finite_written_guards"]=@YES;
            r[@"threadgroup_memory"]=@23040;r[@"static_threadgroup_memory"]=@(state.staticThreadgroupMemoryLength);
            [results addObject:r];emit(r);
        }
        manifest[@"scan"]=results;
    }
    need([sha(body(f.q),QUERY_BYTES)isEqual:sha(q.bytes,QUERY_BYTES)]&&[sha(body(f.kv),KV_BYTES)isEqual:sha(kv.bytes,KV_BYTES)]&&
         [sha(body(f.mask),a.nb31)isEqual:sha(padded_mask.bytes,padded_mask.length)]&&[sha(body(f.sinks),256)isEqual:sha(sinks.bytes,256)],"diagnostic real inputs unchanged");
    finish(@"complete");emit(@{@"kind":@"complete",@"diagnostic_only":@YES,@"performance_evidence":@NO,@"providers":@5,
        @"slice_matches_full_working_set":@(sliced),@"observer_unperturbed":@(unperturbed),@"root_cause_proven":@NO});
    return 0;
}}
