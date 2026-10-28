/* Correctness-only register-resident MMA experiment.  No runtime, model,
 * timing, warmup, fallback, or production registration is involved.  Full
 * production-TU native / probe libraries and an external deployed library are
 * mandatory.  Every provider has its own guarded output and one command buffer.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { GUARD = 64, PROVIDERS = 6 };
struct index_args {
    uint32_t n_comp,n_tokens,n_head,head_dim,pos0,ratio;
    uint64_t q_token_stride,q_head_stride,weights_token_stride,index_row_stride,score_token_stride;
    float scale;
};
struct exact_args { uint32_t rows,heads,dim,cache_type; float scale; uint32_t tokens,pos,ratio,window; };
_Static_assert(sizeof(struct index_args)==72,"DSV4 indexer ABI72");
_Static_assert(__builtin_offsetof(struct index_args,q_token_stride)==24,"DSV4 strides offset");
_Static_assert(__builtin_offsetof(struct index_args,scale)==64,"DSV4 scale offset");
_Static_assert(sizeof(struct exact_args)==36,"actual exact chunk ABI36");
static const unsigned CASES[][4] = {
    {8,16385,32762,0},{9,16385,32761,1},{15,32767,65519,2},
    {31,32768,65505,0},{63,32769,65475,1},{31,65535,131039,2},
    {32,65536,131040,0},{9,65536,131063,1}
};
static NSMutableDictionary *manifest;
static NSMutableArray *files;
static NSMutableDictionary *aliases;
static NSString *output;
static void finish(NSString *status) {
    if(!manifest||!output)return;
    manifest[@"status"]=status;manifest[@"files"]=files;manifest[@"buffer_storage"]=@"sha256_only";
    NSData *d=[NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    if(d)[d writeToFile:[output stringByAppendingPathComponent:@"artifact-manifest.json"] atomically:YES];
}
static void need(BOOL ok,const char *s) {
    if(ok)return;fprintf(stderr,"FAIL: %s\n",s);
    if(manifest)manifest[@"failure"]=[NSString stringWithUTF8String:s];finish(@"evidence_failed");exit(1);
}
static NSString *sha(const void *p,size_t n) {
    CC_SHA256_CTX c;CC_SHA256_Init(&c);const uint8_t *v=p;
    while(n){CC_LONG k=(CC_LONG)(n>0x10000000u?0x10000000u:n);CC_SHA256_Update(&c,v,k);v+=k;n-=k;}
    uint8_t b[32];CC_SHA256_Final(b,&c);char s[65];for(unsigned i=0;i<32;i++)snprintf(s+2*i,3,"%02x",b[i]);
    return [NSString stringWithUTF8String:s];
}
static NSString *canon(NSString *s){return [[s stringByStandardizingPath]stringByResolvingSymlinksInPath];}
static NSData *readfile(NSString *p){NSData *d=[NSData dataWithContentsOfFile:p];need(d!=nil,"complete input file read");return d;}
static NSString *filesha(NSString *p){NSData *d=readfile(p);return sha(d.bytes,d.length);}
static NSString *binary_path(void){uint32_t n=0;_NSGetExecutablePath(NULL,&n);char *b=calloc(n,1);need(b!=NULL,"binary path allocation");need(!_NSGetExecutablePath(b,&n),"binary path");NSString *p=canon([NSString stringWithUTF8String:b]);free(b);return p;}
static void emit(NSDictionary *r){NSData *d=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingSortedKeys error:nil];need(d!=nil,"JSON encode");need(fwrite(d.bytes,1,d.length,stdout)==d.length,"stdout complete write");putchar('\n');fflush(stdout);}
static NSDictionary *raw(NSString *name,const void *p,size_t bytes) {
    return @{@"name":name,@"bytes":@(bytes),@"sha256":sha(p,bytes),@"stored":@NO};
}
static NSDictionary *rawdata(NSString *name,NSData *d){return raw(name,d.bytes,d.length);}
static NSDictionary *guarded_raw(NSString *stem,const void *full,size_t bytes) {
    const uint8_t *b=full;return @{@"body":raw([stem stringByAppendingString:@".body.raw"],b+GUARD,bytes),
      @"prefix_guard":raw([stem stringByAppendingString:@".prefix-guard.raw"],b,GUARD),
      @"suffix_guard":raw([stem stringByAppendingString:@".suffix-guard.raw"],b+GUARD+bytes,GUARD)};
}
static BOOL guards(const void *full,size_t n){const uint8_t *b=full;for(unsigned i=0;i<GUARD;i++)if(b[i]!=0xa5||b[GUARD+n+i]!=0xa5)return NO;return YES;}
static uint32_t word(uint64_t i,uint32_t seed){uint32_t x=(uint32_t)(i+1u)*747796405u+seed*2891336453u;x=((x>>((x>>28)+4))^x)*277803737u;return(x>>22)^x;}
static float value(uint64_t i,uint32_t seed){float v=((int32_t)(word(i,seed)%8191u)-4095)/4096.f;uint32_t b;memcpy(&b,&v,4);b&=0xffff0000u;memcpy(&v,&b,4);return v;}
static float input_value(uint64_t i,uint32_t seed,unsigned profile){
    if(!profile)return value(i,seed);
    float v=((int32_t)(word(i,seed)%8191u)-4095)/4096.f;
    uint32_t b;if(profile==1&&i%193u==0){b=(i/193u)%2u?0x80000000u:0u;memcpy(&v,&b,4);}
    if(profile==2&&i%127u==0){const uint32_t special[]={0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,0x3f800001u,0xbf800001u};b=special[(i/127u)%10u];memcpy(&v,&b,4);}
    return v;
}
static NSMutableData *poison(size_t bytes){NSMutableData *d=[NSMutableData dataWithLength:bytes+2*GUARD];memset(d.mutableBytes,0xa5,d.length);return d;}
static BOOL scope(const struct index_args *a){
    if(a->n_head!=32||a->head_dim!=128||a->ratio!=2||a->n_tokens<8||a->n_tokens>2048||a->n_comp<=512||a->n_comp>65536||(uint64_t)a->pos0+a->n_tokens>131072||a->scale!=1.f/64.f)return NO;
    if(a->q_head_stride<512||a->q_head_stride%4||a->q_head_stride>(UINT64_MAX-512)/31||a->q_token_stride<31*a->q_head_stride+512||a->q_token_stride%4||a->weights_token_stride<128||a->weights_token_stride%4||a->index_row_stride<256||a->index_row_stride%2||a->score_token_stride<(uint64_t)a->n_comp*4||a->score_token_stride%4)return NO;
    return a->q_token_stride<=(SIZE_MAX-2*GUARD)/a->n_tokens&&a->weights_token_stride<=(SIZE_MAX-2*GUARD)/a->n_tokens&&a->index_row_stride<=(SIZE_MAX-2*GUARD)/a->n_comp&&a->score_token_stride<=(SIZE_MAX-2*GUARD)/a->n_tokens;
}
static NSDictionary *args_json(const struct index_args *a){return @{@"n_comp":@(a->n_comp),@"n_tokens":@(a->n_tokens),@"n_head":@(a->n_head),@"head_dim":@(a->head_dim),@"pos0":@(a->pos0),@"ratio":@(a->ratio),@"q_token_stride":@(a->q_token_stride),@"q_head_stride":@(a->q_head_stride),@"weights_token_stride":@(a->weights_token_stride),@"index_row_stride":@(a->index_row_stride),@"score_token_stride":@(a->score_token_stride),@"scale":@(a->scale)};}
static size_t diff(const void *a,const void *b,size_t n){const uint32_t *x=a,*y=b;size_t bad=0;for(size_t i=0;i<n/4;i++)bad+=x[i]!=y[i];return bad;}
static id<MTLComputePipelineState> pso(id<MTLDevice> d,id<MTLLibrary> lib,NSString *name,unsigned tg){id<MTLFunction> f=[lib newFunctionWithName:name];need(f!=nil,"explicit selected PSO exists, no fallback");NSError *error=nil;id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error];if(!p)fprintf(stderr,"PSO error: %s\n",error.localizedDescription.UTF8String);need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&p.staticThreadgroupMemoryLength+tg<=d.maxThreadgroupMemoryLength,"PSO SIMD/memory contract");if(!tg)need(p.staticThreadgroupMemoryLength==0,"register-only candidate has zero TG memory");return p;}
/* Linked into the isolated long-indexer fixture. This invokes the actual
 * production causal Top512 provider on both score tensors. */
#include "myds_metal.h"
#include "myds_v41_gpu.h"

static myds_v41_gpu_scratch_t *topk_scratch;

static void init_topk(NSString *deployed_metallib) {
    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE", "0", 1);
    setenv("MYDS_INDEXER_BATCH_FLOATS", "2097152", 1);
    setenv("MYDS_CHUNK_KEY_MAX", "65536", 1);
    need(myds_metal_init(deployed_metallib.stringByDeletingLastPathComponent.UTF8String),
         "production Metal runtime initialization for topk");
    topk_scratch=myds_v41_gpu_scratch_create(65536u);
    need(topk_scratch!=NULL,"production topk scratch allocation");
}

static NSDictionary *topk_against_exact(NSData *exact,NSData *candidate,
                                        uint32_t nc,uint32_t nt,uint32_t start) {
    const uint32_t top=512u;
    size_t score_bytes=(size_t)nc*nt*4u, selected_bytes=(size_t)top*nt*4u;
    need(nc>top && (uint64_t)nc*nt<=2097152u && exact.length==score_bytes &&
         candidate.length==score_bytes,"topk production batch and input size");
    myds_metal_buffer_t *scores[3],*roots[3],*views[3];
    NSData *inputs[]={exact,exact,candidate};
    myds_v41_gpu_topk_prefix_counters(NULL,true);
    myds_v41_gpu_topk_nan_fuse_counters(NULL,true);
    for(unsigned arm=0;arm<3;arm++) {
        scores[arm]=myds_metal_buffer_alloc(score_bytes);
        roots[arm]=myds_metal_buffer_alloc(selected_bytes+2u*GUARD);
        need(scores[arm]&&roots[arm],"topk buffers");
        memset(myds_metal_buffer_data(roots[arm]),0xa5,selected_bytes+2u*GUARD);
        views[arm]=myds_metal_buffer_sub_view(roots[arm],GUARD,selected_bytes);
        need(views[arm]&&myds_metal_buffer_write(scores[arm],0,inputs[arm].bytes,score_bytes),
             "topk distinct score/output views");
        need(myds_metal_begin_batch()&&
             myds_v41_gpu_topk_causal(views[arm],scores[arm],nc,top,nt,start,2u,topk_scratch)&&
             myds_metal_commit_batch(),"production topk GPU completion");
    }
    myds_v41_gpu_topk_prefix_counters_t pc;
    myds_v41_gpu_topk_nan_fuse_counters_t fc;
    myds_v41_gpu_topk_prefix_counters(&pc,false);
    myds_v41_gpu_topk_nan_fuse_counters(&fc,false);
    need(pc.requests==3&&pc.encoded==3&&!pc.unavailable&&fc.requests==3&&fc.encoded==3&&!fc.unavailable,
         "production prefix/nan-fuse topk paths encoded for all three arms");
    BOOL valid=YES;
    for(unsigned arm=0;arm<3;arm++) {
        const uint8_t *raw=myds_metal_buffer_data(roots[arm]);
        valid&=guards(raw,selected_bytes);
        valid&=!memcmp(myds_metal_buffer_data(scores[arm]),inputs[arm].bytes,score_bytes);
        const int32_t *ids=(const int32_t *)(raw+GUARD);
        uint8_t *seen=malloc(nc);need(seen!=NULL,"topk distinct-id scratch");
        for(uint32_t row=0;row<nt;row++) {
            memset(seen,0,nc);
            uint32_t visible=(start+row+1u)/2u;if(visible>nc)visible=nc;
            for(uint32_t i=0;i<top;i++) {
                int32_t id=ids[(size_t)row*top+i];
                if(id<0||(uint32_t)id>=visible||seen[id]) {valid=NO;continue;}
                seen[id]=1;
            }
        }
        free(seen);
    }
    const int32_t *x=(const int32_t *)myds_metal_buffer_data(views[0]);
    const int32_t *repeat=(const int32_t *)myds_metal_buffer_data(views[1]);
    const int32_t *y=(const int32_t *)myds_metal_buffer_data(views[2]);
    size_t mismatches=0,aa_mismatches=0,first=SIZE_MAX,aa_first=SIZE_MAX;
    for(size_t i=0;i<(size_t)nt*top;i++) {
        if(x[i]!=repeat[i]) {if(aa_first==SIZE_MAX)aa_first=i;aa_mismatches++;}
        if(x[i]!=y[i]) {if(first==SIZE_MAX)first=i;mismatches++;}
    }
    NSDictionary *result=@{@"production_topk":@YES,@"prefix_encoded":@(pc.encoded),
        @"nan_fuse_encoded":@(fc.encoded),@"gpu_batches_completed":@3,@"guards_and_causal_ids_valid":@(valid),
        @"exact_aa_ordered_id_mismatches":@(aa_mismatches),
        @"exact_aa_first_mismatch_index":aa_first==SIZE_MAX?(id)[NSNull null]:@(aa_first),
        @"ordered_id_mismatches":@(mismatches),@"first_mismatch_index":first==SIZE_MAX?(id)[NSNull null]:@(first),
        @"first_exact_id":first==SIZE_MAX?(id)[NSNull null]:@(x[first]),
        @"first_candidate_id":first==SIZE_MAX?(id)[NSNull null]:@(y[first]),
        @"exact_top512_sha256":sha(x,selected_bytes),@"exact_repeat_top512_sha256":sha(repeat,selected_bytes),
        @"candidate_top512_sha256":sha(y,selected_bytes)};
    for(unsigned arm=0;arm<3;arm++){myds_metal_buffer_free(views[arm]);myds_metal_buffer_free(roots[arm]);myds_metal_buffer_free(scores[arm]);}
    return result;
}

int main(int argc,char **argv){@autoreleasepool {
    if(argc==2&&!strcmp(argv[1],"--scope-self-test")){
        struct index_args a={.n_comp=513,.n_tokens=8,.n_head=32,.head_dim=128,.pos0=0,.ratio=2,.q_token_stride=16384,.q_head_stride=512,.weights_token_stride=128,.index_row_stride=256,.score_token_stride=2052,.scale=1.f/64.f};
        need(scope(&a),"scope positive");unsigned tests=1;struct index_args b;
#define REJECT(field,value) b=a;b.field=value;need(!scope(&b),"out-of-scope CPU rejection");tests++;
        REJECT(n_tokens,1);REJECT(n_tokens,7);REJECT(n_tokens,2049);REJECT(n_comp,512);REJECT(n_comp,65537);REJECT(ratio,1);REJECT(n_head,31);REJECT(head_dim,64);REJECT(pos0,131065);REJECT(q_head_stride,508);REJECT(q_token_stride,16380);REJECT(index_row_stride,254);REJECT(q_token_stride,16385);REJECT(weights_token_stride,129);REJECT(index_row_stride,257);REJECT(score_token_stride,2053);REJECT(q_head_stride,UINT64_MAX-3);REJECT(q_token_stride,UINT64_MAX-3);
#undef REJECT
        emit(@{@"kind":@"scope_self_test",@"tests":@(tests),@"gpu_command_buffers":@0,@"passed":@YES});return 0;
    }
    NSMutableDictionary *options=[NSMutableDictionary new];BOOL host=NO;
    for(int i=1;i<argc;i++){NSString *k=[NSString stringWithUTF8String:argv[i]];if([k isEqual:@"--host-only"]){need(!host,"single host-only flag");host=YES;continue;}need(i+1<argc,"option value");need([@[@"--native-metallib",@"--probe-metallib",@"--deployed-metallib",@"--build-identity",@"--harness-source",@"--out-dir"]containsObject:k]&&!options[k],"unique known option");NSString *p=[NSString stringWithUTF8String:argv[++i]];need(p.isAbsolutePath,"absolute paths required");options[k]=canon(p);}
    need(options.count==6,"all six library/identity/source/output options required");output=options[@"--out-dir"];
    need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new output directory");need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil],"output creation");files=[NSMutableArray new];aliases=[NSMutableDictionary new];
    NSString *paths[]={options[@"--deployed-metallib"],options[@"--native-metallib"],options[@"--probe-metallib"]};
    NSString *labels[]={@"deployed",@"native_full_tu",@"probe_full_tu"};NSMutableArray *libraries=[NSMutableArray new];
    for(unsigned i=0;i<3;i++){NSData *d=readfile(paths[i]);[libraries addObject:@{@"index":@(i),@"label":labels[i],@"path":paths[i],@"bytes":@(d.length),@"sha256":sha(d.bytes,d.length)}];}
    NSData *identity=readfile(options[@"--build-identity"]),*source=readfile(options[@"--harness-source"]);NSString *exe=binary_path();
    manifest=[@{@"schema_version":@1,@"probe":@"indexer_register_only_correctness",@"fixture_kind":@"deterministic_synthetic",@"correctness_only":@YES,@"timing_collected":@NO,@"fullmodel":@NO,@"production_registered":@NO,@"adoption_eligible":@NO,@"host_only":@(host),@"guard_bytes_per_side":@(GUARD),@"poison_byte":@165,@"provider_count":@(PROVIDERS),@"case_count":@(sizeof(CASES)/sizeof(CASES[0])),@"libraries":libraries,@"cases":[NSMutableArray new],@"binary_path":exe,@"binary_sha256":filesha(exe),@"harness_source_path":options[@"--harness-source"],@"harness_source_sha256":sha(source.bytes,source.length),@"build_identity_path":options[@"--build-identity"],@"build_identity_sha256":sha(identity.bytes,identity.length)}mutableCopy];
    manifest[@"build_identity_copy"]=rawdata(@"build-identity.json",identity);manifest[@"harness_source_copy"]=rawdata(@"harness-source.m",source);
    NSString *names[]={@"kernel_dsv4_indexer_scores_tiled_bf16",@"kernel_dsv4_indexer_scores_tiled_bf16",@"kernel_dsv4_indexer_scores_tiled_bf16",@"kernel_myds_indexer_scores_register_only_bf16",@"kernel_myds_indexer_score_chunk_exact",@"kernel_myds_indexer_score_chunk_exact"};
    unsigned lib_index[]={0,1,2,2,0,1},tg[]={21504,21504,21504,0,5120,5120};NSMutableArray *bindings=[NSMutableArray new];
    id<MTLDevice> device=nil;id<MTLCommandQueue> queue=nil;id<MTLLibrary> libs[3];id<MTLComputePipelineState> psos[PROVIDERS];
    if(!host){device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device");queue=[device newCommandQueue];need(queue!=nil,"Metal command queue");manifest[@"device"]=device.name;
        for(unsigned i=0;i<3;i++){NSError *e=nil;libs[i]=[device newLibraryWithURL:[NSURL fileURLWithPath:paths[i]] error:&e];need(libs[i]!=nil,"explicit Metal library load");emit(@{@"kind":@"loaded_library",@"index":@(i),@"label":labels[i],@"path":paths[i],@"sha256":libraries[i][@"sha256"],@"bytes":libraries[i][@"bytes"]});}
    }
    for(unsigned p=0;p<PROVIDERS;p++){NSMutableDictionary *b=[@{@"provider":@(p),@"symbol":names[p],@"library_index":@(lib_index[p]),@"library_path":paths[lib_index[p]],@"library_sha256":libraries[lib_index[p]][@"sha256"],@"args_bytes":@(p<4?72:36),@"dynamic_threadgroup_bytes":@(tg[p]),@"threads":@[@32,@4,@1],@"input_layout":p<4?@"strided":@"dense",@"oracle_role":p<4?@"tiled_bit_exact_gate":@"actual_exact_identity_bridge_not_candidate_oracle"}mutableCopy];if(!host){psos[p]=pso(device,libs[lib_index[p]],names[p],tg[p]);b[@"static_threadgroup_bytes"]=@(psos[p].staticThreadgroupMemoryLength);b[@"thread_execution_width"]=@(psos[p].threadExecutionWidth);b[@"max_threads"]=@(psos[p].maxTotalThreadsPerThreadgroup);emit([@{@"kind":@"pipeline",@"binding":b}copy]);}[bindings addObject:b];}manifest[@"bindings"]=bindings;
    if(!host)init_topk(options[@"--deployed-metallib"]);
    unsigned completed=0;BOOL all=YES;NSString *roles[]={@"q",@"weights",@"cache"};
    for(unsigned ci=0;ci<sizeof(CASES)/sizeof(CASES[0]);ci++){@autoreleasepool {
        unsigned nt=CASES[ci][0],nk=CASES[ci][1],pos=CASES[ci][2],pad=CASES[ci][3],profile=ci%3;
        NSString *data_profiles[]={@"bf16_representable_f32",@"full_f32_signed_zero",@"full_f32_signed_zero_subnormal"};
        struct index_args a={0};a.n_comp=nk;a.n_tokens=nt;a.n_head=32;a.head_dim=128;a.pos0=pos;a.ratio=2;a.q_head_stride=512+(pad==2?16:0);a.q_token_stride=32*a.q_head_stride+(pad?16:0);a.weights_token_stride=128+(pad?16:0);a.index_row_stride=256+(pad?16:0);a.score_token_stride=(uint64_t)(nk+(pad?7:0))*4;a.scale=1.f/64.f;
        need(scope(&a),"frozen candidate scope");struct exact_args x={nk,32,128,2,1.f/64.f,nt,pos,2,0};
        size_t sizes[2][3]={{(size_t)nt*a.q_token_stride,(size_t)nt*a.weights_token_stride,(size_t)nk*a.index_row_stride},{(size_t)nt*16384,(size_t)nt*128,(size_t)nk*256}};
        NSMutableData *before[2][3];for(unsigned l=0;l<2;l++)for(unsigned r=0;r<3;r++)before[l][r]=poison(sizes[l][r]);
        for(unsigned t=0;t<nt;t++){for(unsigned h=0;h<32;h++)for(unsigned d=0;d<128;d++){float v=input_value((uint64_t)t*4096+h*128+d,17+ci,profile);memcpy((uint8_t*)before[0][0].mutableBytes+GUARD+(size_t)t*a.q_token_stride+h*a.q_head_stride+d*4,&v,4);memcpy((uint8_t*)before[1][0].mutableBytes+GUARD+(size_t)t*16384+h*512+d*4,&v,4);}for(unsigned h=0;h<32;h++){float v=input_value((uint64_t)t*32+h,29+ci,profile);memcpy((uint8_t*)before[0][1].mutableBytes+GUARD+(size_t)t*a.weights_token_stride+h*4,&v,4);memcpy((uint8_t*)before[1][1].mutableBytes+GUARD+(size_t)t*128+h*4,&v,4);}}
        for(unsigned k=0;k<nk;k++)for(unsigned d=0;d<128;d++){float v=value((uint64_t)k*128+d,37+ci);uint32_t bits;memcpy(&bits,&v,4);uint16_t b=bits>>16;if(profile==2&&((uint64_t)k*128+d)%127u==0){const uint16_t special[]={0u,0x8000u,1u,0x8001u,0x007fu,0x807fu,0x0080u,0x8080u,0x3f81u,0xbf81u};b=special[(((uint64_t)k*128+d)/127u)%10u];}memcpy((uint8_t*)before[0][2].mutableBytes+GUARD+(size_t)k*a.index_row_stride+d*2,&b,2);memcpy((uint8_t*)before[1][2].mutableBytes+GUARD+(size_t)k*256+d*2,&b,2);}
        NSMutableDictionary *c=[@{@"case":@(ci),@"padding_profile":@(pad),@"data_profile":data_profiles[profile],@"cache_type":@"BF16",@"args":args_json(&a),@"args72":raw([NSString stringWithFormat:@"case%03u.args72.raw",ci],&a,72),@"exact_args36":raw([NSString stringWithFormat:@"case%03u.exact-args36.raw",ci],&x,36),@"inputs_before":[NSMutableDictionary new],@"providers":[NSMutableArray new]}mutableCopy];
        for(unsigned l=0;l<2;l++){NSMutableArray *rs=[NSMutableArray new];for(unsigned r=0;r<3;r++){NSString *stem=[NSString stringWithFormat:@"case%03u.%@.%@.before",ci,l?@"dense":@"strided",roles[r]];[rs addObject:guarded_raw(stem,before[l][r].bytes,sizes[l][r])];}c[@"inputs_before"][l?@"dense":@"strided"]=rs;}
        if(!host){id<MTLBuffer> inputs[2][3];for(unsigned l=0;l<2;l++)for(unsigned r=0;r<3;r++){inputs[l][r]=[device newBufferWithBytes:before[l][r].bytes length:before[l][r].length options:MTLResourceStorageModeShared];need(inputs[l][r]!=nil,"guarded input buffer");}
            NSMutableData *logical[PROVIDERS];
            for(unsigned p=0;p<PROVIDERS;p++){@autoreleasepool {
                unsigned l=p<4?0:1;size_t stride=p<4?a.score_token_stride:(size_t)nk*4,bytes=(size_t)nt*stride;
                NSMutableData *poisoned=poison(bytes);id<MTLBuffer> out=[device newBufferWithBytes:poisoned.bytes length:poisoned.length options:MTLResourceStorageModeShared];need(out!=nil,"distinct guarded output buffer");
                id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"one command buffer per provider");MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"compute encoder");[e setComputePipelineState:psos[p]];if(p<4)[e setBytes:&a length:72 atIndex:0];else[e setBytes:&x length:36 atIndex:0];for(unsigned r=0;r<3;r++)[e setBuffer:inputs[l][r] offset:GUARD atIndex:r+1];[e setBuffer:out offset:GUARD atIndex:4];if(tg[p])[e setThreadgroupMemoryLength:tg[p] atIndex:0];unsigned gx=(nk+(p<4?31:15))/(p<4?32:16),gy=(nt+7)/8;[e dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];BOOL gpu=cb.status==MTLCommandBufferStatusCompleted;completed++;
                emit(@{@"kind":@"gpu_completion",@"ordinal":@(completed),@"case":@(ci),@"provider":@(p),@"status":@(cb.status),@"completed":@(gpu),@"symbol":names[p],@"library_sha256":libraries[lib_index[p]][@"sha256"],@"error":cb.error?cb.error.localizedDescription:(id)[NSNull null]});
                NSMutableDictionary *pr=[@{@"provider":@(p),@"completion_ordinal":@(completed),@"gpu_completed":@(gpu),@"symbol":names[p],@"library_index":@(lib_index[p]),@"library_sha256":libraries[lib_index[p]][@"sha256"],@"input_layout":l?@"dense":@"strided",@"grid":@[@(gx),@(gy),@1],@"threads":@[@32,@4,@1],@"dynamic_threadgroup_bytes":@(tg[p]),@"score_stride":@(stride),@"scores":guarded_raw([NSString stringWithFormat:@"case%03u.provider%u.scores",ci,p],out.contents,bytes),@"inputs_after":[NSMutableArray new]}mutableCopy];
                BOOL inputok=YES;for(unsigned r=0;r<3;r++){BOOL eq=!memcmp(inputs[l][r].contents,before[l][r].bytes,before[l][r].length);inputok&=eq;[pr[@"inputs_after"]addObject:guarded_raw([NSString stringWithFormat:@"case%03u.provider%u.%@.after",ci,p,roles[r]],inputs[l][r].contents,sizes[l][r])];}pr[@"inputs_unchanged"]=@(inputok);
                logical[p]=[NSMutableData dataWithLength:(size_t)nt*nk*4];const uint8_t *data=(const uint8_t*)out.contents+GUARD;BOOL padding=YES,causal=YES,finite=YES,written=YES;size_t invisible=0;
                for(unsigned t=0;t<nt;t++){const uint32_t *row=(const uint32_t*)(data+(size_t)t*stride);memcpy((uint8_t*)logical[p].mutableBytes+(size_t)t*nk*4,row,(size_t)nk*4);unsigned visible=(pos+t+1)/2;if(visible>nk)visible=nk;for(unsigned k=0;k<nk;k++){uint32_t b=row[k];written&=b!=0xa5a5a5a5u;if(k>=visible){invisible++;causal&=b==0xff800000u;}else finite&=(b&0x7f800000u)!=0x7f800000u;}for(size_t b=(size_t)nk*4;b<stride;b++)padding&=data[(size_t)t*stride+b]==0xa5;}
                size_t bad=diff(logical[p].bytes,logical[0].bytes,logical[p].length);size_t exactbad=p==5?diff(logical[p].bytes,logical[4].bytes,logical[p].length):0;BOOL guard=guards(out.contents,bytes),ok=gpu&&inputok&&padding&&causal&&finite&&written&&guard&&(p>=4||bad==0)&&(p!=5||exactbad==0);all&=ok;
                pr[@"tiled_mismatches"]=@(bad);pr[@"exact_bridge_mismatches"]=p==5?@(exactbad):(id)[NSNull null];pr[@"padding_unchanged"]=@(padding);pr[@"causal_tail_valid"]=@(causal);pr[@"visible_finite"]=@(finite);pr[@"all_scores_written"]=@(written);pr[@"guards_intact"]=@(guard);pr[@"invisible_minus_inf_count"]=@(invisible);pr[@"numerical_passed"]=@(ok);[c[@"providers"]addObject:pr];emit(@{@"kind":@"provider_result",@"case":@(ci),@"provider":@(p),@"numerical_passed":@(ok),@"tiled_mismatches":@(bad),@"exact_bridge_mismatches":pr[@"exact_bridge_mismatches"]});
            }}
            NSDictionary *topk=topk_against_exact(logical[4],logical[3],nk,nt,pos);
            c[@"actual_exact_vs_long_top512"]=topk;
            all &= [topk[@"guards_and_causal_ids_valid"]boolValue] && [topk[@"exact_aa_ordered_id_mismatches"]unsignedLongLongValue]==0 && [topk[@"ordered_id_mismatches"]unsignedLongLongValue]==0;
            emit(@{@"kind":@"top512_result",@"case":@(ci),@"details":topk});
        }
        [manifest[@"cases"]addObject:c];emit(@{@"kind":@"case_complete",@"case":@(ci),@"args":c[@"args"],@"providers":@(host?0:PROVIDERS)});finish(@"in_progress");
    }}
    for(unsigned i=0;i<3;i++)need([filesha(paths[i])isEqual:libraries[i][@"sha256"]],"libraries unchanged");need([filesha(options[@"--build-identity"])isEqual:manifest[@"build_identity_sha256"]]&&[filesha(options[@"--harness-source"])isEqual:manifest[@"harness_source_sha256"]]&&[filesha(exe)isEqual:manifest[@"binary_sha256"]],"identity/source/binary unchanged");manifest[@"gpu_completion_count"]=@(completed);BOOL numerical=host?NO:all;manifest[@"all_numerical_passed"]=@(numerical);NSString *status=host?@"host_only_complete_not_numerical":all?@"correctness_complete_not_adoption":@"numerical_failed";finish(status);emit(@{@"kind":@"complete",@"status":status,@"case_count":@(sizeof(CASES)/sizeof(CASES[0])),@"gpu_completion_count":@(completed),@"all_numerical_passed":@(numerical),@"timing_collected":@NO,@"adoption_eligible":@NO});if(!host){myds_v41_gpu_scratch_free(topk_scratch);myds_metal_cleanup();}return host||all?0:2;
}}
