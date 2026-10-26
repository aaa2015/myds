/* Private synthetic/component dummy-Q-free KV finish exact-output gate.
 * Build only with build_kv_only_finish_probe.py; frozen_rope_init.inc is copied from
 * the production C source, with its original precise host frequency order.
 * --metallib PROBE --golden-metallib ACTUAL_PRODUCTION --build-manifest JSON
 * --out-dir NEW --synthetic [--host-only] [--timing]
 * Default: correctness only. --timing is an explicitly synthetic component
 * screen after every full-output gate, never a full-model/adoption result.
 * All providers retain the native 80 groups x 32 lanes and three-resource join.
 * Candidate changes only groups0..63 to early return; KV groups64..79 match.
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

enum { GUARD=64, QBYTES=131072, KVBYTES=2048, WINBYTES=262144, MAX_CASES=128 };
struct finish_args { uint32_t pos,pad; float frequencies[32]; };
_Static_assert(sizeof(struct finish_args)==136 && offsetof(struct finish_args,frequencies)==8,"native finish ABI");

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
    fprintf(stderr,"kv_only_finish_probe: %s\n",reason);
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
static const size_t sizes[3]={QBYTES,KVBYTES,WINBYTES};
static NSString *const roles[3]={@"dummy_q",@"kv",@"window"};
static BOOL index_source(unsigned l){return l==2||l==8||l==14||l==20||(l>=24&&l%4==0);}
static BOOL eligible_layer(unsigned l){return l<40&&!index_source(l);}
static uint64_t rng=UINT64_C(0x4ae3079c14ca852b);
static uint64_t random64(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static float round_bf16(float x){uint32_t u;memcpy(&u,&x,4);if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16)&1u);u&=0xffff0000u;memcpy(&x,&u,4);return x;}
static BOOL finite_bf16(const void*p,size_t n){const uint32_t*v=p;for(size_t i=0;i<n/4;i++)if((v[i]&0xffffu)||((v[i]&0x7f800000u)==0x7f800000u))return NO;return YES;}
static size_t differences(const void*a,const void*b,size_t n,size_t*first){const uint32_t*x=a,*y=b;size_t bad=0;*first=SIZE_MAX;for(size_t i=0;i<n/4;i++)if(x[i]!=y[i]){if(!bad)*first=i;bad++;}return bad;}
static void *body_ptr(id<MTLBuffer>b){return(uint8_t*)b.contents+GUARD;}
static BOOL guards(id<MTLBuffer>b,size_t n){const uint8_t*p=b.contents;for(unsigned i=0;i<GUARD;i++)if(p[i]!=0xa5||p[GUARD+n+i]!=0xa5)return NO;return YES;}
static id<MTLBuffer> allocate(id<MTLDevice>d,NSData*initial){need(initial.length+2u*GUARD<=d.maxBufferLength,"Metal maxBufferLength");id<MTLBuffer>b=[d newBufferWithLength:initial.length+2u*GUARD options:MTLResourceStorageModeShared];need(b!=nil,"Metal allocation");memset(b.contents,0xa5,b.length);memcpy(body_ptr(b),initial.bytes,initial.length);return b;}
static void dump(NSString*name,id<MTLBuffer>b,size_t n){raw([name stringByAppendingString:@".raw"],body_ptr(b),n);raw([name stringByAppendingString:@".prefix-guard.raw"],b.contents,GUARD);raw([name stringByAppendingString:@".suffix-guard.raw"],(uint8_t*)b.contents+GUARD+n,GUARD);}
typedef struct {unsigned layer,pos,index;BOOL compressed;NSMutableArray*before;id<MTLBuffer>buf[3][3];NSData*oracle[3];} fixture;
static fixture cases[MAX_CASES];static unsigned case_count;
static struct finish_args args_for(fixture*f){struct finish_args a={.pos=f->pos};pthread_once(&rope_once,init_rope);memcpy(a.frequencies,rope_frequencies[f->compressed?1:0],sizeof(a.frequencies));return a;}
static void add_case(unsigned layer,unsigned pos){need(case_count<MAX_CASES&&eligible_layer(layer)&&pos<1048576u,"eligible layer/position");fixture*f=&cases[case_count];f->layer=layer;f->pos=pos;f->index=case_count++;f->compressed=layer>=2;f->before=[NSMutableArray new];for(unsigned r=0;r<3;r++){NSMutableData*d=[NSMutableData dataWithLength:sizes[r]];float*v=d.mutableBytes;for(size_t j=0;j<sizes[r]/4;j++){int t=(int)(random64()%16385)-8192;float x=round_bf16((float)t/1024.0f);if(r==1){if(f->index%8==0)x=(j&1)?-0.0f:0.0f;if(f->index%8==1)x=round_bf16(x*0x1p-20f);if(f->index%8==2)x=round_bf16(x*0x1p18f);}v[j]=x;}if(r==0&&f->index%3==0){uint32_t special[]={0x7fc00001u,0x7f800000u,0xff800000u,0x80000000u,0x00000001u,0x3f808000u};uint32_t*u=d.mutableBytes;for(unsigned j=0;j<64*64;j++)u[(j/64)*512u+448u+(j%64)]=special[j%6];}[f->before addObject:d];}}
static id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,NSString*n,unsigned groups){NSError*error=nil;id<MTLFunction>f=[l newFunctionWithName:n];need(f!=nil,"explicit native/candidate function missing; no fallback");id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&error];if(!p)fprintf(stderr,"PSO %s: %s\n",n.UTF8String,error.localizedDescription.UTF8String);need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=32,"PSO native 32-lane geometry");need(p.staticThreadgroupMemoryLength<=d.maxThreadgroupMemoryLength,"PSO memory geometry");emit(@{@"kind":@"pipeline",@"symbol":n,@"groups":@[@(groups),@1,@1],@"threads":@[@32,@1,@1],@"static_tg_bytes":@(p.staticThreadgroupMemoryLength),@"dynamic_tg_bytes":@0,@"fallback":@NO});return p;}
static BOOL safe(fixture*f,unsigned p){for(unsigned r=0;r<3;r++){if(!f->buf[p][r]||f->buf[p][r].length!=sizes[r]+2u*GUARD||((uintptr_t)body_ptr(f->buf[p][r])&15u))return NO;for(unsigned j=0;j<r;j++)if(f->buf[p][j]==f->buf[p][r])return NO;}return YES;}
static id<MTLCommandBuffer>run(id<MTLCommandQueue>queue,fixture*f,unsigned p,id<MTLComputePipelineState>ps){need(p<3&&safe(f,p),"full disjoint native buffer ranges before CB");for(unsigned r=0;r<3;r++){memset(f->buf[p][r].contents,0xa5,f->buf[p][r].length);memcpy(body_ptr(f->buf[p][r]),[f->before[r]bytes],sizes[r]);}id<MTLCommandBuffer>cb=[queue commandBuffer];need(cb!=nil,"command buffer");MTLComputePassDescriptor*pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeConcurrent;id<MTLComputeCommandEncoder>e=[cb computeCommandEncoderWithDescriptor:pass];need(e!=nil,"concurrent encoder");struct finish_args a=args_for(f);[e setComputePipelineState:ps];[e setBytes:&a length:sizeof(a) atIndex:0];for(unsigned r=0;r<3;r++)[e setBuffer:f->buf[p][r] offset:GUARD atIndex:r+1];[e dispatchThreadgroups:MTLSizeMake(80,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];id<MTLResource>deps[3]={f->buf[p][0],f->buf[p][1],f->buf[p][2]};[e memoryBarrierWithResources:deps count:3];[e endEncoding];command_ordinal++;[cb commit];[cb waitUntilCompleted];return cb;}
static BOOL validate(fixture*f,unsigned p,NSString*stage,BOOL save){BOOL ok=YES;for(unsigned r=0;r<3;r++){if(p==0&&!f->oracle[r])f->oracle[r]=[NSData dataWithBytes:body_ptr(f->buf[p][r]) length:sizes[r]];const void*expected=(p==2&&r==0)?[f->before[r]bytes]:f->oracle[r].bytes;size_t first=SIZE_MAX,bad=differences(expected,body_ptr(f->buf[p][r]),sizes[r],&first);BOOL g=guards(f->buf[p][r],sizes[r]);BOOL format=r==0||finite_bf16(body_ptr(f->buf[p][r]),sizes[r]);size_t untouched=0;if(r==2){const uint32_t*x=body_ptr(f->buf[p][r]),*y=[f->before[r]bytes];for(unsigned i=0;i<128u*512u;i++)if(i/512u!=f->pos%128u&&x[i]!=y[i])untouched++;}BOOL slot=YES;if(r==2)slot=!memcmp((uint8_t*)body_ptr(f->buf[p][2])+(size_t)(f->pos%128u)*KVBYTES,body_ptr(f->buf[p][1]),KVBYTES);NSString*n=[NSString stringWithFormat:@"%@.case%03u.provider%u.%@",stage,f->index,p,roles[r]];NSDictionary*record=@{@"kind":@"parity",@"stage":stage,@"case":@(f->index),@"layer":@(f->layer),@"position":@(f->pos),@"compressed":@(f->compressed),@"provider":@(p),@"role":roles[r],@"bytes":@(sizes[r]),@"mismatches":@(bad),@"first_mismatch":first==SIZE_MAX?[NSNull null]:@(first),@"guards":@(g),@"finite_bf16_required":@(r!=0),@"finite_bf16":@(format),@"untouched_window_mismatches":@(untouched),@"current_window_slot_equals_complete_kv":@(slot),@"expectation":p==2&&r==0?@"complete-before-bytes-dead-dummy-remains-unmodified":@"actual-golden-output-complete-bytes",@"sha256":sha(body_ptr(f->buf[p][r]),sizes[r])};[parity_records addObject:record];emit(record);BOOL good=!bad&&g&&format&&!untouched&&slot;if(save||!good)dump(n,f->buf[p][r],sizes[r]);ok=ok&&good;}return ok;}
static double median(double*v){double a[9];memcpy(a,v,sizeof(a));for(unsigned i=0;i<9;i++)for(unsigned j=i+1;j<9;j++)if(a[i]>a[j]){double t=a[i];a[i]=a[j];a[j]=t;}return a[4];}
int main(int argc,char**argv){@autoreleasepool{NSString*probe=nil,*gold=nil,*build_path=nil;BOOL synthetic=NO,host=NO,timing=NO;for(int i=1;i<argc;i++){const char*k=argv[i];if(!strcmp(k,"--synthetic")){synthetic=YES;continue;}if(!strcmp(k,"--host-only")){host=YES;continue;}if(!strcmp(k,"--timing")){timing=YES;continue;}need(i+1<argc,"argument value");NSString*v=canonical([NSString stringWithUTF8String:argv[++i]]);if(!strcmp(k,"--metallib"))probe=v;else if(!strcmp(k,"--golden-metallib"))gold=v;else if(!strcmp(k,"--build-manifest"))build_path=v;else if(!strcmp(k,"--out-dir"))output=v;else need(NO,"unknown argument");}need(synthetic&&probe&&gold&&build_path&&output,"explicit synthetic and absolute paths");need(probe.isAbsolutePath&&gold.isAbsolutePath&&output.isAbsolutePath&&build_path.isAbsolutePath,"absolute paths");need(![probe isEqual:gold],"separate actual gold path");need(![[NSFileManager defaultManager]fileExistsAtPath:output],"new output directory");need([[NSFileManager defaultManager]createDirectoryAtPath:output withIntermediateDirectories:NO attributes:nil error:nil],"create output directory");files=[NSMutableArray new];parity_records=[NSMutableArray new];timing_records=[NSMutableArray new];input_records=[NSMutableArray new];NSData*bd=read_file(build_path);NSDictionary*build=[NSJSONSerialization JSONObjectWithData:bd options:0 error:nil];need([build[@"kind"]isEqual:@"kv_only_finish_private_full_tu_build"]&&[build[@"complete_native_prefix_byte_exact"]boolValue]&&[build[@"proof"][@"complete_kv_branch_bytes_unchanged"]boolValue]&&[build[@"proof"][@"reverse_edits_restore_original_body_bytes"]boolValue],"full TU and complete KV expression identity");need([file_sha(probe)isEqual:build[@"metallib_sha256"]]&&[file_sha(binary_path())isEqual:build[@"binary_sha256"]],"running binary and library SHA bound to build");manifest=[@{@"schema_version":@1,@"kind":@"kv_only_finish_synthetic_component",@"host_only":@(host),@"synthetic":@YES,@"runtime_modified":@NO,@"production_registered":@NO,@"full_model_evidence":@NO,@"timing_requested":@(timing),@"build_manifest_path":build_path,@"build_manifest_sha256":sha(bd.bytes,bd.length),@"binary_path":binary_path(),@"binary_sha256":file_sha(binary_path()),@"probe_metallib_path":probe,@"probe_metallib_sha256":file_sha(probe),@"golden_metallib_path":gold,@"golden_metallib_sha256":file_sha(gold),@"abi":build[@"abi"],@"symbols":@[@"kernel_dsv41_qkv_finish",@"kernel_myds_probe_kv_only_finish"],@"providers":@[@"actual-golden-original-qkv-finish-dummy-Q",@"full-TU-native-original-qkv-finish-dummy-Q",@"full-TU-kv-only-finish-unused-Q-removed"],@"limitations":@[@"Synthetic single dispatch component, no ordinary graph overlap or end-to-end speed proof.",@"Dummy Q output is intentionally different; candidate must retain all original before bytes and guards.",@"Semantic complete KV/window outputs and full-TU native dummy Q must exactly match actual golden."]}mutableCopy];raw(@"build-identity.json",bd.bytes,bd.length);NSMutableArray*eligible=[NSMutableArray new];for(unsigned l=0;l<40;l++)if(eligible_layer(l))[eligible addObject:@(l)];need(eligible.count==32&&[eligible isEqual:build[@"eligible_decode_non_index_source_layers"]],"exact 32 layers");unsigned ps[]={0,1,327679};for(unsigned l=0;l<40;l++)if(eligible_layer(l))for(unsigned j=0;j<3;j++)add_case(l,ps[j]);unsigned ls[]={0,3,39},edge[]={127,128,16383,16384,65535,65536};for(unsigned l=0;l<3;l++)for(unsigned j=0;j<6;j++)add_case(ls[l],edge[j]);for(unsigned i=0;i<case_count;i++){fixture*f=&cases[i];struct finish_args a=args_for(f);raw([NSString stringWithFormat:@"case%03u.args136.raw",i],&a,sizeof(a));for(unsigned r=0;r<3;r++)raw([NSString stringWithFormat:@"case%03u.%@.before.raw",i,roles[r]],[f->before[r]bytes],sizes[r]);emit(@{@"kind":@"case",@"case":@(i),@"layer":@(f->layer),@"position":@(f->pos),@"compressed":@(f->compressed),@"args_sha256":sha(&a,sizeof(a)),@"args_bytes":@136});}manifest[@"case_count"]=@(case_count);manifest[@"eligible_layers"]=eligible;emit(@{@"kind":@"host_contract",@"case_count":@(case_count),@"eligible_layers":eligible,@"native_groups":@80,@"candidate_groups":@80,@"args_bytes":@136,@"no_gpu_device_created":@(host)});if(host){finish(@"host_only_complete_no_gpu");return 0;}id<MTLDevice>d=MTLCreateSystemDefaultDevice();need(d!=nil,"Metal device");manifest[@"device_name"]=d.name;manifest[@"device_registry_id"]=@(d.registryID);NSError*error=nil;id<MTLLibrary>libs[2]={[d newLibraryWithURL:[NSURL fileURLWithPath:gold]error:&error],[d newLibraryWithURL:[NSURL fileURLWithPath:probe]error:&error]};need(libs[0]&&libs[1],"load explicit actual-gold and full-TU URLs, no fallback");for(unsigned i=0;i<2;i++)emit(@{@"kind":@"loaded_library",@"provider_library":@(i),@"path":i?probe:gold,@"sha256":file_sha(i?probe:gold),@"direct_url_load":@YES});id<MTLComputePipelineState>pso[3]={pipeline(d,libs[0],@"kernel_dsv41_qkv_finish",80),pipeline(d,libs[1],@"kernel_dsv41_qkv_finish",80),pipeline(d,libs[1],@"kernel_myds_probe_kv_only_finish",80)};id<MTLCommandQueue>queue=[d newCommandQueue];need(queue!=nil,"queue");for(unsigned i=0;i<case_count;i++)for(unsigned p=0;p<3;p++)for(unsigned r=0;r<3;r++)cases[i].buf[p][r]=allocate(d,cases[i].before[r]);BOOL all=YES;for(unsigned i=0;i<case_count;i++)for(unsigned p=0;p<3;p++){id<MTLCommandBuffer>cb=run(queue,&cases[i],p,pso[p]);BOOL completed=cb.status==MTLCommandBufferStatusCompleted;emit(@{@"kind":@"gpu_completion",@"stage":@"phase1",@"case":@(i),@"provider":@(p),@"ordinal":@(command_ordinal),@"completed":@(completed),@"status":@(cb.status),@"error":cb.error.localizedDescription?:[NSNull null]});BOOL v=validate(&cases[i],p,@"phase1",YES);all=all&&v&&completed;}manifest[@"all_complete_outputs_and_guards_pass"]=@(all);need(all,"all phase1 complete raw saved; exact gate refused");if(!timing){finish(@"synthetic_correctness_complete_not_adoption");return 0;}unsigned selected[]={0,7,95};for(unsigned c=0;c<3;c++)for(unsigned ctl=0;ctl<2;ctl++){double v[3][9]={{0}};for(int sample=-1;sample<9;sample++)for(unsigned arm=0;arm<3;arm++){unsigned p=arm==1?2:ctl;fixture*f=&cases[selected[c]];id<MTLCommandBuffer>cb=run(queue,f,p,pso[p]);double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6;BOOL completed=cb.status==MTLCommandBufferStatusCompleted;BOOL parity=validate(f,p,[NSString stringWithFormat:@"timing-c%u-ctl%u-s%d-arm%u",c,ctl,sample,arm],NO);NSDictionary*r=@{@"kind":@"timing",@"case":@(f->index),@"control":@(ctl),@"sample":@(sample),@"warmup":@(sample<0),@"arm":@(arm),@"provider":@(p),@"ordinal":@(command_ordinal),@"dispatches":@1,@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),@"us_per_finish":@(us),@"completed":@(completed),@"output_gate":@(parity),@"diagnostic_only":@YES};[timing_records addObject:r];emit(r);need(completed&&parity&&isfinite(us)&&us>0,"timing failure raw saved; no cohort retry");if(sample>=0)v[arm][sample]=us;}double a1=median(v[0]),b=median(v[1]),a2=median(v[2]);emit(@{@"kind":@"component_summary",@"case":@(selected[c]),@"control":@(ctl),@"a1_us":@(a1),@"b_us":@(b),@"a2_us":@(a2),@"conservative_saving_percent":@(100*(fmin(a1,a2)-b)/fmin(a1,a2)),@"control_drift_percent":@(100*(a2-a1)/a1),@"adoption_gate_evaluated":@NO});}finish(@"synthetic_component_screen_complete_not_adoption");return 0;}}
