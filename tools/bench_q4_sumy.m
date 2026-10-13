/* Standalone Dense Q4_K sumy probe. Uses an independent probe.metallib and
 * NEVER initializes the inference runtime or modifies production state.
 * Build: clang -O3 -Wall -Wextra -fobjc-arc -framework Foundation
 *        -framework Metal tools/bench_q4_sumy.m -o /tmp/bench_q4_sumy
 * Run only in root's exclusive GPU window. Defaults to actual LM-head shape.
 * The candidate PAIR includes prepare, RAW resource barrier and cached GEMV.
 * --golden-metallib supplies authoritative production reference PSOs. Probe
 * library reference clones must match those PSOs before candidate validation.
 * Synthetic weights are valid, nonzero fixtures; --model/--offset can read
 * only a selected real tensor. --out-dir retains complete raw evidence.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct mv_args {
    int32_t ne00,ne01,ne02;
    uint64_t nb00,nb01,nb02,nb03;
    int32_t ne10,ne11,ne12;
    uint64_t nb10,nb11,nb12,nb13;
    int32_t ne0,ne1,nr0;
    int16_t r2,r3;
};
_Static_assert(sizeof(struct mv_args)==112,"Q4 GEMV Metal argument ABI");
struct prepare_args { uint32_t width,pad; };
static void req(int ok,const char *why) {
    if(!ok){fprintf(stderr,"bench_q4_sumy: %s\n",why);exit(1);}
}
static uint64_t rng_state=88172645463325252ull;
static uint64_t rnd(void){rng_state^=rng_state<<13;rng_state^=rng_state>>7;rng_state^=rng_state<<17;return rng_state;}
static uint64_t hash(const void *p,size_t n){const uint8_t *s=p;uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++)h=(h^s[i])*UINT64_C(1099511628211);return h;}
static NSString *sha256(NSData *data){req(data!=nil&&data.length<=UINT32_MAX,"SHA input unavailable/too large");unsigned char h[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data.bytes,(CC_LONG)data.length,h);NSMutableString *s=[NSMutableString stringWithCapacity:64];for(unsigned i=0;i<sizeof(h);i++)[s appendFormat:@"%02x",h[i]];return s;}
static double median(double *x,unsigned n){for(unsigned i=1;i<n;i++){double y=x[i];unsigned j=i;while(j&&x[j-1]>y){x[j]=x[j-1];j--;}x[j]=y;}return x[n/2];}
static unsigned parse_u(const char *p){char *e=NULL;unsigned long n=strtoul(p,&e,10);req(e&&e!=p&&!*e&&n>0&&n<=INT32_MAX,"invalid unsigned argument");return (unsigned)n;}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> lib,NSString *name,int constants){
    NSError *error=nil;id<MTLFunction> fn=nil;
    if(constants){MTLFunctionConstantValues *v=[MTLFunctionConstantValues new];short nsg=4;[v setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];fn=[lib newFunctionWithName:name constantValues:v error:&error];}
    else fn=[lib newFunctionWithName:name];
    if(!fn)fprintf(stderr,"function %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    req(fn!=nil,"missing function; refusing fallback");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:fn error:&error];
    if(!p)fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);
    req(p!=nil&&p.maxTotalThreadsPerThreadgroup>=128,"unsupported pipeline geometry");return p;
}
static void dump(const char *dir,const char *name,const void *p,size_t n){
    if(!dir)return;char path[4096];int z=snprintf(path,sizeof(path),"%s/%s",dir,name);req(z>0&&(size_t)z<sizeof(path),"evidence path too long");FILE *f=fopen(path,"wb");req(f!=NULL,"evidence fopen");req(fwrite(p,1,n,f)==n&&!fclose(f),"evidence write");
}
static void guards(id<MTLBuffer> b,size_t bytes){const uint8_t *p=b.contents;for(unsigned i=0;i<16;i++)req(p[i]==0xa5&&p[16+bytes+i]==0xa5,"output/cache canary changed");}

int main(int argc,char **argv){
    const char *lib_path=NULL,*golden_path=NULL,*model=NULL,*outdir=NULL;uint64_t offset=0;int has_offset=0;
    unsigned rows=129280,width=5120,samples=5,reps=16,seeds=3;
    for(int i=1;i<argc;i++){
        req(i+1<argc,"incomplete argument");const char *v=argv[++i];
        if(!strcmp(argv[i-1],"--metallib"))lib_path=v;
        else if(!strcmp(argv[i-1],"--golden-metallib"))golden_path=v;
        else if(!strcmp(argv[i-1],"--rows"))rows=parse_u(v);
        else if(!strcmp(argv[i-1],"--width"))width=parse_u(v);
        else if(!strcmp(argv[i-1],"--samples"))samples=parse_u(v);
        else if(!strcmp(argv[i-1],"--reps"))reps=parse_u(v);
        else if(!strcmp(argv[i-1],"--seeds"))seeds=parse_u(v);
        else if(!strcmp(argv[i-1],"--model"))model=v;
        else if(!strcmp(argv[i-1],"--offset")){char *e;errno=0;offset=strtoull(v,&e,10);req(!errno&&*v&&!*e&&offset<=INT64_MAX,"invalid tensor offset");has_offset=1;}
        else if(!strcmp(argv[i-1],"--out-dir"))outdir=v;
        else req(0,"unknown argument");
    }
    req(lib_path&&golden_path&&width%256u==0&&samples>=3&&samples<=31&&reps<=1024&&seeds<=32,"missing probe/golden metallib or invalid geometry/sampling");
    req((model!=NULL)==has_offset,"--model requires --offset and conversely");
    if(outdir){struct stat st;req(!stat(outdir,&st)&&S_ISDIR(st.st_mode),"out-dir must already exist");NSError *e=nil;NSArray *files=[[NSFileManager defaultManager] contentsOfDirectoryAtPath:[NSString stringWithUTF8String:outdir] error:&e];req(files!=nil&&files.count==0,"out-dir must be empty");}
    const size_t row_bytes=(size_t)(width/256u)*144u,wbytes=row_bytes*rows,obytes=(size_t)rows*4u,cbytes=(size_t)(width/256u)*8u*16u;
    req(wbytes<=UINT32_MAX,"probe tensor exceeds supported 4 GiB weight limit");
    @autoreleasepool{
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();req(dev!=nil,"no Metal device");
        NSError *error=nil;id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib_path]] error:&error];
        if(!lib)fprintf(stderr,"metallib: %s\n",error.localizedDescription.UTF8String);req(lib!=nil,"load probe metallib");
        id<MTLLibrary> golden=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:golden_path]] error:&error];
        if(!golden)fprintf(stderr,"golden metallib: %s\n",error.localizedDescription.UTF8String);req(golden!=nil,"load authoritative production metallib");
        id<MTLComputePipelineState> prep=pipeline(dev,lib,@"probe_q4_prepare_sumy",0);
        NSArray<id<MTLComputePipelineState>> *ref=@[pipeline(dev,golden,@"kernel_mul_mv_q4_K_dense_f32",1),pipeline(dev,golden,@"kernel_mul_mv_q4_K_dense_f32_bf16",1)];
        NSArray<id<MTLComputePipelineState>> *clone=@[pipeline(dev,lib,@"kernel_mul_mv_q4_K_dense_f32",1),pipeline(dev,lib,@"kernel_mul_mv_q4_K_dense_f32_bf16",1)];
        NSArray<id<MTLComputePipelineState>> *cand=@[pipeline(dev,lib,@"probe_q4_sumy_f32",1),pipeline(dev,lib,@"probe_q4_sumy_bf16",1)];
        id<MTLBuffer> w=[dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared],x=[dev newBufferWithLength:(size_t)width*4u options:MTLResourceStorageModeShared];
        id<MTLBuffer> cache=[dev newBufferWithLength:cbytes+32u options:MTLResourceStorageModeShared],out=[dev newBufferWithLength:obytes+32u options:MTLResourceStorageModeShared];
        req(w&&x&&cache&&out,"allocation");memset(cache.contents,0xa5,cache.length);memset(out.contents,0xa5,out.length);
        if(model){int fd=open(model,O_RDONLY);req(fd>=0,"model open");struct stat st;req(!fstat(fd,&st)&&offset<=(uint64_t)st.st_size&&wbytes<=(uint64_t)st.st_size-offset,"model tensor bounds");for(size_t n=0;n<wbytes;){ssize_t z=pread(fd,(uint8_t *)w.contents+n,wbytes-n,(off_t)(offset+n));if(z<0&&errno==EINTR)continue;req(z>0,"model pread");n+=(size_t)z;}req(!close(fd),"model close");}
        else{uint8_t *p=w.contents;for(size_t b=0;b<wbytes/144u;b++){uint16_t d=0x2400u,dm=(uint16_t)(0x2000u+(b%1024u));memcpy(p+b*144u,&d,2);memcpy(p+b*144u+2,&dm,2);for(unsigned j=4;j<144;j++)p[b*144u+j]=(uint8_t)rnd();}}
        struct mv_args a={.ne00=(int32_t)width,.ne01=(int32_t)rows,.ne02=1,.nb00=144,.nb01=row_bytes,.nb02=wbytes,.nb03=wbytes,.ne10=(int32_t)width,.ne11=1,.ne12=1,.nb10=4,.nb11=(size_t)width*4,.nb12=(size_t)width*4,.nb13=(size_t)width*4,.ne0=(int32_t)rows,.ne1=1,.nr0=2,.r2=1,.r3=1};
        struct prepare_args pa={width,0};id<MTLCommandQueue> queue=[dev newCommandQueue];req(queue!=nil,"command queue");
        void (^prepare)(id<MTLComputeCommandEncoder>)=^(id<MTLComputeCommandEncoder> e){[e setComputePipelineState:prep];[e setBytes:&pa length:sizeof(pa) atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:cache offset:16 atIndex:2];[e dispatchThreadgroups:MTLSizeMake(((width/256u)*8u+127u)/128u,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];};
        void (^gemv)(id<MTLComputeCommandEncoder>,id<MTLComputePipelineState>,int)=^(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,int cached){[e setComputePipelineState:p];[e setBytes:&a length:sizeof(a) atIndex:0];[e setBuffer:w offset:0 atIndex:1];[e setBuffer:x offset:0 atIndex:2];[e setBuffer:out offset:16 atIndex:3];if(cached)[e setBuffer:cache offset:16 atIndex:4];[e setThreadgroupMemoryLength:256 atIndex:0];[e dispatchThreadgroups:MTLSizeMake((rows+7u)/8u,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];};
        /* mode 0 production reference; 1 full candidate pair; 2 consumer-only
         * diagnostic; 3 probe-library reference clone (authoritative gate).
         * Pair includes cache reuse WAR and prepare-to-consumer RAW barriers.
         * All modes include the same output WAW barrier between repetitions. */
        double (^run)(unsigned,int,unsigned)=^double(unsigned kind,int mode,unsigned repeat){id<MTLCommandBuffer> cb=[queue commandBuffer];req(cb!=nil,"command buffer");id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];req(e!=nil,"concurrent encoder");for(unsigned r=0;r<repeat;r++){id<MTLResource> output_resources[]={out};[e memoryBarrierWithResources:output_resources count:1];if(mode==1){id<MTLResource> cr[]={cache};[e memoryBarrierWithResources:cr count:1];prepare(e);[e memoryBarrierWithResources:cr count:1];}gemv(e,mode==3?clone[kind]:(mode?cand[kind]:ref[kind]),mode==1||mode==2);}[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU completion: %s\n",cb.error.localizedDescription.UTF8String);req(cb.status==MTLCommandBufferStatusCompleted&&cb.GPUEndTime>cb.GPUStartTime,"GPU command completion/timestamp");return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeat;};
        printf("{\"kind\":\"identity\",\"device\":\"%s\",\"rows\":%u,\"width\":%u,\"weight_bytes\":%zu,\"cache_bytes\":%zu,\"source\":\"%s\",\"weight_fnv\":\"%016" PRIx64 "\",\"nr0\":2,\"nsg\":4,\"samples\":%u,\"reps\":%u}\n",dev.name.UTF8String,rows,width,wbytes,cbytes,model?"selected_gguf_tensor":"synthetic",hash(w.contents,wbytes),samples,reps);
        dump(outdir,"weights.q4k",w.contents,wbytes);
        for(unsigned seed=1;seed<=seeds;seed++){
            float *xp=x.contents;for(unsigned j=0;j<width;j++){uint32_t v=(uint32_t)rnd();xp[j]=((int)(v%200001u)-100000)/32768.0f;if(j%17u==0)xp[j]=-0.0f;}
            float *oracle=malloc(cbytes);req(oracle!=NULL,"cache oracle allocation");for(unsigned ib=0;ib<width/256u;ib++)for(unsigned it=0;it<8;it++){unsigned at=ib*256u+64u*(it/4u)+8u*(it%4u);float s[4]={0};for(unsigned j=0;j<8;j++){s[0]+=xp[at+j];s[1]+=xp[at+j+32];s[2]+=xp[at+j+128];s[3]+=xp[at+j+160];}memcpy(oracle+(ib*8u+it)*4u,s,sizeof(s));}
            char name[128];snprintf(name,sizeof(name),"seed%u-input.f32",seed);dump(outdir,name,x.contents,x.length);
            for(unsigned kind=0;kind<2;kind++){
                uint8_t *saved=malloc(obytes);req(saved!=NULL,"snapshot allocation");run(kind,0,1);memcpy(saved,(uint8_t *)out.contents+16,obytes);
                snprintf(name,sizeof(name),"seed%u-%s-reference.f32",seed,kind?"bf16":"f32");dump(outdir,name,saved,obytes);
                run(kind,3,1);req(!memcmp(saved,(uint8_t *)out.contents+16,obytes),"probe-library reference clone differs from authoritative production output");guards(out,obytes);
                snprintf(name,sizeof(name),"seed%u-%s-reference-clone.f32",seed,kind?"bf16":"f32");dump(outdir,name,(uint8_t *)out.contents+16,obytes);
                run(kind,1,1);req(!memcmp(cache.contents+16,oracle,cbytes),"sum-cache differs from independent sequential CPU oracle");req(!memcmp(saved,(uint8_t *)out.contents+16,obytes),"candidate differs from reference F32 bytes");guards(out,obytes);guards(cache,cbytes);
                size_t nz=0;const float *yp=(const float *)((uint8_t *)out.contents+16);for(unsigned j=0;j<rows;j++){req(isfinite(yp[j]),"nonfinite output");nz+=yp[j]!=0;}req(nz!=0,"unexpected zero output");
                printf("{\"kind\":\"parity\",\"seed\":%u,\"epilogue\":\"%s\",\"bytes\":%zu,\"bit_exact\":true,\"reference_clone_bit_exact\":true,\"finite\":true,\"canary\":true,\"output_fnv\":\"%016" PRIx64 "\",\"cache_fnv\":\"%016" PRIx64 "\"}\n",seed,kind?"bf16":"f32",obytes,hash(yp,obytes),hash(oracle,cbytes));
                snprintf(name,sizeof(name),"seed%u-%s-candidate.f32",seed,kind?"bf16":"f32");dump(outdir,name,yp,obytes);snprintf(name,sizeof(name),"seed%u-cache.f32",seed);dump(outdir,name,(uint8_t *)cache.contents+16,cbytes);
                if(seed==1){const char *arms[]={"A1","B_pair","A2","B_consumer_only"};double med[4];for(unsigned arm=0;arm<4;arm++){int mode=arm==1?1:(arm==3?2:0);run(kind,mode,1);req(!memcmp(saved,(uint8_t *)out.contents+16,obytes),"warmup output changed");guards(out,obytes);guards(cache,cbytes);double t[31];for(unsigned z=0;z<samples;z++){t[z]=run(kind,mode,reps);req(isfinite(t[z])&&t[z]>0,"invalid timing sample");req(!memcmp(saved,(uint8_t *)out.contents+16,obytes),"timing output changed");guards(out,obytes);guards(cache,cbytes);printf("{\"kind\":\"timing\",\"epilogue\":\"%s\",\"arm\":\"%s\",\"sample\":%u,\"gpu_us\":%.9f,\"finite\":true,\"canary\":true,\"full_output_exact\":true}\n",kind?"bf16":"f32",arms[arm],z,t[z]);}med[arm]=median(t,samples);}printf("{\"kind\":\"summary\",\"epilogue\":\"%s\",\"a1_us\":%.9f,\"pair_us\":%.9f,\"a2_us\":%.9f,\"consumer_only_us\":%.9f,\"pair_change_pct\":%.6f,\"control_drift_pct\":%.6f}\n",kind?"bf16":"f32",med[0],med[1],med[2],med[3],(med[1]/((med[0]+med[2])/2)-1)*100,(med[2]/med[0]-1)*100);}
                free(saved);
            }
            free(oracle);
        }
        if(outdir){
            NSString *dir=[NSString stringWithUTF8String:outdir];
            NSArray<NSString *> *files=[[[NSFileManager defaultManager] contentsOfDirectoryAtPath:dir error:nil] sortedArrayUsingSelector:@selector(compare:)];
            req(files!=nil,"manifest inventory");NSMutableDictionary *raw=[NSMutableDictionary new];
            for(NSString *name in files){NSData *data=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:name]];req(data!=nil,"manifest raw read");raw[name]=@{@"bytes":@(data.length),@"sha256":sha256(data)};}
            NSData *gold_data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:golden_path]];
            NSData *probe_data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:lib_path]];
            NSData *binary_data=[NSData dataWithContentsOfFile:[NSProcessInfo processInfo].arguments[0]];
            NSDictionary *manifest=@{@"device":dev.name,@"rows":@(rows),@"width":@(width),@"seeds":@(seeds),@"samples":@(samples),@"reps":@(reps),@"nr0":@2,@"nsg":@4,@"golden_metallib_sha256":sha256(gold_data),@"probe_metallib_sha256":sha256(probe_data),@"binary_sha256":sha256(binary_data),@"reference_clone_gate":@"all seeds and both epilogues full F32 bytes",@"candidate_gate":@"production reference full F32 bytes and sequential sum cache oracle",@"timing_checks":@"every warmup and formal sample output/canary/finite timestamp checked",@"scope":@"isolated repeated component; no full-model speed claim",@"source":model?@"selected_gguf_tensor":@"synthetic",@"tensor_offset":@(offset),@"raw_files":raw};
            NSData *json=[NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];req(json!=nil,"manifest JSON serialization");dump(outdir,"manifest.json",json.bytes,json.length);
        }
    }
    return 0;
}
