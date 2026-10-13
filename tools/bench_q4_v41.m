/* Isolated Q4 V4.1 fixed-shape screening. No inference runtime/service init.
 * Build: clang -O3 -Wall -Wextra -fobjc-arc -Iinclude -framework Foundation
 * -framework Metal tools/bench_q4_v41.m src/myds_gguf.c -o /tmp/bench_q4_v41
 * Required: --metallib PROBE --golden-metallib PRODUCTION --shape head|qb|ob
 * --out-dir NEW_DIRECTORY --model GGUF [--layer 0 --rotations 4].
 * --synthetic is explicit; --host-only validates argv/GGUF without Metal.
 * A1/A2 use real production PSOs. Clone and candidates must first match them.
 * hot and rotating are isolated components, not cold-DRAM/full-model proof.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "myds_gguf.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { GUARD=64 };
struct mv_args {
    int32_t ne00,ne01,ne02; uint64_t nb00,nb01,nb02,nb03;
    int32_t ne10,ne11,ne12; uint64_t nb10,nb11,nb12,nb13;
    int32_t ne0,ne1,nr0; int16_t r2,r3;
};
_Static_assert(sizeof(struct mv_args)==112,"Q4 production ABI");
static NSMutableDictionary *artifact;
static NSMutableArray *files;
static NSString *evidence_dir;
static void finish_manifest(NSString *status) {
    if(!artifact||!evidence_dir)return;
    artifact[@"status"]=status;artifact[@"files"]=files;
    NSData *j=[NSJSONSerialization dataWithJSONObject:artifact options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    if(j)[j writeToFile:[evidence_dir stringByAppendingPathComponent:@"artifact-manifest.json"] atomically:YES];
}
static void req(BOOL ok,const char *message) {
    if(ok)return;
    fprintf(stderr,"bench_q4_v41: %s\n",message);
    if(artifact)artifact[@"failure"]=[NSString stringWithUTF8String:message];
    finish_manifest(@"failed");exit(1);
}
static NSString *sha(const void *data,size_t bytes) {
    CC_SHA256_CTX c;CC_SHA256_Init(&c);const uint8_t *p=data;
    while(bytes){CC_LONG n=(CC_LONG)(bytes>0x10000000u?0x10000000u:bytes);CC_SHA256_Update(&c,p,n);p+=n;bytes-=n;}
    unsigned char d[32];CC_SHA256_Final(d,&c);char s[65];
    for(unsigned i=0;i<32;i++)snprintf(s+2*i,3,"%02x",d[i]);return [NSString stringWithUTF8String:s];
}
static NSString *file_sha(NSString *p) {
    NSData *d=[NSData dataWithContentsOfFile:p options:NSDataReadingMappedIfSafe error:nil];
    req(d!=nil,"identity file read");return sha(d.bytes,d.length);
}
static NSString *hex(const void *p,size_t n) {
    const uint8_t *s=p;NSMutableString *v=[NSMutableString stringWithCapacity:2*n];
    for(size_t i=0;i<n;i++)[v appendFormat:@"%02x",s[i]];
    return v;
}
static void emit(NSDictionary *d) {
    NSData *j=[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingSortedKeys error:nil];req(j!=nil,"stdout JSON");
    fwrite(j.bytes,1,j.length,stdout);putchar('\n');fflush(stdout);
}
static void raw(NSString *name,const void *p,size_t n) {
    NSString *path=[evidence_dir stringByAppendingPathComponent:name];
    req(![[NSFileManager defaultManager]fileExistsAtPath:path],"refuse raw overwrite");
    FILE *f=fopen(path.fileSystemRepresentation,"wb");req(f!=NULL,"raw open");
    req(fwrite(p,1,n,f)==n&&!fclose(f),"raw write");
    [files addObject:@{@"path":name,@"bytes":@(n),@"sha256":sha(p,n)}];
}
static unsigned number(const char *s,unsigned lo,unsigned hi) {
    char *e=NULL;unsigned long n=strtoul(s,&e,10);req(*s&&e&&!*e&&n>=lo&&n<=hi,"numeric range");return (unsigned)n;
}
static uint64_t rng=UINT64_C(88172645463325252);
static uint64_t random64(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static BOOL canary(id<MTLBuffer>b,size_t n){const uint8_t*p=b.contents;for(unsigned i=0;i<GUARD;i++)if(p[i]!=0xa5||p[GUARD+n+i]!=0xa5)return NO;return YES;}
static BOOL finite(const void*p,size_t n){const float*x=p;for(size_t i=0;i<n;i++)if(!isfinite(x[i]))return NO;return YES;}
static BOOL written(const void*p,size_t n){const uint32_t*x=p;for(size_t i=0;i<n;i++)if(x[i]==0xa5a5a5a5u)return NO;return YES;}
static double med(const double*x,unsigned n){double a[31];memcpy(a,x,n*sizeof(*x));for(unsigned i=1;i<n;i++){double t=a[i];unsigned j=i;while(j&&a[j-1]>t){a[j]=a[j-1];j--;}a[j]=t;}return n%2?a[n/2]:(a[n/2-1]+a[n/2])/2;}
static id<MTLComputePipelineState> pso(id<MTLDevice>d,id<MTLLibrary>l,NSString*n) {
    MTLFunctionConstantValues*v=[MTLFunctionConstantValues new];short nsg=4;[v setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError*e=nil;id<MTLFunction>f=[l newFunctionWithName:n constantValues:v error:&e];
    if(!f)fprintf(stderr,"PSO %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);
    req(f!=nil,"missing explicit PSO; no fallback");
    id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];
    req(p!=nil&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"PSO geometry");
    return p;
}
int main(int argc,char**argv) {
 @autoreleasepool {
    NSString *libpath=nil,*goldpath=nil,*model=nil,*shape=nil,*dir=nil,*inputfile=nil;
    unsigned layer=0,rotations=4,seeds=3,samples=5,reps=1;BOOL host=NO,synthetic=NO;
    for(int i=1;i<argc;i++){
        const char *k=argv[i];
        if(!strcmp(k,"--host-only")){host=YES;continue;}
        if(!strcmp(k,"--synthetic")){synthetic=YES;continue;}
        req(i+1<argc,"incomplete argv");const char*v=argv[++i];
        if(!strcmp(k,"--metallib"))libpath=[NSString stringWithUTF8String:v];
        else if(!strcmp(k,"--golden-metallib"))goldpath=[NSString stringWithUTF8String:v];
        else if(!strcmp(k,"--model"))model=[NSString stringWithUTF8String:v];
        else if(!strcmp(k,"--shape"))shape=[NSString stringWithUTF8String:v];
        else if(!strcmp(k,"--out-dir"))dir=[NSString stringWithUTF8String:v];
        else if(!strcmp(k,"--input-f32"))inputfile=[NSString stringWithUTF8String:v];
        else if(!strcmp(k,"--layer"))layer=number(v,0,39);
        else if(!strcmp(k,"--rotations"))rotations=number(v,1,16);
        else if(!strcmp(k,"--seeds"))seeds=number(v,1,32);
        else if(!strcmp(k,"--samples"))samples=number(v,3,31);
        else if(!strcmp(k,"--reps"))reps=number(v,1,64);
        else req(NO,"unknown argv");
    }
    req(shape&&((model!=nil)!=synthetic),"choose --model or explicit --synthetic and a shape");
    unsigned rows=0,width=0;BOOL bf16=NO;NSString*stem=nil,*tensor=nil;
    if([shape isEqual:@"head"]){rows=129280;width=5120;stem=@"head";tensor=@"output.weight";}
    else if([shape isEqual:@"qb"]){rows=32768;width=1280;bf16=YES;stem=@"qb";tensor=@"attn_q_b";}
    else if([shape isEqual:@"ob"]){rows=5120;width=8192;bf16=YES;stem=@"ob";tensor=@"attn_output_b";}
    else req(NO,"unsupported shape");
    req([shape isEqual:@"head"]||layer+rotations<=40,"rotating GGUF layer range");
    const size_t rowbytes=(size_t)(width/256u)*144u,wbytes=rowbytes*rows,obytes=(size_t)rows*4u,ibytes=(size_t)width*4u;
    req(rows%8u==0&&wbytes<UINT32_MAX&&obytes<UINT32_MAX&&ibytes<UINT32_MAX,"uint32 legal shape bounds");
    gguf_file_t*gf=model?gguf_open(model.fileSystemRepresentation):NULL;req(!model||gf!=NULL,"GGUF read-only parse");
    NSMutableArray *descriptors=[NSMutableArray new];
    for(unsigned slot=0;slot<rotations;slot++){
        NSString*n=[shape isEqual:@"head"]?tensor:[NSString stringWithFormat:@"blk.%u.%@.weight",layer+slot,tensor];
        const gguf_tensor_t*t=gf?gguf_find_tensor(gf,n.UTF8String):NULL;
        req(!gf||(t&&t->type==GGML_TYPE_Q4_K&&t->n_dims==2&&t->dims[0]==width&&t->dims[1]==rows&&t->size_bytes==wbytes),"real tensor name/type/dimensions/length");
        [descriptors addObject:@{@"slot":@(slot),@"tensor":n,@"absolute_offset":gf?@(gf->tensor_data_offset+t->offset):(id)[NSNull null],@"bytes":@(wbytes)}];
    }
    NSData*captured=inputfile?[NSData dataWithContentsOfFile:inputfile]:nil;
    req(!inputfile||(captured&&captured.length==ibytes&&finite(captured.bytes,width)),"captured activation full finite F32 length");
    if(host){emit(@{@"kind":@"host_only",@"pass":@YES,@"gpu_executed":@NO,@"shape":shape,@"rows":@(rows),@"width":@(width),@"descriptors":descriptors});if(gf)gguf_close(gf);return 0;}
    req(libpath&&goldpath&&dir,"required probe/golden libraries and new output directory");
    req(![[NSFileManager defaultManager]fileExistsAtPath:dir],"output directory must not exist");
    req([[NSFileManager defaultManager]createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil],"create evidence directory");
    evidence_dir=dir;files=[NSMutableArray new];artifact=[NSMutableDictionary dictionary];
    artifact[@"schema_version"]=@1;artifact[@"argv"]=[NSProcessInfo processInfo].arguments;
    artifact[@"shape"]=shape;artifact[@"rows"]=@(rows);artifact[@"width"]=@(width);artifact[@"nr0"]=@2;artifact[@"nsg"]=@4;
    artifact[@"rotations"]=@(rotations);artifact[@"samples"]=@(samples);artifact[@"seeds"]=@(seeds);artifact[@"reps"]=@(reps);
    artifact[@"synthetic_weights"]=@(synthetic);artifact[@"activation_source"]=captured?@"captured_F32":@"seeded_F32";
    artifact[@"model_path"]=model?:[NSNull null];artifact[@"cold_dram_proven"]=@NO;artifact[@"fullmodel_speedup_proven"]=@NO;
    if(gf){
        struct stat st;req(!fstat(gf->fd,&st),"GGUF descriptor identity");
        artifact[@"model_identity"]=@{@"bytes":@((uint64_t)st.st_size),@"inode":@((uint64_t)st.st_ino),
            @"mtime_ns":@((uint64_t)st.st_mtimespec.tv_sec*1000000000ull+st.st_mtimespec.tv_nsec),
            @"header_bytes":@(gf->tensor_data_offset),@"header_sha256":sha(gf->mmap_addr,gf->tensor_data_offset),@"full_model_sha256_computed":@NO};
    }
    artifact[@"candidate_metallib_sha256"]=file_sha(libpath);artifact[@"production_metallib_sha256"]=file_sha(goldpath);
    artifact[@"binary_sha256"]=file_sha([NSProcessInfo processInfo].arguments[0]);
    id<MTLDevice>d=MTLCreateSystemDefaultDevice();req(d!=nil,"Metal device");
    req(wbytes+2*GUARD<=d.maxBufferLength,"weight range exceeds device maxBufferLength");
    artifact[@"device"]=d.name;NSError*error=nil;
    id<MTLLibrary>lib=[d newLibraryWithURL:[NSURL fileURLWithPath:libpath] error:&error];req(lib!=nil,"probe library");
    id<MTLLibrary>gold=[d newLibraryWithURL:[NSURL fileURLWithPath:goldpath] error:&error];req(gold!=nil,"production library");
    NSString*reference=bf16?@"kernel_mul_mv_q4_K_dense_f32_bf16":@"kernel_mul_mv_q4_K_dense_f32";
    NSString*epilogue=bf16?@"bf16":@"f32";
    NSArray *names=@[reference,reference,[NSString stringWithFormat:@"kernel_myds_q4_v41_%@_addr_%@",stem,epilogue],[NSString stringWithFormat:@"kernel_myds_q4_v41_%@_nb_%@",stem,epilogue]];
    NSArray *psos=@[pso(d,gold,names[0]),pso(d,lib,names[1]),pso(d,lib,names[2]),pso(d,lib,names[3])];
    NSMutableArray*identities=[NSMutableArray new];
    for(unsigned k=0;k<4;k++){id<MTLComputePipelineState>p=psos[k];[identities addObject:@{@"index":@(k),@"kernel":names[k],@"production_library":@(k==0),@"NSG600":@4,@"max_threads":@(p.maxTotalThreadsPerThreadgroup),@"static_tg_bytes":@(p.staticThreadgroupMemoryLength)}];}
    artifact[@"psos"]=identities;artifact[@"epilogue"]=epilogue;artifact[@"dynamic_tg_bytes"]=@256;
    NSMutableArray *weights=[NSMutableArray new],*inputs=[NSMutableArray new],*outputs=[NSMutableArray new],*weightShas=[NSMutableArray new];
    NSMutableSet *savedWeights=[NSMutableSet new];
    for(unsigned slot=0;slot<rotations;slot++){
        id<MTLBuffer>w=[d newBufferWithLength:wbytes+2*GUARD options:MTLResourceStorageModeShared];
        id<MTLBuffer>x=[d newBufferWithLength:ibytes+2*GUARD options:MTLResourceStorageModeShared];
        id<MTLBuffer>y=[d newBufferWithLength:obytes+2*GUARD options:MTLResourceStorageModeShared];
        req(w&&x&&y,"complete guarded buffers");memset(w.contents,0xa5,w.length);memset(x.contents,0xa5,x.length);memset(y.contents,0xa5,y.length);
        NSString*n=descriptors[slot][@"tensor"];
        if(gf){const gguf_tensor_t*t=gguf_find_tensor(gf,n.UTF8String);memcpy((uint8_t*)w.contents+GUARD,t->data,wbytes);}
        else {uint8_t*p=(uint8_t*)w.contents+GUARD;for(size_t b=0;b<wbytes/144u;b++){uint16_t v=0x2400u,mi=(uint16_t)(0x1800u+b%1024u);memcpy(p+b*144u,&v,2);memcpy(p+b*144u+2,&mi,2);for(unsigned z=4;z<144;z++)p[b*144u+z]=(uint8_t)random64();}}
        NSString*h=sha((uint8_t*)w.contents+GUARD,wbytes);[weightShas addObject:h];
        NSMutableDictionary*desc=[descriptors[slot] mutableCopy];desc[@"sha256"]=h;
        NSString*filename=[NSString stringWithFormat:@"weight-%@.q4k",h];
        desc[@"raw_file"]=filename;descriptors[slot]=desc;
        if(![savedWeights containsObject:h]){raw(filename,(uint8_t*)w.contents+GUARD,wbytes);[savedWeights addObject:h];}
        [weights addObject:w];[inputs addObject:x];[outputs addObject:y];
    }
    if(gf)gguf_close(gf);artifact[@"weights"]=descriptors;
    id<MTLCommandQueue>q=[d newCommandQueue];req(q!=nil,"queue");
    const struct mv_args a={.ne00=(int32_t)width,.ne01=(int32_t)rows,.ne02=1,.nb00=144,.nb01=rowbytes,.nb02=wbytes,.nb03=wbytes,.ne10=(int32_t)width,.ne11=1,.ne12=1,.nb10=4,.nb11=ibytes,.nb12=ibytes,.nb13=ibytes,.ne0=(int32_t)rows,.ne1=1,.nr0=2,.r2=1,.r3=1};
    artifact[@"production_args_hex"]=hex(&a,sizeof(a));
    __block double last_gpu_start=0,last_gpu_end=0;
    double(^run)(unsigned,unsigned,unsigned)=^double(unsigned k,unsigned active,unsigned repeat){
        for(unsigned slot=0;slot<active;slot++){id<MTLBuffer>y=outputs[slot];memset((uint8_t*)y.contents+GUARD,0xa5,obytes);}
        id<MTLCommandBuffer>cb=[q commandBuffer];req(cb!=nil,"command buffer");
        id<MTLComputeCommandEncoder>e=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];req(e!=nil&&e.dispatchType==MTLDispatchTypeConcurrent,"actual concurrent encoder");
        for(unsigned r=0;r<repeat;r++)for(unsigned slot=0;slot<active;slot++){
            id<MTLBuffer>w=weights[slot],x=inputs[slot],y=outputs[slot];
            id<MTLResource>join[]={y};[e memoryBarrierWithResources:join count:1];
            [e setComputePipelineState:psos[k]];[e setBytes:&a length:sizeof(a) atIndex:0];
            [e setBuffer:w offset:GUARD atIndex:1];[e setBuffer:x offset:GUARD atIndex:2];[e setBuffer:y offset:GUARD atIndex:3];
            [e setThreadgroupMemoryLength:256 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(rows/8u,1,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];
        }
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"GPU completion: %s\n",cb.error.localizedDescription.UTF8String);
        req(cb.status==MTLCommandBufferStatusCompleted&&isfinite(cb.GPUStartTime)&&isfinite(cb.GPUEndTime)&&cb.GPUEndTime>cb.GPUStartTime,"completed CB/timestamp");
        last_gpu_start=cb.GPUStartTime;last_gpu_end=cb.GPUEndTime;
        return (last_gpu_end-last_gpu_start)*1e6/(active*repeat);
    };
    emit(@{@"kind":@"identity",@"shape":shape,@"rows":@(rows),@"width":@(width),@"rotations":@(rotations),@"samples":@(samples),@"seeds":@(seeds),@"reps":@(reps),@"psos":identities,@"weights":descriptors,@"device":d.name,@"cold_dram_proven":@NO});
    NSMutableArray *frozen=[NSMutableArray new];
    for(unsigned seed=0;seed<seeds;seed++){
        for(unsigned slot=0;slot<rotations;slot++){id<MTLBuffer>x=inputs[slot];float*p=(float*)((uint8_t*)x.contents+GUARD);
            if(captured)memcpy(p,captured.bytes,ibytes);
            else for(unsigned j=0;j<width;j++){p[j]=((int)(random64()%200001u)-100000)/32768.f;if(j%17u==0)p[j]=-0.f;if(seed%3==1)p[j]*=16.f;if(seed%3==2)p[j]=ldexpf(p[j],(int)(j%13)-6);}
            raw([NSString stringWithFormat:@"seed%u-slot%u-input.f32",seed,slot],p,ibytes);
        }
        [frozen removeAllObjects];
        for(unsigned k=0;k<4;k++){
            (void)run(k,rotations,1);BOOL exact=YES;
            for(unsigned slot=0;slot<rotations;slot++){id<MTLBuffer>y=outputs[slot];const uint8_t*p=(const uint8_t*)y.contents+GUARD;
                raw([NSString stringWithFormat:@"seed%u-slot%u-k%u-output.guarded",seed,slot,k],y.contents,y.length);
                const BOOL fin=finite(p,rows),g=canary(y,obytes),wrote=written(p,rows);
                if(k==0)[frozen addObject:[NSData dataWithBytes:p length:obytes]];
                NSData*oracle=frozen[slot];BOOL same=!memcmp(oracle.bytes,p,obytes);exact&=fin&&g&&same&&wrote;
                if(bf16){const uint32_t*u=(const uint32_t*)p;for(unsigned j=0;j<rows;j++)exact&=(u[j]&0xffffu)==0;}
                emit(@{@"kind":@"parity",@"seed":@(seed),@"slot":@(slot),@"pso":@(k),@"finite":@(fin),@"guards":@(g),@"poison_free":@(wrote),@"full_output_same":@(same),@"sha256":sha(p,obytes),@"production_golden":@(k==0)});
            }
            req(exact,"full production golden/clone/candidate bytes, finite or guards gate");
        }
    }
    // Each formal output is compared with the last complete production oracle.
    for(unsigned candidate=2;candidate<4;candidate++)for(unsigned scheme=0;scheme<2;scheme++){
        unsigned active=scheme?rotations:1;NSString*schemeName=scheme?@"rotating":@"hot";double m[3];
        for(unsigned arm=0;arm<3;arm++){
            unsigned k=arm==1?candidate:0;NSString*armName=arm==0?@"A1":arm==1?@"B":@"A2";double t[31];
            for(unsigned sample=0;sample<=samples;sample++){
                double us=run(k,active,reps);BOOL same=YES,g=YES,fin=YES,wrote=YES;
                for(unsigned slot=0;slot<active;slot++){id<MTLBuffer>y=outputs[slot];const uint8_t*p=(const uint8_t*)y.contents+GUARD;NSData*oracle=frozen[slot];
                    same&=!memcmp(oracle.bytes,p,obytes);g&=canary(y,obytes);fin&=finite(p,rows);wrote&=written(p,rows);
                    raw([NSString stringWithFormat:@"timing-c%u-%@-%@-s%u-slot%u-output.guarded",candidate,schemeName,armName,sample,slot],y.contents,y.length);
                }
                emit(@{@"kind":@"timing",@"candidate":@(candidate),@"scheme":schemeName,@"arm":armName,@"pso":@(k),@"sample":@(sample),@"warmup":@(sample==0),@"gpu_us":@(us),@"gpu_start_s":@(last_gpu_start),@"gpu_end_s":@(last_gpu_end),@"dispatches":@(active*reps),@"full_output_same":@(same),@"guards":@(g),@"finite":@(fin),@"poison_free":@(wrote),@"sample_finite":@(isfinite(us)&&us>0),@"production_control":@(k==0)});
                req(same&&g&&fin&&wrote&&isfinite(us)&&us>0,"timing full output/guards/finite");if(sample)t[sample-1]=us;
            }
            m[arm]=med(t,samples);
        }
        emit(@{@"kind":@"summary",@"candidate":@(candidate),@"scheme":schemeName,@"a1_us":@(m[0]),@"b_us":@(m[1]),@"a2_us":@(m[2]),@"time_saved_pct":@((1-m[1]/((m[0]+m[2])/2))*100),@"control_drift_pct":@((m[2]/m[0]-1)*100),@"fullmodel_speedup_proven":@NO});
    }
    for(unsigned slot=0;slot<rotations;slot++){
        id<MTLBuffer>w=weights[slot],x=inputs[slot];
        uint8_t wg[2*GUARD];memcpy(wg,w.contents,GUARD);memcpy(wg+GUARD,(uint8_t*)w.contents+GUARD+wbytes,GUARD);
        raw([NSString stringWithFormat:@"weight-slot%u-guards.bin",slot],wg,sizeof(wg));
        raw([NSString stringWithFormat:@"final-slot%u-input.guarded",slot],x.contents,x.length);
        req(canary(w,wbytes)&&canary(x,ibytes),"input/weight guards");
        req([sha((uint8_t*)w.contents+GUARD,wbytes)isEqual:weightShas[slot]],"weight bytes changed");
    }
    artifact[@"all_output_gates_passed"]=@YES;artifact[@"input_guards_passed"]=@YES;
    finish_manifest(@"complete");emit(@{@"kind":@"complete",@"pass":@YES,@"raw_files":@(files.count),@"gpu_executed":@YES});
 }
 return 0;
}

