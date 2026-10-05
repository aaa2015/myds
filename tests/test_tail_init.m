/* Compile the actual production translation unit unchanged. Internal reads
 * prove cleanup clears state. A test-only Objective-C interceptor below makes
 * one real Metal PSO creation API fail; there are no production runtime hooks. */
#include "myds_metal.m"
#include <limits.h>
#include <objc/runtime.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
static int fail_mandatory_pso;
static unsigned mandatory_failure_hits;
static Method fixture_original_pso_method,fixture_interceptor_method;
@interface NSObject (MydsTailInitFixture)
- (id<MTLComputePipelineState>)myds_fixture_newComputePipelineStateWithFunction:(id<MTLFunction>)function
    error:(NSError *__autoreleasing *)error NS_RETURNS_RETAINED;
@end
@implementation NSObject (MydsTailInitFixture)
- (id<MTLComputePipelineState>)myds_fixture_newComputePipelineStateWithFunction:(id<MTLFunction>)function
    error:(NSError *__autoreleasing *)error {
    if(fail_mandatory_pso && [function.name isEqualToString:@"kernel_rms_norm_mul_f32_4"]){
        ++mandatory_failure_hits;
        if(error)*error=[NSError errorWithDomain:@"myds.tail-init.fixture" code:1 userInfo:nil];
        return nil;
    }
    /* During interception this selector resolves to the original +1-returning
     * Metal IMP. NS_RETURNS_RETAINED preserves its new-method ARC contract. */
    return [self myds_fixture_newComputePipelineStateWithFunction:function error:error];
}
@end
static void install_mandatory_interceptor(void) {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();REQUIRE(device);
    Class cls=object_getClass(device);
    SEL original=@selector(newComputePipelineStateWithFunction:error:);
    SEL replacement=@selector(myds_fixture_newComputePipelineStateWithFunction:error:);
    Method old=class_getInstanceMethod(cls,original);
    Method hook=class_getInstanceMethod([NSObject class],replacement);REQUIRE(old&&hook);
    /* Make both methods belong to the concrete device class. Exchanging an
     * inherited Method would otherwise mutate its superclass implementation. */
    (void)class_addMethod(cls,original,method_getImplementation(old),method_getTypeEncoding(old));
    REQUIRE(class_addMethod(cls,replacement,method_getImplementation(hook),method_getTypeEncoding(hook)));
    fixture_original_pso_method=class_getInstanceMethod(cls,original);
    fixture_interceptor_method=class_getInstanceMethod(cls,replacement);
    method_exchangeImplementations(fixture_original_pso_method,fixture_interceptor_method);
}
static void remove_mandatory_interceptor(void) {
    method_exchangeImplementations(fixture_original_pso_method,fixture_interceptor_method);
    fixture_original_pso_method=fixture_interceptor_method=NULL;
}
static void require_cold(const char *phase) {
    REQUIRE(!g_initialized);
#define COLD(x) do { if ((x)!=nil) { fprintf(stderr,"FAIL cold phase=%s object=%s\n",phase,#x);exit(1); } } while(0)
#include "cold_id_checks.inc"
#undef COLD
    REQUIRE(g_native_pipelines==nil);
    REQUIRE(g_inflight_cbs==nil);
    REQUIRE(g_native_pso_cache_count==0);
    REQUIRE(g_specialized_pso_cache_count==0);
    REQUIRE(g_specialized_pso_cache_next==0);
    REQUIRE(g_moe_id_map_bytes==0);
    REQUIRE(g_shared_event_val==0);
    REQUIRE(g_qkv_finish_available==-1);
    printf("PASS state cleared: %s\n",phase);fflush(stdout);
}
static void assert_library(int has_tail) {
    REQUIRE(g_initialized&&g_library&&g_queue&&g_device);
    REQUIRE(([g_library newFunctionWithName:@"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_tail_cull"]!=nil)==has_tail);
    REQUIRE([g_library newFunctionWithName:@"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16"]!=nil);
    REQUIRE(g_pso_moe_mm_iq2_pair_swiglu_f16!=nil);
}
static uint64_t hash_bytes(const void *data,size_t n) {
    const unsigned char *p=data;uint64_t h=14695981039346656037ull;
    for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;
}
static void run_moe(float *reference,int save,const char *phase) {
    const uint32_t dim=256,width=256,E=16,K=6,B=40;
    const size_t gate_bytes=(size_t)E*width*66u,down_bytes=(size_t)E*dim*84u;
    const size_t out_bytes=(size_t)B*dim*4u,mid_bytes=(size_t)B*K*width*4u;
    myds_metal_buffer_t *out=myds_metal_buffer_alloc(out_bytes),*x=myds_metal_buffer_alloc(out_bytes);
    myds_metal_buffer_t *gate=myds_metal_buffer_alloc(gate_bytes),*up=myds_metal_buffer_alloc(gate_bytes);
    myds_metal_buffer_t *down=myds_metal_buffer_alloc(down_bytes);
    myds_metal_buffer_t *ids=myds_metal_buffer_alloc((size_t)B*K*4u),*weights=myds_metal_buffer_alloc((size_t)B*K*4u);
    myds_metal_buffer_t *sg=myds_metal_buffer_alloc(mid_bytes),*su=myds_metal_buffer_alloc(mid_bytes);
    myds_metal_buffer_t *sm=myds_metal_buffer_alloc(mid_bytes),*sd=myds_metal_buffer_alloc((size_t)B*K*dim*4u);
    REQUIRE(out&&x&&gate&&up&&down&&ids&&weights&&sg&&su&&sm&&sd);
    memset(myds_metal_buffer_data(out),0xff,out_bytes);
    float *xp=myds_metal_buffer_data(x),*wp=myds_metal_buffer_data(weights);
    int32_t *ip=myds_metal_buffer_data(ids);
    for(uint32_t i=0;i<B*dim;++i)xp[i]=0.2f+(int)(i*31u%67u)/128.0f;
    const uint32_t tails[]={1,16,17,31,32,33};
    for(uint32_t t=0;t<B;++t)for(uint32_t k=0;k<K;++k){
        ip[(size_t)t*K+k]=(int32_t)(t<tails[k]?k:k+6u);
        wp[(size_t)t*K+k]=0.1f+0.01f*((t+2*k)%11u);
    }
    uint8_t *gp=myds_metal_buffer_data(gate),*upv=myds_metal_buffer_data(up),*dp=myds_metal_buffer_data(down);
    memset(gp,0x22,gate_bytes);memset(upv,0x35,gate_bytes);memset(dp,0x22,down_bytes);
    for(uint32_t r=0;r<E*width;++r){
        _Float16 g=(_Float16)(0.013f+0.0001f*(r%13u)),u=(_Float16)(0.009f+0.0001f*(r%17u));
        memcpy(gp+(size_t)r*66u,&g,2);memcpy(upv+(size_t)r*66u,&u,2);
    }
    for(uint32_t r=0;r<E*dim;++r){
        _Float16 d=(_Float16)(0.007f+0.00007f*(r%17u));
        memcpy(dp+(size_t)r*84u+80,&d,2);dp[(size_t)r*84u+82]=dp[(size_t)r*84u+83]=0;
    }
    myds_metal_dispatch_count_reset();
    REQUIRE(myds_metal_routed_moe_batch(out,x,gate,up,down,ids,weights,sg,su,sm,sd,
        dim,width,E,K,B,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,10.0f));
    REQUIRE(myds_metal_dispatch_count()==4); /* proves MM selected, not MV fallback */
    const float *actual=myds_metal_buffer_data(out);unsigned nonzero=0;
    for(size_t i=0;i<out_bytes/4;++i){REQUIRE(isfinite(actual[i]));nonzero+=actual[i]!=0;}
    REQUIRE(nonzero);
    if(save)memcpy(reference,actual,out_bytes);else REQUIRE(!memcmp(reference,actual,out_bytes));
    /* Populate a native PSO cache as well, then validate cleanup resets it. */
    REQUIRE(myds_metal_v41_qkv_fusion_available());
    REQUIRE(g_native_pipelines.count>0);
    printf("PASS nonzero MM path: %s dispatches=4 FNV=%016llx\n",phase,(unsigned long long)hash_bytes(actual,out_bytes));fflush(stdout);
    myds_metal_buffer_t *all[]={out,x,gate,up,down,ids,weights,sg,su,sm,sd};
    for(size_t i=0;i<sizeof(all)/sizeof(*all);++i)myds_metal_buffer_free(all[i]);
}
static void fail_explicit(const char *dir,const char *phase) {
    REQUIRE(!myds_metal_init(dir));
    require_cold(phase);
    REQUIRE(!myds_metal_v41_qkv_fusion_available());
    myds_metal_cleanup();
    require_cold("caller cleanup after failed init");
}
int main(int argc,char **argv) {
    if(argc!=3){fprintf(stderr,"usage: test_tail_init MISSING_TAIL_LIB_DIR FULL_LIB_DIR\n");return 2;}
    @autoreleasepool {
        REQUIRE(unsetenv("MYDS_DISABLE_MOE_MM")==0);
        REQUIRE(setenv("MYDS_MOE_MM_MIN_TOKENS","1",1)==0);
        REQUIRE(unsetenv("MYDS_MOE_STEP_PROF")==0);
        float reference[40*256];
        require_cold("process start");
        REQUIRE(unsetenv("MYDS_MOE_IQ2_TAIL_CULL")==0);
        REQUIRE(myds_metal_init(argv[1]));assert_library(0);
        run_moe(reference,1,"unset env/missing tail -> original MM fallback");
        myds_metal_cleanup();require_cold("default fallback cleanup");
        REQUIRE(setenv("MYDS_MOE_IQ2_TAIL_CULL","1",1)==0);
        fail_explicit(argv[1],"explicit tail missing first failure");
        fail_explicit(argv[1],"explicit tail missing repeated failure");
        REQUIRE(setenv("MYDS_MOE_IQ2_TAIL_CULL","0",1)==0);
        REQUIRE(myds_metal_init(argv[1]));assert_library(0);
        run_moe(reference,0,"explicit zero/missing tail -> original MM");
        myds_metal_cleanup();require_cold("explicit zero cleanup");
        REQUIRE(setenv("MYDS_MOE_IQ2_TAIL_CULL","1",1)==0);
        REQUIRE(myds_metal_init(argv[2]));assert_library(1);
        run_moe(reference,0,"same process/full library retry -> explicit tail MM");
        myds_metal_cleanup();require_cold("explicit tail success cleanup");
        REQUIRE(unsetenv("MYDS_MOE_IQ2_TAIL_CULL")==0);
        REQUIRE(myds_metal_init(argv[2]));assert_library(1);
        run_moe(reference,0,"unset env/full library -> default tail MM");
        myds_metal_cleanup();require_cold("default tail success cleanup");
        REQUIRE(setenv("MYDS_MOE_IQ2_TAIL_CULL","1",1)==0);
        install_mandatory_interceptor();fail_mandatory_pso=1;
        REQUIRE(!myds_metal_init(argv[2]));
        fail_mandatory_pso=0;REQUIRE(mandatory_failure_hits==1);
        require_cold("mandatory PSO API failure cleanup");
        REQUIRE(!myds_metal_v41_qkv_fusion_available());
        remove_mandatory_interceptor();
        fail_explicit(argv[1],"missing tail remains strict after mandatory PSO failure");
        REQUIRE(myds_metal_init(argv[2]));assert_library(1);
        run_moe(reference,0,"full library retry after mandatory PSO failure");
        myds_metal_cleanup();require_cold("mandatory-failure retry cleanup");
        fail_explicit(argv[1],"missing tail failure after warmed native cache");
        /* begin_batch intentionally performs a fresh default-path init. Make
         * that path deterministic, then prove a failed explicit init has not
         * poisoned the lazy-init contract. Close the empty batch before the
         * standalone MoE call, which performs its own commit/wait. */
        char saved_cwd[PATH_MAX];
        REQUIRE(getcwd(saved_cwd,sizeof(saved_cwd))!=NULL);
        NSString *full_dir=[[[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath] stringByResolvingSymlinksInPath];
        REQUIRE([full_dir.lastPathComponent isEqualToString:@"bin"]);
        REQUIRE(chdir(full_dir.stringByDeletingLastPathComponent.fileSystemRepresentation)==0);
        REQUIRE(myds_metal_begin_batch());assert_library(1);
        REQUIRE(myds_metal_commit_batch());
        run_moe(reference,0,"begin_batch/full default path -> lazy init after explicit failure");
        myds_metal_cleanup();require_cold("lazy init success cleanup");
        REQUIRE(chdir(saved_cwd)==0);
        printf("ALL TAIL INIT/FALLBACK/RETRY FIXTURES PASS\n");
    }
    return 0;
}
