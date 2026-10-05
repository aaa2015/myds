/* Opt-in GPU test; root schedules execution during the exclusive test window.
 * Includes the unchanged production runtime to inject a real fused-PSO creation
 * failure without adding a production hook. Inputs/weights are seeded fixtures,
 * not captured model activations. Full-model IDs/bytes are checked separately.
 */
#include "myds_metal.m"
#include "myds_v41_decode.h"
#include <objc/runtime.h>

#define REQUIRE(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#c); exit(1); } } while (0)
static int missing_fused_pso;
static unsigned pso_failure_hits;
static Method original_pso_method, interceptor_method;
@interface NSObject (MydsDecodeQ8Fixture)
- (id<MTLComputePipelineState>)myds_decode_fixture_newComputePipelineStateWithFunction:(id<MTLFunction>)function
    error:(NSError *__autoreleasing *)error NS_RETURNS_RETAINED;
@end
@implementation NSObject (MydsDecodeQ8Fixture)
- (id<MTLComputePipelineState>)myds_decode_fixture_newComputePipelineStateWithFunction:(id<MTLFunction>)function
    error:(NSError *__autoreleasing *)error {
    if (missing_fused_pso && ([function.name isEqualToString:@"kernel_mul_mv_q8_0_f32_bf16"] ||
                             [function.name isEqualToString:@"kernel_mul_mv_q8_0_f32_bf16_r4"])) {
        ++pso_failure_hits;
        if (error) *error = [NSError errorWithDomain:@"myds.decode-q8.fixture" code:1 userInfo:nil];
        return nil;
    }
    return [self myds_decode_fixture_newComputePipelineStateWithFunction:function error:error];
}
@end
static void install_interceptor(void) {
    Class cls = object_getClass(g_device);
    SEL original = @selector(newComputePipelineStateWithFunction:error:);
    SEL replacement = @selector(myds_decode_fixture_newComputePipelineStateWithFunction:error:);
    Method old = class_getInstanceMethod(cls, original);
    Method hook = class_getInstanceMethod([NSObject class], replacement);
    REQUIRE(old && hook);
    (void)class_addMethod(cls,original,method_getImplementation(old),method_getTypeEncoding(old));
    REQUIRE(class_addMethod(cls,replacement,method_getImplementation(hook),method_getTypeEncoding(hook)));
    original_pso_method = class_getInstanceMethod(cls,original);
    interceptor_method = class_getInstanceMethod(cls,replacement);
    method_exchangeImplementations(original_pso_method,interceptor_method);
}
static void remove_interceptor(void) {
    method_exchangeImplementations(original_pso_method,interceptor_method);
    original_pso_method = interceptor_method = NULL;
}
struct guarded { myds_metal_buffer_t *owner, *view; size_t bytes, offset; };
static struct guarded allocate_at(size_t bytes, size_t offset) {
    struct guarded b = {.bytes=bytes,.offset=offset};
    b.owner = myds_metal_buffer_alloc(bytes + offset + 16); REQUIRE(b.owner);
    memset(myds_metal_buffer_data(b.owner),0xa5,bytes+offset+16);
    b.view = myds_metal_buffer_sub_view(b.owner,offset,bytes); REQUIRE(b.view); return b;
}
static struct guarded allocate(size_t bytes) {
    return allocate_at(bytes,16);
}
static void canary(struct guarded b) {
    const uint8_t *p = myds_metal_buffer_data(b.owner);
    for (size_t i=0;i<b.offset;i++) REQUIRE(p[i]==0xa5);
    for (size_t i=0;i<16;i++) REQUIRE(p[b.offset+b.bytes+i]==0xa5);
}
static void release(struct guarded b) {
    myds_metal_buffer_free(b.view); myds_metal_buffer_free(b.owner);
}
static void input_fill(myds_metal_buffer_t *b) {
    float *p = myds_metal_buffer_data(b);
    for(size_t i=0;i<myds_metal_buffer_size(b)/4;i++)
        p[i] = ((int)((i*19u+17u)%257u)-128)/512.0f;
}
static void weight_fill(myds_metal_buffer_t *b, ggml_type type) {
    const size_t bytes = myds_metal_buffer_size(b);
    if (type == GGML_TYPE_Q8_0) {
        uint8_t *p=myds_metal_buffer_data(b);
        for(size_t i=0;i<bytes/34;i++) {
            uint16_t d=(uint16_t)(0x2400u+(i%4)*0x100u); memcpy(p+i*34,&d,2);
            for(size_t j=0;j<32;j++) p[i*34+2+j]=(uint8_t)(int8_t)((int)((i*13+j*29)%255)-127);
        }
    } else if (type == GGML_TYPE_F16) {
        uint16_t *p=myds_metal_buffer_data(b);
        for(size_t i=0;i<bytes/2;i++) p[i]=(uint16_t)(0x3000u+(i%17));
    } else input_fill(b);
}
static void exact(struct guarded a, struct guarded b, uint32_t rows) {
    const float *x=myds_metal_buffer_data(a.view), *y=myds_metal_buffer_data(b.view);
    size_t nonzero=0;
    for(uint32_t i=0;i<rows;i++) {
        REQUIRE(isfinite(x[i]) && isfinite(y[i])); nonzero += x[i] != 0;
        if(memcmp(x+i,y+i,4)) {
            uint32_t u,v; memcpy(&u,x+i,4); memcpy(&v,y+i,4);
            fprintf(stderr,"mismatch row=%u baseline=%08x candidate=%08x\n",i,u,v); exit(1);
        }
    }
    REQUIRE(nonzero); canary(a); canary(b);
}
static void shape_at(uint32_t rows, uint32_t width, ggml_type type,
    int batch, bool trace_on, unsigned expected_dispatches, size_t output_offset) {
    size_t row_bytes=type==GGML_TYPE_Q8_0 ? (size_t)width/32*34 : (size_t)width*(type==GGML_TYPE_F16?2:4);
    struct guarded w=allocate((size_t)rows*row_bytes), x=allocate((size_t)width*4),
        a=allocate_at((size_t)rows*4,output_offset), b=allocate_at((size_t)rows*4,output_offset);
    weight_fill(w.view,type); input_fill(x.view);
    if(batch) REQUIRE(myds_metal_begin_batch());
    myds_metal_barrier(); myds_metal_dispatch_count_reset();
    REQUIRE(myds_metal_gemv(a.view,w.view,x.view,rows,width,type));
    REQUIRE(myds_metal_v41_bf16(a.view,rows));
    REQUIRE(myds_metal_dispatch_count()==2);
    myds_metal_barrier(); myds_metal_dispatch_count_reset();
    REQUIRE(myds_v41_decode_gemv_bf16(b.view,w.view,x.view,rows,width,type,trace_on));
    REQUIRE(myds_metal_dispatch_count()==expected_dispatches);
    if(batch) REQUIRE(myds_metal_commit_batch());
    exact(a,b,rows); canary(w); canary(x);
    if(type==GGML_TYPE_Q8_0) {
        myds_metal_buffer_t *short_out=myds_metal_buffer_sub_view(b.view,0,(size_t)(rows-1)*4);
        if(rows>1) REQUIRE(!myds_v41_decode_gemv_bf16(short_out,w.view,x.view,rows,width,type,trace_on));
        REQUIRE(!myds_v41_decode_gemv_bf16(NULL,w.view,x.view,rows,width,type,trace_on));
        REQUIRE(!myds_v41_decode_gemv_bf16(b.view,w.view,x.view,rows,width-1,type,trace_on));
        myds_metal_buffer_free(short_out);
    }
    printf("PASS decode projection type=%d rows=%u width=%u batch=%d trace=%d output_offset=%zu dispatches=%u exact/view/canary\n",
        type,rows,width,batch,trace_on,output_offset,expected_dispatches); fflush(stdout);
    release(w); release(x); release(a); release(b);
}
static void shape(uint32_t rows, uint32_t width, ggml_type type,
    int batch, bool trace_on, unsigned expected_dispatches) {
    shape_at(rows,width,type,batch,trace_on,expected_dispatches,16);
}
static void rejected_output_offset(size_t offset) {
    const uint32_t rows=13,width=64;
    struct guarded w=allocate((size_t)rows*width/32*34), x=allocate((size_t)width*4),
        a=allocate_at((size_t)rows*4,offset), b=allocate_at((size_t)rows*4,offset);
    weight_fill(w.view,GGML_TYPE_Q8_0); input_fill(x.view);
    REQUIRE(myds_metal_begin_batch());
    myds_metal_barrier(); myds_metal_dispatch_count_reset();
    REQUIRE(myds_metal_gemv(a.view,w.view,x.view,rows,width,GGML_TYPE_Q8_0));
    REQUIRE(!myds_metal_v41_bf16(a.view,rows));
    REQUIRE(myds_metal_dispatch_count()==1);
    myds_metal_barrier(); myds_metal_dispatch_count_reset();
    REQUIRE(!myds_v41_decode_gemv_bf16(b.view,w.view,x.view,rows,width,GGML_TYPE_Q8_0,false));
    REQUIRE(myds_metal_dispatch_count()==1);
    REQUIRE(myds_metal_commit_batch());
    /* Both old/new calls fail at the same BF16 alignment check after the
     * original GEMV. Verify those permitted partial F32 side effects too. */
    exact(a,b,rows); canary(w); canary(x);
    printf("PASS decode output_offset=%zu: original BF16 alignment rejection propagated, one GEMV dispatch, exact F32 side effects/canary\n",offset);
    fflush(stdout); release(w); release(x); release(a); release(b);
}
int main(int argc, char **argv) {
    REQUIRE(argc==2);
    REQUIRE(myds_metal_init(argv[1]));
    const char *flag=getenv("MYDS_DECODE_Q8_FUSE_BF16");
    const int enabled=!(flag && !strcmp(flag,"0"));
    const uint32_t shapes[][2]={{1,32},{3,64},{5,128},{9,256},
        {5120,8192},{2304,5120},{5120,2304}};
    for(size_t i=0;i<sizeof(shapes)/sizeof(*shapes);i++)
        for(int batch=0;batch<2;batch++)
            shape(shapes[i][0],shapes[i][1],GGML_TYPE_Q8_0,batch,false,enabled?1:2);
    shape(13,64,GGML_TYPE_Q8_0,1,true,2);
    shape(4,128,GGML_TYPE_F16,1,false,2);
    shape(4,128,GGML_TYPE_F32,1,false,2);
    for(size_t offset=4;offset<16;offset+=4) rejected_output_offset(offset);
    if(enabled) {
        install_interceptor(); missing_fused_pso=1;
        [g_native_pipelines removeAllObjects];
        shape(13,64,GGML_TYPE_Q8_0,1,false,2);
        REQUIRE(pso_failure_hits>0);
        missing_fused_pso=0; remove_interceptor();
        shape(13,64,GGML_TYPE_Q8_0,1,false,1);
        printf("PASS fused PSO unavailable -> original GPU projection + BF16; successful retry, failure hits=%u\n",pso_failure_hits);
    }
    myds_metal_cleanup(); printf("PASS decode Q8 GPU fixture enabled=%d\n",enabled); return 0;
}
