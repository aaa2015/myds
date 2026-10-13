/* Actual Metal HC-stage preflight with Foundation-only fake protocol objects.
 * Never creates a Metal device/queue/encoder or executes a shader. Compile:
 * clang -O2 -Wall -Wextra -Iinclude -Isrc -fobjc-arc -Wl,-dead_strip
 *   tests/test_moe_hc_stages_host.m -framework Foundation -framework Metal
 *   -o /tmp/test_moe_hc_stages_host
 */
#include "myds_metal.m"
#include <assert.h>

@interface HCTestDevice : NSObject
@property NSUInteger maxThreadgroupMemoryLength;
@property MTLSize maxThreadsPerThreadgroup;
@end
@implementation HCTestDevice @end
@interface HCTestPipeline : NSObject
@property id device;
@property NSUInteger maxTotalThreadsPerThreadgroup;
@property NSUInteger threadExecutionWidth;
@property NSUInteger staticThreadgroupMemoryLength;
@end
@implementation HCTestPipeline @end
@interface HCTestBuffer : NSObject
@property id device;
@property NSUInteger length;
@property void *contents;
@end
@implementation HCTestBuffer @end

static myds_metal_buffer_t b[7];
static int preflight(uint32_t dim,float eps,float hc_eps) {
    return myds_metal_v41_hc_mix_stages_preflight(&b[2],&b[3],&b[4],&b[5],&b[6],&b[0],&b[1],dim,eps,hc_eps);
}
int main(void) {
    @autoreleasepool {
        HCTestDevice *device=[HCTestDevice new],*other=[HCTestDevice new];
        device.maxThreadgroupMemoryLength=32768;
        device.maxThreadsPerThreadgroup=MTLSizeMake(1024,1024,64);
        HCTestPipeline *norm=[HCTestPipeline new],*project=[HCTestPipeline new],*split=[HCTestPipeline new];
        for (HCTestPipeline *p in @[norm,project,split]) {
            p.device=device;p.maxTotalThreadsPerThreadgroup=1024;p.threadExecutionWidth=32;
        }
        NSMutableArray<HCTestBuffer *> *owners=[NSMutableArray new];
        const size_t bytes[]={81920,96,96,81920,983040,12,96};
        for(unsigned i=0;i<7;++i) {
            HCTestBuffer *obj=[HCTestBuffer new];obj.device=device;obj.length=bytes[i]+32;
            obj.contents=malloc(obj.length);assert(obj.contents);memset(obj.contents,0xa5,obj.length);
            [owners addObject:obj];
            b[i]=(myds_metal_buffer_t){.mtl_buf=(id<MTLBuffer>)obj,.size=bytes[i],.offset=16,
                .ptr=(uint8_t *)obj.contents+16};
        }
        g_device=(id<MTLDevice>)device;g_initialized=true;g_auto_barrier=0;g_concurrent_mode=1;g_kp_on=0;
        g_batch_cb=(id<MTLCommandBuffer>)[NSObject new];
        g_pso_v41_norm_plain=(id<MTLComputePipelineState>)norm;
        g_pso_v41_hc_split=(id<MTLComputePipelineState>)split;
        g_native_pipelines=[NSMutableDictionary new];
        g_native_pipelines[@"kernel_mul_mv_f16_f32_4_8"]=(id<MTLComputePipelineState>)project;
        const uint64_t dispatch_before=g_dispatch_count;
        assert(preflight(5120,1e-5f,1e-5f));
        g_initialized=false;assert(!preflight(5120,1e-5f,1e-5f));g_initialized=true;
        id<MTLCommandBuffer> saved_cb=g_batch_cb;g_batch_cb=nil;assert(!preflight(5120,1e-5f,1e-5f));g_batch_cb=saved_cb;
        g_auto_barrier=1;assert(!preflight(5120,1e-5f,1e-5f));g_auto_barrier=0;
        g_concurrent_mode=0;assert(!preflight(5120,1e-5f,1e-5f));g_concurrent_mode=1;
        g_kp_on=1;assert(!preflight(5120,1e-5f,1e-5f));g_kp_on=0;
        assert(!preflight(4096,1e-5f,1e-5f));assert(!preflight(5120,NAN,1e-5f));
        assert(!preflight(5120,1e-5f,-1));assert(!preflight(5120,-1,1e-5f));
        for(unsigned i=0;i<7;++i) {
            const size_t saved=b[i].size;b[i].size=bytes[i]-1;assert(!preflight(5120,1e-5f,1e-5f));b[i].size=saved;
            HCTestBuffer *obj=owners[i];obj.device=other;assert(!preflight(5120,1e-5f,1e-5f));obj.device=device;
        }
        size_t saved_offset=b[0].offset;void *saved_ptr=b[0].ptr;
        b[0].offset=17;assert(!preflight(5120,1e-5f,1e-5f));b[0].offset=saved_offset;
        b[0].ptr=(void *)(UINTPTR_MAX-15);assert(!preflight(5120,1e-5f,1e-5f));b[0].ptr=saved_ptr;
        b[0].ptr=(uint8_t *)saved_ptr+16;assert(!preflight(5120,1e-5f,1e-5f));b[0].ptr=saved_ptr;
        HCTestBuffer *obj0=owners[0];void *saved_contents=obj0.contents;
        obj0.contents=(void *)(UINTPTR_MAX-7);assert(!preflight(5120,1e-5f,1e-5f));obj0.contents=saved_contents;
        obj0.contents=NULL;assert(!preflight(5120,1e-5f,1e-5f));obj0.contents=saved_contents;
        myds_metal_buffer_t saved_mix=b[1];b[1]=b[2];assert(!preflight(5120,1e-5f,1e-5f));b[1]=saved_mix;
        HCTestBuffer *mix_obj=owners[1];void *mix_contents=mix_obj.contents;NSUInteger mix_length=mix_obj.length;
        mix_obj.contents=owners[2].contents;mix_obj.length=owners[2].length;b[1].ptr=b[2].ptr;
        assert(!preflight(5120,1e-5f,1e-5f));mix_obj.contents=mix_contents;mix_obj.length=mix_length;b[1]=saved_mix;
        norm.device=other;assert(!preflight(5120,1e-5f,1e-5f));norm.device=device;
        norm.threadExecutionWidth=16;assert(!preflight(5120,1e-5f,1e-5f));norm.threadExecutionWidth=32;
        norm.maxTotalThreadsPerThreadgroup=1023;assert(!preflight(5120,1e-5f,1e-5f));norm.maxTotalThreadsPerThreadgroup=1024;
        project.maxTotalThreadsPerThreadgroup=255;assert(!preflight(5120,1e-5f,1e-5f));project.maxTotalThreadsPerThreadgroup=1024;
        split.maxTotalThreadsPerThreadgroup=0;assert(!preflight(5120,1e-5f,1e-5f));split.maxTotalThreadsPerThreadgroup=1024;
        device.maxThreadsPerThreadgroup=MTLSizeMake(1023,1024,64);assert(!preflight(5120,1e-5f,1e-5f));
        device.maxThreadsPerThreadgroup=MTLSizeMake(1024,7,64);assert(!preflight(5120,1e-5f,1e-5f));
        device.maxThreadsPerThreadgroup=MTLSizeMake(1024,1024,64);
        norm.staticThreadgroupMemoryLength=32768;assert(!preflight(5120,1e-5f,1e-5f));norm.staticThreadgroupMemoryLength=0;
        g_native_pipelines[@"kernel_mul_mv_f16_f32_4_8"]=nil;
        assert(!preflight(5120,1e-5f,1e-5f)); /* nil library, no cached PSO. */
        g_native_pipelines[@"kernel_mul_mv_f16_f32_4_8"]=(id<MTLComputePipelineState>)project;
        assert(preflight(5120,1e-5f,1e-5f));
        assert(g_dispatch_count==dispatch_before && !g_batch_enc);
        for(HCTestBuffer *obj in owners) {
            for(NSUInteger i=0;i<obj.length;++i)assert(((uint8_t *)obj.contents)[i]==0xa5);
            free(obj.contents);obj.contents=NULL;
        }
        g_batch_cb=nil;g_device=nil;g_initialized=false;g_native_pipelines=nil;
        g_pso_v41_norm_plain=nil;g_pso_v41_hc_split=nil;
        puts("PASS: actual HC Metal preflight rejects device/geometry/scope/range/mapping/alias/missingPSO; zero dispatch/writes; Foundation fakes only");
    }
    return 0;
}
