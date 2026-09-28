#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static const char *base_source =
"#include <metal_stdlib>\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#ifndef QK_K\n"
"#define QK_K 256\n"
"#endif\n"
"#define N_SIMDWIDTH 32\n"
"#ifndef N_R0_Q8_0\n"
"#define N_R0_Q8_0 2\n"
"#endif\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"struct block_q8_K {\n"
"    float d;\n"
"    int8_t qs[QK_K];\n"
"    int16_t bsums[QK_K / 16];\n"
"};\n"
"\n";

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device found\n");
            return 1;
        }
        printf("Metal device: %s\n", [[device name] UTF8String]);

        NSArray<NSString *> *files = @[
            @"metal/flash_attn.metal",
            @"metal/dense.metal",
            @"metal/moe.metal",
            @"metal/dsv4_hc.metal",
            @"metal/unary.metal",
            @"metal/dsv4_kv.metal",
            @"metal/dsv41.metal",
            @"metal/dsv4_rope.metal",
            @"metal/dsv4_misc.metal",
            @"metal/argsort.metal",
            @"metal/cpy.metal",
            @"metal/concat.metal",
            @"metal/get_rows.metal",
            @"metal/sum_rows.metal",
            @"metal/softmax.metal",
            @"metal/repeat.metal",
            @"metal/glu.metal",
            @"metal/norm.metal",
            @"metal/bin.metal",
            @"metal/set_rows.metal"
        ];

        NSMutableString *full_source = [NSMutableString stringWithUTF8String:base_source];
        for (NSString *file in files) {
            NSError *read_err = nil;
            NSString *content = [NSString stringWithContentsOfFile:file encoding:NSUTF8StringEncoding error:&read_err];
            if (!content) {
                fprintf(stderr, "Failed to read %s: %s\n", [file UTF8String], [[read_err localizedDescription] UTF8String]);
                return 1;
            }
            [full_source appendFormat:@"\n#line 1 \"%@\"\n%@", file, content];
        }

        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = YES;
        NSError *compile_err = nil;
        printf("Compiling Metal library for DeepSeek kernels...\n");
        id<MTLLibrary> library = [device newLibraryWithSource:full_source options:options error:&compile_err];
        if (!library) {
            fprintf(stderr, "Metal shader compilation failed: %s\n", [[compile_err localizedDescription] UTF8String]);
            return 1;
        }
        printf("Metal library successfully compiled! Function count: %lu\n", (unsigned long)[[library functionNames] count]);

        // Probe critical DeepSeek functions
        NSArray<NSString *> *probe_functions = @[
            @"kernel_rms_norm_f32_4",
            @"kernel_rms_norm_mul_f32_4",
            @"kernel_dsv4_head_rms_norm_rope_tail_f32",
            @"kernel_swiglu",
            @"kernel_router_select_f32",
            @"kernel_moe_sum6",
            @"kernel_softmax_f32"
        ];

        for (NSString *fn_name in probe_functions) {
            id<MTLFunction> fn = [library newFunctionWithName:fn_name];
            if (!fn) {
                printf("  [WARN] Function %s not found\n", [fn_name UTF8String]);
            } else {
                NSError *pso_err = nil;
                id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&pso_err];
                if (pso) {
                    printf("  [OK] %s (maxThreadsPerThreadgroup=%lu)\n", [fn_name UTF8String], (unsigned long)pso.maxTotalThreadsPerThreadgroup);
                } else {
                    printf("  [FAIL] %s pipeline creation error: %s\n", [fn_name UTF8String], [[pso_err localizedDescription] UTF8String]);
                }
            }
        }
    }
    return 0;
}
