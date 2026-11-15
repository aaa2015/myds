/* tools/evo_pso_probe.m — Live Apple Silicon Metal PSO & hardware ceiling probe.
 *
 * Inspects every kernel in a compiled myds.metallib against the local Metal GPU
 * without allocating model weights or touching any running inference service.
 * Outputs deterministic JSON to stdout:
 *   - Hardware envelope (GPU name, GPU core count via IOKit, unified memory,
 *     max threadgroup memory, Apple GPU family & Dynamic Caching availability)
 *   - Per-kernel PSO limits (maxTotalThreadsPerThreadgroup, threadExecutionWidth,
 *     staticThreadgroupMemoryLength, smem-bound max TGs/core, register pressure tier)
 *
 * Usage:
 *   bin/evo_pso_probe [bin/myds.metallib] [--filter SUBSTR] [--fc NAME_OR_IDX=INT ...]
 */
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

static uint64_t query_sysctl_u64(const char *name) {
    uint64_t val = 0;
    size_t len = sizeof(val);
    if (sysctlbyname(name, &val, &len, NULL, 0) == 0) return val;
    return 0;
}

static NSString *query_sysctl_str(const char *name) {
    size_t len = 0;
    if (sysctlbyname(name, NULL, &len, NULL, 0) != 0 || len == 0) return @"";
    char *buf = calloc(1, len + 1);
    if (!buf) return @"";
    NSString *res = @"";
    if (sysctlbyname(name, buf, &len, NULL, 0) == 0) {
        res = [NSString stringWithUTF8String:buf] ?: @"";
    }
    free(buf);
    return res;
}

static uint32_t query_gpu_core_count(void) {
    io_iterator_t iter = IO_OBJECT_NULL;
    kern_return_t kr = IOServiceGetMatchingServices(
        kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iter);
    if (kr != KERN_SUCCESS || iter == IO_OBJECT_NULL) return 0;
    uint32_t cores = 0;
    io_object_t service = IO_OBJECT_NULL;
    while ((service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        CFTypeRef prop = IORegistryEntryCreateCFProperty(
            service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (prop) {
            if (CFGetTypeID(prop) == CFNumberGetTypeID()) {
                int64_t v = 0;
                if (CFNumberGetValue((CFNumberRef)prop, kCFNumberSInt64Type, &v) && v > 0) {
                    cores = (uint32_t)v;
                }
            }
            CFRelease(prop);
        }
        IOObjectRelease(service);
        if (cores > 0) break;
    }
    IOObjectRelease(iter);
    return cores;
}

static int detect_apple_gpu_family(id<MTLDevice> dev) {
    /* MTLGPUFamilyApple1 == 1001 .. MTLGPUFamilyApple9 == 1009, Apple10 == 1010 ... */
    int best = 0;
    for (int fam = 1; fam <= 14; ++fam) {
        MTLGPUFamily f = (MTLGPUFamily)(1000 + fam);
        @try {
            if ([dev supportsFamily:f]) best = fam;
        } @catch (__unused NSException *ex) {
            break;
        }
    }
    return best;
}

/* Sensible V4.1 production defaults for function-constant specialized kernels
 * when probing default PSO footprint. Callers can override any constant via
 * --fc NAME=INT or --fc INDEX=INT. */
static int64_t default_fc_value(NSString *name) {
    NSString *low = [name lowercaseString];
    if ([low hasSuffix:@"_dv"]) return 512;
    if ([low hasSuffix:@"_ns10"]) return 576;
    if ([low hasSuffix:@"_ns20"]) return 512;
    if ([low hasSuffix:@"_ncpsg"]) return 32;
    if ([low hasSuffix:@"_nqptg"]) return 8;
    if ([low hasSuffix:@"_nsg"]) return 4;
    if ([low hasSuffix:@"_nxpsg"] || [low hasSuffix:@"_nwg"]) return 1;
    if ([low containsString:@"has_mask"]) return 1;
    return 0;
}

static void bind_fc(MTLFunctionConstantValues *vals, MTLDataType dtype,
                    NSUInteger idx, int64_t raw_val) {
    switch (dtype) {
        case MTLDataTypeBool: {
            bool v = (raw_val != 0);
            [vals setConstantValue:&v type:dtype atIndex:idx];
            break;
        }
        case MTLDataTypeShort: {
            int16_t v = (int16_t)raw_val;
            [vals setConstantValue:&v type:dtype atIndex:idx];
            break;
        }
        case MTLDataTypeUShort: {
            uint16_t v = (uint16_t)raw_val;
            [vals setConstantValue:&v type:dtype atIndex:idx];
            break;
        }
        case MTLDataTypeInt: {
            int32_t v = (int32_t)raw_val;
            [vals setConstantValue:&v type:dtype atIndex:idx];
            break;
        }
        case MTLDataTypeUInt: {
            uint32_t v = (uint32_t)raw_val;
            [vals setConstantValue:&v type:dtype atIndex:idx];
            break;
        }
        default: {
            int32_t v = (int32_t)raw_val;
            [vals setConstantValue:&v type:MTLDataTypeInt atIndex:idx];
            break;
        }
    }
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        NSString *libPath = @"bin/myds.metallib";
        NSString *filter = nil;
        NSMutableDictionary<NSString *, NSNumber *> *fcOverrides = [NSMutableDictionary dictionary];

        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
                filter = [NSString stringWithUTF8String:argv[++i]];
            } else if (strcmp(argv[i], "--fc") == 0 && i + 1 < argc) {
                const char *spec = argv[++i];
                const char *eq = strchr(spec, '=');
                if (!eq || eq == spec || *(eq + 1) == '\0') {
                    fprintf(stderr, "evo_pso_probe: invalid --fc '%s' (expected KEY=INT)\n", spec);
                    return 2;
                }
                NSString *k = [[NSString alloc] initWithBytes:spec
                                                       length:(NSUInteger)(eq - spec)
                                                     encoding:NSUTF8StringEncoding];
                char *end = NULL;
                long long v = strtoll(eq + 1, &end, 0);
                if (!end || *end != '\0') {
                    fprintf(stderr, "evo_pso_probe: invalid integer in --fc '%s'\n", spec);
                    return 2;
                }
                fcOverrides[k] = @(v);
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                fprintf(stderr, "Usage: %s [bin/myds.metallib] [--filter SUBSTR] [--fc KEY=INT ...]\n", argv[0]);
                return 0;
            } else if (argv[i][0] == '-') {
                fprintf(stderr, "evo_pso_probe: unknown flag '%s'\n", argv[i]);
                return 2;
            } else {
                libPath = [NSString stringWithUTF8String:argv[i]];
            }
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "evo_pso_probe: no Metal device available\n");
            return 1;
        }

        NSURL *url = [NSURL fileURLWithPath:libPath];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) {
            fprintf(stderr, "evo_pso_probe: failed to load '%s': %s\n",
                    [libPath UTF8String],
                    err ? [[err localizedDescription] UTF8String] : "unknown error");
            return 1;
        }

        uint64_t memBytes = query_sysctl_u64("hw.memsize");
        uint32_t gpuCores = query_gpu_core_count();
        int appleFamily = detect_apple_gpu_family(dev);
        NSUInteger maxTgMem = dev.maxThreadgroupMemoryLength;

        NSArray<NSString *> *names = [[lib functionNames] sortedArrayUsingSelector:@selector(compare:)];
        NSMutableArray<NSDictionary *> *kernels = [NSMutableArray arrayWithCapacity:names.count];
        NSUInteger compiledCount = 0;
        NSUInteger failedCount = 0;

        for (NSString *fnName in names) {
            if (filter && ![fnName localizedCaseInsensitiveContainsString:filter]) continue;

            id<MTLFunction> baseFn = [lib newFunctionWithName:fnName];
            if (!baseFn) {
                failedCount++;
                [kernels addObject:@{@"name": fnName, @"compiled": @NO, @"error": @"newFunctionWithName returned nil"}];
                continue;
            }

            NSDictionary<NSString *, MTLFunctionConstant *> *fcDict = baseFn.functionConstantsDictionary;
            id<MTLFunction> targetFn = baseFn;
            NSMutableDictionary<NSString *, NSNumber *> *bound = [NSMutableDictionary dictionary];

            if (fcDict.count > 0) {
                MTLFunctionConstantValues *fcVals = [[MTLFunctionConstantValues alloc] init];
                for (NSString *cName in fcDict) {
                    MTLFunctionConstant *fc = fcDict[cName];
                    if (!fc.required && !fcOverrides[cName] &&
                        !fcOverrides[[NSString stringWithFormat:@"%lu", (unsigned long)fc.index]]) {
                        continue;
                    }
                    NSString *idxKey = [NSString stringWithFormat:@"%lu", (unsigned long)fc.index];
                    NSNumber *ov = fcOverrides[cName] ?: fcOverrides[idxKey];
                    int64_t val = ov ? [ov longLongValue] : default_fc_value(cName);
                    bind_fc(fcVals, fc.type, fc.index, val);
                    bound[cName] = @(val);
                }
                NSError *fcErr = nil;
                targetFn = [lib newFunctionWithName:fnName constantValues:fcVals error:&fcErr];
                if (!targetFn) {
                    failedCount++;
                    [kernels addObject:@{
                        @"name": fnName,
                        @"compiled": @NO,
                        @"error": fcErr ? (fcErr.localizedDescription ?: @"FC specialize failed") : @"FC specialize failed"
                    }];
                    continue;
                }
            }

            NSError *psoErr = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:targetFn error:&psoErr];
            if (!pso) {
                failedCount++;
                [kernels addObject:@{
                    @"name": fnName,
                    @"compiled": @NO,
                    @"error": psoErr ? (psoErr.localizedDescription ?: @"PSO compile failed") : @"PSO compile failed"
                }];
                continue;
            }

            compiledCount++;
            NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
            NSUInteger simdWidth = pso.threadExecutionWidth;
            NSUInteger staticSmem = pso.staticThreadgroupMemoryLength;
            NSUInteger maxTgsBySmem = (staticSmem > 0 && maxTgMem > 0) ? (maxTgMem / staticSmem) : 32;
            if (maxTgsBySmem > 32) maxTgsBySmem = 32;
            NSString *regTier = (maxThreads >= 1024) ? @"low"
                              : (maxThreads >= 512)  ? @"moderate"
                                                     : @"high_or_spill_risk";

            NSMutableDictionary *entry = [@{
                @"name": fnName,
                @"compiled": @YES,
                @"max_total_threads_per_tg": @(maxThreads),
                @"thread_execution_width": @(simdWidth),
                @"static_tg_mem_bytes": @(staticSmem),
                @"max_tgs_per_core_by_smem": @(maxTgsBySmem),
                @"register_pressure_tier": regTier,
            } mutableCopy];
            if (bound.count > 0) {
                entry[@"function_constants_bound"] = bound;
            }
            [kernels addObject:entry];
        }

        NSDictionary *root = @{
            @"kind": @"evo_pso_probe",
            @"device": @{
                @"gpu_name": dev.name ?: @"",
                @"cpu_brand": query_sysctl_str("machdep.cpu.brand_string"),
                @"gpu_cores": @(gpuCores),
                @"unified_mem_bytes": @(memBytes),
                @"unified_mem_gib": @((double)memBytes / (double)(1ULL << 30)),
                @"recommended_max_working_set_bytes": @(dev.recommendedMaxWorkingSetSize),
                @"max_tg_mem_bytes": @(maxTgMem),
                @"apple_gpu_family_max": @(appleFamily),
                @"has_dynamic_caching": @(appleFamily >= 9),
            },
            @"metallib": @{
                @"path": libPath,
                @"total_functions": @(names.count),
                @"matched_functions": @(kernels.count),
                @"compiled_count": @(compiledCount),
                @"failed_count": @(failedCount),
            },
            @"kernels": kernels,
        };

        NSData *json = [NSJSONSerialization dataWithJSONObject:root
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:&err];
        if (!json) {
            fprintf(stderr, "evo_pso_probe: JSON serialization failed\n");
            return 1;
        }
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
        return failedCount == 0 ? 0 : 1;
    }
}
