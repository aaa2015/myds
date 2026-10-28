/* Ordinary, uninstrumented setup+append driver. Building never runs a model.
 * The explicit --run option is required before a model can be loaded. */
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <math.h>
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_configfile.h"

static void need(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_indexer_mask_warm: %s\n", what); exit(1); }
}
static NSData *read_file(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:nil];
    need(data != nil, "cannot read input"); return data;
}
static NSString *digest(const void *bytes, size_t length) {
    CC_SHA256_CTX ctx; CC_SHA256_Init(&ctx);
    const uint8_t *p = bytes;
    while (length) { CC_LONG n = (CC_LONG)MIN(length, (size_t)1048576); CC_SHA256_Update(&ctx,p,n); p+=n; length-=n; }
    uint8_t result[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(result,&ctx);
    NSMutableString *hex=[NSMutableString stringWithCapacity:64];
    for (unsigned i=0;i<sizeof(result);i++) [hex appendFormat:@"%02x",result[i]];
    return hex;
}
static NSString *file_digest(NSString *path) { NSData *data=read_file(path); return digest(data.bytes,data.length); }
extern char **environ;
static int compare_env(const void *a,const void *b) {
    return strcmp(*(const char *const *)a,*(const char *const *)b);
}
/* Canonical digest of the complete MYDS_ environment, including empty
 * values; NUL terminators prevent ambiguous key/value concatenation. */
static NSDictionary *environment_identity(void) {
    size_t n=0;
    for (char **p=environ; p && *p; ++p) if (!strncmp(*p,"MYDS_",5)) n++;
    const char **items=calloc(n ? n : 1,sizeof(*items)); need(items!=NULL,"environment allocation");
    size_t i=0;
    for (char **p=environ; p && *p; ++p) if (!strncmp(*p,"MYDS_",5)) items[i++]=*p;
    qsort(items,n,sizeof(*items),compare_env);
    NSMutableData *bytes=[NSMutableData new];
    for (i=0;i<n;i++) [bytes appendBytes:items[i] length:strlen(items[i])+1];
    free(items);
    return @{@"count":@(n),@"sha256":digest(bytes.bytes,bytes.length),
             @"encoding":@"sorted raw MYDS_ key=value bytes including NUL terminator"};
}
static NSDictionary *mask_skip_counts(myds_v41_gpu_mask_skip_counters_t counts) {
    return @{@"requests":@(counts.requests),@"encoded":@(counts.encoded),
             @"unavailable":@(counts.unavailable),@"failures":@(counts.failures)};
}
static NSDictionary *write_payload(NSString *directory, NSString *name, NSData *data) {
    NSString *path=[directory stringByAppendingPathComponent:name];
    need([data writeToFile:path options:NSDataWritingWithoutOverwriting error:nil], "cannot write output payload");
    return @{@"path":name,@"bytes":@(data.length),@"sha256":digest(data.bytes,data.length)};
}
static NSString *executable_path(void) {
    uint32_t n=0; _NSGetExecutablePath(NULL,&n); char *p=calloc(n,1);
    need(p && _NSGetExecutablePath(p,&n)==0, "cannot determine executable path");
    NSString *path=[[NSString stringWithUTF8String:p] stringByResolvingSymlinksInPath]; free(p); return path;
}
struct output { void *bytes; void *ids; double first; };
static void output_callback(const char *bytes,size_t length,int32_t token,void *opaque) {
    struct output *out=opaque;
    if (!out->first) out->first=myds_metal_host_time();
    [(__bridge NSMutableData *)out->bytes appendBytes:bytes length:length];
    [(__bridge NSMutableData *)out->ids appendBytes:&token length:sizeof(token)];
}
static NSDictionary *phase(myds_context_t *ctx, NSString *name, NSString *prompt_path,
                           NSString *tokens_path, NSString *directory, BOOL cold) {
    NSData *prompt=read_file(prompt_path),*tokens=read_file(tokens_path);
    need(prompt.length && !memchr(prompt.bytes,0,prompt.length), "prompt must be nonempty and NUL-free");
    need(tokens.length && tokens.length%sizeof(int32_t)==0 && tokens.length/sizeof(int32_t)<=INT32_MAX,
         "token input must be nonempty int32le");
    NSMutableData *z=[prompt mutableCopy]; uint8_t zero=0; [z appendBytes:&zero length:1];
    int32_t n=myds_tokenizer_encode(ctx->model->tokenizer,z.bytes,NULL,0,true);
    need(n>0 && (size_t)n*sizeof(int32_t)==tokens.length, "token count mismatch");
    NSMutableData *actual=[NSMutableData dataWithLength:tokens.length];
    need(myds_tokenizer_encode(ctx->model->tokenizer,z.bytes,actual.mutableBytes,n,true)==n &&
         memcmp(actual.bytes,tokens.bytes,tokens.length)==0, "full tokenizer IDs differ");
    uint32_t before=myds_context_get_pcache_len(ctx), prompt_tokens=0;
    NSMutableData *result=[NSMutableData new],*ids=[NSMutableData new];
    struct output out={(__bridge void *)result,(__bridge void *)ids,0};
    myds_stop_reason_t stop=MYDS_STOP_ERROR;
    double begin=myds_metal_host_time();
    int success=myds_generate_bytes_ex(ctx,z.bytes,output_callback,&out,&stop,&prompt_tokens,NULL);
    double end=myds_metal_host_time();
    need(success==1 && stop!=MYDS_STOP_ERROR && stop!=MYDS_STOP_CANCELLED, "generation failed or cancelled");
    myds_prefill_stats_t stats; myds_last_prefill_stats(ctx,&stats);
    need(prompt_tokens==(uint32_t)n && stats.prompt_tokens==(uint32_t)n, "generation token count mismatch");
    need(cold ? stats.cached_tokens==0 : stats.cached_tokens>0, "cold/warm prefix-cache expectation failed");
    need(isfinite(end-begin) && end>begin && isfinite(stats.prefill_seconds) && stats.prefill_seconds>=0, "invalid timings");
    myds_decode_stats_t decode; myds_last_decode_stats(ctx,&decode);
    return @{@"phase":name,@"prompt_path":prompt_path,@"prompt_bytes":@(prompt.length),
        @"prompt_sha256":digest(prompt.bytes,prompt.length),@"tokens_path":tokens_path,
        @"tokens_bytes":@(tokens.length),@"tokens_sha256":digest(tokens.bytes,tokens.length),
        @"actual_tokens":@(n),@"cached_tokens":@(stats.cached_tokens),@"prefill_tokens":@(stats.prefill_tokens),
        @"prefill_seconds":@(stats.prefill_seconds),@"generation_seconds":@(end-begin),
        @"first_callback_seconds":out.first ? @(out.first-begin) : (id)NSNull.null,
        @"pcache_len_before":@(before),@"pcache_len_after":@(myds_context_get_pcache_len(ctx)),
        @"stop_reason":@(stop),@"output":write_payload(directory,[name stringByAppendingString:@"-output.bin"],result),
        @"output_token_ids":write_payload(directory,[name stringByAppendingString:@"-output-token-ids.i32le.bin"],ids),
        @"draft_cycles":@(decode.draft_cycles),@"pld_cycles":@(decode.pld_cycles),@"batch2_calls":@(decode.batch2_calls)};
}
int main(int argc,char **argv) { @autoreleasepool {
    if (argc==2 && !strcmp(argv[1],"--help")) {
        puts("Usage: bench_indexer_mask_warm --run MODEL BASE_PROMPT BASE_TOKENS APPEND_PROMPT APPEND_TOKENS NEW_OUTPUT_DIR MAX_SEQ_LEN CONFIG EXPECTED_MYDS_ENV_SHA256 EXPECT_MASK_SKIP(on|off)\nTokens are little-endian int32 and must match the actual model tokenizer. CONFIG uses production precedence; the complete resulting MYDS_ environment must match the expected digest. No model is loaded without --run."); return 0;
    }
    need(argc==12 && !strcmp(argv[1],"--run"), "use --help; explicit --run and ten arguments required");
    NSString *model=[NSString stringWithUTF8String:argv[2]],*base=[NSString stringWithUTF8String:argv[3]],
        *base_ids=[NSString stringWithUTF8String:argv[4]],*append=[NSString stringWithUTF8String:argv[5]],
        *append_ids=[NSString stringWithUTF8String:argv[6]],*directory=[NSString stringWithUTF8String:argv[7]],
        *config=[NSString stringWithUTF8String:argv[9]],*expected_env=[NSString stringWithUTF8String:argv[10]];
    for (NSString *p in @[model,base,base_ids,append,append_ids,directory,config]) need(p.isAbsolutePath,"all paths must be absolute");
    char *end=NULL; unsigned long maxseq=strtoul(argv[8],&end,10);
    need(end && *end==0 && end!=argv[8] && maxseq>0 && maxseq<=UINT32_MAX,"invalid max sequence length");
    need(expected_env.length==64,"expected environment SHA256 required");
    NSString *config_hash=file_digest(config);
    int config_entries=myds_configfile_load(config.fileSystemRepresentation,true);
    need(config_entries>0 && [config_hash isEqual:file_digest(config)],"config load or identity failed");
    NSDictionary *effective_env=environment_identity();
    need([effective_env[@"sha256"] isEqual:expected_env],"complete MYDS_ environment digest differs");
    bool expect_skip=!strcmp(argv[11],"on");
    need(expect_skip || !strcmp(argv[11],"off"),"expect mask skip must be on or off");
    const char *skip_env=getenv("MYDS_INDEXER_MASK_SKIP");
    need(expect_skip == (skip_env && !strcmp(skip_env,"1")),"expected mask-skip mode differs from effective environment");
    NSFileManager *fm=[NSFileManager defaultManager];
    need(![fm fileExistsAtPath:directory] && [fm createDirectoryAtPath:directory withIntermediateDirectories:NO attributes:nil error:nil],"output directory must be new");
    struct stat st; need(stat(model.fileSystemRepresentation,&st)==0 && st.st_size>=1048576,"model stat/header");
    myds_model_t *m=myds_model_load(model.fileSystemRepresentation); need(m && m->gf && m->gf->mmap_addr,"model load");
    myds_config_t cfg={.max_seq_len=(uint32_t)maxseq,.max_new_tokens=1,.temperature=0,.top_p=1,.seed=42};
    myds_context_t *ctx=myds_context_create(m,&cfg); need(ctx!=NULL,"context create");
    myds_v41_gpu_mask_skip_counters(NULL,true);
    NSDictionary *cold=phase(ctx,@"cold",base,base_ids,directory,YES);
    myds_v41_gpu_mask_skip_counters_t cold_counts,warm_counts;
    myds_v41_gpu_mask_skip_counters(&cold_counts,true);
    NSDictionary *warm=phase(ctx,@"warm_append",append,append_ids,directory,NO);
    myds_v41_gpu_mask_skip_counters(&warm_counts,false);
    need(!cold_counts.failures && !warm_counts.failures, "mask-skip dispatch failure observed");
    need(expect_skip ? warm_counts.encoded>0 : (!cold_counts.encoded && !warm_counts.encoded),
         "mask-skip dispatch witness does not match expected mode");
    NSDictionary *post_env=environment_identity();
    need([post_env[@"sha256"] isEqual:expected_env] && [config_hash isEqual:file_digest(config)],
         "effective environment or configuration changed during run");
    const char *lib=myds_metal_loaded_library_path(); need(lib!=NULL,"loaded metallib path missing");
    NSString *library=[NSString stringWithUTF8String:lib],*exe=executable_path();
    NSMutableDictionary *env=[NSMutableDictionary new];
    for (NSString *key in @[@"MYDS_METAL_CONCURRENT",@"MYDS_MOE_CONCURRENT",@"MYDS_PREFIX_CACHE",@"MYDS_INDEXER_REGISTER_ONLY",@"MYDS_INDEXER_TILED",@"MYDS_INDEXER_MASK_SKIP",@"MYDS_INDEXER_BATCH_FLOATS",@"MYDS_PREFILL_ATTN"]) {
        const char *value=getenv(key.UTF8String); env[key]=value ? [NSString stringWithUTF8String:value] : (id)NSNull.null;
    }
    NSDictionary *summary=@{@"schema_version":@1,@"kind":@"ordinary_warm_prefill_screen",@"observer":@NO,
        @"executable_path":exe,@"executable_sha256":file_digest(exe),@"metallib_path":library,
        @"metallib_sha256":file_digest(library),@"model_path":model,@"model_bytes":@((uint64_t)st.st_size),
        @"model_header_bytes":@1048576,@"model_header_sha256":digest(m->gf->mmap_addr,1048576),
        @"parameters":@{@"max_seq_len":@(maxseq),@"max_new_tokens":@1,@"temperature":@0,@"top_p":@1,@"seed":@42},
        @"config_path":config,@"config_sha256":config_hash,@"config_loaded_entries":@(config_entries),
        @"effective_myds_environment":effective_env,@"post_run_myds_environment":post_env,@"selected_environment":env,
        @"environment_note":@"null means unset; actual provider and code defaults require independent verification",
        @"mask_skip_expected":expect_skip?@"on":@"off",
        @"mask_skip_cold_counters":mask_skip_counts(cold_counts),
        @"mask_skip_warm_counters":mask_skip_counts(warm_counts),@"phases":@[cold,warm]};
    NSData *json=[NSJSONSerialization dataWithJSONObject:summary options:NSJSONWritingSortedKeys|NSJSONWritingPrettyPrinted error:nil];
    need(json && [json writeToFile:[directory stringByAppendingPathComponent:@"summary.json"] options:NSDataWritingWithoutOverwriting error:nil],"write summary");
    myds_context_free(ctx); myds_model_free(m); puts("warm screen completed"); return 0;
} }
