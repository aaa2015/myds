/* Full-model diagnostic capture of three real prefill indexer groups.
 * Build src/myds_v41.c and src/myds_v41_gpu.c with MYDS_INDEXER_CAPTURE.
 * The observer adds waits; all elapsed times from this process are diagnostic. */
#include "myds_internal.h"
#include "myds_v41.h"
#include "myds_configfile.h"
#include "reference/trace_input.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern int myds_indexer_capture_begin(const char *directory);
extern int myds_indexer_capture_finish(void);
static int new_path(char *path,size_t capacity,const char *dir,const char *name) {
    int n=snprintf(path,capacity,"%s/%s",dir,name);
    return n>0 && (size_t)n<capacity;
}
static int json_string(FILE *f,const char *s) {
    if(!s)return fputs("null",f)>=0;
    if(fputc('"',f)==EOF)return 0;
    for(const unsigned char *p=(const unsigned char *)s;*p;p++) {
        if(*p=='"' || *p=='\\') {if(fputc('\\',f)==EOF || fputc(*p,f)==EOF)return 0;}
        else if(*p<32) {if(fprintf(f,"\\u%04x",*p)<0)return 0;}
        else if(fputc(*p,f)==EOF)return 0;
    }
    return fputc('"',f)!=EOF;
}
static int enabled(const char *name) {
    const char *v=getenv(name);return v && *v && strcmp(v,"0");
}
static uint32_t context_limit(void) {
    const char *v=getenv("MYDS_CTX_SIZE");if(!v)return 327680u;
    char *end=NULL;unsigned long n=strtoul(v,&end,10);
    return *v && end && !*end && n>=32768u && n<=1048576u ? (uint32_t)n : 0;
}
int main(int argc,char **argv) {
    if(argc!=8 || strcmp(argv[2],"--tokens-file") || strcmp(argv[4],"--out-dir") ||
       strcmp(argv[6],"--config") || argv[1][0]!='/' || argv[3][0]!='/' || argv[5][0]!='/' || argv[7][0]!='/') {
        fprintf(stderr,"Usage: %s ABS_MODEL --tokens-file ABS --out-dir NEW_ABS --config ABS\n",argv[0]);return 2;
    }
    /* Match main.c's actual overwrite behavior, independently of the stale
     * priority prose in myds_configfile.h. Record only effective MYDS_* below. */
    if(myds_configfile_load(argv[7],true)<0 || getenv("MYDS_TRACE_PREFIX") ||
       enabled("MYDS_PROFILE_PREFILL") || enabled("MYDS_PROFILE_SPLIT") ||
       enabled("MYDS_PROFILE_SPLIT2") || enabled("MYDS_PROFILE_SPLIT3") ||
       enabled("MYDS_PROFILE_MOE") || enabled("MYDS_KPROF") ||
       enabled("MYDS_INDEXER_KEY_WINDOW") || enabled("MYDS_INDEXER_SGBAR"))return 2;
    const uint32_t capacity=context_limit();if(!capacity)return 2;
    trace_tokens tokens={0};
    if(!trace_tokens_parse(&tokens,4,argv) || tokens.count!=32768u) {
        free(tokens.ids);fprintf(stderr,"Exactly 32768 real tokenizer IDs required\n");return 2;
    }
    struct stat model_stat,config_stat,token_stat;
    if(stat(argv[1],&model_stat) || stat(argv[7],&config_stat) || stat(argv[3],&token_stat)) {
        free(tokens.ids);return 2;
    }
    myds_model_t *model=myds_model_load(argv[1]);
    if(!model){free(tokens.ids);return 1;}
    for(size_t i=0;i<tokens.count;i++)if((uint32_t)tokens.ids[i]>=model->vocab_size) {
        free(tokens.ids);myds_model_free(model);return 2;
    }
    const myds_config_t cfg={.max_seq_len=capacity,.max_new_tokens=1,
        .temperature=0,.top_p=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);
    int result=1;
    if(ctx && myds_indexer_capture_begin(argv[5])) {
        char path[4096];int recorded=0;
        if(new_path(path,sizeof(path),argv[5],"input-token-ids.i32")) {
            FILE *file=fopen(path,"wx");
            if(file){recorded=fwrite(tokens.ids,sizeof(*tokens.ids),tokens.count,file)==tokens.count;
                if(fclose(file))recorded=0;}
        }
        if(!new_path(path,sizeof(path),argv[5],"effective-myds.env") ||
           myds_configfile_dump_env(path)<0)recorded=0;
        if(new_path(path,sizeof(path),argv[5],"capture-meta.json")) {
            FILE *file=fopen(path,"wx");
            if(!file)recorded=0;
            else {
                if(fprintf(file,"{\"schema_version\":1,\"capture_kind\":\"myds_real_prefill_indexer\","
                    "\"tokens\":32768,\"context_capacity\":%u,\"layers\":[2,8,14],"
                    "\"chunk_global_start\":30720,\"batch_floats\":2097152,"
                    "\"model_size\":%llu,\"model_inode\":%llu,\"model_mtime_s\":%lld,\"model_mtime_ns\":%ld,"
                    "\"config_size\":%llu,\"token_file_size\":%llu,\"direct_prefill_tokens\":true,"
                    "\"real_model_forward\":true,\"config_overwrites_environment\":true,"
                    "\"observer_commit_wait\":true,\"performance_from_capture\":false,\"loaded_metallib_path\":",
                    capacity,(unsigned long long)model_stat.st_size,(unsigned long long)model_stat.st_ino,
                    (long long)model_stat.st_mtimespec.tv_sec,model_stat.st_mtimespec.tv_nsec,
                    (unsigned long long)config_stat.st_size,(unsigned long long)token_stat.st_size)<0)recorded=0;
                if(!myds_metal_loaded_library_path() || !json_string(file,myds_metal_loaded_library_path()) ||
                   fputs("}\n",file)<0)recorded=0;
                if(fclose(file))recorded=0;
            }
        }else recorded=0;
        fprintf(stderr,"INDEXER_REAL_PREFILL_BEGIN tokens=32768 context=%u raw_output=%s\n",capacity,argv[5]);
        int prefill=recorded ? myds_v41_prefill_tokens(ctx,tokens.ids,32768,NULL) : -1;
        int complete=myds_indexer_capture_finish();
        if(prefill>=0 && complete && recorded) {
            puts("{\"kind\":\"capture_complete\",\"cases\":3,\"layers\":[2,8,14],"
                 "\"tokens\":32768,\"real_model_forward\":true,\"performance_from_capture\":false}");result=0;
        }
    }
    myds_context_free(ctx);myds_model_free(model);free(tokens.ids);return result;
}
