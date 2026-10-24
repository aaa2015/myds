/* Real 16401-token prefill observer. No benchmark timings are produced. */
#define _POSIX_C_SOURCE 200809L
#include "myds_internal.h"
#include "myds_v41.h"
#include "reference/trace_input.h"
#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

extern char **environ;
extern int myds_moe_workspace_capture_begin(const char *directory);
extern int myds_moe_workspace_capture_finish(void);

static void json_string(FILE *out,const char *s) {
    if(!s){fputs("null",out);return;}fputc('"',out);
    for(const unsigned char *p=(const unsigned char *)s;*p;p++) {
        if(*p=='"' || *p=='\\'){fputc('\\',out);fputc(*p,out);}
        else if(*p<32)fprintf(out,"\\u%04x",*p);else fputc(*p,out);
    }
    fputc('"',out);
}
static int file_sha(const char *path,size_t limit,char result[65]) {
    FILE *f=fopen(path,"rb");if(!f)return 0;
    unsigned char block[65536],digest[32];CC_SHA256_CTX c;CC_SHA256_Init(&c);
    while(limit){size_t n=fread(block,1,limit<sizeof(block)?limit:sizeof(block),f);if(!n)break;
        CC_SHA256_Update(&c,block,(CC_LONG)n);limit-=n;}
    int ok=!ferror(f);if(fclose(f)!=0)ok=0;if(!ok)return 0;
    CC_SHA256_Final(digest,&c);for(unsigned i=0;i<32;i++)snprintf(result+i*2u,3,"%02x",digest[i]);result[64]=0;return 1;
}
static int capture_json(const char *directory,const char *model_path,const char *tokens_path,myds_model_t *model) {
    char path[4096],header[65],tokensha[65],recordsha[65];struct stat st;
    if(stat(model_path,&st)!=0 || st.st_size<1048576 || !file_sha(model_path,1048576,header) ||
       !file_sha(tokens_path,SIZE_MAX,tokensha))return 0;
    int n=snprintf(path,sizeof(path),"%s/capture-records.jsonl",directory);
    if(n<0 || (size_t)n>=sizeof(path) || !file_sha(path,SIZE_MAX,recordsha))return 0;
    FILE *records=fopen(path,"rb");if(!records)return 0;
    n=snprintf(path,sizeof(path),"%s/capture.json",directory);
    FILE *out=n>0 && (size_t)n<sizeof(path)?fopen(path,"wx"):NULL;
    if(!out){fclose(records);return 0;}
    fprintf(out,"{\"schema_version\":1,\"capture_kind\":\"myds_prefill_moe_workspace\",\"model_bytes\":%llu,"
        "\"model_header_bytes\":1048576,\"model_header_sha256\":\"%s\",\"source\":{"
        "\"kind\":\"actual_native_prefill_routed_moe_batch\",\"model_path\":",
        (unsigned long long)st.st_size,header);json_string(out,model_path);
    fputs(",\"tokens_path\":",out);json_string(out,tokens_path);
    fprintf(out,",\"tokens_sha256\":\"%s\",\"records_sha256\":\"%s\",\"actual_tokens\":16401,"
        "\"max_seq_len\":327680,\"trace_on\":false,\"performance_from_capture\":false,\"actual_metallib_path\":",tokensha,recordsha);
    json_string(out,myds_metal_loaded_library_path());fputs(",\"environment\":{",out);
    unsigned count=0;
    for(char **p=environ;p && *p;p++)if(!strncmp(*p,"MYDS_",5)) {
        const char *equal=strchr(*p,'=');if(!equal)continue;
        size_t bytes=(size_t)(equal-*p);char *name=malloc(bytes+1);if(!name){fclose(records);fclose(out);return 0;}
        memcpy(name,*p,bytes);name[bytes]=0;if(count++)fputc(',',out);json_string(out,name);fputc(':',out);json_string(out,equal+1);free(name);
    }
    fputs("}},\"weight_source\":\"model_backed_complete_tensor_intervals_not_exported_weight_payloads\",\"tensors\":[",out);
    count=0;
    const unsigned layers[]={0,19,39};const char *kinds[]={"gate","up","down"};
    for(unsigned li=0;li<3;li++)for(unsigned k=0;k<3;k++) {
        char name[128];snprintf(name,sizeof(name),"blk.%u.ffn_%s_exps.weight",layers[li],kinds[k]);
        const gguf_tensor_t *t=gguf_find_tensor(model->gf,name);
        if(!t || t->n_dims!=3 || t->type!=(k==2?GGML_TYPE_Q2_K:GGML_TYPE_IQ2_XXS) ||
           t->dims[0]!=(k==2?2304u:5120u) || t->dims[1]!=(k==2?5120u:2304u) || t->dims[2]!=384u) {
            fclose(records);fclose(out);return 0;
        }
        if(count++)fputc(',',out);
        fprintf(out,"{\"layer\":%u,\"name\":\"%s\",\"type\":%u,\"dims\":[%llu,%llu,%llu],"
            "\"offset\":%llu,\"bytes\":%llu,\"model_backed\":true}",layers[li],name,(unsigned)t->type,
            (unsigned long long)t->dims[0],(unsigned long long)t->dims[1],(unsigned long long)t->dims[2],
            (unsigned long long)(model->gf->tensor_data_offset+t->offset),(unsigned long long)t->size_bytes);
    }
    fputs("],\"cases\":[",out);char *line=NULL;size_t capacity=0;ssize_t bytes;count=0;
    while((bytes=getline(&line,&capacity,records))>0) {
        while(bytes>0 && (line[bytes-1]=='\n'||line[bytes-1]=='\r'))bytes--;
        if(!bytes)continue;if(count++)fputc(',',out);if(fwrite(line,1,(size_t)bytes,out)!=(size_t)bytes){free(line);fclose(records);fclose(out);return 0;}
    }
    free(line);int ok=!ferror(records) && count>=3;fputs("]}\n",out);
    if(fclose(records)!=0)ok=0;if(fclose(out)!=0)ok=0;return ok;
}

int main(int argc,char **argv) {
    if(argc!=6 || strcmp(argv[2],"--tokens-file") || strcmp(argv[4],"--out-dir") ||
       argv[1][0]!='/' || argv[3][0]!='/' || argv[5][0]!='/') {
        fprintf(stderr,"Usage: %s MODEL --tokens-file ABS --out-dir NEW_ABS\n",argv[0]);return 2;
    }
    for(char **p=environ;p && *p;p++)if(!strncmp(*p,"MYDS_TRACE",10)||!strncmp(*p,"MYDS_SKIP_",10)||
       !strncmp(*p,"MYDS_PROFILE",12)||!strncmp(*p,"MYDS_KPROF",10)||!strncmp(*p,"MYDS_MOE_SOLO=",14)||
       !strncmp(*p,"MYDS_MOE_STEP_PROF=",19))return 2;
    trace_tokens tokens={0};if(!trace_tokens_parse(&tokens,4,argv) || tokens.count!=16401) {
        free(tokens.ids);fprintf(stderr,"Exactly 16401 actual tokenizer IDs required\n");return 2;
    }
    if(mkdir(argv[5],0700)!=0){free(tokens.ids);perror("new capture directory");return 2;}
    myds_model_t *model=myds_model_load(argv[1]);if(!model){free(tokens.ids);return 1;}
    for(size_t i=0;i<tokens.count;i++)if(tokens.ids[i]<0 || (uint32_t)tokens.ids[i]>=model->vocab_size) {
        myds_model_free(model);free(tokens.ids);return 2;
    }
    myds_config_t cfg={.max_seq_len=327680,.max_new_tokens=1,.temperature=0,.top_p=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);int result=1;
    if(ctx && myds_moe_workspace_capture_begin(argv[5])) {
        int prefill=myds_v41_prefill_tokens(ctx,tokens.ids,16401,NULL);
        int complete=myds_moe_workspace_capture_finish();
        if(prefill==1 && complete && capture_json(argv[5],argv[1],argv[3],model)) {
            puts("{\"kind\":\"capture_complete\",\"capture_kind\":\"myds_prefill_moe_workspace\","
                 "\"layers\":[0,19,39],\"actual_tokens\":16401,\"real_model_forward\":true,\"performance_from_capture\":false}");result=0;
        }
    }
    myds_context_free(ctx);myds_model_free(model);free(tokens.ids);return result;
}
