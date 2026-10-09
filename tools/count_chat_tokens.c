/* CPU-only input audit for bench_phase. Link the actual server prompt renderer,
 * GGUF metadata reader and tokenizer; no model loader or Metal initialization.
 * macOS build (dead stripping keeps only reachable parser/tokenizer code):
 * clang -O2 -Wall -Wextra -std=c11 -Iinclude -Ithird_party \
 *   tools/count_chat_tokens.c src/myds_server.c src/myds_gguf.c \
 *   src/myds_tokenizer.c src/myds_log.c -Wl,-dead_strip -lpthread -lm \
 *   -o bin/count_chat_tokens
 * Usage: count_chat_tokens MODEL REQUEST_JSON OUTPUT_PREFIX
 * Writes OUTPUT_PREFIX.prompt.txt and OUTPUT_PREFIX.tokens.i32le.bin for SHA256.
 */
#define _POSIX_C_SOURCE 200809L
#include "myds_server_internal.h"
#include "myds_tokenizer.h"
#include "myds_vjob.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* parse_chat_prompt can reach image parsing in the server TU. This probe is
 * deliberately text-only: never link or initialize the visual/GPU runtime. */
int myds_vjob_sink_add(myds_vjob_sink_t *sink,const uint8_t *encoded,size_t len,
                        char *error,size_t error_cap) {
    (void)sink;(void)encoded;(void)len;
    if(error_cap)snprintf(error,error_cap,"CPU tokenizer probe accepts text only");
    return 0;
}

static char *read_request(const char *path,size_t *bytes) {
    FILE *f=fopen(path,"rb");if(!f)return NULL;
    if(fseek(f,0,SEEK_END)){fclose(f);return NULL;}
    long n=ftell(f);
    if(n<0||n>64*1024*1024||fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
    char *body=malloc((size_t)n+1u);
    if(!body){fclose(f);return NULL;}
    if(fread(body,1,(size_t)n,f)!=(size_t)n||memchr(body,0,(size_t)n)){
        free(body);fclose(f);return NULL;
    }
    fclose(f);body[n]=0;*bytes=(size_t)n;return body;
}
static int write_bytes(const char *prefix,const char *suffix,const void *data,size_t bytes) {
    size_t n=strlen(prefix)+strlen(suffix)+1u;
    char *path=malloc(n);if(!path)return 0;
    snprintf(path,n,"%s%s",prefix,suffix);
    FILE *f=fopen(path,"wb");free(path);if(!f)return 0;
    int ok=fwrite(data,1,bytes,f)==bytes;
    if(fclose(f))ok=0;return ok;
}
int main(int argc,char **argv) {
    if(argc!=4){fprintf(stderr,"usage: %s MODEL REQUEST_JSON OUTPUT_PREFIX\n",argv[0]);return 2;}
    size_t body_bytes=0;char *body=read_request(argv[2],&body_bytes);
    if(!body){perror("request read");return 2;}
    /* Identical capacity and parser call to bench_phase. */
    size_t prompt_cap=body_bytes*4u+8192u;
    if(prompt_cap<65536u)prompt_cap=65536u;
    char *prompt=malloc(prompt_cap);
    if(!prompt||!parse_chat_prompt(body,prompt,prompt_cap)){
        fprintf(stderr,"parse_chat_prompt failed\n");free(prompt);free(body);return 2;
    }
    free(body);
    gguf_file_t *gf=gguf_open(argv[1]);
    myds_tokenizer_t *tok=gf?myds_tokenizer_load(gf):NULL;
    int status=1;int32_t *ids=NULL;unsigned char *encoded=NULL;
    if(!tok){fprintf(stderr,"GGUF tokenizer metadata load failed\n");goto done;}
    /* Identical add_bos=true to myds_generate_internal. The count-only call
     * does not inspect tensor payloads or prepare Engram rows. */
    int32_t count=myds_tokenizer_encode(tok,prompt,NULL,0,true);
    if(count<=0||count>1048576){fprintf(stderr,"invalid token count %d\n",count);goto done;}
    ids=malloc((size_t)count*sizeof(*ids));encoded=malloc((size_t)count*4u);
    if(!ids||!encoded||myds_tokenizer_encode(tok,prompt,ids,count,true)!=count)goto done;
    for(int32_t i=0;i<count;i++)for(unsigned j=0;j<4;j++)
        encoded[(size_t)i*4u+j]=(unsigned char)((uint32_t)ids[i]>>(j*8u));
    const size_t prompt_bytes=strlen(prompt);
    if(!write_bytes(argv[3],".prompt.txt",prompt,prompt_bytes)||
       !write_bytes(argv[3],".tokens.i32le.bin",encoded,(size_t)count*4u)){
        perror("token audit output");goto done;
    }
    printf("{\"prompt_renderer\":\"parse_chat_prompt\",\"tokenizer\":\"target_gguf_metadata\","
           "\"add_bos\":true,\"prompt_tokens\":%d,\"prompt_bytes\":%zu,\"request_bytes\":%zu,"
           "\"bos_id\":%d,\"eos_id\":%d,\"token_encoding\":\"little_endian_int32\","
           "\"gguf_metadata_prefix_bytes\":%llu,"
           "\"model_loader_called\":false,\"metal_initialized\":false}\n",
           count,prompt_bytes,body_bytes,tok->bos_id,tok->eos_id,
           (unsigned long long)gf->tensor_data_offset);
    status=0;
done:
    free(encoded);free(ids);myds_tokenizer_free(tok);gguf_close(gf);free(prompt);return status;
}
