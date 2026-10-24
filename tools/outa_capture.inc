/* Stage-only Output_A input observer, included by myds_v41.c solely under
 * MYDS_OUTA_CAPTURE. The normal object must not contain these symbols.
 *
 * Call immediately before myds_metal_attn_output_a_bf16, after the explicit
 * !attention_rope_consumed rope_round branch. The captured heads are the
 * actual inverse-RoPE/BF16 boundary, including when RF consumed that operation.
 * The observer adds commit/wait boundaries and supplies no performance proof.
 */
#include <CommonCrypto/CommonDigest.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { OUTA_CAPTURE_LAYERS=40, OUTA_CAPTURE_TOKENS=16,
       OUTA_CAPTURE_FLOATS=64*512, OUTA_CAPTURE_BYTES=64*512*4 };
static struct {
    const char *directory;
    FILE *records;
    unsigned calls, count, seen, rope_consumed_calls;
    int failed;
} outa_capture;

static void outa_capture_sha256(const void *data,size_t bytes,char result[65]) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data,(CC_LONG)bytes,digest);
    for(unsigned i=0;i<sizeof(digest);i++)snprintf(result+2*i,3,"%02x",digest[i]);
}

int myds_outa_capture_begin(const char *directory) {
    if(!directory || directory[0]!='/' || outa_capture.directory ||
       mkdir(directory,0700)!=0)return 0;
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/capture-records.jsonl",directory);
    if(n<0 || (size_t)n>=sizeof(path))return 0;
    FILE *records=fopen(path,"wx");
    if(!records)return 0;
    outa_capture.directory=directory;
    outa_capture.records=records;
    return 1;
}

static int myds_outa_capture_input(const myds_metal_buffer_t *heads,
        uint32_t layer,uint32_t pos,bool need_logits,bool ordinary_decode_allowed,
        bool trace_on,bool attention_rope_consumed,bool compressed,ggml_type type) {
    if(!outa_capture.directory)return 1;
    /* Every real layer invocation must be observed; no selected-layer ordinal
     * inference or missing/repeated forward is accepted at finish. */
    if(outa_capture.failed || layer!=outa_capture.calls%OUTA_CAPTURE_LAYERS ||
       pos!=outa_capture.calls/OUTA_CAPTURE_LAYERS || pos>=OUTA_CAPTURE_TOKENS ||
       !need_logits || !ordinary_decode_allowed || trace_on ||
       type!=GGML_TYPE_Q4_K || !heads ||
       myds_metal_buffer_size(heads)<OUTA_CAPTURE_BYTES) {
        outa_capture.failed=1;
        return 0;
    }
    outa_capture.calls++;
    outa_capture.rope_consumed_calls+=(unsigned)attention_rope_consumed;
    const unsigned li=layer==0?0:layer==19?1:layer==39?2:UINT32_MAX;
    const unsigned pi=pos==0?0:pos==1?1:pos==7?2:pos==15?3:UINT32_MAX;
    if(li==UINT32_MAX || pi==UINT32_MAX)return 1;
    const unsigned bit=1u<<(li*4u+pi);
    if(outa_capture.seen&bit){outa_capture.failed=1;return 0;}
    /* Completion is mandatory before reading the shared input. Restore the
     * active batch after the CPU observer, including on a dump failure. */
    if(!myds_metal_batch_active() || !myds_metal_commit_batch()) {
        outa_capture.failed=1;
        return 0;
    }
    const uint32_t *words=myds_metal_buffer_data((myds_metal_buffer_t *)heads);
    int ok=words!=NULL;
    for(unsigned i=0;ok && i<OUTA_CAPTURE_FLOATS;i++)
        if((words[i]&0x7f800000u)==0x7f800000u || (words[i]&0xffffu))ok=0;
    char filename[96],path[4096],digest[65]={0};
    int n=snprintf(filename,sizeof(filename),"layer-%u-pos-%u-input-f32.bin",layer,pos);
    ok=ok && n>0 && (size_t)n<sizeof(filename);
    n=snprintf(path,sizeof(path),"%s/%s",outa_capture.directory,filename);
    ok=ok && n>0 && (size_t)n<sizeof(path);
    if(ok) {
        outa_capture_sha256(words,OUTA_CAPTURE_BYTES,digest);
        FILE *file=fopen(path,"wx");
        if(!file)ok=0;
        else {
            ok=fwrite(words,1,OUTA_CAPTURE_BYTES,file)==OUTA_CAPTURE_BYTES;
            if(fclose(file)!=0)ok=0;
        }
    }
    const int resumed=myds_metal_begin_batch();
    ok=ok && resumed;
    if(ok) {
        ok=fprintf(outa_capture.records,
            "{\"layer\":%u,\"position\":%u,\"call_ordinal\":%u,"
            "\"need_logits\":true,\"ordinary_decode_allowed\":true,\"trace_on\":false,"
            "\"attention_rope_consumed\":%s,\"compressed\":%s,\"weight_type\":%u,"
            "\"boundary\":\"post_inverse_rope_bf16_before_output_a\","
            "\"input_f32\":{\"path\":\"%s\",\"bytes\":%u,\"sha256\":\"%s\"},"
            "\"all_f32_finite\":true,\"all_bf16_bits\":true,"
            "\"observer_commit_wait\":true,\"observer_batch_restored\":true,"
            "\"performance_from_capture\":false}\n",
            layer,pos,outa_capture.calls,attention_rope_consumed?"true":"false",
            compressed?"true":"false",(unsigned)type,filename,OUTA_CAPTURE_BYTES,digest)>0;
        if(fflush(outa_capture.records)!=0)ok=0;
    }
    if(!ok || !resumed){outa_capture.failed=1;return 0;}
    outa_capture.seen|=bit;
    outa_capture.count++;
    return 1;
}

int myds_outa_capture_finish(void) {
    int ok=outa_capture.records && !outa_capture.failed &&
        outa_capture.calls==OUTA_CAPTURE_LAYERS*OUTA_CAPTURE_TOKENS &&
        outa_capture.count==12u && outa_capture.seen==0xfffu;
    if(outa_capture.records && fclose(outa_capture.records)!=0)ok=0;
    outa_capture.records=NULL;
    if(!outa_capture.directory)return 0;
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/capture-observer-summary.json",outa_capture.directory);
    FILE *file=n>0 && (size_t)n<sizeof(path)?fopen(path,"wx"):NULL;
    if(!file)return 0;
    if(fprintf(file,
        "{\"schema_version\":1,\"capture_kind\":\"myds_decode_output_a\","
        "\"status\":\"%s\",\"observed_layer_calls\":%u,\"selected_cases\":%u,"
        "\"selected_mask\":%u,\"rope_consumed_layer_calls\":%u,"
        "\"forward_tokens\":16,\"layers\":[0,19,39],\"positions\":[0,1,7,15],"
        "\"input_bytes_per_case\":%u,\"all_f32_finite\":%s,\"all_bf16_bits\":%s,"
        "\"public_ordinary_decode\":true,\"trace_on\":false,"
        "\"performance_from_capture\":false}\n",
        ok?"complete":"failed",outa_capture.calls,outa_capture.count,
        outa_capture.seen,outa_capture.rope_consumed_calls,OUTA_CAPTURE_BYTES,
        ok?"true":"false",ok?"true":"false")<=0)ok=0;
    if(fclose(file)!=0)ok=0;
    return ok;
}
