/* Real-weight ordinary decode Output_A input observer. Capture timings are
 * diagnostic only: the stage-only hook inserts 12 commit/wait boundaries.
 * Usage: capture_output_a MODEL --tokens-file ABS --out-dir NEW_ABS
 */
#include "myds_internal.h"
#include "myds_v41_gpu.h"
#include "reference/trace_input.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern char **environ;
extern int myds_outa_capture_begin(const char *directory);
extern int myds_outa_capture_finish(void);

static int unmodified_forward_environment(void) {
    /* The production configuration is supplied by the exclusive outer runner.
     * Do not enable tracing or diagnostic skip paths to obtain these inputs. */
    const char *prefixes[]={"MYDS_TRACE", "MYDS_SKIP_", "MYDS_PROFILE",
        "MYDS_KPROF", "MYDS_GAP_PROF", "MYDS_CAPTURE_"};
    for(char **entry=environ;entry && *entry;entry++)
        for(unsigned i=0;i<sizeof(prefixes)/sizeof(prefixes[0]);i++)
            if(!strncmp(*entry,prefixes[i],strlen(prefixes[i]))) {
                const char *eq=strchr(*entry,'=');
                fprintf(stderr,"Refusing diagnostic environment variable %.*s\n",
                    eq?(int)(eq-*entry):(int)strlen(*entry),*entry);
                return 0;
            }
    return 1;
}

int main(int argc,char **argv) {
    if(argc!=6 || strcmp(argv[2],"--tokens-file") || strcmp(argv[4],"--out-dir") ||
       argv[1][0]!='/' || argv[3][0]!='/' || argv[5][0]!='/') {
        fprintf(stderr,"Usage: %s MODEL --tokens-file ABS --out-dir NEW_ABS\n",argv[0]);
        return 2;
    }
    if(!unmodified_forward_environment())return 2;
    trace_tokens tokens={0};
    if(!trace_tokens_parse(&tokens,4,argv) || tokens.count!=16) {
        free(tokens.ids);fprintf(stderr,"Exactly 16 real tokenizer IDs required\n");return 2;
    }
    myds_model_t *model=myds_model_load(argv[1]);
    if(!model){free(tokens.ids);return 1;}
    for(size_t i=0;i<tokens.count;i++)if((uint32_t)tokens.ids[i]>=model->vocab_size) {
        free(tokens.ids);myds_model_free(model);return 2;
    }
    myds_config_t cfg={.max_seq_len=327680,.max_new_tokens=1,
        .temperature=0,.top_p=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);
    int result=1;
    if(ctx && myds_outa_capture_begin(argv[5])) {
        const int rope_mode=myds_v41_attention_rope_fuse_enabled();
        const int register_mode=myds_v41_gpu_attention_packed_register_enabled();
        myds_v41_attention_rope_fuse_counters(NULL,true);
        myds_v41_gpu_attention_packed_register_counters(NULL,true);
        fprintf(stderr,"OUTPUT_A_REAL_ORDINARY_DECODE_BEGIN tokens=16 context=327680 trace=off\n");
        unsigned completed=0;
        for(unsigned pos=0;pos<16;pos++) {
            /* Public forward always requests logits, preserving ordinary
             * RF/overlap eligibility. Teacher forcing supplies the next real
             * prompt token; the sampled return is checked, never fed back. */
            int32_t next=myds_forward_token(ctx,tokens.ids[pos],pos);
            if(next<0 || (uint32_t)next>=model->vocab_size)break;
            completed++;
        }
        const int complete=myds_outa_capture_finish();
        myds_metal_decode_candidate_counters_t rf={0};
        myds_v41_gpu_attention_packed_register_counters_t ar={0};
        myds_v41_attention_rope_fuse_counters(&rf,false);
        myds_v41_gpu_attention_packed_register_counters(&ar,false);
        char path[4096];
        int n=snprintf(path,sizeof(path),"%s/capture-provider.json",argv[5]);
        FILE *record=n>0 && (size_t)n<sizeof(path)?fopen(path,"wx"):NULL;
        int recorded=0;
        /* A requested RF/register provider must actually encode every layer.
         * Mode 0 remains an explicit source identity, not an RF success claim. */
        const int provider=rope_mode>=0 && register_mode>=0 && !rf.failures && !ar.failures &&
            (rope_mode!=1 || (rf.requests==640 && rf.encoded==640)) &&
            (register_mode!=1 || (ar.requests==640 && ar.encoded==640));
        if(record) {
            recorded=fprintf(record,
                "{\"schema_version\":1,\"capture_kind\":\"myds_decode_output_a\","
                "\"forward_tokens_completed\":%u,\"max_context\":327680,"
                "\"public_forward_need_logits\":true,\"ordinary_decode_allowed\":true,"
                "\"trace_on\":false,\"skip_flags_present\":false,"
                "\"rope_mode\":%d,\"register_mode\":%d,"
                "\"rf_requests\":%llu,\"rf_encoded\":%llu,\"rf_failures\":%llu,"
                "\"register_requests\":%llu,\"register_encoded\":%llu,"
                "\"register_failures\":%llu,\"register_direct_encoded\":%llu,"
                "\"register_staged_encoded\":%llu,\"provider_contract_passed\":%s,"
                "\"performance_from_capture\":false}\n",
                completed,rope_mode,register_mode,(unsigned long long)rf.requests,
                (unsigned long long)rf.encoded,(unsigned long long)rf.failures,
                (unsigned long long)ar.requests,(unsigned long long)ar.encoded,
                (unsigned long long)ar.failures,(unsigned long long)ar.direct_encoded,
                (unsigned long long)ar.staged_encoded,provider?"true":"false")>0;
            if(fclose(record)!=0)recorded=0;
        }
        if(completed==16 && complete && recorded && provider) {
            puts("{\"kind\":\"capture_complete\",\"capture_kind\":\"myds_decode_output_a\","
                "\"cases\":12,\"layers\":[0,19,39],\"positions\":[0,1,7,15],"
                "\"tokens\":16,\"input_bytes_per_case\":131072,"
                "\"real_model_forward\":true,\"ordinary_decode\":true,"
                "\"all_f32_finite\":true,\"all_bf16_bits\":true,"
                "\"performance_from_capture\":false}");
            result=0;
        }
    }
    myds_context_free(ctx);myds_model_free(model);free(tokens.ids);
    return result;
}
