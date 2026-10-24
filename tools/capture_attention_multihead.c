/* Real-weight prefill input observer. Its timing is diagnostic only. */
#include "myds_internal.h"
#include "myds_v41_gpu.h"
#include "reference/trace_input.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int myds_attention_multihead_capture_begin(const char *directory);
extern int myds_attention_multihead_capture_finish(void);

int main(int argc,char **argv) {
    if(argc!=6 || strcmp(argv[2],"--tokens-file") || strcmp(argv[4],"--out-dir") ||
       argv[1][0]!='/' || argv[3][0]!='/' || argv[5][0]!='/') {
        fprintf(stderr,"Usage: %s MODEL --tokens-file ABS --out-dir NEW_ABS\n",argv[0]);
        return 2;
    }
    if(getenv("MYDS_TRACE_PREFIX") || getenv("MYDS_PROFILE_PREFILL") ||
       getenv("MYDS_CAPTURE_ATTENTION_MULTIHEAD"))return 2;
    trace_tokens tokens={0};
    if(!trace_tokens_parse(&tokens,4,argv) || tokens.count!=4096) {
        free(tokens.ids);fprintf(stderr,"Exactly 4096 real tokenizer IDs required\n");return 2;
    }
    myds_model_t *model=myds_model_load(argv[1]);
    if(!model){free(tokens.ids);return 1;}
    for(size_t i=0;i<tokens.count;i++)if((uint32_t)tokens.ids[i]>=model->vocab_size){
        free(tokens.ids);myds_model_free(model);return 2;
    }
    myds_config_t cfg={.max_seq_len=327680,.max_new_tokens=1,
        .temperature=0,.top_p=1,.seed=42};
    myds_context_t *ctx=myds_context_create(model,&cfg);
    int result=1;
    if(ctx && myds_attention_multihead_capture_begin(argv[5])) {
        myds_v41_gpu_attention_counters(NULL,true);
        fprintf(stderr,"ATTENTION_MULTIHEAD_REAL_PREFILL_BEGIN tokens=4096 context=327680\n");
        int prefill=myds_v41_prefill_tokens(ctx,tokens.ids,4096,NULL);
        int complete=myds_attention_multihead_capture_finish();
        myds_v41_gpu_attention_counters_t counters={0};
        myds_v41_gpu_attention_counters(&counters,false);
        char path[4096];
        int n=snprintf(path,sizeof(path),"%s/capture-provider.json",argv[5]);
        FILE *record=n>0 && (size_t)n<sizeof(path)?fopen(path,"wx"):NULL;
        int recorded=0;
        if(record) {
            recorded=fprintf(record,
                "{\"gather_calls\":%llu,\"head_first_attempts\":%llu,"
                "\"head_first_encoded\":%llu,\"head_first_failures\":%llu,"
                "\"head_first_cached_fallbacks\":%llu,\"matrix_encoded\":%llu,"
                "\"matrix_fallbacks\":%llu,\"ordinary_production_prefill\":true}\n",
                (unsigned long long)counters.gather_calls,
                (unsigned long long)counters.head_first_attempts,
                (unsigned long long)counters.head_first_encoded,
                (unsigned long long)counters.head_first_failures,
                (unsigned long long)counters.head_first_cached_fallbacks,
                (unsigned long long)counters.matrix_encoded,
                (unsigned long long)counters.matrix_fallbacks)>0;
            if(fclose(record)!=0)recorded=0;
        }
        int provider=counters.head_first_encoded>=3 &&
            counters.head_first_attempts==counters.head_first_encoded &&
            !counters.head_first_failures && !counters.head_first_cached_fallbacks &&
            !counters.matrix_encoded && !counters.matrix_fallbacks;
        if(prefill>=0 && complete && recorded && provider) {
            puts("{\"kind\":\"capture_complete\",\"cases\":3,\"layers\":[2,8,19],"
                 "\"tokens\":4096,\"rows_per_case\":2048,\"keys\":640,"
                 "\"real_model_forward\":true,\"performance_from_capture\":false}");
            result=0;
        }
    }
    myds_context_free(ctx);myds_model_free(model);free(tokens.ids);
    return result;
}
