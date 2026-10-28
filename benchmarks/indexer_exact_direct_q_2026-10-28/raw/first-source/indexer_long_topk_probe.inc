/* Linked into the isolated long-indexer fixture. This invokes the actual
 * production causal Top512 provider on both score tensors. */
#include "myds_metal.h"
#include "myds_v41_gpu.h"

static myds_v41_gpu_scratch_t *topk_scratch;

static void init_topk(NSString *deployed_metallib) {
    setenv("MYDS_INDEXER_TOPK_PREFIX_MERGE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_FUSE", "1", 1);
    setenv("MYDS_INDEXER_TOPK_NAN_SHUFFLE", "0", 1);
    setenv("MYDS_INDEXER_BATCH_FLOATS", "2097152", 1);
    setenv("MYDS_CHUNK_KEY_MAX", "65536", 1);
    need(myds_metal_init(deployed_metallib.stringByDeletingLastPathComponent.UTF8String),
         "production Metal runtime initialization for topk");
    const char *loaded=myds_metal_loaded_library_path();
    need(loaded!=NULL,"topk runtime reports actual loaded library");
    NSString *expected=canon(deployed_metallib),*actual=canon([NSString stringWithUTF8String:loaded]);
    need([actual isEqualToString:expected],"topk runtime loaded the exact nominated library path");
    NSString *expected_sha=filesha(expected),*actual_sha=filesha(actual);
    need([expected_sha isEqualToString:actual_sha],"topk runtime loaded library SHA matches");
    manifest[@"topk_runtime_library"]=@{@"path":actual,@"sha256":actual_sha,@"path_and_sha_match":@YES};
    topk_scratch=myds_v41_gpu_scratch_create(65536u);
    need(topk_scratch!=NULL,"production topk scratch allocation");
}

static NSDictionary *topk_against_exact(NSData *exact,NSData *candidate,
                                        uint32_t nc,uint32_t nt,uint32_t start) {
    const uint32_t top=512u;
    size_t score_bytes=(size_t)nc*nt*4u, selected_bytes=(size_t)top*nt*4u;
    need(nc>top && (uint64_t)nc*nt<=2097152u && exact.length==score_bytes &&
         candidate.length==score_bytes,"topk production batch and input size");
    myds_metal_buffer_t *scores[3],*roots[3],*views[3];
    NSData *inputs[]={exact,exact,candidate};
    myds_v41_gpu_topk_prefix_counters(NULL,true);
    myds_v41_gpu_topk_nan_fuse_counters(NULL,true);
    for(unsigned arm=0;arm<3;arm++) {
        scores[arm]=myds_metal_buffer_alloc(score_bytes);
        roots[arm]=myds_metal_buffer_alloc(selected_bytes+2u*GUARD);
        need(scores[arm]&&roots[arm],"topk buffers");
        memset(myds_metal_buffer_data(roots[arm]),0xa5,selected_bytes+2u*GUARD);
        views[arm]=myds_metal_buffer_sub_view(roots[arm],GUARD,selected_bytes);
        need(views[arm]&&myds_metal_buffer_write(scores[arm],0,inputs[arm].bytes,score_bytes),
             "topk distinct score/output views");
        need(myds_metal_begin_batch()&&
             myds_v41_gpu_topk_causal(views[arm],scores[arm],nc,top,nt,start,2u,topk_scratch)&&
             myds_metal_commit_batch(),"production topk GPU completion");
    }
    myds_v41_gpu_topk_prefix_counters_t pc;
    myds_v41_gpu_topk_nan_fuse_counters_t fc;
    myds_v41_gpu_topk_prefix_counters(&pc,false);
    myds_v41_gpu_topk_nan_fuse_counters(&fc,false);
    need(pc.requests==3&&pc.encoded==3&&!pc.unavailable&&fc.requests==3&&fc.encoded==3&&!fc.unavailable,
         "production prefix/nan-fuse topk paths encoded for all three arms");
    BOOL valid=YES;
    for(unsigned arm=0;arm<3;arm++) {
        const uint8_t *raw=myds_metal_buffer_data(roots[arm]);
        valid&=guards(raw,selected_bytes);
        valid&=!memcmp(myds_metal_buffer_data(scores[arm]),inputs[arm].bytes,score_bytes);
        const int32_t *ids=(const int32_t *)(raw+GUARD);
        uint8_t *seen=malloc(nc);need(seen!=NULL,"topk distinct-id scratch");
        for(uint32_t row=0;row<nt;row++) {
            memset(seen,0,nc);
            uint32_t visible=(start+row+1u)/2u;if(visible>nc)visible=nc;
            for(uint32_t i=0;i<top;i++) {
                int32_t id=ids[(size_t)row*top+i];
                if(id<0||(uint32_t)id>=visible||seen[id]) {valid=NO;continue;}
                seen[id]=1;
            }
        }
        free(seen);
    }
    const int32_t *x=(const int32_t *)myds_metal_buffer_data(views[0]);
    const int32_t *repeat=(const int32_t *)myds_metal_buffer_data(views[1]);
    const int32_t *y=(const int32_t *)myds_metal_buffer_data(views[2]);
    size_t mismatches=0,aa_mismatches=0,first=SIZE_MAX,aa_first=SIZE_MAX;
    for(size_t i=0;i<(size_t)nt*top;i++) {
        if(x[i]!=repeat[i]) {if(aa_first==SIZE_MAX)aa_first=i;aa_mismatches++;}
        if(x[i]!=y[i]) {if(first==SIZE_MAX)first=i;mismatches++;}
    }
    NSDictionary *result=@{@"production_topk":@YES,@"prefix_encoded":@(pc.encoded),
        @"nan_fuse_encoded":@(fc.encoded),@"gpu_batches_completed":@3,@"guards_and_causal_ids_valid":@(valid),
        @"exact_aa_ordered_id_mismatches":@(aa_mismatches),
        @"exact_aa_first_mismatch_index":aa_first==SIZE_MAX?(id)[NSNull null]:@(aa_first),
        @"ordered_id_mismatches":@(mismatches),@"first_mismatch_index":first==SIZE_MAX?(id)[NSNull null]:@(first),
        @"first_exact_id":first==SIZE_MAX?(id)[NSNull null]:@(x[first]),
        @"first_candidate_id":first==SIZE_MAX?(id)[NSNull null]:@(y[first]),
        @"exact_top512_sha256":sha(x,selected_bytes),@"exact_repeat_top512_sha256":sha(repeat,selected_bytes),
        @"candidate_top512_sha256":sha(y,selected_bytes)};
    for(unsigned arm=0;arm<3;arm++){myds_metal_buffer_free(views[arm]);myds_metal_buffer_free(roots[arm]);myds_metal_buffer_free(scores[arm]);}
    return result;
}
