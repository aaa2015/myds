/* Included only in a MYDS_INDEXER_CAPTURE diagnostic GPU object.
 * Commit/wait observations are correctness evidence, never performance data. */
#include <stdio.h>
#include <sys/stat.h>

struct indexer_capture_args72 {
    uint32_t n_comp,n_tokens,n_head,head_dim,pos0,ratio;
    uint64_t q_token_stride,q_head_stride,weights_token_stride,index_row_stride,score_token_stride;
    float scale;
};
_Static_assert(sizeof(struct indexer_capture_args72)==72,"indexer tiled ABI");
static struct {
    const char *directory;
    FILE *records,*selected_records;
    uint32_t selected_seen,selected_count,topk_active,topk_count;
    struct {char name[128];uint32_t grid[3],threads[3];size_t smem;int accepted;} topk[32];
    myds_v41_gpu_topk_prefix_counters_t prefix_before;
    myds_v41_gpu_topk_nan_fuse_counters_t nan_before;
    myds_v41_gpu_topk_nan_shuffle_counters_t shuffle_before;
    uint32_t layer,base_pos,c_start,c_len,off,batch_floats,seen,count;
    int failed;
} idx_capture={.layer=UINT32_MAX};

int myds_indexer_capture_begin(const char *directory) {
    if(!directory || directory[0]!='/' || idx_capture.directory || mkdir(directory,0700))return 0;
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/capture-records.jsonl",directory);
    if(n<0 || (size_t)n>=sizeof(path))return 0;
    idx_capture.records=fopen(path,"wx");
    if(!idx_capture.records)return 0;
    n=snprintf(path,sizeof(path),"%s/selected-records.jsonl",directory);
    if(n<0 || (size_t)n>=sizeof(path))return 0;
    idx_capture.selected_records=fopen(path,"wx");
    if(!idx_capture.selected_records)return 0;
    idx_capture.directory=directory;
    return 1;
}
void myds_indexer_capture_set_group(uint32_t layer,uint32_t base_pos,uint32_t c_start,
    uint32_t c_len,uint32_t off,uint32_t batch_floats) {
    idx_capture.layer=layer;idx_capture.base_pos=base_pos;idx_capture.c_start=c_start;
    idx_capture.c_len=c_len;idx_capture.off=off;idx_capture.batch_floats=batch_floats;
}
static const void *indexer_capture_data(const myds_metal_buffer_t *b) {
    return myds_metal_buffer_data((myds_metal_buffer_t *)b);
}
static int indexer_capture_blob(const char *kind,const void *data,size_t bytes) {
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/layer-%u-%s.bin",idx_capture.directory,idx_capture.layer,kind);
    if(!data || !bytes || n<0 || (size_t)n>=sizeof(path))return 0;
    FILE *f=fopen(path,"wx");if(!f)return 0;
    int ok=fwrite(data,1,bytes,f)==bytes;
    if(fclose(f))ok=0;
    return ok;
}
static uint32_t indexer_capture_bit(void) {
    return idx_capture.layer==2?1u:idx_capture.layer==8?2u:idx_capture.layer==14?4u:0u;
}
static int indexer_capture_before(const myds_metal_buffer_t *q,const myds_metal_buffer_t *w,
    const myds_metal_buffer_t *cache,uint32_t cols,uint32_t heads,uint32_t rows,
    uint32_t pos,uint32_t ratio,bool bf16) {
    if(!idx_capture.directory)return 0;
    const uint32_t bit=indexer_capture_bit();
    /* Freeze the first real subgroup of the 30K..32K chunk, independently for
     * each owner/source layer. No padding, synthetic enlargement or row slicing.
     * For production 2048 chunks and 2M floats this is normally 128x15424.
     * The actual tuple is recorded and accepted rather than asserted to be 256x16K. */
    if(!bit || idx_capture.seen&bit || idx_capture.base_pos+idx_capture.c_start!=30720u || idx_capture.off)return 0;
    if(idx_capture.failed || heads!=32u || ratio!=2u || pos!=30720u || rows<8u ||
       cols<15000u || cols>16384u || idx_capture.batch_floats!=2097152u ||
       (uint64_t)cols*rows>idx_capture.batch_floats || indexer_key_window())return -1;
    const size_t qb=(size_t)rows*heads*128u*4u,wb=(size_t)rows*heads*4u;
    const size_t kb=(size_t)cols*128u*(bf16?2u:4u);
    if(myds_metal_buffer_size(q)<qb || myds_metal_buffer_size(w)<wb ||
       myds_metal_buffer_size(cache)<kb || !myds_metal_batch_active() ||
       !myds_metal_commit_batch())return -1;
    /* Parent producers are complete. Raw files cover every consumed input byte. */
    int ok=indexer_capture_blob("q-f32",indexer_capture_data(q),qb) &&
        indexer_capture_blob("weights-f32",indexer_capture_data(w),wb) &&
        indexer_capture_blob("cache",indexer_capture_data(cache),kb);
    struct indexer_capture_args72 a={0};
    a.n_comp=cols;a.n_tokens=rows;a.n_head=heads;a.head_dim=128;a.pos0=pos;a.ratio=ratio;
    a.q_token_stride=heads*128u*4u;a.q_head_stride=512u;a.weights_token_stride=heads*4u;
    a.index_row_stride=128u*(bf16?2u:4u);a.score_token_stride=cols*4u;a.scale=1.f/64.f;
    if(ok)ok=indexer_capture_blob("args72",&a,sizeof(a));
    if(!myds_metal_begin_batch())return -1;
    if(!ok){idx_capture.failed=1;return -1;}
    return 1;
}
static int indexer_capture_after(int capture,int result,const char *provider,
    const void *args,size_t argbytes,const myds_metal_buffer_t *scores,
    uint32_t cols,uint32_t rows,uint32_t pos,uint32_t ratio,bool bf16,
    uint32_t gx,uint32_t gy,size_t smem) {
    if(capture<=0 || !result)return result;
    if(!myds_metal_batch_active() || !myds_metal_commit_batch())return 0;
    const size_t sb=(size_t)cols*rows*4u;
    int ok=myds_metal_buffer_size(scores)>=sb &&
        indexer_capture_blob("native-args",args,argbytes) &&
        indexer_capture_blob("scores-f32",indexer_capture_data(scores),sb);
    if(ok && fprintf(idx_capture.records,
        "{\"schema_version\":1,\"layer\":%u,\"base_pos\":%u,\"c_start\":%u,\"c_len\":%u,"
        "\"group_off\":%u,\"batch_floats\":%u,\"n_comp\":%u,\"n_tokens\":%u,\"n_heads\":32,"
        "\"head_dim\":128,\"pos0\":%u,\"ratio\":%u,\"cache_bf16\":%s,\"key_window\":0,"
        "\"provider\":\"%s\",\"native_args_bytes\":%zu,\"grid\":[%u,%u,1],\"threads\":[32,4,1],"
        "\"dynamic_threadgroup_bytes\":%zu,\"q_bytes\":%zu,\"weights_bytes\":%zu,\"cache_bytes\":%zu,"
        "\"scores_bytes\":%zu,\"observer_commit_wait\":true,\"performance_from_capture\":false,"
        "\"full_scores\":true,\"synthetic\":false}\n",
        idx_capture.layer,idx_capture.base_pos,idx_capture.c_start,idx_capture.c_len,
        idx_capture.off,idx_capture.batch_floats,cols,rows,pos,ratio,bf16?"true":"false",provider,argbytes,gx,gy,smem,
        (size_t)rows*32u*128u*4u,(size_t)rows*32u*4u,(size_t)cols*128u*(bf16?2u:4u),sb)<0)ok=0;
    if(ok && fflush(idx_capture.records))ok=0;
    if(!myds_metal_begin_batch())return 0;
    if(!ok){idx_capture.failed=1;return 0;}
    idx_capture.seen|=indexer_capture_bit();idx_capture.count++;
    return 1;
}
/* The actual causal Top512 consumer runs after the score capture. Its full
 * ordered IDs and accepted shader names are retained as an independent oracle.
 * layers 2/8/14 have no block-mask/filter between scores and this TopK. */
int myds_indexer_capture_topk_begin(void) {
    const uint32_t bit=indexer_capture_bit();
    if(!idx_capture.directory || !bit || !(idx_capture.seen&bit) || idx_capture.selected_seen&bit ||
       idx_capture.base_pos+idx_capture.c_start!=30720u || idx_capture.off)return 1;
    if(idx_capture.topk_active || idx_capture.failed)return 0;
    idx_capture.topk_active=1;idx_capture.topk_count=0;
    myds_v41_gpu_topk_prefix_counters(&idx_capture.prefix_before,false);
    myds_v41_gpu_topk_nan_fuse_counters(&idx_capture.nan_before,false);
    myds_v41_gpu_topk_nan_shuffle_counters(&idx_capture.shuffle_before,false);
    return 1;
}
static void indexer_capture_note_dispatch(const char *name,const uint32_t grid[3],
    const uint32_t threads[3],size_t smem,int accepted) {
    if(!idx_capture.topk_active)return;
    if(idx_capture.topk_count>=32u){idx_capture.failed=1;return;}
    const unsigned i=idx_capture.topk_count++;
    int n=snprintf(idx_capture.topk[i].name,sizeof(idx_capture.topk[i].name),"%s",name);
    if(n<0 || (size_t)n>=sizeof(idx_capture.topk[i].name)){idx_capture.failed=1;return;}
    memcpy(idx_capture.topk[i].grid,grid,12);memcpy(idx_capture.topk[i].threads,threads,12);
    idx_capture.topk[i].smem=smem;idx_capture.topk[i].accepted=accepted;
}
int myds_indexer_capture_selected(const myds_metal_buffer_t *ids,uint32_t cols,
    uint32_t rows,uint32_t pos,uint32_t ratio,uint32_t top) {
    if(!idx_capture.topk_active)return 1;
    if(idx_capture.failed || !idx_capture.topk_count || top!=512u || ratio!=2u || pos!=30720u ||
       !floats(ids,(uint64_t)rows*top) || !myds_metal_batch_active() || !myds_metal_commit_batch())return 0;
    myds_v41_gpu_topk_prefix_counters_t prefix;
    myds_v41_gpu_topk_nan_fuse_counters_t nan;
    myds_v41_gpu_topk_nan_shuffle_counters_t shuffle;
    myds_v41_gpu_topk_prefix_counters(&prefix,false);
    myds_v41_gpu_topk_nan_fuse_counters(&nan,false);
    myds_v41_gpu_topk_nan_shuffle_counters(&shuffle,false);
    int ok=indexer_capture_blob("selected-i32",indexer_capture_data(ids),(size_t)rows*top*4u);
    FILE *f=idx_capture.selected_records;
    if(ok && fprintf(f,"{\"layer\":%u,\"n_tokens\":%u,\"n_comp\":%u,\"pos0\":%u,\"ratio\":%u,\"top\":512,"
        "\"ordered_ids_complete\":true,\"mask_filter_between_scores_and_topk\":false,"
        "\"prefix_requests\":%llu,\"prefix_encoded\":%llu,\"prefix_unavailable\":%llu,"
        "\"nan_fuse_encoded\":%llu,\"nan_shuffle_encoded\":%llu,\"dispatches\":[",
        idx_capture.layer,rows,cols,pos,ratio,
        (unsigned long long)(prefix.requests-idx_capture.prefix_before.requests),
        (unsigned long long)(prefix.encoded-idx_capture.prefix_before.encoded),
        (unsigned long long)(prefix.unavailable-idx_capture.prefix_before.unavailable),
        (unsigned long long)(nan.encoded-idx_capture.nan_before.encoded),
        (unsigned long long)(shuffle.encoded-idx_capture.shuffle_before.encoded))<0)ok=0;
    for(unsigned i=0;ok && i<idx_capture.topk_count;i++) {
        if(fprintf(f,"%s{\"kernel\":\"%s\",\"grid\":[%u,%u,%u],\"threads\":[%u,%u,%u],"
            "\"dynamic_threadgroup_bytes\":%zu,\"host_accepted\":%s}",i?",":"",idx_capture.topk[i].name,
            idx_capture.topk[i].grid[0],idx_capture.topk[i].grid[1],idx_capture.topk[i].grid[2],
            idx_capture.topk[i].threads[0],idx_capture.topk[i].threads[1],idx_capture.topk[i].threads[2],
            idx_capture.topk[i].smem,idx_capture.topk[i].accepted?"true":"false")<0)ok=0;
    }
    if(ok && (fputs("],\"observer_commit_wait\":true,\"performance_from_capture\":false}\n",f)<0 || fflush(f)))ok=0;
    if(!myds_metal_begin_batch())return 0;
    idx_capture.topk_active=0;
    if(!ok){idx_capture.failed=1;return 0;}
    idx_capture.selected_seen|=indexer_capture_bit();idx_capture.selected_count++;
    return 1;
}
int myds_indexer_capture_finish(void) {
    int ok=idx_capture.records && idx_capture.selected_records && !idx_capture.failed &&
        idx_capture.seen==7u && idx_capture.count==3u && idx_capture.selected_seen==7u && idx_capture.selected_count==3u;
    if(idx_capture.records && fclose(idx_capture.records))ok=0;
    if(idx_capture.selected_records && fclose(idx_capture.selected_records))ok=0;
    idx_capture.records=NULL;idx_capture.selected_records=NULL;
    return ok;
}
