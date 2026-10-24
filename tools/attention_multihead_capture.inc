/* Build-only observer. Included solely by the separate capture GPU object.
 * The observer adds a commit/wait and is never used for performance timing. */
#include <stdio.h>
#include <sys/stat.h>

static struct {
    const char *directory;
    FILE *records;
    uint32_t layer, seen, count;
} mh_capture = {.layer=UINT32_MAX};

int myds_attention_multihead_capture_begin(const char *directory) {
    if (!directory || directory[0]!='/' || mh_capture.directory ||
        mkdir(directory,0700)!=0) return 0;
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/capture-records.jsonl",directory);
    if(n<0 || (size_t)n>=sizeof(path))return 0;
    mh_capture.records=fopen(path,"wx");
    if(!mh_capture.records)return 0;
    mh_capture.directory=directory;
    return 1;
}

void myds_attention_multihead_capture_set_layer(uint32_t layer) {
    mh_capture.layer=layer;
}

static int mh_capture_blob(const char *kind,const void *data,size_t bytes) {
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/layer-%u-%s.bin",mh_capture.directory,mh_capture.layer,kind);
    if(!data || n<0 || (size_t)n>=sizeof(path))return 0;
    FILE *file=fopen(path,"wx");
    if(!file)return 0;
    int ok=fwrite(data,1,bytes,file)==bytes;
    if(fclose(file)!=0)ok=0;
    return ok;
}

static const void *mh_capture_readonly_data(const myds_metal_buffer_t *buffer) {
    /* The public accessor predates const buffer handles; the observer only reads. */
    return myds_metal_buffer_data((myds_metal_buffer_t *)buffer);
}

static int mh_capture_prefill(const struct flash_vec_args *args,
        const myds_metal_buffer_t *q,const myds_metal_buffer_t *kv,
        const myds_metal_buffer_t *mask,const myds_metal_buffer_t *sinks,
        uint32_t base_pos,uint32_t c_start,uint32_t rows,uint32_t n_comp,
        uint32_t top_k,uint32_t ratio,int gathered,int matrix,int head_first) {
    if(!mh_capture.directory)return 1;
    if(mh_capture.layer!=2 && mh_capture.layer!=8 && mh_capture.layer!=19)return 1;
    if(base_pos!=0 || c_start!=2048)return 1;
    uint32_t bit=mh_capture.layer==2?1u:mh_capture.layer==8?2u:4u;
    if((mh_capture.seen&bit) || rows!=2048 || n_comp<512 || top_k!=512 || ratio!=2 ||
       !gathered || matrix || !head_first || args->ne11!=640 ||
       args->nb_kvq!=655360 || args->nb31<1280 || args->nb31%2)return 0;
    /* Wait for the real Q projection, gather and mask producers. Never inspect
     * a shared buffer while its command buffer remains in flight. */
    int active=myds_metal_batch_active();
    if(!active || !myds_metal_commit_batch())return 0;
    const size_t qbytes=(size_t)rows*64u*512u*4u;
    const size_t kvbytes=(size_t)rows*640u*512u*2u;
    if(myds_metal_buffer_size(q)<qbytes || myds_metal_buffer_size(kv)<kvbytes ||
       myds_metal_buffer_size(mask)<(size_t)(rows-1)*args->nb31+1280 ||
       myds_metal_buffer_size(sinks)<256)return 0;
    if(!mh_capture_blob("args",args,sizeof(*args)) ||
       !mh_capture_blob("q-f32",mh_capture_readonly_data(q),qbytes) ||
       !mh_capture_blob("kv-f16",mh_capture_readonly_data(kv),kvbytes) ||
       !mh_capture_blob("sinks-f32",mh_capture_readonly_data(sinks),256))return 0;
    /* Preserve exactly the 640 consumed half values per query. The original
     * mask stride remains in args; unconsumed stride padding is not evidence. */
    uint16_t *packed=malloc((size_t)rows*1280u);
    const uint8_t *source=mh_capture_readonly_data(mask);
    if(!packed || !source){free(packed);return 0;}
    for(uint32_t row=0;row<rows;row++)
        memcpy(packed+(size_t)row*640u,source+(size_t)row*args->nb31,1280u);
    int ok=mh_capture_blob("mask-f16",packed,(size_t)rows*1280u);
    free(packed);
    if(!ok)return 0;
    if(fprintf(mh_capture.records,
       "{\"layer\":%u,\"base_pos\":%u,\"c_start\":%u,\"rows\":%u,\"keys\":640,"
       "\"n_comp\":%u,\"top_k\":%u,\"ratio\":%u,\"mask_source_stride\":%llu,"
       "\"gathered\":true,\"head_first\":true,\"matrix\":false,"
       "\"observer_commit_wait\":true,\"performance_from_capture\":false}\n",
       mh_capture.layer,base_pos,c_start,rows,n_comp,top_k,ratio,
       (unsigned long long)args->nb31)<0 || fflush(mh_capture.records)!=0)return 0;
    mh_capture.seen|=bit;mh_capture.count++;
    return myds_metal_begin_batch();
}

int myds_attention_multihead_capture_finish(void) {
    int ok=mh_capture.records && mh_capture.seen==7u && mh_capture.count==3u;
    if(mh_capture.records && fclose(mh_capture.records)!=0)ok=0;
    mh_capture.records=NULL;
    return ok;
}
