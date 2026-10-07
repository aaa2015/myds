#include "myds_v41_decode.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static pthread_once_t fuse_once = PTHREAD_ONCE_INIT;
static int fuse;

static void init_fuse(void) {
    const char *value = getenv("MYDS_DECODE_Q8_FUSE_BF16");
    fuse = !(value && strcmp(value, "0") == 0);
}

int myds_v41_decode_gemv_bf16(myds_metal_buffer_t *out,
    const myds_metal_buffer_t *weight, const myds_metal_buffer_t *input,
    uint32_t rows, uint32_t width, ggml_type type, bool trace_on) {
    pthread_once(&fuse_once, init_fuse);
    if (fuse && !trace_on && type == GGML_TYPE_Q8_0 &&
        myds_metal_buffer_offset(out) % 16u == 0)
        return myds_metal_v41_gemv_bf16(out, weight, input, rows, width, type);
    if (!myds_metal_gemv(out, weight, input, rows, width, type)) return 0;
    if (!myds_metal_get_auto_barrier()) {
        myds_metal_barrier_buffer(out);
    }
    return myds_metal_v41_bf16(out, rows);
}
