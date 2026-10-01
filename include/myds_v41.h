#ifndef MYDS_V41_H
#define MYDS_V41_H

#include "myds.h"
#include <stdbool.h>

typedef struct myds_v41_model myds_v41_model_t;
typedef struct myds_v41_context myds_v41_context_t;

/* Native, text-only V4.1 graph. Kept behind the diagnostic loader until its
 * independently captured stage tensors and logits satisfy the release gate. */
myds_v41_model_t *myds_v41_model_create(myds_model_t *model, const char *path);
void myds_v41_model_free(myds_v41_model_t *model);
myds_v41_context_t *myds_v41_context_create(myds_context_t *context);
void myds_v41_context_free(myds_v41_context_t *context);

/* ── 前缀快照（供服务端调度复用状态）──
 * 窗口环只有 128 槽、事后无法回溯，因此 prefill 经过边界时就把状态存下来。
 * 新请求的公共前缀只要覆盖到某个快照点，就能从那里续跑 —— 覆盖
 * 「客户端每次都当新对话、只带系统提示词」这一类（严格续写之外的）用法。 */
uint32_t myds_v41_snapshot_best(const myds_v41_context_t *context, uint32_t common);
bool     myds_v41_snapshot_restore(myds_v41_context_t *context, uint32_t position);
void     myds_v41_snapshot_clear(myds_v41_context_t *context);
int32_t myds_v41_forward(myds_context_t *context, int32_t token,
                        uint32_t position, bool need_logits);
int myds_v41_prefill_tokens(myds_context_t *context, const int32_t *tokens,
                            uint32_t n_tokens, volatile int *cancel_flag);

#endif
