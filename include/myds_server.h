#ifndef MYDS_SERVER_H
#define MYDS_SERVER_H

#include "myds.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Start an OpenAI-compatible HTTP server.
 * Returns 0 on clean exit, non-zero on error.
 */
int myds_server_start(myds_context_t *ctx, const char *host, int port, const char *model_alias);

#ifdef __cplusplus
}
#endif

#endif /* MYDS_SERVER_H */
