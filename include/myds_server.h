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

/* Register the /debug/state implementation. Full binaries call this with the
 * myds_debugstate.c functions; a build that does not link that file leaves it
 * unregistered, and the endpoint answers 404 rather than failing to link. */
#include <stddef.h>
#include <stdbool.h>
struct myds_context;
typedef void (*myds_debug_state_provider_t)(char *out, size_t cap,
                                            const struct myds_context *ctx);
typedef bool (*myds_debug_predicate_t)(void);
typedef bool (*myds_debug_loopback_t)(const char *ip);
typedef bool (*myds_debug_token_t)(const char *presented);
void myds_server_set_debug_provider(myds_debug_state_provider_t state,
                                    myds_debug_predicate_t enabled,
                                    myds_debug_loopback_t is_loopback,
                                    myds_debug_token_t token_ok);
