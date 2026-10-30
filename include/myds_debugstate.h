/* myds_debugstate.h — implementation of GET /debug/state.
 *
 * Deliberately a separate translation unit. The endpoint needs the knob
 * inventory, the register-only counters and the loaded library path, and tying
 * myds_server.c to all three would drag those dependencies into every test that
 * includes server.c for unrelated reasons (that already happened once: a
 * three-object dependency broke four lightweight test targets).
 *
 * myds_server.c therefore declares this weakly and calls it only if linked. A
 * test that does not link this file still compiles and still serves requests;
 * it just reports that debug state is unavailable.
 */
#ifndef MYDS_DEBUGSTATE_H
#define MYDS_DEBUGSTATE_H

#include <stddef.h>
#include <stdbool.h>

struct myds_context;

/* Is the endpoint allowed at all? Requires MYDS_DEBUG_ENDPOINT=1. */
bool myds_debug_enabled(void);

/* The endpoint's only access control: the server has no authentication. */
bool myds_debug_is_loopback(const char *ip);

/* SHA256 of a file as lowercase hex, or "" when unreadable. Callers emit null
 * rather than a zero digest so "unknown" cannot be mistaken for "identity". */
void myds_debug_sha256_file(const char *path, char out[65]);

/* One JSON object. Fields without a source are null, never 0. */
void myds_debug_state_json(char *out, size_t cap, const struct myds_context *ctx);

#endif /* MYDS_DEBUGSTATE_H */
