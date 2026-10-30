/* Verify the /debug/state payload without a model or a GPU.
 *
 * Two things here can be wrong in a way that still compiles and still looks
 * plausible, which is exactly why they get a test:
 *
 *   1. the hand-rolled SHA-256 in myds_server.c -- if it is wrong, every
 *      identity the endpoint reports is fiction, and the whole point of the
 *      endpoint is identity;
 *   2. the JSON shape -- a printf whose argument types drift emits a body a
 *      parser silently drops, and then a reader concludes "the counter is zero"
 *      rather than "the line was never valid".
 *
 * The expected digest is passed in from the caller (computed by shasum), so this
 * test does not re-implement the thing it checks.
 */
#define _DARWIN_C_SOURCE
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Include the implementation under test directly: server.c only forwards
 * to it through a provider pointer, so testing server.c would test the
 * plumbing rather than the payload. */
#include "../src/myds_debugstate.c"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <file> <expected-sha256-hex>\n", argv[0]);
        return 2;
    }
    int failures = 0;
    char digest[65];
    myds_debug_sha256_file(argv[1], digest);
    const int sha_ok = strcmp(digest, argv[2]) == 0;
    printf("{\"kind\":\"sha256\",\"file\":\"%s\",\"computed\":\"%s\",\"expected\":\"%s\",\"pass\":%s}\n",
           argv[1], digest, argv[2], sha_ok ? "true" : "false");
    if (!sha_ok) ++failures;

    /* Shape with every source available. */
    ;
    char state[2048];
    myds_debug_state_json(state, sizeof(state), NULL);
    printf("{\"kind\":\"state\",\"payload\":%s}\n", state);

    /* Unreadable file must yield empty, not a zero-filled digest that looks real. */
    char missing[65];
    myds_debug_sha256_file("/definitely/not/here", missing);
    printf("{\"kind\":\"missing\",\"computed\":\"%s\",\"pass\":%s}\n",
           missing, missing[0] == 0 ? "true" : "false");
    if (missing[0] != 0) ++failures;  /* a zero-filled digest would look real */

    /* Loopback classification: the endpoint's only access control. */
    printf("{\"kind\":\"loopback\",\"127\":%s,\"::1\":%s,\"v4mapped\":%s,"
           "\"lan\":%s,\"null\":%s}\n",
           myds_debug_is_loopback("127.0.0.1") ? "true" : "false",
           myds_debug_is_loopback("::1") ? "true" : "false",
           myds_debug_is_loopback("::ffff:127.0.0.1") ? "true" : "false",
           myds_debug_is_loopback("192.168.1.50") ? "true" : "false",
           myds_debug_is_loopback(NULL) ? "true" : "false");
    /* The endpoint's only access control. A false positive here publishes state. */
    const int loop_ok = myds_debug_is_loopback("127.0.0.1") &&
                        myds_debug_is_loopback("::1") &&
                        myds_debug_is_loopback("::ffff:127.0.0.1") &&
                        !myds_debug_is_loopback("192.168.1.50") &&
                        !myds_debug_is_loopback("10.0.0.7") &&
                        !myds_debug_is_loopback(NULL);
    if (!loop_ok) ++failures;
    /* The payload must be non-empty and carry every top-level section. */
    const int shape_ok = state[0] == '{' && strstr(state, "\"build\"") &&
                         strstr(state, "\"knobs\"") && strstr(state, "\"counters\"");
    if (!shape_ok) ++failures;
    printf("{\"kind\":\"summary\",\"failures\":%d,\"pass\":%s}\n",
           failures, failures == 0 ? "true" : "false");
    return failures == 0 ? 0 : 1;
}
