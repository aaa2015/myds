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

    /* Loopback classification.
     *
     * The listener is AF_INET and the peer address comes from inet_ntop(AF_INET)
     * into char[16], so an IPv6 string can never reach this function. The earlier
     * version asserted "::1" was accepted -- testing a path production cannot take,
     * which reads as IPv6 coverage that does not exist. Non-IPv4 input must now be
     * rejected, and that is what gets asserted. */
    printf("{\"kind\":\"loopback\",\"127.0.0.1\":%s,\"127.0.0.2\":%s,\"ipv6_rejected\":%s,"
           "\"lan\":%s,\"null\":%s}\n",
           myds_debug_is_loopback("127.0.0.1") ? "true" : "false",
           myds_debug_is_loopback("127.0.0.2") ? "true" : "false",
           !myds_debug_is_loopback("::1") ? "true" : "false",
           myds_debug_is_loopback("192.168.1.50") ? "true" : "false",
           myds_debug_is_loopback(NULL) ? "true" : "false");
    const int loop_ok = myds_debug_is_loopback("127.0.0.1") &&
                        myds_debug_is_loopback("127.0.0.2") &&
                        !myds_debug_is_loopback("::1") &&            /* unreachable, must not claim it */
                        !myds_debug_is_loopback("::ffff:127.0.0.1") &&
                        !myds_debug_is_loopback("192.168.1.50") &&
                        !myds_debug_is_loopback("10.0.0.7") &&
                        !myds_debug_is_loopback("127.0.0.1.evil") && /* no prefix match */
                        !myds_debug_is_loopback("1271.0.0.1") &&
                        !myds_debug_is_loopback(NULL);
    if (!loop_ok) ++failures;

    /* Token gate. Fail-closed when unset; exact match when set; a prefix must not
     * pass, since a proxy operator could otherwise brute-force it one byte at a time. */
    unsetenv("MYDS_DEBUG_TOKEN");
    const int token_unset_ok = !myds_debug_token_configured() &&
                               !myds_debug_token_ok("") &&
                               !myds_debug_token_ok(NULL) &&
                               !myds_debug_token_ok("anything");
    setenv("MYDS_DEBUG_TOKEN", "s3cret", 1);
    const int token_set_ok = myds_debug_token_configured() &&
                             myds_debug_token_ok("s3cret") &&
                             !myds_debug_token_ok("s3cre") &&
                             !myds_debug_token_ok("s3cretX") &&
                             !myds_debug_token_ok("") &&
                             !myds_debug_token_ok(NULL);
    unsetenv("MYDS_DEBUG_TOKEN");
    printf("{\"kind\":\"token\",\"fail_closed_when_unset\":%s,\"exact_match\":%s}\n",
           token_unset_ok ? "true" : "false", token_set_ok ? "true" : "false");
    if (!token_unset_ok || !token_set_ok) ++failures;

    /* Truncation must produce a valid object that says so, never a half-written
     * one. The old EMIT ran `used` past cap and silently dropped everything after. */
    char tiny[192];
    myds_debug_state_json(tiny, sizeof tiny, NULL);
    const int trunc_ok = tiny[0] == '{' && strstr(tiny, "truncated") != NULL;
    printf("{\"kind\":\"truncation\",\"reports_instead_of_cutting\":%s,\"body\":\"%s\"}\n",
           trunc_ok ? "true" : "false", tiny);
    if (!trunc_ok) ++failures;
    /* The payload must be non-empty and carry every top-level section. */
    const int shape_ok = state[0] == '{' && strstr(state, "\"build\"") &&
                         strstr(state, "\"knobs\"") && strstr(state, "\"counters\"");
    if (!shape_ok) ++failures;
    printf("{\"kind\":\"summary\",\"failures\":%d,\"pass\":%s}\n",
           failures, failures == 0 ? "true" : "false");
    return failures == 0 ? 0 : 1;
}
