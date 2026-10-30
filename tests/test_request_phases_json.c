/* Exercise the request_phases formatter without a model.
 *
 * Why this exists: a printf whose argument types drift from its format string
 * compiles cleanly and produces unparsable output. Emitting a broken line is
 * worse than emitting none, because a parser will silently drop it and the run
 * looks as if the stage was never measured. So the formatter is a pure function
 * and this test prints its output for an external json.loads to judge.
 *
 *   ./bin/test_request_phases_json            -> prints one JSON object per case
 */
#define _DARWIN_C_SOURCE
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/myds_server.c"

int main(void) {
    /* Fully measured request. */
    myds_request_phases_t full = {
        .queue_s = 0.0, .render_s = 0.001234, .setup_s = 0.000456,
        .engine_s = 12.5, .prefill_s = 11.25, .decode_s = 1.25,
        .unattributed_s = 0.0, .prompt_tokens = 32242, .cached_tokens = 0,
        .out_tokens = 160, .stream = 0,
    };
    char out[512];
    myds_request_phases_json(&full, out, sizeof(out));
    printf("%s\n", out);

    /* Engine never ran: everything after queue must be null, not 0. */
    myds_request_phases_t none = {
        .queue_s = 0.5, .render_s = -1.0, .setup_s = -1.0, .engine_s = -1.0,
        .prefill_s = -1.0, .decode_s = -1.0, .unattributed_s = NAN,
        .prompt_tokens = 0, .cached_tokens = 0, .out_tokens = 0, .stream = 1,
    };
    myds_request_phases_json(&none, out, sizeof(out));
    printf("%s\n", out);

    /* Render measured, engine started but no token ever arrived. */
    myds_request_phases_t partial = {
        .queue_s = 0.0, .render_s = 0.002, .setup_s = 0.001, .engine_s = -1.0,
        .prefill_s = 3.0, .decode_s = -1.0, .unattributed_s = NAN,
        .prompt_tokens = 8099, .cached_tokens = 8100, .out_tokens = 1, .stream = 0,
    };
    myds_request_phases_json(&partial, out, sizeof(out));
    printf("%s\n", out);

    /* Negative unattributed is meaningful (t_last precedes the engine's return);
     * it must survive as a number so a wrong split is visible rather than hidden. */
    myds_request_phases_t negative = {
        .queue_s = 0.0, .render_s = 0.001, .setup_s = 0.001, .engine_s = 0.5,
        .prefill_s = 0.52, .decode_s = 0.0, .unattributed_s = -0.02,
        .prompt_tokens = 10, .cached_tokens = 0, .out_tokens = 2, .stream = 0,
    };
    myds_request_phases_json(&negative, out, sizeof(out));
    printf("%s\n", out);
    return 0;
}
