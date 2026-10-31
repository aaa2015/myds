/* Unit tests for MYDS_STEER_ALPHA resolution.
 *
 * These exist because the per-profile form is the one line of this feature that can
 * fail silently: a name mismatch makes every profile fall back to its file default,
 * which looks exactly like "the env var had no effect".
 */
#include "myds_steer_alpha.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(const char *what, float got, float want)
{
    const int ok = (fabsf(got - want) < 1e-6f);
    if (!ok) failures++;
    printf("  %-58s got=%-8.3f want=%-8.3f %s\n",
           what, got, want, ok ? "PASS" : "FAIL");
}

int main(void)
{
    const char *p27 = "/Users/x/.lmstudio/models/glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf";
    const char *p39 = "/Users/x/.lmstudio/models/glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf";

    printf("scalar form (back-compatible: one value for every profile)\n");
    check("scalar 3.5 / glp27", myds_steer_alpha_for("3.5", p27), 3.5f);
    check("scalar 3.5 / glp39", myds_steer_alpha_for("3.5", p39), 3.5f);
    check("scalar 0.75 / glp39", myds_steer_alpha_for("0.75", p39), 0.75f);
    check("scalar 1 / glp27", myds_steer_alpha_for("1", p27), 1.0f);

    printf("per-profile map\n");
    check("map glp27:1.0 / glp27", myds_steer_alpha_for("glp27:1.0,glp39:0.75", p27), 1.0f);
    check("map glp39:0.75 / glp39", myds_steer_alpha_for("glp27:1.0,glp39:0.75", p39), 0.75f);
    check("map order swapped / glp39", myds_steer_alpha_for("glp39:0.75,glp27:1.0", p39), 0.75f);
    check("map order swapped / glp27", myds_steer_alpha_for("glp39:0.75,glp27:1.0", p27), 1.0f);

    printf("unlisted profile falls back to file default (-1)\n");
    check("map glp39 only / glp27", myds_steer_alpha_for("glp39:0.75", p27), -1.0f);
    check("map glp27 only / glp39", myds_steer_alpha_for("glp27:1.0", p39), -1.0f);
    check("empty env / glp27", myds_steer_alpha_for("", p27), -1.0f);
    check("NULL env / glp27", myds_steer_alpha_for(NULL, p27), -1.0f);

    printf("whitespace tolerance\n");
    check("spaces around entries / glp27", myds_steer_alpha_for(" glp27:2.0 , glp39:0.5 ", p27), 2.0f);
    check("spaces around entries / glp39", myds_steer_alpha_for(" glp27:2.0 , glp39:0.5 ", p39), 0.5f);
    check("space after colon / glp27", myds_steer_alpha_for("glp27: 2.5", p27), 2.5f);

    printf("name must match exactly, not as a prefix\n");
    check("glp2 vs glp27 must miss", myds_steer_alpha_for("glp2:9.0", p27), -1.0f);
    check("glp27extra must miss", myds_steer_alpha_for("glp27extra:9.0", p27), -1.0f);
    check("glp271 must miss", myds_steer_alpha_for("glp271:9.0", p27), -1.0f);

    printf("malformed input must not crash or guess\n");
    check("garbage", myds_steer_alpha_for("nonsense", p27), -1.0f);
    check("trailing colon", myds_steer_alpha_for("glp27:", p27), 0.0f);
    check("map with empty tail entry", myds_steer_alpha_for("glp39:0.5,", p27), -1.0f);

    printf("unknown profile matched by basename\n");
    {
        const char *custom = "/models/my-custom-vector.gguf";
        check("basename match", myds_steer_alpha_for("my-custom-vector.gguf:0.25", custom), 0.25f);
        check("basename miss", myds_steer_alpha_for("glp27:1.0", custom), -1.0f);
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
