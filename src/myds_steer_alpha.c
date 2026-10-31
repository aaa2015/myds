/* GLP steering alpha resolution: MYDS_STEER_ALPHA parsing.
 *
 * Extracted from main.c so it can be unit-tested without loading a 150 GB model.
 * See include/myds_steer_alpha.h for the two accepted forms and why per-profile
 * is needed at all.
 */
#include "myds_steer_alpha.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Profile name exactly as myds_steer.c derives st->name. Keep these in step. */
static void profile_name_of(const char *path, char *out, size_t cap)
{
    if (strstr(path, "GLP-27") || strstr(path, "glp-27") || strstr(path, "glp27")) {
        snprintf(out, cap, "glp27");
    } else if (strstr(path, "GLP-39") || strstr(path, "glp-39") || strstr(path, "glp39")) {
        snprintf(out, cap, "glp39");
    } else {
        const char *slash = strrchr(path, '/');
        snprintf(out, cap, "%s", slash ? slash + 1 : path);
    }
}

float myds_steer_alpha_for(const char *env, const char *profile_path)
{
    if (!env || !*env || !profile_path) return -1.0f;

    /* Form 1: a bare float applies to every profile. */
    char *end = NULL;
    float v = strtof(env, &end);
    if (end && *end == '\0') return v;

    /* Form 2: name:alpha[,name:alpha...] */
    char pname[64];
    profile_name_of(profile_path, pname, sizeof(pname));
    const size_t nlen = strlen(pname);

    for (const char *q = env; *q; ) {
        while (*q == ',' || isspace((unsigned char)*q)) q++;
        if (!*q) break;
        const char *colon = strchr(q, ':');
        if (!colon) break;               /* malformed entry: stop, keep defaults */
        if ((size_t)(colon - q) == nlen && strncmp(q, pname, nlen) == 0) {
            return strtof(colon + 1, NULL);
        }
        const char *comma = strchr(colon, ',');
        if (!comma) break;
        q = comma + 1;
    }
    return -1.0f;                        /* unlisted profile: its own file default */
}
