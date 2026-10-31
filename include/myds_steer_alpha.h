#ifndef MYDS_STEER_ALPHA_H
#define MYDS_STEER_ALPHA_H

/*
 * Resolve the GLP steering alpha for one profile.
 *
 * `env` is the raw MYDS_STEER_ALPHA value, which accepts two forms:
 *
 *   "3.5"                     one value for every profile (the original behaviour)
 *   "glp27:1.0,glp39:0.75"    per profile; a profile not listed keeps its file default
 *
 * Returns the resolved alpha, or -1.0f meaning "use the profile's own default".
 *
 * Per-profile exists because one alpha cannot be correct for both vectors: on this
 * stack glp39 was measured at 0.75 and glp27 at 1.0 (§135–§137). Sharing a single
 * value forces one of them off the operating point it was verified at.
 *
 * Name matching mirrors myds_steer.c, which is what assigns st->name. If those two
 * ever disagree, the map silently stops matching and every profile falls back to its
 * file default -- a failure that is invisible except in the startup alpha log line.
 */
float myds_steer_alpha_for(const char *env, const char *profile_path);

#endif /* MYDS_STEER_ALPHA_H */
