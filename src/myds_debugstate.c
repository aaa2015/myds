/* myds_debugstate.c — GET /debug/state, kept out of myds_server.c on purpose.
 * See include/myds_debugstate.h for why it is a separate translation unit. */
#include "myds_debugstate.h"
#include "myds.h"
#include "myds_knobs.h"
#include "myds_v41_gpu.h"
#include "myds_metal.h"
#include "myds_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

/* ---- /debug/state ----------------------------------------------------------
 *
 * Why this exists: until now the only way to read the register-only counters was
 * to attach lldb to the live process (§105.9-1). That is not reproducible by
 * anyone who is not sitting at this machine, and it cannot be used during a
 * window without perturbing it. The counters already have an accessor; what was
 * missing was a way to reach it.
 *
 * Three properties are not negotiable:
 *
 *   1. OFF unless MYDS_DEBUG_ENDPOINT=1, and loopback-only even then. The server
 *      has no authentication at all, so an always-on state dump would publish
 *      configuration and traffic shape to anyone who can reach the port.
 *   2. The counter accessor is called with reset=false. With reset=true it does
 *      an atomic_exchange to zero, which would silently destroy the arm separation
 *      that every A/B in the ledger depends on (§105: "requests constant 2897,
 *      encoded 0 vs 2025"). A debug read must never subtract evidence.
 *   3. Read-only. No command buffer, no commit, no residency change.
 */
bool myds_debug_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("MYDS_DEBUG_ENDPOINT");
        cached = (value && *value && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

bool myds_debug_is_loopback(const char *ip) {
    if (!ip) return false;
    return strcmp(ip, "127.0.0.1") == 0 || strcmp(ip, "::1") == 0 ||
           strncmp(ip, "::ffff:127.", 11) == 0;
}

/* SHA256 of a file, computed once. Returns "" when unreadable, so the caller can
 * emit null and a consumer can tell "identity unknown" from "identity differs". */
void myds_debug_sha256_file(const char *path, char out[65]) {
    out[0] = 0;
    if (!path || !*path) return;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    /* Minimal streaming SHA-256: a debug endpoint may not add a dependency. */
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint64_t total = 0;
    unsigned char block[64];
    size_t have = 0;
    #define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
    #define COMPRESS(buf) do {                                                     \
        uint32_t w[64];                                                            \
        for (int i = 0; i < 16; ++i)                                               \
            w[i] = ((uint32_t)(buf)[4*i] << 24) | ((uint32_t)(buf)[4*i+1] << 16) | \
                   ((uint32_t)(buf)[4*i+2] << 8) | (uint32_t)(buf)[4*i+3];          \
        for (int i = 16; i < 64; ++i) {                                            \
            uint32_t s0 = ROTR(w[i-15],7) ^ ROTR(w[i-15],18) ^ (w[i-15] >> 3);      \
            uint32_t s1 = ROTR(w[i-2],17) ^ ROTR(w[i-2],19) ^ (w[i-2] >> 10);       \
            w[i] = w[i-16] + s0 + w[i-7] + s1;                                     \
        }                                                                          \
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];         \
        for (int i = 0; i < 64; ++i) {                                             \
            uint32_t S1 = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);                     \
            uint32_t ch = (e & f) ^ ((~e) & g);                                    \
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];                              \
            uint32_t S0 = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);                     \
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);                             \
            uint32_t t2 = S0 + mj;                                                 \
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;                        \
        }                                                                          \
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;   \
    } while (0)
    for (;;) {
        const size_t got = fread(block + have, 1, 64 - have, f);
        total += got;
        have += got;
        if (have < 64) break;
        COMPRESS(block);
        have = 0;
    }
    fclose(f);
    const uint64_t bits = total * 8u;
    block[have++] = 0x80;
    if (have > 56) {
        while (have < 64) block[have++] = 0;
        COMPRESS(block);
        have = 0;
    }
    while (have < 56) block[have++] = 0;
    for (int i = 7; i >= 0; --i) block[have++] = (unsigned char)((bits >> (i * 8)) & 0xffu);
    COMPRESS(block);
    #undef COMPRESS
    #undef ROTR
    for (int i = 0; i < 8; ++i) snprintf(out + i * 8, 9, "%08x", h[i]);
    out[64] = 0;
}

/* Emit one JSON object describing the live process. Fields whose source is
 * unavailable are null, never a plausible zero. */
void myds_debug_state_json(char *out, size_t cap, const struct myds_context *ctx) {
    char binary[65] = {0}, metallib[65] = {0};
    char binary_path[4096] = {0};
#if defined(__APPLE__)
    uint32_t size = sizeof(binary_path);
    if (_NSGetExecutablePath(binary_path, &size) != 0) binary_path[0] = 0;
#endif
    myds_debug_sha256_file(binary_path[0] ? binary_path : NULL, binary);
    myds_debug_sha256_file(myds_metal_loaded_library_path(), metallib);

    /* reset=false: see point 2 above. */
    myds_v41_gpu_indexer_register_only_counters_t ro;
    memset(&ro, 0, sizeof(ro));
    myds_v41_gpu_indexer_register_only_counters(&ro, false);

    /* Knob accounting, reusing the same inventory --print-config reports. */
    static myds_knob_seen_t seen[1024];
    const int n = myds_knobs_from_environ(seen, sizeof(seen) / sizeof(seen[0]));
    int unread = 0;
    for (int i = 0; i < n; ++i) if (n > 0 && !seen[i].knob) ++unread;

    size_t used = 0;
    #define EMIT(...) do {                                                       \
        if (used < cap) {                                                        \
            const int w = snprintf(out + used, cap - used, __VA_ARGS__);         \
            if (w > 0) used += (size_t)w;                                        \
        }                                                                        \
    } while (0)

    EMIT("{\"build\":{\"binary_sha256\":%s%s%s,\"metallib_sha256\":%s%s%s,",
         binary[0] ? "\"" : "null", binary[0] ? binary : "", binary[0] ? "\"" : "",
         metallib[0] ? "\"" : "null", metallib[0] ? metallib : "", metallib[0] ? "\"" : "");
    EMIT("\"binary_path\":\"%s\"},", binary_path[0] ? binary_path : "");

    EMIT("\"knobs\":{\"known\":%zu,\"set\":%d,\"unread\":%d},",
         myds_knob_count(), n, unread);

    EMIT("\"counters\":{\"indexer_register_only\":{\"requests\":%llu,\"encoded\":%llu,"
         "\"fallbacks\":%llu,\"failures\":%llu,\"f32_dispatches\":%llu,"
         "\"bf16_dispatches\":%llu},\"prefix_cache_len\":%u},",
         (unsigned long long)ro.requests, (unsigned long long)ro.encoded,
         (unsigned long long)ro.fallbacks, (unsigned long long)ro.failures,
         (unsigned long long)ro.f32_dispatches, (unsigned long long)ro.bf16_dispatches,
         ctx ? myds_context_get_pcache_len(ctx) : 0u);

    EMIT("\"state_capture\":\"point-in-time; prefix-cache miss reasons are log-only, "
         "not counted here\"}\n");
    #undef EMIT
}

