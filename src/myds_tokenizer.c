#include "myds_tokenizer.h"
#include "myds_tokenizer_unicode.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int32_t left, right, result, rank; } merge_t;
typedef struct {
    int32_t *vocab_hash;
    size_t vocab_cap, max_token_len;
    merge_t *merges;
    size_t merge_cap;
    int32_t byte_ids[256];
    int32_t *special_ids;
    size_t n_special;
    bool byte_bpe, joyai;
    bool special_prefix[256];
} tokenizer_impl_t;

static uint32_t hash_bytes(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) h = (h ^ (uint8_t)s[i]) * 16777619u;
    return h;
}
static size_t table_capacity(size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap *= 2;
    return cap;
}
static int32_t vocab_find(const myds_tokenizer_t *tok, const char *s, size_t n) {
    const tokenizer_impl_t *p = tok->impl;
    size_t slot = hash_bytes(s, n) & (p->vocab_cap - 1);
    while (p->vocab_hash[slot] >= 0) {
        int32_t id = p->vocab_hash[slot];
        if (tok->vocab[id].len == n && !memcmp(tok->vocab[id].text, s, n)) return id;
        slot = (slot + 1) & (p->vocab_cap - 1);
    }
    return -1;
}
static uint64_t hash_pair(int32_t a, int32_t b) {
    uint64_t h = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    h ^= h >> 30; h *= UINT64_C(0xbf58476d1ce4e5b9);
    h ^= h >> 27; h *= UINT64_C(0x94d049bb133111eb);
    return h ^ (h >> 31);
}
static const merge_t *merge_find(const tokenizer_impl_t *p, int32_t a, int32_t b) {
    if (!p->merge_cap) return NULL;
    size_t slot = hash_pair(a, b) & (p->merge_cap - 1);
    while (p->merges[slot].rank >= 0) {
        if (p->merges[slot].left == a && p->merges[slot].right == b) return &p->merges[slot];
        slot = (slot + 1) & (p->merge_cap - 1);
    }
    return NULL;
}
static uint32_t byte_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    if (b <= 32) return 256 + b;
    if (b <= 160) return 256 + 33 + b - 127;
    return 323; /* byte 173 */
}
static int codepoint_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return (int)cp;
    if (cp >= 256 && cp <= 288) return (int)(cp - 256);
    if (cp >= 289 && cp <= 322) return (int)(cp - 289 + 127);
    return cp == 323 ? 173 : -1;
}
static size_t utf8_read(const char *s, size_t n, uint32_t *cp) {
    uint8_t c = (uint8_t)s[0];
    size_t k = c < 128 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 : c <= 0xef && c >= 0xe0 ? 3 : c <= 0xf4 && c >= 0xf0 ? 4 : 0;
    if (!k || k > n) { *cp = UINT32_MAX; return 1; }
    uint32_t v = c & (k == 1 ? 127 : k == 2 ? 31 : k == 3 ? 15 : 7);
    for (size_t i = 1; i < k; ++i) {
        if (((uint8_t)s[i] & 0xc0) != 0x80) { *cp = UINT32_MAX; return 1; }
        v = (v << 6) | ((uint8_t)s[i] & 63);
    }
    if ((k == 2 && v < 128) || (k == 3 && v < 2048) || (k == 4 && v < 65536) ||
        v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff)) { *cp = UINT32_MAX; return 1; }
    *cp = v; return k;
}
static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}
static bool decode_entry(myds_token_entry_t *v, bool byte_bpe) {
    v->decoded = malloc((size_t)v->len + 1);
    if (!v->decoded) return false;
    if (!v->special && v->len == 6 && !memcmp(v->text, "<0x", 3) && v->text[5] == '>' &&
        hex_digit(v->text[3]) >= 0 && hex_digit(v->text[4]) >= 0) {
        v->decoded[0] = (char)((hex_digit(v->text[3]) << 4) | hex_digit(v->text[4]));
        v->decoded_len = 1;
    } else if (byte_bpe && !v->special) {
        for (size_t off = 0; off < v->len;) {
            uint32_t cp; size_t k = utf8_read(v->text + off, v->len - off, &cp);
            int b = codepoint_byte(cp);
            if (b < 0) return false;
            v->decoded[v->decoded_len++] = (char)b;
            off += k;
        }
    } else {
        memcpy(v->decoded, v->text, v->len);
        v->decoded_len = v->len;
    }
    v->decoded[v->decoded_len] = 0;
    return true;
}

myds_tokenizer_t *myds_tokenizer_load(const gguf_file_t *gf) {
    if (!gf) return NULL;
    const gguf_kv_t *tokens = gguf_find_key(gf, "tokenizer.ggml.tokens");
    bool fallback = !tokens;
    if (!fallback && (tokens->type != GGUF_TYPE_ARRAY || tokens->value.val_arr.elem_type != GGUF_TYPE_STRING ||
        !tokens->value.val_arr.count || tokens->value.val_arr.count > INT32_MAX)) return NULL;
    myds_tokenizer_t *tok = calloc(1, sizeof(*tok));
    if (!tok) return NULL;
    tokenizer_impl_t *p = calloc(1, sizeof(*p));
    if (!p) { free(tok); return NULL; }
    tok->impl = p;
    tok->vocab_size = fallback ? 256 : (uint32_t)tokens->value.val_arr.count;
    tok->vocab = calloc(tok->vocab_size, sizeof(*tok->vocab));
    p->vocab_cap = table_capacity(tok->vocab_size);
    p->vocab_hash = malloc(p->vocab_cap * sizeof(*p->vocab_hash));
    p->special_ids = malloc(tok->vocab_size * sizeof(*p->special_ids));
    if (!tok->vocab || !p->vocab_hash || !p->special_ids) goto fail;
    memset(p->vocab_hash, 0xff, p->vocab_cap * sizeof(*p->vocab_hash));
    tok->bos_id = (int32_t)gguf_get_u32(gf, "tokenizer.ggml.bos_token_id", 1);
    tok->eos_id = (int32_t)gguf_get_u32(gf, "tokenizer.ggml.eos_token_id", fallback ? 0 : 2);
    tok->pad_id = (int32_t)gguf_get_u32(gf, "tokenizer.ggml.padding_token_id", 0);
    p->byte_bpe = !fallback && !strcmp(gguf_get_str(gf, "tokenizer.ggml.model", ""), "gpt2");
    const char *pre = gguf_get_str(gf, "tokenizer.ggml.pre", "gpt-2");
    p->joyai = !strcmp(pre, "joyai-llm") || !strcmp(pre, "deepseek-v3");
    if (p->byte_bpe && !p->joyai && strcmp(pre, "gpt-2") && strcmp(pre, "gpt2")) {
        fprintf(stderr, "myds_tokenizer: unsupported byte-BPE pre-tokenizer '%s'\n", pre);
        goto fail;
    }
    const gguf_kv_t *scores = gguf_find_key(gf, "tokenizer.ggml.scores");
    const gguf_kv_t *types = gguf_find_key(gf, "tokenizer.ggml.token_type");
    if (scores && (scores->type != GGUF_TYPE_ARRAY || scores->value.val_arr.elem_type != GGUF_TYPE_FLOAT32 ||
        scores->value.val_arr.count != tok->vocab_size)) goto fail;
    if (types && (types->type != GGUF_TYPE_ARRAY || types->value.val_arr.elem_type != GGUF_TYPE_INT32 ||
        types->value.val_arr.count != tok->vocab_size)) goto fail;
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        myds_token_entry_t *v = &tok->vocab[i];
        const char *s = fallback ? NULL : ((char **)tokens->value.val_arr.data)[i];
        size_t n = fallback ? 1 : s ? strlen(s) : 0;
        if (n > UINT32_MAX) goto fail;
        v->text = malloc(n + 1);
        if (!v->text) goto fail;
        if (fallback) v->text[0] = (char)i; else if (n) memcpy(v->text, s, n);
        v->text[n] = 0; v->len = (uint32_t)n;
        v->score = scores ? ((float *)scores->value.val_arr.data)[i] : -(float)i;
        int type = types ? ((int32_t *)types->value.val_arr.data)[i] : 1;
        v->special = type == 3 || type == 4 || type == 5;
        if (!fallback && !types && ((int32_t)i == tok->bos_id || (int32_t)i == tok->eos_id || (int32_t)i == tok->pad_id))
            v->special = true;
        if (n > p->max_token_len) p->max_token_len = n;
        size_t slot = hash_bytes(v->text, n) & (p->vocab_cap - 1);
        while (p->vocab_hash[slot] >= 0) slot = (slot + 1) & (p->vocab_cap - 1);
        p->vocab_hash[slot] = (int32_t)i;
    }
    for (int b = 0; b < 256; ++b) {
        char s[8]; size_t n;
        if (p->byte_bpe) {
            uint32_t cp = byte_codepoint((uint8_t)b);
            if (cp < 128) { s[0] = (char)cp; n = 1; }
            else { s[0] = (char)(0xc0 | (cp >> 6)); s[1] = (char)(0x80 | (cp & 63)); n = 2; }
        } else { s[0] = (char)b; n = 1; }
        p->byte_ids[b] = vocab_find(tok, s, n);
        if (p->byte_ids[b] < 0) {
            snprintf(s, sizeof(s), "<0x%02X>", b);
            p->byte_ids[b] = vocab_find(tok, s, 6);
        }
    }
    if (p->byte_bpe) {
        const gguf_kv_t *merges = gguf_find_key(gf, "tokenizer.ggml.merges");
        if (!merges || merges->type != GGUF_TYPE_ARRAY || merges->value.val_arr.elem_type != GGUF_TYPE_STRING ||
            merges->value.val_arr.count > INT32_MAX) goto fail;
        p->merge_cap = table_capacity((size_t)merges->value.val_arr.count);
        p->merges = malloc(p->merge_cap * sizeof(*p->merges));
        if (!p->merges) goto fail;
        memset(p->merges, 0xff, p->merge_cap * sizeof(*p->merges));
        for (size_t i = 0; i < merges->value.val_arr.count; ++i) {
            const char *s = ((char **)merges->value.val_arr.data)[i];
            const char *sep = s ? strchr(s, ' ') : NULL;
            if (!sep || sep == s || !sep[1]) goto fail;
            int32_t a = vocab_find(tok, s, (size_t)(sep - s));
            int32_t b = vocab_find(tok, sep + 1, strlen(sep + 1));
            size_t n = strlen(s) - 1;
            char *joined = malloc(n + 1);
            if (!joined) goto fail;
            memcpy(joined, s, (size_t)(sep - s));
            memcpy(joined + (sep - s), sep + 1, strlen(sep + 1));
            int32_t result = vocab_find(tok, joined, n);
            free(joined);
            if (a < 0 || b < 0 || result < 0) goto fail;
            size_t slot = hash_pair(a, b) & (p->merge_cap - 1);
            while (p->merges[slot].rank >= 0 && !(p->merges[slot].left == a && p->merges[slot].right == b))
                slot = (slot + 1) & (p->merge_cap - 1);
            if (p->merges[slot].rank < 0) p->merges[slot] = (merge_t){ a, b, result, (int32_t)i };
        }
    }
    /* Some GGUF converters label added (non-control) tokens NORMAL instead
     * of USER_DEFINED. Such tokens are absent from the byte alphabet and from
     * every merge result, so BPE cannot produce them. Retain them literally and
     * recognize them as added tokens, including DeepSeek role/DSML markers. */
    uint8_t *reachable = p->byte_bpe ? calloc(tok->vocab_size, 1) : NULL;
    if (p->byte_bpe && !reachable) goto fail;
    if (reachable) {
        for (int b = 0; b < 256; ++b) if (p->byte_ids[b] >= 0) reachable[p->byte_ids[b]] = 1;
        for (size_t i = 0; i < p->merge_cap; ++i)
            if (p->merges[i].rank >= 0) reachable[p->merges[i].result] = 1;
    }
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        myds_token_entry_t *v = &tok->vocab[i];
        if (reachable && !reachable[i]) v->special = true;
        if (v->special && v->len) {
            p->special_ids[p->n_special++] = (int32_t)i;
            p->special_prefix[(uint8_t)v->text[0]] = true;
        }
        if (!decode_entry(v, p->byte_bpe)) { free(reachable); goto fail; }
    }
    free(reachable);
    return tok;
fail:
    fprintf(stderr, "myds_tokenizer: invalid, unsupported, or unallocatable tokenizer metadata\n");
    myds_tokenizer_free(tok);
    return NULL;
}

void myds_tokenizer_free(myds_tokenizer_t *tok) {
    if (!tok) return;
    tokenizer_impl_t *p = tok->impl;
    if (p) { free(p->vocab_hash); free(p->merges); free(p->special_ids); free(p); }
    if (tok->vocab) for (uint32_t i = 0; i < tok->vocab_size; ++i) { free(tok->vocab[i].text); free(tok->vocab[i].decoded); }
    free(tok->vocab); free(tok);
}

typedef struct { int32_t *ids, capacity, count; } output_t;
static int emit(output_t *out, int32_t id) {
    if (id < 0) return MYDS_TOKENIZER_ERROR_UNKNOWN_BYTE;
    if (out->count == INT32_MAX) return MYDS_TOKENIZER_ERROR_INVALID;
    if (out->ids && out->count < out->capacity) out->ids[out->count] = id;
    ++out->count; return 0;
}
typedef struct { int32_t id; size_t prev, next, version; } symbol_t;
typedef struct { size_t left, right, left_version, right_version; const merge_t *merge; } candidate_t;
typedef struct { candidate_t *items; size_t size, capacity; } heap_t;
static bool candidate_before(candidate_t a, candidate_t b) {
    return a.merge->rank < b.merge->rank || (a.merge->rank == b.merge->rank && a.left < b.left);
}
static int heap_offer(heap_t *h, const tokenizer_impl_t *p, symbol_t *s, size_t left, size_t n) {
    if (left >= n || s[left].next >= n) return 0;
    size_t right = s[left].next;
    const merge_t *m = merge_find(p, s[left].id, s[right].id);
    if (!m) return 0;
    if (h->size == h->capacity) {
        size_t cap = h->capacity ? h->capacity * 2 : 32;
        if (cap > SIZE_MAX / sizeof(*h->items)) return MYDS_TOKENIZER_ERROR_NOMEM;
        void *items = realloc(h->items, cap * sizeof(*h->items));
        if (!items) return MYDS_TOKENIZER_ERROR_NOMEM;
        h->items = items; h->capacity = cap;
    }
    candidate_t c = {left, right, s[left].version, s[right].version, m};
    size_t i = h->size++;
    while (i && candidate_before(c, h->items[(i - 1) / 2])) { h->items[i] = h->items[(i - 1) / 2]; i = (i - 1) / 2; }
    h->items[i] = c; return 0;
}
static candidate_t heap_pop(heap_t *h) {
    candidate_t result = h->items[0], last = h->items[--h->size];
    size_t i = 0;
    while (i * 2 + 1 < h->size) {
        size_t child = i * 2 + 1;
        if (child + 1 < h->size && candidate_before(h->items[child + 1], h->items[child])) ++child;
        if (!candidate_before(h->items[child], last)) break;
        h->items[i] = h->items[child]; i = child;
    }
    if (h->size) h->items[i] = last;
    return result;
}
static int bpe_piece(const myds_tokenizer_t *tok, const char *text, size_t n, output_t *out) {
    const tokenizer_impl_t *p = tok->impl;
    if (!n) return 0;
    if (n > SIZE_MAX / sizeof(symbol_t)) return MYDS_TOKENIZER_ERROR_NOMEM;
    symbol_t *s = malloc(n * sizeof(*s));
    heap_t h = {0}; int err = 0;
    if (!s) return MYDS_TOKENIZER_ERROR_NOMEM;
    for (size_t i = 0; i < n; ++i) {
        int32_t id = p->byte_ids[(uint8_t)text[i]];
        if (id < 0) { err = MYDS_TOKENIZER_ERROR_UNKNOWN_BYTE; goto done; }
        s[i] = (symbol_t){id, i ? i - 1 : n, i + 1, 0};
    }
    for (size_t i = 0; i + 1 < n; ++i) if ((err = heap_offer(&h, p, s, i, n))) goto done;
    while (h.size) {
        candidate_t c = heap_pop(&h);
        size_t a = c.left, b = c.right;
        if (s[a].version != c.left_version || s[b].version != c.right_version || s[a].next != b) continue;
        s[a].id = c.merge->result; s[a].next = s[b].next; ++s[a].version; ++s[b].version;
        if (s[a].next < n) s[s[a].next].prev = a;
        if ((err = heap_offer(&h, p, s, s[a].prev, n)) || (err = heap_offer(&h, p, s, a, n))) goto done;
    }
    for (size_t i = 0; i < n; i = s[i].next) if ((err = emit(out, s[i].id))) break;
done:
    free(s); free(h.items); return err;
}

/* Unicode classes are fixed to the generated Unicode version, not process locale. */
typedef struct { uint32_t cp; size_t next; unsigned cls; } character_t;
static character_t char_at(const char *s, size_t n, size_t pos) {
    character_t c = {UINT32_MAX, pos, 0};
    if (pos < n) { c.next += utf8_read(s + pos, n - pos, &c.cp); c.cls = myds_unicode_class(c.cp); }
    return c;
}
static bool cjk(uint32_t c) { return (c >= 0x4e00 && c <= 0x9fa5) || (c >= 0x3040 && c <= 0x30ff); }
static bool ascii_alpha(uint32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static bool ascii_punct(uint32_t c) { return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126); }
static bool newline(uint32_t c) { return c == '\r' || c == '\n'; }
static size_t consume_class(const char *s, size_t n, size_t pos, unsigned mask) {
    while (pos < n) { character_t c = char_at(s, n, pos); if (!(c.cls & mask)) break; pos = c.next; }
    return pos;
}
/* Third JoyAI split expression, or the standard GPT-2 expression. */
static size_t text_piece_end(const char *s, size_t n, size_t pos, bool joyai) {
    character_t c = char_at(s, n, pos), next = char_at(s, n, c.next);
    if (!joyai && c.cp == '\'') {
        static const char *suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
            size_t k = strlen(suffixes[i]);
            if (k <= n - pos && !memcmp(s + pos, suffixes[i], k)) return pos + k;
        }
    }
    if (joyai && ascii_punct(c.cp) && ascii_alpha(next.cp)) {
        pos = next.next;
        while (pos < n) { character_t a = char_at(s, n, pos); if (!ascii_alpha(a.cp)) break; pos = a.next; }
        return pos;
    }
    unsigned letters = U_LETTER | (joyai ? U_MARK : 0);
    if (c.cls & letters) return consume_class(s, n, c.next, letters);
    if ((joyai ? (!newline(c.cp) && !(c.cls & (U_LETTER | U_PUNCT))) : c.cp == ' ') && (next.cls & letters))
        return consume_class(s, n, next.next, letters);
    if (!joyai && ((c.cls & U_NUMBER) || (c.cp == ' ' && (next.cls & U_NUMBER))))
        return consume_class(s, n, c.cp == ' ' ? next.next : c.next, U_NUMBER);
    character_t punct = c.cp == ' ' ? next : c;
    bool is_punct = joyai ? !!(punct.cls & U_PUNCT) : punct.next > (c.cp == ' ' ? c.next : pos) && !(punct.cls & (U_SPACE | U_LETTER | U_NUMBER));
    if (is_punct) {
        size_t end = punct.next;
        while (end < n) {
            character_t a = char_at(s, n, end);
            if (joyai ? !(a.cls & U_PUNCT) : !!(a.cls & (U_SPACE | U_LETTER | U_NUMBER))) break;
            end = a.next;
        }
        if (joyai) while (end < n) { character_t a = char_at(s, n, end); if (!newline(a.cp)) break; end = a.next; }
        return end;
    }
    if (c.cls & U_SPACE) {
        size_t end = pos, previous = pos, last_nl = 0;
        while (end < n) { character_t a = char_at(s, n, end); if (!(a.cls & U_SPACE)) break; previous = end; end = a.next; if (newline(a.cp)) last_nl = end; }
        if (joyai && last_nl) return last_nl;
        return end < n && previous > pos ? previous : end;
    }
    /* Split(Isolated) preserves unmatched text too. */
    return c.next;
}
static int encode_bpe(const myds_tokenizer_t *tok, const char *s, size_t n, output_t *out) {
    const tokenizer_impl_t *p = tok->impl;
    size_t pos = 0;
    while (pos < n) {
        size_t end = pos;
        character_t c = char_at(s, n, pos);
        if (p->joyai && (c.cls & U_NUMBER)) {
            for (int i = 0; i < 3 && end < n; ++i) { c = char_at(s, n, end); if (!(c.cls & U_NUMBER)) break; end = c.next; }
        } else if (p->joyai && cjk(c.cp)) {
            do { end = c.next; c = char_at(s, n, end); } while (end < n && cjk(c.cp));
        } else {
            size_t limit = n;
            /* JoyAI's number and CJK splits precede the remaining regex; a
             * leading space or a Latin word cannot consume those spans. */
            if (p->joyai) {
                for (size_t scan = c.next; scan < n;) {
                    character_t a = char_at(s, n, scan);
                    if ((a.cls & U_NUMBER) || cjk(a.cp)) { limit = scan; break; }
                    scan = a.next;
                }
            }
            /* Process this entire span once; rescanning its end for every
             * word would make long text with no digits quadratic. */
            while (pos < limit) {
                end = text_piece_end(s, limit, pos, p->joyai);
                int err = bpe_piece(tok, s + pos, end - pos, out);
                if (err) return err;
                pos = end;
            }
            continue;
        }
        int err = bpe_piece(tok, s + pos, end - pos, out);
        if (err) return err;
        pos = end;
    }
    return 0;
}
static int encode_plain(const myds_tokenizer_t *tok, const char *s, size_t n, output_t *out) {
    const tokenizer_impl_t *p = tok->impl;
    for (size_t pos = 0; pos < n;) {
        size_t k = n - pos < p->max_token_len ? n - pos : p->max_token_len;
        int32_t id = -1;
        for (; k; --k) if ((id = vocab_find(tok, s + pos, k)) >= 0) break;
        if (id < 0) { id = p->byte_ids[(uint8_t)s[pos]]; k = 1; }
        int err = emit(out, id); if (err) return err;
        pos += k;
    }
    return 0;
}

int32_t myds_tokenizer_encode_bytes(const myds_tokenizer_t *tok, const char *text, size_t len,
                                    int32_t *out_tokens, int32_t max_tokens, bool add_bos) {
    if (!tok || !tok->impl || (!text && len) || max_tokens < 0 || len > INT32_MAX) return MYDS_TOKENIZER_ERROR_INVALID;
    const tokenizer_impl_t *p = tok->impl;
    output_t out = {out_tokens, max_tokens, 0}; int err;
    if (add_bos && tok->bos_id >= 0 && (uint32_t)tok->bos_id < tok->vocab_size) {
        const myds_token_entry_t *bos = &tok->vocab[tok->bos_id];
        bool already_present = bos->special && bos->len && bos->len <= len && !memcmp(text, bos->text, bos->len);
        if (!already_present && (err = emit(&out, tok->bos_id))) return err;
    }
    if (!p->byte_bpe) { err = encode_plain(tok, text, len, &out); return err ? err : out.count; }
    size_t plain = 0, pos = 0;
    while (pos < len) {
        if (!p->special_prefix[(uint8_t)text[pos]]) { ++pos; continue; }
        int32_t special = -1; size_t best = 0;
        for (size_t i = 0; i < p->n_special; ++i) {
            int32_t id = p->special_ids[i]; const myds_token_entry_t *v = &tok->vocab[id];
            if (v->len > best && v->len <= len - pos && !memcmp(text + pos, v->text, v->len)) { special = id; best = v->len; }
        }
        if (special < 0) { ++pos; continue; }
        if ((err = encode_bpe(tok, text + plain, pos - plain, &out)) || (err = emit(&out, special))) return err;
        pos += best; plain = pos;
    }
    err = encode_bpe(tok, text ? text + plain : "", len - plain, &out);
    return err ? err : out.count;
}
int32_t myds_tokenizer_encode(const myds_tokenizer_t *tok, const char *text, int32_t *out_tokens,
                              int32_t max_tokens, bool add_bos) {
    if (!text) return MYDS_TOKENIZER_ERROR_INVALID;
    return myds_tokenizer_encode_bytes(tok, text, strlen(text), out_tokens, max_tokens, add_bos);
}
const char *myds_tokenizer_decode_bytes(const myds_tokenizer_t *tok, int32_t token_id, size_t *len) {
    if (len) *len = 0;
    if (!tok || token_id < 0 || (uint32_t)token_id >= tok->vocab_size) return "";
    if (len) *len = tok->vocab[token_id].decoded_len;
    return tok->vocab[token_id].decoded ? tok->vocab[token_id].decoded : "";
}
const char *myds_tokenizer_decode_piece(const myds_tokenizer_t *tok, int32_t token_id) {
    return myds_tokenizer_decode_bytes(tok, token_id, NULL);
}
