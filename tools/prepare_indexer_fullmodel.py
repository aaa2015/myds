#!/usr/bin/env python3
"""Prepare a checked include from the unchanged Output_A full-model driver.

Only evidence export/subject hooks change. The base timed generation call,
fresh-context lifecycle, prompt renderer, cache settings, raw logits and byte
callback remain unchanged. The original harness is never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(source):
    begin = source.index("static bool export_state(")
    end = source.index("\nint main(", begin)
    old_state = source[begin:end]
    if source.count("static bool export_state(") != 1 or not old_state.rstrip().endswith("}"):
        raise ValueError("unique complete legacy state exporter required")
    if "ctx->anchor_len>1024u" not in old_state or "ctx->anchor_tokens," not in old_state:
        raise ValueError("legacy anchor layout changed; review adapter")
    changes = [(old_state, '#include "indexer_fullmodel_state.inc"\n')]
    changes += [
        ('    fprintf(json, "{\\"kind\\":\\"metadata\\",\\"model\\":");',
         '    fprintf(json, "{\\"kind\\":\\"metadata\\",\\"subject\\":\\"prefill_indexer_k_resident\\",\\"model\\":");'),
        ("        c->first = t;", "        c->first = t;\n        bench_indexer_first_callback();"),
        ("        bench_outa_get(NULL,true);", "        bench_indexer_before_sample();\n        bench_outa_get(NULL,true);"),
        ("                           logits_finite && logits_hash == expected_logits;",
         "                           logits_finite && logits_hash == expected_logits && bench_indexer_provider_valid();"),
        ('        decode_metrics(json, ctx);', '        bench_indexer_sample_json(json, run);\n        decode_metrics(json, ctx);'),
    ]
    adapted = source
    for old, new in changes:
        if adapted.count(old) != 1:
            raise ValueError("unique base driver anchor required: " + old[:90])
        adapted = adapted.replace(old, new, 1)
    restored = adapted
    for old, new in reversed(changes):
        if restored.count(new) != 1:
            raise ValueError("unique reverse adapter anchor required")
        restored = restored.replace(new, old, 1)
    if restored != source:
        raise ValueError("unlisted base driver change")
    return adapted, {"base_sha256": digest(source), "generated_sha256": digest(adapted),
                     "reverse_restores_base_byte_for_byte": True,
                     "edits": [{"before_sha256": digest(a), "after": b} for a, b in changes]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.base.resolve() == args.out.resolve():
        raise ValueError("base driver must remain unchanged")
    output, proof = prepare(args.base.read_text())
    args.out.write_text(output)
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
