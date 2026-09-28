# DeepSeek V4.1 reference engine and tensor trace

These tools construct an isolated, deterministic ds4 reference from commit
`6f873ad7f5940ab8556abc41ae510853118ecc4a`. They read Git objects or a verified
source archive. They never copy a ds4 working tree, overwrite an existing build,
start a model during preparation, or change a running service.

The reference is a separate implementation, not the previous myds binary. A
correctness change must be assessed against reference tokens, intermediate
tensors and logits. Agreement with an old myds build only establishes behavior
preservation.

## Prepare and build

On the source machine, export the pinned sources and build the reference:

```sh
python3 tools/reference/prepare_ds4.py \
  --source-repo ~/Projects/ds4 \
  --output /tmp/myds-ds4-reference \
  --export-archive /tmp/ds4-reference-6f873ad7.tar.gz \
  --build
```

The selected source closure has 82 files and 11,279,442 bytes; its gzip archive
is about 1.7 MiB. The uncompressed Git archive SHA256 is pinned in the script:

```text
bb67197f7444b6c43a045c4b444451c3ef74ce2b9df57b0e5c010f2975198b94
```

The pinned commit is not assumed to exist in the remote machine's Git history.
Copy `tools/reference/` and the exported archive to the inference machine, then:

```sh
python3 tools/reference/prepare_ds4.py \
  --source-archive /tmp/ds4-reference-6f873ad7.tar.gz \
  --output /tmp/myds-ds4-reference \
  --build
```

The archive SHA256 is verified before extraction. The output includes the
patched reference CLI `ds4`, public C API archive `libds4.a`, exact-token harness
`trace_ds4`, all matching Metal sources, and `REFERENCE_MANIFEST.json` with
source-file and trace-patch hashes. Compile on the target Mac to select its CPU
architecture. Do not run the binary from a different project's directory:
ds4 loads its `metal/*.metal` kernels relative to the working directory.

## Exact-token trace

Run this only on the designated inference machine when its model memory is
available. Preparation itself does not load a model. Choose the same token IDs
for this harness and myds's `tools/trace_model.c`; no tokenizer, BOS insertion or
chat template participates in this harness.

```sh
cd /tmp/myds-ds4-reference
DS4_METAL_DISABLE_V41_LAYER_PREFILL=1 \
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/ds4-ref \
DS4_METAL_GRAPH_DUMP_LAYER=0 \
DS4_METAL_GRAPH_DUMP_POS=0 \
./trace_ds4 \
  ~/.lmstudio/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  42
```

Replace `42` with the exact agreed token. Additional space-separated token IDs
are supported. The harness uses at least 128 context slots, prints each token frontier
argmax/logit and finite-value count, and fails if any frontier logit is nonfinite.
The `DS4_METAL_DISABLE_V41_LAYER_PREFILL=1` switch makes the trace use the scalar
V4.1 path, which the patch instruments. This diagnostic mode deliberately
synchronizes the GPU between captures and must not be used for timing results.
The harness writes each frontier's logits directly through `ds4_session_copy_logits`; the graph may already have advanced its internal position when
it calculates the output head, so the graph hook alone is not sufficient for a
position-filtered final capture.

Filenames are `<prefix>_<stage>-<layer>_pos<position>.bin`, containing little
endian float32 values. Global embedding and logits use layer 0 so the same layer
filter includes them. `DS4_METAL_GRAPH_DUMP_NAME` optionally selects stage names;
without it every listed stage is captured. For a single token, layer 0, position
0, the expected files are:

| Stage | Float values | Meaning |
|---|---:|---|
| `embedding` | 20,480 | Four repeated 5,120-element HC streams; compare all four native streams individually |
| `hc_attn_input` | 5,120 | Previous sublayer's weighted HC reduction; initial weights are `[1,0,0,0]` |
| `attn_norm` | 5,120 | Weighted RMSNorm, epsilon `1e-20`, then BF16 round |
| `q_a_raw` | 1,280 | Q_A projection **after** the model's mandatory BF16 round |
| `q_a_norm` | 1,280 | Weighted Q_A norm, then BF16 round |
| `q_b` | 32,768 | Q_B projection, BF16 round, **no additional per-head RMSNorm** |
| `kv_raw` | 512 | KV projection, then BF16 round |
| `kv_norm` | 512 | KV weighted norm, then BF16 round |
| `kv_rope` | 512 | RoPE applied, before FP8 cache quantization |
| `kv_quant` | 512 | FP8 E8M0 raw-cache values represented as F32 |
| `q_rope` | 32,768 | Query after V4.1 RoPE |
| `attn_out` | 5,120 | Attention output projection, BF16 round, before HC residual expansion |
| `layer_out` | 20,480 | All four HC residual streams after the FFN; compare all four native streams; one-vector outputs fail |
| `logits` | 129,280 | Full vocabulary logits, without a BF16 rounding step at the output head |

The reference's generic V4 path already had dump helpers, but its V4.1 graph did
not call them. `ds4-v41-trace.patch` adds calls to the V4.1 scalar path and retains
the existing operation order. The hooks are no-ops when no dump prefix is set.
Use the unpatched pinned sources for subsequent performance measurements.

For complete text tokenization and logits diagnostics, the isolated reference
CLI also provides:

```sh
./ds4 -m "$MODEL" --raw -p 'A' --dump-tokens
./ds4 --metal -m "$MODEL" -c 128 --raw -p 'A' --dump-logits /tmp/logits.json
./ds4 --metal -m "$MODEL" -c 128 --raw -p 'The capital of France is' \
  --temp 0 -n 16 --dump-logprobs /tmp/logprobs.json --logprobs-top-k 20
```

`--dump-tokens` reads GGUF metadata and tokenizer data without loading the model
for inference. Do not substitute `--first-token-test` or the legacy
`--metal-graph-test`: the V4.1 loader explicitly rejects those legacy diagnostics.

## Multi-position native-engine probes

The selected project direction is the independent myds native engine. This
pinned ds4 executable is an external correctness oracle only; it is not linked
into myds and is not a production backend.

Both harnesses accept `MODEL --tokens-file PATH`. The file contains decimal,
nonnegative token IDs separated by whitespace; `#` starts a line comment. It
must contain at least one ID and at most 1,048,576 IDs. Invalid text, overflow,
missing files and empty sequences fail before any model loads. No tokenizer,
automatic BOS, chat template, sampling or generated continuation participates.
Keep the token file with the resulting traces and record its SHA256.

Use at least 130 exact input tokens to capture zero-based positions 1, 8, 64
and 129. More than 128 inputs exercise the sliding-window boundary; this is a
necessary coverage point, not evidence that compression/indexer logic is correct.
Run the two engines sequentially on the designated inference machine, in
isolated directories and without replacing production port 8181:

```sh
cd /tmp/myds-ds4-reference
DS4_METAL_DISABLE_V41_LAYER_PREFILL=1 \
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/native-probe/reference \
DS4_METAL_GRAPH_DUMP_LAYER=0,1,14 \
DS4_METAL_GRAPH_DUMP_POS=1,8,64,129 \
./trace_ds4 "$MODEL" --tokens-file /tmp/native-probe/tokens.txt

cd /tmp/myds-native-candidate
MYDS_TRACE_PREFIX=/tmp/native-probe/myds \
MYDS_TRACE_LAYER=0,1,14 MYDS_TRACE_POS=1,8,64,129 \
./bin/trace_model "$MODEL" --tokens-file /tmp/native-probe/tokens.txt

python3 tools/compare_traces.py /tmp/native-probe \
  --positions 1,8,64,129 --layers 0,1,14 \
  --tokens-file /tmp/native-probe/tokens.txt \
  --output /tmp/native-probe/comparison.json
```

`*_LAYER` and `*_POS` accept a single decimal index, a comma-separated list, or
`all`. Unset filters capture every index. The reference synchronizes the first
input token to initialize its checkpoint, then evaluates every supplied token
one at a time. Logits are copied through the public API after **each** supplied
token, with that token's actual position; the ambiguous internal graph head
position is no longer used. JSONL progress lines report every input ID and
argmax, regardless of which positions are selected for binary dumps.

Additional HC stage files are `hc_attn_mix` / `hc_ffn_mix` (24 raw projection
values), `hc_attn_split` / `hc_ffn_split` (24 values: pre4, post4, comb4x4),
`hc_attn_pre` (the four carried weights consumed by attention), `after_attn`
(four residual streams), `hc_ffn_input` and `ffn_norm` (one 5120-element row).
The attention reduction consumes the previous FFN's pre weights, initialized
as `[1,0,0,0]`; the FFN reduction consumes the current attention split's pre
weights. These are different from each block's own post/comb expansion.

The comparator reports each selected position/layer separately. Equal-shaped
`embedding`, `after_attn` and `layer_out` tensors receive per-stream max absolute
error, RMSE and mismatch counts. A native single-stream tensor against a
four-stream reference remains a **shape failure**; the comparator never repeats
or averages it to manufacture a pass. At positions >= 1, `layer_out` must also
contain pairwise-distinct streams on both sides. This assertion does **not**
apply to embedding: every token starts with four identical embedding streams,
including later positions. Layer 0's first pre-reduction can therefore still be
an identity even far into the input. The roadmap's original general assertion
about later positions is only valid for post-HC residual outputs.

Missing stages, nonfinite values, differing shapes and numerical differences
are failures. Use `--stages attn_norm,q_a_raw,...` for a named partial milestone;
the selected list is evident in the report. Defaults remain `atol=0, rtol=0`.
Nonzero tolerance requires an explicit `--tolerance-reason`, chosen from the
numerical contract before inspecting the comparison. This records a choice;
it does not establish that the choice is justified.

No-model regression checks:

```sh
python3 -m unittest discover -s tools/reference -p test_trace_tools.py -v
```

These cover malformed token inputs/filters, all four requested positions and
three layers, missing artifacts, shape mismatches, per-stream errors, duplicated
post-HC streams, repeated valid embeddings and nonfinite values. Actual model
traces remain separate evidence and must be collected on the inference machine.
