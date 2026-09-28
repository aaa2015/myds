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
are supported. The harness uses at least 128 context slots, prints the final
argmax/logit and finite-value count, and fails if any final logit is nonfinite.
The `DS4_METAL_DISABLE_V41_LAYER_PREFILL=1` switch makes the trace use the scalar
V4.1 path, which the patch instruments. This diagnostic mode deliberately
synchronizes the GPU between captures and must not be used for timing results.
The harness writes final logits directly through `ds4_session_copy_logits` at
`tokens.len - 1`; the graph may already have advanced its internal position when
it calculates the output head, so the graph hook alone is not sufficient for a
position-filtered final capture.

Filenames are `<prefix>_<stage>-<layer>_pos<position>.bin`, containing little
endian float32 values. Global embedding and logits use layer 0 so the same layer
filter includes them. `DS4_METAL_GRAPH_DUMP_NAME` optionally selects stage names;
without it every listed stage is captured. For a single token, layer 0, position
0, the expected files are:

| Stage | Float values | Meaning |
|---|---:|---|
| `embedding` | 20,480 | Four repeated 5,120-element HC streams; compare myds embedding to the first stream |
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
| `layer_out` | 20,480 | All four HC residual streams after the FFN; not directly comparable to one myds vector |
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

## Maintaining a native ds4 backend in myds

Static linking through `ds4.h` is the smallest practical integration boundary.
The build's eight core objects are `ds4.o`, `ds4_image.o`, `ds4_distributed.o`,
`ds4_tp.o`, `ds4_ssd.o`, `ds4_metal.o`, `ds4_layer_pack.o` and `ds4_engram.o`.
The build tool already archives and links these for `trace_ds4`. Required macOS
libraries are Foundation, Metal, libm and pthread; no Python package is required
at inference time.

| myds responsibility | Public ds4 API |
|---|---|
| Model lifetime | `ds4_engine_open`, `ds4_engine_close` |
| Inference context | `ds4_session_create`, `ds4_session_free` |
| Rendered prompt tokenization | `ds4_tokenize_rendered_chat` |
| Native chat construction | `ds4_chat_begin`, `ds4_chat_append_message`, `ds4_chat_append_assistant_prefix` |
| Prefill and prefix reuse | `ds4_session_sync` with the full token prefix |
| One-token decode | `ds4_session_eval` |
| Logit inspection | `ds4_session_copy_logits`, `ds4_session_top_logprobs` |
| Sampling | `ds4_session_sample`, `ds4_sample_logits`, `ds4_session_argmax` |
| Decode text bytes | `ds4_token_text` returns allocated bytes plus explicit length; caller frees them |
| Stop detection | `ds4_token_is_stop` or think-mode-specific variant |
| Prefill cancellation | `ds4_session_set_cancel`, with interrupted return code 2 |
| Prefill progress | `ds4_session_set_progress` |

Integration constraints established from the pinned source:

* V4.1 supports Metal and single-GPU CUDA per rank, with optional network tensor
  parallelism. CPU, distributed layer slices, legacy graph diagnostics and
  directional steering are rejected. Maximum context is 1,048,576 and power
  must be 100 for this model path. macOS integration should select Metal.
* Keep one active engine/backend and serialize GPU execution. ds4 uses global
  shape and GPU state; it is not a general independent multi-model library.
  Initializing both myds Metal and ds4 Metal for the same request is unnecessary.
* Bind the 26 `DS4_METAL_*_SOURCE` paths to the **vendored** kernel directory, or
  patch its source-root resolver. Reading myds's unrelated `metal/` directory
  would silently mix kernel versions. Avoid process-wide `chdir` in the server.
* For repetition penalties, copy logits and apply myds policy before calling
  `ds4_sample_logits`. Preserve explicit token fragment lengths through UTF-8
  assembly and JSON escaping. Some fragments contain incomplete UTF-8 bytes.
* `ds4_encode_chat_prompt` correctly handles a single system/user turn. The
  public append functions are primitives, not the complete multi-turn renderer:
  they do not automatically merge consecutive users or terminate historical
  assistant messages with EOS. The full renderer is the static
  `render_deepseek41_chat_ordered` in `ds4_server.c`; extract it as a reusable
  module or validate a port against its known-answer fixtures. Reusing a
  V3-style rendered string is not a correctness oracle.
* A changed prompt that is not a prefix is rebuilt by `session_sync`. Single-step
  `myds_forward_token(..., pos)` needs adapter-owned history/reset semantics,
  since `ds4_session_eval` advances the current session position.
* Some malformed model/config paths call `ds4_die`/`exit`; embedding does not
  provide a fully recoverable loader boundary. A process adapter provides crash
  isolation but adds IPC, cancellation, startup and token/logit transport work.
  The current controlled single-model deployment can use static linking; a
  multi-tenant loader should reconsider isolation.

A vendored source archive plus SHA256 manifest gives a standalone reproducible
build. Alternatively, pin a published Git commit/submodule. This particular
commit exists in the local source repository and is not assumed published; do
not depend on `../ds4`, its branch name, its dirty files or its existing objects.
The archive does not contain model weights. ds4 and the included Iris image
reader are MIT licensed; retain both `LICENSE` and `third_party/iris/LICENSE`.

Validation completed here: pinned Git export, patch application, Metal CLI
compilation, static-library/harness linking, help invocation, and verified
archive-based extraction. The direct logit-writer's frontier position, byte
output and position filtering were checked without loading a model. Actual
model traces are collected separately by the
operator of the inference machine.
