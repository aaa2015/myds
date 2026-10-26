# Conditional native/HTTP business and cancellation stage — CPU-only draft

Prepared 2026-10-25. This is a source audit and proposed protocol, not an executed experiment. It authorizes no GPU submission, server launch, remote command, deployment, or production modification. Execution depends on independently accepted long Q and P and root's explicit decision. Historical r5 results provide workflow ideas only; they do not pass any gate below.

## Frozen subject and controls

Use `/tmp/myds-indexer-k-runtime-20261025/source`, with all 548 source files checked against `build-manifest.json` inventory SHA `7791166d7a9a6ef29eaa2336c665919ee041ec1eae47e32027d6a656b44adec7`. Preserve the seed inventory and core/full patches. Do not rebuild from the current changing worktree. No `MYDS_INDEXER_CAPTURE` or other observer macro belongs in the ordinary performance executable.

The private runtime remains default off. Candidate eligibility is the intersection of ordinary prefill, no trace, active non-exclusive batch, nheads=32, ratio=2, 512<ncomp<=16384, 8<=rows<=2048, end=start+rows<=32768, ncomp=end/2, key-window=0, SGBAR absent/0, valid scratch budget, complete accessible disjoint output/read spans, and both candidate PSOs preflighted. Public APIs and scalar decode do not enter this provider. Missing eligible PSO/dispatch failure is an error, not reference fallback. Six host counters are requests/encoded/fallbacks/failures/f32/bf16; encoded denotes accepted encoding, not GPU completion. Normal off/ineligible calls contribute requests but not fallback or encoded.

Keep three artifact arms distinct: actual deployed metallib `3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9` with mode 0; full-TU private metallib `9df7ed2a5212d54d4475480bf6388d2f66ce24c93bb3a8b003f2b789806d6982` with mode 0; the same private library with mode 1. The new HTTP executable gets its own SHA even when its engine objects match the native fullmodel executable. Loaded-path/log evidence must identify the actual library in every process; expected file location alone is insufficient.

## Independent HTTP CPU build without replacing the approved library

Frozen Makefile line 220 makes the normal executable depend on `$(METALLIB)`; its normal Metal source list lacks the private append. A default `make` can therefore overwrite the approved library. Proposed external make fragment, created outside the frozen source only after authorization:

```make
include Makefile
$(HTTP_BIN): $(OBJS_C) $(OBJS_M) $(BUILD_DIR)/main.o | $(BIN_DIR)
	$(CC) $(filter %.o,$^) $(LDFLAGS) -o $@
```

Proposed CPU command (not executed here):

```sh
make -C "$PRIVATE/source" -f "$HTTP_STAGE/http-cpu-only.mk" \
  BUILD_DIR="$HTTP_STAGE/obj" BIN_DIR="$HTTP_STAGE/bin" \
  HTTP_BIN="$HTTP_STAGE/bin/myds-http" "$HTTP_STAGE/bin/myds-http"
```

Use a clean object directory, the frozen Makefile's deployment flags and C/ObjC/link flags, and capture full compiler inputs/version/SDK/argv/logs/outputs. Copy the approved frozen library beside this executable as the unique `myds.metallib`; never invoke the ordinary Metal rule. Use an empty owned cwd. A gold executable/library pair and private executable/library pair live in separate directories. Verify minos and hashes, then prove CPU-only `--dump-config` output before any GPU-authorized execution. No compile-time capture synchronization is required for the plain server.

## Configuration, request identity, and native oracle

Frozen `src/main.c:119–159` loads `--config` with overwrite=true before model/API argument parsing. Environment overrides cannot be assumed to win. Prepare separate complete mode-0/mode-1 configuration copies from the frozen 62-entry production config, changing only the selector and explicitly owned evidence paths. Record base-file SHA, overlay/full-file bytes, effective `MYDS_*` values, and absence of other experiment selectors. `MYDS_RESP_STATE` must point to the stage's own response index; otherwise `src/myds_server.c:283–287` can write `$HOME/.cache/myds/responses.idx`. Do not change HOME. If a future frozen config enables disk-KV, define separate owned disk state and inventory/restoration before launch; do not silently reuse production cache.

Future example argv, requiring paths/port to be frozen first:

```sh
"$HTTP_STAGE/bin/myds-http" --config "$MODE_CONFIG" -m "$MODEL" \
  --host 127.0.0.1 --port "$OWNED_PORT" --alias myds-private \
  -c 327680 -n 64 --temp 0 --top-p 1 --top-k 0 --seed 42 --no-thinking
```

This example defines a non-thinking greedy workload. It is not permission to rewrite the real business JSON. Freeze the actual business request bytes, chat-rendered prompt bytes, complete token IDs/count, model/tokenizer identities, and decode semantics first. If actual business requests thinking, the server forces temperature=1 in that branch (`myds_server.c:3795–3800`), whereas the existing native wrapper hardcodes greedy temp=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1, seed=42 (`tools/bench_output_a_fullmodel.c:539–540`) and default false thinking. Such a wrapper is not a native oracle until a separately reviewed driver matches the actual request. HTTP JSON seed is not a substitute for the server argv seed.

For an explicitly agreed non-thinking workload, freeze max_tokens=64, temperature=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1, presence_penalty=0, enable_thinking=false, stream=true, and stream_options.include_usage=true. The native oracle must call the same `parse_chat_prompt` path and same config fields. Require 64 generated tokens and length termination if that is the frozen workload contract; if EOS terminates earlier, preserve and classify it rather than retroactively changing the gate.

The short actual-business request may have ncomp<=512 and be correctly ineligible. Its encoded=0 is then applicability/safety evidence, not a failed speed measurement. Existing fullmodel mode-1 first-callback contract encoded>0 must remain intact for long eligible requests. Do not weaken it to admit a short business request; use a separate business-safety classification/harness. Eligible native long requests must retain first encoded>0, actual symbol witnesses and decode encoded delta=0.

## Available witnesses and missing HTTP diagnostic plumbing

Available without engine edits: chat rendering, response bytes, HTTP usage, stop reason internally, existing engine prefill/decode statistics, loaded-library path getter, six global candidate getters, serialized request inference lock. The server currently does not emit candidate getters, token IDs, full logits or raw state. `myds_server_sse_callback` discards token_id at line 3202. The ordinary native wrapper's schema-2 state exporter is tool code, not an HTTP engine observer.

If this stage is authorized, a separately reviewed private diagnostic clone/driver is needed for: reset after inference lock acquisition and before generation (`myds_server.c:3623–3635`); first token callback six-counter snapshot; token-ID capture; final counters/logits and scoped state before cleanup/config restoration; high precision engine timing and stop reason; cancel/transport status. Add no GPU commit/wait to the normal server. Diagnostic exports after the engine's existing completion wait are correctness evidence; any extra wait/checkpoint version must be identified separately and excluded from ordinary timing.

The existing `tools/indexer_fullmodel_state.inc` schema 2 covers a fresh-context final state and current multi-slot anchors. It is not a ready-made partial-cancel or HTTP exporter: its hardcoded fresh-context claim and explicit exclusions (historical snapshot Engram tails, inactive disk data, padding/scratch/intermediate logits) must be reviewed for every new state scope. Compare complete raw payloads and metadata under a declared semantic scope; allocation capacity after a cancelled larger request may differ from a fresh oracle and must be reported separately, not silently mask an entire context record.

The server log rounds prefill/decode durations to one decimal second (`myds_server.c:4295–4303`). It cannot support a 2% performance decision. Until a reviewed high precision observer/protocol exists, this HTTP stage is correctness, wire, cache and recovery evidence. Native Q/P performance does not automatically prove HTTP speed.

## Raw wire and normal request sequence

Use loopback `/v1/chat/completions` SSE first. Its manual header, fixed chat completion id and body framing permit literal byte comparison; non-streaming has wall-clock `created`, and Responses endpoints have dynamic ids/timestamps. Do not remove or normalize dynamic fields and call the result raw parity. If additional endpoints are tested, keep complete raw bytes and declare any semantic comparison separately.

Capture exact request TCP bytes and every response header/body byte with an owned raw-socket client until EOF, plus separate monotonic timestamps. Preserve raw bytes before parsing; do not use `tools/bench_http.py` line decode/strip output as a raw oracle. That tool may inspire timing/client logic only. `tests/test_http_bytes.py` is a useful CPU transport fixture with the actual handler but a mock generator; it is not real-model evidence. An initial empty assistant role SSE event is emitted before generation (`myds_server.c:3910–3916`), so first wire byte is not token TTFT. Report first nonempty generated content/reasoning separately.

Freeze all requests and sequence before execution. Fresh/cold HTTP uses a newly started owned server context for each arm; warm/prefix-cache scenarios use the same context and the same declared preceding requests. There is no audited public reset endpoint, so do not substitute an assumed reset for fresh state or disable the frozen production prefix cache. Preserve ordinary max context and all 62 base configuration values.

Normal gates: all output bytes/token IDs under native+HTTP equivalence; complete final vocabulary logits where exported; declared final live-cache/Engram/RNG/published snapshot state bytes; all guard/input protection applicable to observers; exact ordered raw SSE wire across matched arms; actual loaded artifacts, effective config and provider scope; zero failures and forbidden fallback. Preserve every arm/sample and raw evidence before evaluation. HTTP A1/B/A2 sample count/drift/speed rules require their own frozen protocol; they are not established by this draft or by old r5.

## HTTP disconnect and native prefill cancellation are separate gates

Source reality: `client_disconnected` uses nonblocking socket peek (`myds_server.c:2764–2769`), but the normal checks occur in token callbacks (`3209`, `3228`). During a quiet prefill, the SSE writer can sleep on an empty queue; it checks the socket while sending queued packets. Closing a socket during prefill therefore does not prove that the engine observed a prefill cancellation. A failed initial SSE write can cancel before generation starts, which proves no partially executed prefill. Capture this distinction explicitly.

1. HTTP transport/decode cancel: a client closes after a predeclared number of generated content tokens. Preserve the exact received wire prefix and server-side cancellation/stop/transport witnesses. Require no successful completion/DONE after cancellation, request cleanup and inference-lock release, valid response-registry status, and successful next request using the same process and context. Verify post-recovery logits/output/state against the declared same-history control; do not restart the server or repair model state to make recovery pass.
2. Native partial-prefill cancel: a separate current-source-compatible driver passes a real cancel flag to `myds_generate_bytes_ex`. `myds_v41_prefill_tokens` checks it at layer/chunk boundaries (`myds_v41.c:7539,7611`), and cancellation commits outstanding work then returns -2 (`8603–8605`). `myds_engine.c` invalidates the prefix-cache length on that return. A CPU poll of an encoded counter can prove some candidate work was accepted, but cannot prove an exact owner/layer boundary. If deterministic checkpoint proof is needed, build a separate private compile-time diagnostic observer; explicitly classify any extra commit/wait as diagnostic, not ordinary asynchronous timing.
3. Cancellation state gates: prove observed cancellation, outstanding-work completion before host reads/free, no published partial checkpoint, CPU RNG and Engram consistency, valid previous snapshots/window ownership, and same-context recovery without manual repair. Non-greedy native cancellation can make the prefill RNG-preservation claim meaningful; seed/config transitions must be predeclared rather than silently reset as a repair. After recovery compare defined semantic state, full final logits and generated IDs/bytes. Export partially updated live state as partial, not a complete successful oracle.

Existing `tools/test_moe_workspace_cancel.c`/promptclear test files reference obsolete missing workspace APIs and old flat anchor arrays in this frozen source; they are not directly runnable current-source evidence. Historical r5 used anchors149/150 and owner0/layer2 + owner3/layer20 diagnostic observers with an extra commit/wait. Those small anchors are not automatically eligible for today's ncomp>512 candidate. Only its workflow can be reused after obtaining and independently reviewing that bundle. The historical memory records an unfinished combined gate, not six accepted arms; remote paths have not been verified in this CPU-only pass.

## Required preparation before root can choose execution

Freeze actual business JSON/native rendering and whether thinking is intended; classify candidate eligibility. Build/review the isolated CPU HTTP target and effective-config manifest. Implement/review the minimum HTTP/native diagnostic witnesses in a new private clone if full state/cancel proof is required. Freeze the normal/warm/disconnect/native-cancel request matrix, complete raw/state schema, independent reader, artifact controls and owned restoration/lock protocol before GPU authorization. Keep all old Q_B rejection, old failed R drift and unmatched cohorts unchanged. No production default or runtime source changes follow from this draft.
