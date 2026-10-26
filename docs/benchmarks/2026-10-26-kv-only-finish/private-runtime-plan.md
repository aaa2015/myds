# Private decode KV-only runtime and fullmodel driver plan

Status: CPU-only design, 2026-10-25. No implementation or derived runtime snapshot has been created by this plan. Existing548 source files, Q_B v1, KV-only final-v1 tools/bin/library and all P/Q artifacts remain frozen. Root must independently accept the component and explicitly trigger implementation; M2 fullmodel GPU work additionally requires the target-device component gate and root's GPU window. No GPU/remote/server/deployment action is authorized here.

## Source and library identity

Derive a new owned directory, proposed `/tmp/myds-decode-kv-only-runtime-20261025/{seed,source,bin,build,tests,evidence}`. Seed from the exact548-file `/tmp/myds-indexer-k-runtime-20261025/source` inventory SHA `7791166d7a9a6ef29eaa2336c665919ee041ec1eae47e32027d6a656b44adec7`. Before copy, rehash every file against its build-manifest; preserve original inventory/source/config manifests and note this is a dirty-source export rather than a clean-HEAD equivalence claim. After copy, require all548 seed files exact before any edits. Save final complete derived inventory, every bounded diff, reverse patch application proof and build input/command manifests. Do not edit parent seed/source, repo src/header, existing Makefile, original drivers or state exporter.

Use the same component-screened library bytes, not a fresh append/rename: `/tmp/myds-kv-only-finish-probe-final-v1-20261025/probe.metallib`, SHA `d12ddb1ec806f1daa3848e9687e468aba3ea80d1e91aa3d4081c16eb4852facd`. Selected symbol is exactly `kernel_myds_probe_kv_only_finish`; original native prefix is `bc6603c97ec24af7597f6b10c7ed2a7bfe1344379733a0a4df780250a13f3224`. Copy it as unique exe-adjacent `myds.metallib` for the private control/candidate directories. Actual deployed383710… stays a distinct gold control. Do not invoke the ordinary Metal build rule or change shader registration. The seed contains the default-off K-resident implementation, but that selector remains0/absent and its symbols are not in this KV-only library; no combined optimization experiment is proposed.

## Minimal private source changes

1. `source/include/myds_v41_gpu.h`: add a purpose-specific private scoped encoder declaration, strict mode/symbol getters and six host-counter getter. Keep existing public `myds_v41_gpu_qkv_finish` declaration and behavior unchanged.
2. `source/src/myds_v41_gpu.c`: implement mode/counters plus scoped KV-only encoder using the existing generic dispatch/preflight bridge. Add mode validity to existing scratch/context preflight, so invalid config fails before model inference. Original qkv_finish whole function and all other public APIs remain byte-identical.
3. `source/src/myds_v41_moe.c`: change exactly the one existing non-index-source KV_REUSE call at current line2441 to the purpose-specific scoped function, passing the same partial/Q, KV, window, pos/compressed plus scratch/layer/ordinary-scope witness. Keep the `2560` index-source dummy call and every actual-Q call unchanged. Preserve all surrounding barrier calls, resource arrays, sequence, preflight, mode save/restore, failure cleanup and real Q_B/RoPE scheduling.
4. Add new standalone tools/make fragment/adapter/tests only in this derived source. Do not reuse the old prefill provider name or gate by text substitution without reviewing semantics. No `src/myds_metal.m` edit should be required: generic dynamic-symbol dispatch and constants preflight already support the frozen new PSO.

## Selector, eligibility and failure policy

Proposed flag `MYDS_DECODE_KV_ONLY_FINISH`: absent or exact`0` => reference; exact`1` => enabled; every other nonempty/empty spelling => invalid. Use the existing process-immutable, thread-safe mode-init convention; configure before model/context creation. Separate-process CPU fixtures cover parsing, not runtime getenv mutation. Counter reset is allowed only at the existing idle benchmark reset boundary.

The function is only published from the ordinary scalar non-index-source KV_REUSE branch. It independently checks layer<40 and layer not2/8/14/20/24/28/32/36, ordinary/no-trace/need-logits witness, KV_REUSE1, RF1, register-attention1, FLASH_DEC0, active non-exclusive concurrent batch with auto-barrier0, and compatible existing base pipeline mode. Kprof/profiling scopes and legacy/public/spec/batch2 paths retain reference. A conservative initial position limit is `pos<327680`, matching current deployed max context and the component long-position range; larger legal positions should remain reference rather than claiming untested coverage. Root should approve this final predicate before implementation.

Native ranges are required for the complete existing partial allocation4,210,688B, KV2048B and window262144B, alignment16, valid declared offsets/ranges; partial must be the scratch's owned partial allocation supplied by this caller. Check complete actual data-address ranges for pairwise overlap, not merely distinct handles. The caller's existing full-span check also includes the real ctx.q and every scratch/read/write resource, and remains unchanged. A cloned subview with an aliased byte range does not gain safety from a different wrapper handle. Do not scan in-flight tensor values on CPU, add a synchronization point, remove partial allocation or weaken the existing lifetime contract.

Mode0 returns the untouched original qkv_finish with identical buffers, args and fence boundaries. Mode1 normal ineligible scopes use that same reference and are counted as reference, not fallback. Once the narrow candidate is eligible, a malformed span, missing PSO, failed preflight or failed dispatch records failure and returns0; never retry the original finish. The existing caller propagates failure to the outer graph's commit/invalidates-state cleanup. No GPU wait or readback is added.

For eligible calls, preflight exact symbol with threads(32,1,1), dynamic TG0, no constants, every time required by the existing library lifecycle policy (no permanent ready bool surviving teardown/reinit). Dispatch original args136 bytes (`pos,pad=0,frequencies[32]`), with the exact source-once frequency table copied by bits, buffer order args0/partialQ1/KV2/window3, grid80×1×1 and32×1×1 lanes. The screened shader returns for group<64 and retains the complete original KV branch for64..79. All original outer resource joins remain, including partial even though this candidate no longer writes it.

## Six counters with truthful semantics

Proposed type `myds_v41_gpu_decode_kv_only_finish_counters_t` has `requests,eligible,encoded,references,fallbacks,failures`. They are relaxed atomic host counters. No dtype split is invented: this is one F32 KV/window provider, not two F32/BF16 scorer variants.

- requests: every call to the new scoped API, including mode0 and normal ineligible reference calls.
- eligible: mode1 calls that satisfy the narrow policy and attempt candidate span/PSO/dispatch validation. It is not GPU completion.
- encoded: generic bridge accepted the candidate dispatch into the active batch. Only increment after successful dispatch return.
- references: scoped calls that take the exact original reference because mode0 or normal ineligibility; increment for the selected path, with reference failure also counted in failures.
- fallbacks: forbidden eligible retry witness, initialized0 and never incremented by this implementation. Any nonzero is a benchmark failure.
- failures: invalid mode/candidate span/preflight/dispatch failures, or a chosen reference returning failure. No failure is recategorized as an ineligible reference.

Formal mode1 eligible samples require requests=eligible=encoded and references=fallbacks=failures=0; mode0 requires encoded=eligible=fallbacks=failures=0 and references=requests. Normal ineligible tests are separate safety/applicability classifications, not formal speed samples. Read/reset all fields through a host-only getter; never claim encoded means GPU success. Final output/logit/state validation supplies completion/correctness evidence independently.

## First callback and decode counts are different from the prefill driver

`myds_engine.c:3705–3750` prefills n_prompt-1 IDs. The first generated callback follows a scalar forward of the last prompt ID. At the current unique32-layer scoped call site this is32 candidate calls before the first callback, then32 per subsequent ordinary scalar forward. Bulk chunk prefill does not call the new API. Therefore for a frozen successful64-callback, no-EOS/no-spec fresh request:

- through-first callback: requests32; candidate eligible/encoded32 or mode0 references32;
- after-first decode delta: requests2016=32×63; candidate eligible/encoded2016 or mode0 references2016;
- final totals: requests2048=32×64; candidate eligible/encoded2048 or mode0 references2048.

These numbers must be proved from actual counters and observed callbacks, not generated from a configured64 and passed off as measurement. Every field must be monotonic before computing final-minus-first; regression produces null/invalid rather than unsigned wrap. The benchmark should reject formal requests with EOS/cancel/error, fewer than64 callbacks, DSpark/spec mode or unsupported config. A one-token safety request has legitimate decode delta0 and must be classified outside the64-token performance gate.

First-callback scope string should be explicit: `chunk_prefill_plus_last_prompt_scalar_forward_through_first_output_callback`. Do not name it pure prefill. After-first scope is `N_minus_1_output_intervals_ordinary_scalar_decode`. Existing measured `decode_ms_per_token=(last_callback-first_callback)/(N-1)` remains unchanged, so it measures the interval where the2016 decode encodings occur.

## New fullmodel subject and reversible driver generation

New files proposed: `tools/bench_decode_kv_only_fullmodel.c`, `prepare_decode_kv_only_fullmodel.py`, `decode_kv_only_fullmodel_state.inc`, `decode_kv_only_runtime_adapter.inc`, and `decode_kv_only_fullmodel.mk`.

Emitted `subject` is exactly `decode_kv_only_finish` in outer configuration metadata, base request/model metadata, each sample and state manifest. Provider object is `gpu_decode_kv_only_provider`; selected symbol is one string, not fabricated symbol_f32/symbol_bf16. State schema stays2 with all existing fields/raw payload meanings; scope becomes `fresh_context_final_state_after_prefill_and_decode` and limitations remain explicit. A checked two-string subject/scope adaptation can retain the complete exporter source/byte layout; save reverse proof restoring the old exporter exactly. Internal historical helper identifiers can remain if they emit no false subject; rename them only via a listed reversible edit if needed.

Reuse the unchanged `bench_output_a_fullmodel.c` renderer, greedy config, generation callback, fresh-context-per-sample lifecycle, cache/PLD configuration, start/end clock placement, final logits/raw output and storage metrics. New generator must enumerate every evidence-hook edit and reverse them to recover the complete original base driver bytes. In particular, never move generation/timing/context/free work, insert a GPU wait or change cache/RNG behavior to obtain counters. Save the generated include, original base SHA, adapted SHA and reverse-edit proof. New wrapper emits decode metrics and calls a distinct real runtime adapter; API unavailable/mode invalid/empty selected symbol must refuse an executed run.

Reset counters immediately before the original generation call at the existing idle reset boundary. Snapshot once in the original first callback, and final host read immediately after generation. Provider gate gets observed c.count and stop reason so the exact32×N contract cannot accidentally admit candidate work solely before the first callback or a late isolated decode-only encoding. No step infers a prefill-only decode delta0 rule from the old indexer wrapper.

The existing config discipline is retained: absolute stable config loaded(true) before mode/model/context/API parsing; original62 entries remain unchanged, PREFIX_CACHE1/PLD1; only new selector and owned evidence sinks added to the experimental environment/config copies. Save raw config plus effective MYDS-only environment. First --preflight performs no model/device/dispatch and validates empty distinct sinks and available adapter. --run requires native identity, all expected selectors, actual loaded library, matching mode, and complete raw sinks. PSO success is never substituted for a real adoption count.

## CPU fixtures and later acceptance

Source-side fixtures should include mode absence/0/1 and invalid spellings; exact32 eligible versus8 index-source exclusions; pos0/1/127/128/327679 and larger-normal-ineligible position; all trace/spec/no-logits/RF0/FLASH1/kprof/nonactive batch conditions; complete range/offset/alignment/alias cases using distinct handles and actual backend range helpers or explicit mock scope; missing PSO/preflight rejection/dispatch failure with no original retry; reference failure; all entry/exit encoder modes and existing fences unchanged.

Driver fixtures must reject missing adapter, invalid mode, missing/empty/wrong symbol, prefill-only candidate counts, decode-only missing-first counts, 31/33 layer counts, incorrect64→63 decode factor, counter regression, reference/fallback/failure leakage and unexpected index-source calls. Positive fixtures include exact32 first plus2016 decode for mode1 and exact2048 references for mode0; one-token safety is positive only under its separately declared scope. Freeze independent reader/schema tests before any runtime GPU result.

CPU build uses a new explicit make fragment target with no METALLIB prerequisite, outside original Makefile targets; clean new object/bin directories, frozen deployment/compiler flags, exact original Metal library copies. Record binary/load-library SHA, full source/patch inventory and normal preprocessor evidence that no capture/diagnostic extra-wait macro is compiled in. Fullmodel controls must use the same new CPU binary with flag0/1 plus actual gold library control as frozen by root; no reuse of the old prefill executable's subject or thresholds.

Before adoption: target-device component gate, same-build gold/native identity, full final logits/token/output/state byte gates, actual first/decode counts and selected symbol, and an independently frozen two-workload decode A1/B/A2 performance gate. Parent Q/P prefill passes, KV-only component correctness, old serialized qkv_finish time and deleted work counts do not establish decode speed. Native/HTTP business64 and cancellation/recovery remain separate later gates using the reviewed conditional HTTP plan.
