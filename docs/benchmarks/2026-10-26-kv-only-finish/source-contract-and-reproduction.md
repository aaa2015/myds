# KV-only finish private component v1

This is a different source transformation from rejected Q_B→forward-RoPE v1. It removes an unused computation on attention scratch; it does not move actual Q arithmetic into Q_B or change actual Q dot/RoPE expressions. No runtime selector, production source, header or Metal registration was edited. CPU and Metal compilation plus host-only evidence are complete. The author submitted no GPU command buffers, made no remote changes and launched no server.

## Frozen delivery

Use only `/tmp/myds-kv-only-finish-probe-final-v1-20261025`. Do not use `/tmp/myds-kv-only-finish-probe-build-v1-20261025`: that is an earlier unused16TG CPU draft, has no GPU evidence, and was superseded before any run.

- `bench_kv_only_finish_probe`: SHA `02fbae976b68ec8c24ec6b0298e4452e5369629069d24156bae70bdf77452660`.
- `probe.metallib`: SHA `d12ddb1ec806f1daa3848e9687e468aba3ea80d1e91aa3d4081c16eb4852facd`.
- Complete production shader prefix: 1,574,167 bytes, SHA `bc6603c97ec24af7597f6b10c7ed2a7bfe1344379733a0a4df780250a13f3224`.
- Candidate symbol: `kernel_myds_probe_kv_only_finish`; gold/native symbol: `kernel_dsv41_qkv_finish`.
- `source-identity.json` binds every compiler input, complete native prefix, new append, original host frequency initializer, compile commands/flags, binary and libraries. The original native template was not extracted as a substitute oracle.
- `component-correctness-protocol-v1.json`, `source-contract-audit.json`, `delivery-manifest-v1.json` live in `/tmp/myds-kv-only-finish-probe-20261025`.

The append changes exactly two source spans: kernel name and replacement of the entire `if(group<64u){...Q RoPE...return;}` branch with `if(group<64u) return;`. A reverse transformation restores the original full kernel bytes. The whole remaining KV branch is byte-identical, including all precise trig, BF16 boundaries, FP8 scale/dequant/quant and KV/window writes. Every arm retains args136/pos/pad/frequencies offsets, buffer slots0..3, 80 TG×32 lane, no dynamic shared memory, and the same three-resource post-dispatch join. The original native prefix and all original functions remain intact.

## Actual source contract

Frozen private source `/tmp/myds-indexer-k-runtime-20261025/source` has the reviewed548-file inventory `7791166d7a9a6ef29eaa2336c665919ee041ec1eae47e32027d6a656b44adec7` and production62-entry config SHA `0a3524b030a73916f7684d841b912c6e978c680825a3c3520ca18188f35a2cff`.

Ordinary scalar forward enters `myds_v41_decode_attn_indexer_scoped` from `src/myds_v41.c:4213`. With KV_REUSE1 on a non-index-source layer, `src/myds_v41_moe.c:2433–2456` uses `scratch.partial` as Q solely to reuse the QKV finish's KV/window half. The actual `ctx.q` comes from Q_B and independent forward RoPE. Original `metal/dsv41.metal:131–139` nevertheless reads/writes64 tail F32 values/head in dummy scratch and executes precise sin/cos for32 pairs/head.

The proposed initial integration scope is exactly32 non-index-source layers: all40 except2/8/14/20/24/28/32/36, ordinary scalar+need_logits, no trace, KV_REUSE1, RF/register attention1, FLASH_DEC0, exact pre-existing pipeline/span/active-batch guards. No runtime integration exists in this delivery.

At this scope, the subsequent register attention buffers (`src/myds_v41_gpu.c:2223–2269`) do not contain partial, and the direct implementation returns from that register path (`2610–2623`). Therefore no dependency on a later overwrite is needed to prove dead dummy values. Other source paths that read partial are FLASH_DEC split/reduce and generic vec/reduce, which have their own full partial producer then fence/reducer. They are excluded from this initial proposal. Do not delete the partial allocation or its WAR/producer/consumer barriers.

All finish call sites are listed in source-contract-audit.json. Current default scheduling makes40 dummy finishes/token:32 non-index-source plus8 index-source. Only32 are proposed initially. Actual-Q callers in fallback/reference, public/spec/batch2 and disabled-KV branches remain unchanged. The second dummy call at `moe.c:2560` belongs to index-source handling and is also excluded initially.

Legal SWA widths are1..128: pos0 has one valid key; there is no legal zero-key scalar attention state. Test cases cover0/1/127/128 and long positions. Before candidate entry, `decode_layer_buffers_valid` includes full4,210,688B partial and real Q/KV/window ranges; `decode_spans_disjoint` compares view-start data addresses and complete buffer sizes, not handle equality. The CPU fixture passes distinct mock handles with identical data starts and offset4096 overlapping ranges and proves rejection before finish. It uses simulated host addresses; it is not actual MTL allocation/lifetime execution. GPU harness allocations are separate fresh retained `newBuffer` resources, never subviews, so its noalias claim is explicitly limited to those allocations.

A stage error returns failure, resets KV reuse state and restores entry auto-barrier/concurrent modes. The outer scalar forward keeps its existing commit-and-invalidates-graph fail path. This proposal does not retry, remove resources from joins, or rely on the resource list to allow dispatches to cross a barrier. On a later attention failure the removed dummy values still have no reader; existing failure synchronization/lifetime rules remain mandatory and must be tested if runtime integration is later authorized.

## CPU evidence

`source-host-fixture.stdout.txt` executes the actual frozen scheduler with recording stubs. The pre-existing full fixture passes. Extra checks cover5 positions×all40 layers=200 successful scopes,32 long-position non-index-source scopes, four failures(qkv_finish/Q_B/actual-Q RoPE/prestage_window) restoring encoder modes, and two distinct-handle overlapping-range counterexamples rejected before finish. These are source/binding/failure-routing evidence, not shader math or real Metal lifetime proof.

`/tmp/myds-kv-only-finish-probe-host-only-v1-20261025` contains114 cases and457 files,45,089,653 bytes total, all sizes/SHA rechecked. It returned before device creation with0 command buffers. The cases are every32 eligible layers at pos0/1/327679 plus layers0/3/39 at127/128/16383/16384/65535/65536. KV/window inputs are finite BF16 with zero/negative-zero/small/large patterns; dead dummy input also covers NaN/Inf/subnormals to prove it is not a required semantic input. Any gold/native identity failure on those adversarial dead bytes is conservatively rejected rather than ignored.

Rebuild command (CPU/Metal compile only, new output directory required):

```sh
python3 /tmp/myds-kv-only-finish-probe-20261025/tools/build_kv_only_finish_probe.py \
  --source-root /tmp/myds-indexer-k-runtime-20261025/source \
  --out-dir /tmp/NEW-kv-only-finish-probe-build --cpu-minver 26.0
```

For exact frozen-binary target checks copy the delivered binary/library/sourceidentity unmodified rather than rebuilding on a new compiler. Retain compile log, minos, loaded paths and complete copied file SHA evidence.

## Correctness-only GPU command for root's controlled window

This is a proposed command, not executed by the author. Root owns GPU authorization, lock, device identity, competition checks and result collection. M2 must wait for its Ql/P window. Freeze a fresh output path and stdout/stderr before execution; omit `--timing`:

```sh
/tmp/myds-kv-only-finish-probe-final-v1-20261025/bench_kv_only_finish_probe \
  --synthetic \
  --metallib /tmp/myds-kv-only-finish-probe-final-v1-20261025/probe.metallib \
  --golden-metallib /tmp/myds-indexer-r2-real-local-20261025/deployed/myds.metallib \
  --build-manifest /tmp/myds-kv-only-finish-probe-final-v1-20261025/source-identity.json \
  --out-dir /tmp/NEW-kv-only-finish-GPU-raw
```

Each library is loaded from its explicit URL; every required symbol/PSO/geometry is witnessed and missing functions fail rather than falling back. Phase1 submits114×3=342 CB, one dispatch each, and saves every case/provider's three complete post-buffers and all guards even if an earlier numerical comparison fails. Then it rejects before any timing. Gold is actual deployed383710…; probe-native and candidate both use the full-TU probe.

Mandatory raw gates:

- Gold and probe-native complete dummy Q, KV and whole window must match exactly.
- Candidate complete KV512F32 and window65536F32 must match gold exactly.
- Candidate complete dummy Q128KiB must equal original before bytes. It intentionally differs from native dummy output; native dummy output is not its expected result.
- All A5 guards; every non-current window slot unchanged; current window slot exactly equals complete KV; semantic KV/window finite/BF16.
- Complete input/args/output file sizes/SHA, loaded-library/PSO witnesses and successful342 completion statuses.

Raw names: `caseNNN.{dummy_q,kv,window}.before.raw`, `caseNNN.args136.raw`; post `phase1.caseNNN.providerP.{dummy_q,kv,window}.raw` plus `.prefix-guard.raw`/`.suffix-guard.raw`. Manifest holds file size/SHA, parity, timings and total CB count. Completion/error, loaded-path and PSO records are in complete stdout JSONL and must be retained alongside manifest/stderr. The protocol's broad phrase about all records in manifest is to be read with this explicit location clarification; no source or binary change was made after delivery.

No component timing or adoption is claimed. The optional tool timing switch remains outside this correctness-only protocol; only a separately frozen future protocol can authorize it. Both devices require independent judgments. Successful components would still need actual runtime adopted-call witnesses, real Q/KV/window/attention/final-logit/token/state gates and fullmodel performance; short cases alone cannot pass adoption.

## Work count and practical risk

For32 layers, the source deletes65,536 dummy pair rotations,65,536 precise sin plus65,536 precise cos calls and1MiB logical combined dummy read/write per token. All80 groups still dispatch, and no dispatch/barrier is saved. This is not a physical DRAM measurement. Q_B runs in the same cohort and may dominate, so even complete deletion could produce no visible wall-time gain. The old ledger's serialized0.265ms/29call qkv_finish figure is not a current upper bound. There is no defensible >2% fullmodel forecast from source alone.
