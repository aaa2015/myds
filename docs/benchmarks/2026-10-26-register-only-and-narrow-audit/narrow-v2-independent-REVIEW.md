# Narrow weights CPU audit v2: independent saved-evidence review

Verdict: **PASS — CPU representation evidence and retained artifact consistency.**

The review reads only the saved report/archive/artifacts. It does not import or run the audit tool, reread the model, submit GPU work, or modify frozen evidence.

| Family | Actual tensors/type | Source bytes | BF16/code bytes | F32 scale bytes |
|---|---|---:|---:|---:|
| Router | 40 × F32, 5120×384 | 314,572,800 | 157,286,400 | 0 |
| Index-QB | 8 × F16, 1280×4096 | 83,886,080 | 41,943,040 | 163,840 |
| Engram projection | 2 × F16, 6144×25600 | 629,145,600 | 314,572,800 | 1,228,800 |
| Total | 50 | 1,027,604,480 | 513,802,240 | 1,392,640 |

All 50 expected names, actual types/shapes/spans and source/reconstructed SHA pairs match the frozen audit contract. The totals cover 435,159,040 elements and 348,160 32×32 blocks. All failure, nonfinite and reconstruction-difference counters are zero. Exhaustive classification includes 12,226,883 source F16 subnormals, 7,300 negative zeros and 7,369 positive zeros. Each projection's all-valid mask SHA was independently derived from its expected block count; family count and source-manifest hashes were independently recomputed. Ten deliberately corrupted report fixtures were all rejected.

The CPU-qualified payload is **490 MiB** of BF16/codes plus **1.328125 MiB** of scales: **515,194,880 B / 491.328125 MiB** total. This excludes allocator alignment/metadata and conversion staging. Retaining original matrices for prefill makes the narrow copy additional allocation; physical memory headroom was not measured.

The actual metadata binds exact `deepseek41`, `num_hidden_layers=40`, `hidden_size=5120`, index layers `[2,8,14,20,24,28,32,36]`, Engram layers `[1,14]`. Original GGUF is a single 362,142,629,888-byte file with 1,046 tensors, 67 metadata entries, 16 KiB alignment. Model stat identity is unchanged across the saved run. Whole-model SHA remains explicitly unavailable.

Every retained source, worker binary, original upstream patch, compiler command/log, artifact manifest, archive member and durable evidence copy was byte-checked. Actual worker is arm64 Mach-O dylib SHA `644f47df4fde60210b13b23128a074cdff5786a98a83d50115264418a600f4a8`, with only libSystem linked. The compiler is Xcode clang; the recorded SDKROOT is the CommandLineTools SDK, as preserved in the manifest.

**Limit:** original tensor/header raw bytes are not in the small evidence package, so this review does not independently rehash the model's payloads. It checks the pinned audit's recorded hashes and the source/artifact chain. CPU reversibility does not establish GPU GEMV arithmetic/reduction/rounding equivalence, runtime ownership/lifetime safety, performance, or production adoption.

Machine-readable proof: `review.json` (SHA `bbb3a5995817f59bb0304fa0584bee113bc42ba029ee5423a2334b9b25c6516d`). Independent saved-evidence reader: `review.py` (SHA `94b3f1d0e72a6aae181901744276126dfdcd71594f96c8b8f0238287a5ebe2de`).
