# 持久归档与只读复核

两份成功归档的完整路径、文件数、字节数、SHA256和full-manifest身份见[durable-archives.json](durable-archives.json)。它们已同时保存于Mac Studio的`/Users/fangqingyuan/Benchmarks/myds/2026-10-26-metal-performance/`；M4补充包和两份full manifest在本机相同路径也保留。

| 包 | 内容 | bytes | SHA256 |
|---|---|---:|---|
| m2-evidence.tar.gz | M2的R2引用、R3诊断、匹配组件、两候选全模型raw、源码/制品和日志 |410470514|`8eed83847903254c6efcb6ac55bb842b7726c3edd5224a8038ebf507b95900b1`|
| m4-evidence.tar.gz | M4组件raw、host fixtures、545/548 seed、私有源码/bin与修复证明 |132645112|`e4a5dc46ee207633d1404432e66572f1c2c1a1e2879f6fe4998e39dc18e6a746`|

M2包有22705个文件/395个目录，逻辑文件字节11255161100；M4包有8828个文件/113个目录，逻辑文件字节470729171。逻辑字节包含重复payload，不代表物理I/O。只对不可变raw按完整SHA/size/mode作透明硬链接去重，源码和binary保留独立regular entries；空目录保留。校验器逐个重读所有stored regular payload的SHA，并验证每个hardlink指向此前同SHA/size的entry。未删除或修改原始证据。

GGUF模型的362GB payload没有打包；现有模型身份仍为stat与保存前缀，不声称整模型SHA。完整manifest列出各自roots和每个归档文件的路径/bytes/SHA。解压时保留两包的根目录名，用reader的`--path-map`把原`/tmp`、`/private/tmp`及local mirror映射到解压位置；Python core位于归档R3的`source/tools`，可显式加入`PYTHONPATH`。复核输出必须是新文件，重建或再跑benchmark必须使用新的输出目录，避免写入已冻结的hardlinked raw。

M2首次归档因Python tarfile自动识别原始硬链接，而清单未声明该自动link，验证拒收。失败包及错误保留，v2仅修归档entry的类型记录，重新打包/验证原数据，没有重新生成模型或GPU样本。[修复记录](archive-repair.json)。这属于归档完整性修复，未改任何候选、性能阈值或结果。
