# 私有 indexer K-resident runtime 交付

本目录是 root 在 matched 正式组件门通过后明确授权的私有实现。没有改生产工作树，没有执行 GPU/remote/部署。原只读方案 `/tmp/myds-indexer-k-runtime-plan-20261025.md` 不变；当前 `build-manifest.json` 的 `narrow_scope` 是后续授权的收紧差异。

改动仅三份原文件（GPU C、唯一 publisher C call、GPU header）与三个新增文件（adapter、两份实际源 CPU fixture）。原 `indexscores_chunk` 和两个 public wrapper 的函数文本逐bytes不变。所有 shader、build_metallib.py 与 Metal backend 全bytes不变。55份 matched R3 的 src/include/third_party 与seed一致。seed来自当前dirty工作树完整构建源码导出，不冒称clean HEAD。

窄条件为 ratio2 / 512<n_comp<=16384 / 8<=rows<=2048 / end<=32768 / n_comp=end/2，ordinary !trace、active batch、非exclusive kprof、keywindow0/SGBAR0、scratch max/key/budget、完整native spans、writer-reader noalias与两个PSO preflight全部相交。ratio1、小/大表、其他正常不适用回原selector，不计fallback。eligible缺PSO/dispatch失败return0，publisher立即−1，禁止逐token重试。

strictflag是 MYDS_INDEXER_K_RESIDENT absent/0/1，默认0；每进程配置载入后第一次getter固定。六atomic请求/编码/fallback/失败/F32/BF16与现driver逐字段适配。requests含off/ineligible，encoded是active-batch接受编码而非GPU完成。旧publicAPI与decode不编码候选。

验证：48个mode1实际selector控制；17独立进程parser/default/window/非法actual call；实际publisher源fixture确认scorer失败−1且无重复RoPE/消费者，trace正确传参；原fullmodel driver12 CPU tests；fullmodel CPU完整build；mode0/1 --preflight（无model/device）；normal GPU/publisher预处理没有observer标识。fullbuild保留seed里7条既有warnings，GPU新增代码无warning。publisher fixture仅压制seed既有unused/indentation类warnings；普通完整build原样保留日志。

`runtime-core.patch` 可单独review接入；`runtime-with-host-fixtures.patch` 包括两份fixtures；`patch-byteproof.json` 在第三个临时副本正向应用后全548文件一致，反向后恢复545 seed文件。

私有 bin/bench_indexer_fullmodel 已CPU链接，bin/myds.metallib 是已批准full-TU probe的原bytes拷贝，源码prefix/append与已批准R2 full-build逐bytes一致。本轮未加载库，不是actual-loaded witness。root须在M2重新冻结真实build/argv/config/MYDS环境/loadedlib路径SHA，执行独立GPU数值与fullmodel协议。

目标CPU重建示例（仅build）：

```sh
make -C "$STAGE/source" -f tools/indexer_fullmodel.mk \
  BUILD_DIR="$STAGE/obj" BIN_DIR="$STAGE/bin" \
  INDEXER_ADAPTER_HEADER="$STAGE/source/tools/indexer_k_runtime_adapter.inc" \
  "$STAGE/bin/bench_indexer_fullmodel"
```

不要默认make/覆盖生产bin。运行使用空cwd和exe旁唯一批准metallib；普通build不带MYDS_INDEXER_CAPTURE。mode0/1必须独立进程；普通合法ineligible不应被改名为fallback。旧短请求不具备adoption时独立判适用性/安全，不能放宽mode1 first_callback encoded>0与decode编码0。数值门仍需actual默认exact→候选的完整ordered TopK/attention/logits/tokens/live/published state；不能因TILED→candidate scores0ULP就宣称默认exact分数全部一致。
