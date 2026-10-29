# 独占测量窗口：谁负责停/恢复，谁负责核对

## 结论先说

**停服与恢复只有一个实现**：`tools/run_register_only_service_window.py`。
**只读核对**用 `tools/gpu_window.py`（它不停服）。

这两件事分开，是因为项目已经为「同一件事有两个真相源」付过太多代价。

## 为什么 §108 的窗口做不了

`start_fleet.sh` 拉起来的进程**不是停下的那个**：

- 生产是 `nice=5`，`start_fleet.sh` 不设 nice；
- 生产的 stdin 是 `fleet.conf` —— 来自 `start_fleet.sh` 里
  `while IFS='|' read ... done < "$CONFIG_FILE"` 这个循环的重定向，子进程继承它。

原来的窗口基类直接拒绝这种情况（`need(nice == 0, 'nondefault scheduling needs
explicit preservation implementation')`），于是真实模型独占测试根本开不了。

## 本次改动：基类支持任意 niceness 与原始 stdio

`tools/run_register_only_service_window.py`：

1. **preflight**：不再拒绝 `nice != 0`，改为捕获实际值；同时捕获 fd 0/1/2 的目标路径。
2. **restore**：
   - 用 `preexec_fn` 里的 `os.setpriority(PRIO_PROCESS, 0, nice)` 恢复**绝对**优先级。
     `nice -n N` 是**增量**，当窗口运行者本身 nice>0 时会得到 `N + 当前值`，不能用来表达「正好 N」。
   - stdin 重新指向原来的文件；stdout/stderr 以 append 方式打开原来的路径。
     **恢复出来的进程只有在它写的地方和原来一样时才等价**；证据改由路径与摘要记录。
3. **verify**：恢复后核对 nice 与 stdio 目标都与快照一致。

其余原有保证不变：精确 argv/环境/cwd、受保护制品逐个备份并 SHA 校验、有客户端连接时拒绝、
不 SIGKILL 原进程、失败时保留 lease。

## 只读核对：gpu_window.py

```bash
# 窗口前留档
python3 tools/gpu_window.py snapshot --port 8187 --json /tmp/before.json

# 窗口后核对（0=一致，1=不一致，2=前置条件缺失）
python3 tools/gpu_window.py verify --port 8187 --json /tmp/before.json \
  --residency /tmp/window.pm.txt

# 单独判驻留门（0=PASS，1=FAIL，2=没有样本）
python3 tools/gpu_window.py residency --residency /tmp/window.pm.txt
```

它记录 `pid`、`nice`、`argv`、`cwd`、`stdio[0..2]`、继承的 `MYDS_*`、
binary/metallib/config 的 SHA256、`/health`。

**两条容易写错的地方**：

1. `None == None` 看起来像通过。没有 `--config` 或找不到 `myds.metallib` 时 SHA 两边都是
   `None`，比较结果「一致」。这类字段报 `UNVERIFIED`，不报 PASS。
2. 工具自己注入的环境变量（`MYDS_FLEET_DIR` 等）不该参与比较，否则工具的存在本身
   会被报成恢复缺陷。

## 驻留门

`powermetrics --samplers gpu_power` 的 `GPU HW active residency` 中位数 < 95% 判 FAIL（§106.3）。
理由：**窗口跑完而 GPU 是空的，等于什么都没测。**

## 自测

- `gpu_window.py` 在替身服务上验过：snapshot 成功、verify PASS、篡改 config → exit 1、
  恢复 config → exit 0、驻留 16.42% → exit 1。
- `run_register_only_service_window.py` 通过了全部 8 个 window 单元测试中的 7 个；
  `test_run_output_a_fullmodel_window_cpu.py` 的 3 个 case 在**改动前就已失败**
  （stash 前后同为 3），与本改动无关，需单独处理。

**尚未对 macstudio 生产做过真实窗口** —— 那需要一次批准。
