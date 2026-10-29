# 独占测量窗口：gpu_window.py

**用途**：停生产服务 → 在真实模型上测 → 恢复 → **证明恢复了**。

## 为什么需要它

`run_kprof_window.sh` 一类脚本停服后靠 `start_fleet.sh` 拉起来，但**拉起来的进程不是停下的那个**：

- 生产是 `nice=5`，`start_fleet.sh` 不设 nice；
- 生产的 stdin 是 `fleet.conf`，来自 `start_fleet.sh` 里
  `while IFS='|' read ... done < "$CONFIG_FILE"` 这个循环的重定向，子进程继承它。

账本 §108 记录过：因为这个，上一轮真实模型独占测试根本做不了。**恢复路径改变了机器，就不是测量，是副作用。**

## 三个子命令

```bash
# 只读：把当前状态抓成 JSON
python3 tools/gpu_window.py snapshot --port 8187 --json /tmp/before.json

# 对比当前进程与快照（0=一致，1=不一致）
python3 tools/gpu_window.py verify --port 8187 --json /tmp/before.json

# 完整窗口（必须在 macstudio 上跑，环境变量显式给出恢复路径）
MYDS_WINDOW_CONFIRM=1 \
MYDS_FLEET_DIR=$HOME/Projects/myfleet \
MYDS_FLEET_START=$HOME/Projects/myfleet/start_fleet.sh \
python3 tools/gpu_window.py run --port 8187 --json /tmp/window.json \
  --powermetrics /tmp/window.pm.txt \
  -- python3 tools/bench_env_ab.py --arm A:... --arm B:...
```

没有 `MYDS_WINDOW_CONFIRM=1` 时 `run` 拒绝停服。

## 它记录什么

| 项 | 为什么 |
|---|---|
| `pid` / `argv` | 停的是哪个进程 |
| `nice` | 生产是 5，`start_fleet.sh` 不设 ⇒ 必须显式恢复并核对 |
| `stdio[0..2]` | §108 记的坑：恢复后 fd 语义变了 |
| `cwd` | 相对路径的 `-m` / `--config` 指向什么 |
| 继承的 `MYDS_*` | 配置在文件里，但环境里还有 2 项（如 `MYDS_DSPARK`） |
| binary / metallib / config 的 SHA256 | 制品与配置身份 |
| `/health` | 服务确实活了 |

## 两条容易写错的地方

1. **`nice -n N` 是增量不是赋值。** 如果工具本身以 nice>0 运行，恢复出来的服务会是
   `N + 当前值`。所以先 `nice -n N` 启动，再 `renice` 强制到绝对值，最后核对。
2. **`None == None` 看起来像通过。** 没有 `--config` 或找不到 `myds.metallib` 时，
   SHA 比较会两边都是 `None`。这类字段报 `UNVERIFIED`，不报 PASS。

## 驻留门

`--powermetrics` 在窗口内采样 GPU 活跃驻留；中位数 < 95% 判 FAIL（§106.3）。
理由：窗口跑完而 GPU 是空的，等于什么都没测。
需要 `sudo -n powermetrics` 可用；不可用时记 `unavailable`，不假装测过。

## 自测

`gpu_window.py` 的停/恢复逻辑已在一个替身服务上验证过五条：干净周期 PASS、
verify PASS、篡改 config → exit 1、恢复 config → exit 0、无确认位拒绝执行。
驻留门用一个 2 秒空转命令确认会触发（16.42% → FAIL）。

**尚未对 macstudio 生产做过真实窗口** —— 那需要一次批准。
