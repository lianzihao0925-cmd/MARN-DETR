# C3D：定位 MARN 的训练开销

C3C 实测 Baseline 109.06 张/秒、MARN 75.78 张/秒。当前尚未达到 MARN ≥80 的要求。本包先定位瓶颈，不改变原型数量、查询预算、匹配、损失、写库策略或任何已安装源码。

远程放置目录：`/root/autodl-tmp/project_marn-detr/experiments/marn_c3d/`。本聊天取得 SSH 连接后由助手执行；连接尚未建立时，不代表已在服务器运行。

执行入口（只运行一次）：

```bash
conda activate marn-detr
cd /root/autodl-tmp/project_marn-detr/experiments/marn_c3d
python -u profile_c3d.py
```

沿用 C3C 数据与训练配置，仅做一轮 MARN：10 批预热、10 批诊断，共 320 张图；其中采集 5 批 CPU/CUDA profiler。读取本地官方权重，不下载，不遍历全量 COCO，不写训练权重。读取、候选收集、写库和 EMA 的计时采用进程内临时包装，退出时恢复，磁盘源码不变。Profiler 会干扰速度，不能用本次吞吐判断是否达到 80 张/秒。

| 文件 | 用途和覆盖行为 | 删除时机 |
|---|---|---|
| `profile_c3d.py` | 诊断入口，校验并复用 C3C 脚本；不安装源码 | 诊断验收后可删，建议保留 |
| `benchmark_c3c.py` | 原样复制的已通过 C3C 测速框架，由入口限制为 20 批 | 与入口一起保留；本目录不要直接运行它 |
| `source_guard.json` | 校验服务器为 C3C 对应源码，版本不一致时拒绝训练 | 与入口一起保留 |
| `README.md` | 本说明 | 建议保留 |
| `profile_outputs/时间戳/` | 每次新建，不覆盖历史结果；含日志、实际配置、结果及下列诊断文件 | 完成版本保留；失败版本确认不需排错后可删 |
| `profile_scopes.json` | 各 MARN 阶段主机耗时，含现有 GPU 同步等待；嵌套区间不能相加 | 保留 |
| `profile_cpu.txt`、`profile_cuda.txt`、`profile_events.json` | CPU/CUDA 算子耗时，用于区分计算、同步和大量小算子 | 保留 |
| `c3d_report_时间戳.zip` | 上传用报告，无模型/数据集 | 原始结果已保留且读取完成后可删 |

无 `.ckpt`、`.pth`、大体积 Chrome trace 输出。工具初始化失败仍可能没有完整报告，此时需保留终端错误。`C3D_THROUGHPUT_COMPLETED` 仅表示继承框架的诊断运行结束；`profile_scopes.json` 中 `profile_finished=true` 才确认采集完成。

本地只做语法、转换和报告逻辑检查；没有本地 PyTorch/CUDA，服务器实测尚待远程连接建立。
