# C3C：baseline / MARN 实际训练吞吐对照

C3B 报告已通过：EMA 与库同步、两类推理权重的严格重载、完整 optimizer/scheduler/EMA/bank 恢复，以及恢复后的两次反传均通过。本包进入吞吐验收，不修改任何已安装源码。

## 放置与执行

将本包的 `marn_c3c` 文件夹上传到：

`/root/autodl-tmp/project_marn-detr/experiments/marn_c3c/`

执行一次即可，无需安装器、无需重新跑 C3A/C3B：

```bash
conda activate marn-detr
cd /root/autodl-tmp/project_marn-detr/experiments/marn_c3c
python -u benchmark_c3c.py
```

自动保存日志，最后打印 `C3C_THROUGHPUT_COMPLETED` 和 `C3C_REPORT_ZIP`。上传打印路径对应的 `c3c_report_时间戳.zip` 即可。失败也尽量生成报告包；上传报告后再处理，不必连续重跑。

## 测量内容

3 组配对，每组包含 baseline 和 MARN，运行顺序交替以减小缓存和温度偏差。各组种子分别是 42、43、44；组内使用相同种子。

每次都从已校验的本地官方 Small 权重重新初始化；MARN 从空库开始，在预热期间通过真实训练匹配写入。沿用 baseline 的固定 512、增强、batch=16、累计=1、BF16、8 workers、prefetch=2、pin/persistent 和 EMA 配置。MARN 原型库策略沿用已通过 C3B 的 `online_bank_exact_copy_v1`。

每次预热 10 批，计时 50 批，即计时 800 张图。共 6 次短测，包含预热总训练读取 5760 张图。每个短测只有 60 批，不遍历整个 COCO，也不保存训练结果。预期为数分钟级，实际时间随性能变化。

计时从第 10 批的 EMA 完成后开始，到第 60 批的 EMA 完成且 CUDA 同步后结束，包含：

- DataLoader 等待、传输和原生解包；
- 检测前向、官方匹配与 loss、反向、裁剪和优化器；
- MARN 的读取、去重、写入与 EMA 更新。

初始化、worker 启动、预热、验证、保存和退出清理不在计时内。因未计入验证和保存，报告中的单 epoch 时间只是训练部分的估算，不能直接当作大纲要求的完整 epoch 时间。

## 比较公平性与结果解释

- baseline 使用原版训练模块和原版 EMA 回调，MARN 使用通过 C3B 的训练模块及 EMA 回调。
- 两者使用相同的 MARNDataModule；baseline 忽略额外 ID 字段。通过 CPU 上的轻量目标字段哈希核对全部 60 批的目标数据、顺序与增强结果；前两个预热批次另外核对图像像素。全程像素哈希没有放进测速窗口。
- 两者都保留正常训练 loss 日志路径，关闭额外训练精度计算和逐步输出；测量期间不克隆整库、不做 C3B 那样的逐张量相等断言。
- 计时窗口仅增加小型 CPU 目标哈希及 loss 有限性累计；两组使用同一逻辑。这些开销计入结果。
- 报告记录原型库在计时开始和结束时的槽位/类别覆盖、接受读取数和真实写入数，确保不是在空库旁路上测出速度。短预热不能保证所有 80 类都已填满，因此还需结合实际覆盖解释结果。
- 报告输出三次均值、样本标准差、显存峰值和 MARN/baseline 速度比例。大纲目标分别是 baseline >=100 images/s、MARN >=80 images/s。
- `measurements_completed: true` 表示测速完成；`mean_meets_target` 才表示对应均值达到目标。未达标也会完整保留测量结果，不把它变成无报告的异常。
- 当前的单 epoch 短日程、warmup_steps=1 和少量更新仅用于性能诊断。这三组种子不是正式精度实验的三种子证据。

如果未达标，下一步根据报告优化真正的耗时部分。如果达标，再完成正式 best/last 和全量评价接入，准备 prototype 与参数匹配 query-only 对照。当前暂不进入 UAMR、MA-NAS 或长训练。

## 各文件的用途与删除时机

本包只有三个文件，都放到 `experiments/marn_c3c/`：

| 文件 | 用途和覆盖行为 | 保留/删除 |
|---|---|---|
| `benchmark_c3c.py` | 训练吞吐对照脚本；读取已安装源码和本地权重，不改模型源码，不覆盖已有结果 | 建议保留用于复测 |
| `source_guard.json` | 校验已通过 C3B 的版本，避免测到另一份代码；源码不一致时在训练前停止 | 与脚本一起保留 |
| `README.md` | 操作、计时口径与文件管理说明 | 建议保留 |

运行后生成：

| 位置 | 用途 | 删除时机 |
|---|---|---|
| `benchmark_outputs/时间戳/c3c_benchmark.log` | 完整运行日志 | 保留完成版本 |
| 同目录 `c3c_result.json` | 每次和汇总测速结果、槽位覆盖、目标是否达标 | 保留 |
| 同目录 `protocol.json` | 实际配置、计时范围与源码哈希 | 与结果一起保留 |
| `c3c_report_时间戳.zip` | 上传用的小报告包，不包含模型或数据集 | 我读取且原始结果保留后可删 |

不会生成 `.ckpt` 或 `.pth`。每次运行新建时间戳目录，不覆盖先前测速。

本地已检查语法、源码版本对应关系、计时批次边界和统计汇总逻辑。本地没有 PyTorch/CUDA，实际吞吐以服务器结果为准。

## C3B 已可删除的三个临时权重

这份 C3B 已确认通过：`20261016_213451_158182`。其临时权重合计约 **739.35 MiB**，不用于正式初始化。可以在服务器执行以下命令，只删除这三份文件：

```bash
python - <<'PY'
import json
from pathlib import Path

folder = Path('/root/autodl-tmp/project_marn-detr/experiments/marn_c3b/check_outputs/20261016_213451_158182')
result = json.loads((folder / 'c3b_result.json').read_text(encoding='utf-8'))
assert result.get('passed') is True and result.get('stage') == 'complete'
for name in ('c3b_epoch_boundary.ckpt', 'regular_test.pth', 'ema_test.pth'):
    path = folder / name
    if path.is_file():
        size = path.stat().st_size
        path.unlink()
        print(f'DELETED {name}: {size / 1024**2:.2f} MiB')
    else:
        print(f'ALREADY_ABSENT {name}')
PY
```

保留 C3B 的 `c3b_check.log`、`c3b_result.json`、`protocol.json`，以及已安装的 `src/rfdetr/marn/` 和源码备份。C3B 报告 zip 的上传副本现在可删。

官方权重、完整 formal baseline 结果、当前实现源码都继续保留，不删除整个 `runs/`、`experiments/` 或 `src/rfdetr/marn/`。
