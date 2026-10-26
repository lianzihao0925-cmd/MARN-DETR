# OCPM 最小可比较实验报告

## 结论

本轮没有生成新的 A–H 实验结果。当前工作环境无法运行训练或评估：缺少 `torch`、`torchvision` 和 `pytorch_lightning`，没有可用的 `.ckpt`/`.pth` checkpoint，也没有 GPU 运行工具。因此不能把旧目录中的单 seed 结果包装成当前方案的 A–H 对照。

本轮未修改训练主配置，也未启动会改变训练状态的实验。

## 运行前检查

| 检查项 | 结果 |
|---|---|
| PyTorch | 不可用：`ModuleNotFoundError: No module named 'torch'` |
| torchvision | 不可用 |
| PyTorch Lightning | 不可用 |
| GPU 工具 | `nvidia-smi` 不可用 |
| 当前目录 checkpoint | `.ckpt=0`，`.pth=0` |
| 当前分支代码的可运行 smoke | 无法启动 |

## A–H 状态

| 条件 | 定义 | 当前状态 | 可用结果 |
|---|---|---|---|
| A | 官方检测器，无 OCPM | 未在当前分支重跑 | 旧 `baseline_official` 有 AP/AP50/AP75，但无类别 AP 和 bank 指标 |
| B | OCPM 随机读取，旧质量逻辑 | 只有旧实验目录结果 | 单 seed 旧 random arm，不能代表当前代码 |
| C | 随机读取，新字段但不启用新质量分 | 未运行 | 无 |
| D | 随机读取，启用 `Q_write` 写入/维护 | 未运行 | 无 |
| E | 关闭读取但保留 bank 写入 | 未运行 | 无 |
| F | 读取 shuffled prototype | 未运行 | 无 |
| G | UAMR 关闭 | 旧 random/query-only 可作参考 | 不是当前新标签实验 |
| H | 使用本轮配对收益标签的 UAMR | 未运行 | 旧 online arm 使用旧实验 probe，不等同于本轮标签 |

## 旧结果，仅作背景

以下结果来自 `experiments/marn_c6/online_screen`，是 seed 42 的旧单次短期实验，代码 hash、训练实现和当前分支不同。它们不能用于确认新字段或新 `Q_write` 的收益。

| 旧 arm | AP | AP50 | AP75 | APs | APm | APl | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| official baseline | 0.529611 | 0.720446 | 0.572168 | — | — | — | 旧基线文件没有尺度 AP |
| query_only regular | 0.527754 | 0.719443 | 0.568947 | 0.243557 | 0.508560 | 0.716766 | 无 bank 写入 |
| random regular | 0.527729 | 0.719247 | 0.567928 | 0.247832 | 0.507696 | 0.718720 | 旧随机 arm |
| online regular | 0.527955 | 0.720169 | 0.568557 | 0.245690 | 0.508352 | 0.717439 | 旧 UAMR probe |

旧 random arm 的末次 bank 审计记录为 `valid_slots=320`、`covered_classes=80`、`matches=1365`、`eligible=781`、`update=7`、`replace=57`；旧 online arm 为 `valid_slots=320`、`covered_classes=80`、`update=3`、`replace=63`。这些只说明旧试验的 bank 状态，不包含原型年龄分布、重复率或完整读取次数。

旧 online arm 的 probe 记录为 `probe_examples=6896`、`probe_positive=3459`，但没有完整的正/零/负比例；其标签生成代码位于旧实验目录，不能作为本轮新 `uamr.py` 的验证结果。

## 需求指标缺口

当前没有可合法填写的以下指标：

- 类别级 AP；
- C/D/E/F 的 AP、尺度 AP 和类别覆盖对照；
- 原型重复率；
- 原型年龄分布；
- 每个条件的完整读取次数；
- 本轮配对标签的正、零、负收益比例；
- D/E/F 的额外计算成本；
- 多 seed 方差和置信区间。

这些字段不能从 bank 填充率、类别匹配率或相似度推断。

## 是否允许进入正式 smoke

当前不允许把 A–H 结果写成实验结论。恢复运行所需条件是：

1. 在带 PyTorch、数据集和 GPU 的环境中提供同一初始 checkpoint；
2. 固定官方训练主配置、数据顺序、seed、评估 backend 和训练步数；
3. 为 A–H 分别记录完整 AP、类别 AP、bank、读取、收益和耗时日志；
4. 至少重复多个 seed 后再讨论稳定收益；
5. H 必须调用本轮正式配对收益标签，且相关性不足时保持路由关闭。
