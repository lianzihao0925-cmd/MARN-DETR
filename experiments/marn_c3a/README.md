# MARN C3A：正式数据管线中的目标身份

本包承接已经通过的 C2 集成测试。目标是确认原始 COCO annotation ID 经过正式增强、原生批处理与解包后，仍与对应的框和类别一致，供后续原型写入去重使用。

这是 CPU 数据验收，不构造检测模型、不加载权重、不运行训练、不生成 checkpoint，也不评价 AP 或训练吞吐。脚本在导入 PyTorch 前隐藏 CUDA。AutoDL 实例本身的计费不由这个脚本控制。

## 放置与运行

将本目录的五个文件上传到：

`/root/autodl-tmp/project_marn-detr/experiments/marn_c3a/`

进入目录后先安装一次，再运行检查：

```bash
conda activate marn-detr
cd /root/autodl-tmp/project_marn-detr/experiments/marn_c3a
python install_c3a.py && python -u check_c3a.py
```

脚本自己保存日志，无需另加 tee。成功时打印 `C3A_DATA_PIPELINE_CHECK_COMPLETED`，最后打印 `C3A_REPORT_ZIP` 和实际报告路径。失败也会尽量生成报告 zip 并返回非零退出码。检查可重跑，每次新建带时间戳的输出，保留前次证据；安装不可重复覆盖。

上传打印出来的 `c3a_report_时间戳.zip` 即可。它包含检查结果、日志、数据配置，以及下一步 EMA/检查点适配需要的少量源码，不含数据集图片、标注 JSON 或模型权重。

## 各文件用途与删除时机

| 文件 | 用途、替换行为 | 保留/删除 |
|---|---|---|
| `install_c3a.py` | 一次性安装；校验上传源码的 SHA256，备份后修改 `src/rfdetr/datasets/coco.py`，新增 `src/rfdetr/marn/data.py` | 建议与实验记录保留；确认安装和检查通过后可删除实验目录中的副本 |
| `data.py` | MARN 专用 DataModule，沿用官方增强和 DataLoader，只为训练集启用原始 ID；安装器复制它到源码目录 | **安装后的源码副本必须保留**；实验目录内同名副本可在归档本包后删除 |
| `check_c3a.py` | 合成边界、真实增强成对比较、原生打包/解包及 4-worker DataLoader 验收 | 建议保留以便复现，体积很小 |
| `source_guard.json` | 安装器使用的版本校验清单，防止把补丁应用到另一份源码 | 与安装器一起保留；不单独删除后再安装 |
| `README.md` | 本阶段范围、操作和文件管理说明 | 建议保留 |

运行后新增：

| 位置 | 用途 | 删除时机 |
|---|---|---|
| `experiments/marn_c3a/check_outputs/时间戳/c3a_result.json` | 验收结论、检查数量、实际批处理类型 | 保留通过版本作为证据 |
| 同目录 `c3a_check.log` | 全部输出或失败堆栈 | 保留通过版本；失败问题解决后，旧失败版本可删 |
| 同目录 `resolved_data_config.json` | 本次 CPU 检查的配置 | 与通过结果一起保留 |
| `experiments/marn_c3a/c3a_report_时间戳.zip` | 供上传的一份报告及源码副本 | 我读取完且原始结果保留后可删 |
| `runs/source_backups/marn_c3a_时间戳/` | 修改前的 `coco.py` 和前后哈希 | 目前保留，供回滚使用 |

## 具体修改与验收范围

1. 在 `ConvertCoco.__call__` 中添加默认关闭的 `marn_keep_object_ids` 开关。启用时将原始 annotation ID 转为 int64，并复用原版的 crowd 过滤结果和退化框过滤掩码。不开启时不增加目标字段。
2. `MARNDataModule` 在官方 `setup('fit')` 之后、创建 workers 之前开启训练集开关。验证集保持原版；不改类别编号，不用过滤后的数组下标代替原始 ID。
3. 不改任何增强实现。合成测试覆盖同类不同目标、crowd、退化框、图外框、裁剪后空目标及大整数 ID。若环境已安装 Albumentations，也检查该后端的实际裁剪。
4. 从已完成 baseline 的 `runs/baseline_10ep_seed42/resolved_train_config.json` 读取数据协议，不猜默认值。CPU 检查仅调整 worker、设备、预取、固定内存与输出目录等资源字段；保留增强和 `pack_targets` 配置。
5. 对真实样本复制增强器内部状态并重设全局随机状态，逐项检查启用 ID 前后图像和所有原有目标字段完全相等；核对 ID 对应的原始图像、类别。再检查原生 collate 和 CPU transfer 的完整目标字典，以及 6 批、每批 16 图的多 worker 读取。
6. 当前支持 COCO 普通检测、CPU 增强、未填充的目标。其他数据类型、Kornia、target padding 需要另行适配，不静默接受。

本地已验证：上传包的哈希、脚本语法、实际源码补丁、备份一致性、重复安装拒绝、源码漂移拒绝，并通过 AST 比较确认原文件除新增 opt-in 逻辑外无其他修改。本地没有 PyTorch；数值与数据行为以服务器运行本检查的结果为准。

## 后续顺序

C3A 通过后进行 C3B：明确并实现 EMA 原型库策略、完整检查点和轻量权重的 MARN 重建，再用真实 DataLoader 做有界 GPU 短测。当前 C2 的 EMA 禁用保护仍保留；本包没有宣称解决 EMA。

正式机制实验随后比较原型内容与参数匹配的 query-only 对照。当前还没有进入 UAMR、MA-NAS 或新一轮长训练。

## 前一阶段文件管理

- 已完成使命的 `collect_c3_context.py`、上传用 `c3_context.zip` 可以删除；它们只是采集器和源码副本，不是运行依赖。
- C2 已通过后，旧的 `finish_c2.py`、空的 `c2_finish.log`、旧 `c2_check.log`、`check_outputs/c2_boundary.ckpt` 与 `check_outputs/c2_epoch_boundary.ckpt` 可以删除。两个 checkpoint 只是短测产物，不用于正式初始化。
- 保留 `c2_check_native.log`、通过的 `c2_result.json`、当前 `src/rfdetr/marn/`、源码备份、官方预训练权重，以及已完成的 formal baseline 目录。
- 不整目录删除 `runs/`、`experiments/` 或 `src/rfdetr/marn/`。
