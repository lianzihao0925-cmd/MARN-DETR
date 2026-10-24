# MARN-DETR

RF-DETR Small 上的 MARN 研究工程与实验归档：OCPM 跨图原型库、单次有界 MGQR 查询修正、检索监督和在线 UAMR 原型。

这是截至 2026-10-24 的工作区备份。当前实验没有建立稳定的精度收益；不宣称 SOTA、充分收敛或整个提纲已经实现。MA-NAS、T=2 和可选一致性项未完成。

## 目录

| 路径 | 内容 |
|---|---|
| `rf-detr/` | 服务器实际 RF-DETR 工作区，含 `src/rfdetr/marn/` 及检测器接入修改；保留上游许可证 |
| `experiments/` | C1–C6 等阶段的实际程序、配置、工程检查、评价和诊断结果 |
| `runs/` | 基线与四轮机制配对的配置、指标、审计和日志 |
| `docs/` | 研究进展、实验结论与文件管理说明 |
| `environment/` | 服务器安装版本和恢复环境说明 |
| `artifacts/manifest.json` | 每个权重/断点的原路径、大小、SHA256 和 Release 下载链接 |
| `artifacts/source_manifest.json` | 从服务器复制的普通文件校验清单 |
| `artifacts/upstream.json` | 上游来源、commit、分支和上传时工作区状态 |
| `tools/restore_weights.py` | 下载权重并恢复目录，写入前核验大小和 SHA256 |

## 权重与完整断点

大文件保存在本仓库的 [`archive-20261024` Release](https://github.com/lianzihao0925-cmd/MARN-DETR/releases/tag/archive-20261024)，Git 中不保存大权重。相同 SHA256 的文件只上传一次，清单记录全部原路径。工程短测权重也有保留时会归档，不能把它们当正式模型。

```bash
git clone https://github.com/lianzihao0925-cmd/MARN-DETR.git
cd MARN-DETR
# 先只恢复一个正式实验，避免一次下载全部历史权重：
python tools/restore_weights.py --destination . --match experiments/marn_c6/online_screen
# 若需要整个归档，去掉 --match：
# python tools/restore_weights.py --destination .
```

恢复工具复用已匹配文件；发现同名但不同内容会停止。正式 `.ckpt` 是完整训练状态；推理 `.pth` 必须按相应阶段的自定义加载函数读取，不能一律用原生 RFDETR 加载。C4 的格式为 `marn_mechanism_inference_v1`，后续阶段请阅读各自实验模块。

官方 Small 初始权重的归档位置为 `pretrained/rf-detr-small.pth`。原服务器上它位于 `/root/autodl-tmp/models/roboflow/rf-detr-small.pth`，MD5 为 `fb37061c1af7bace359c91b723a8d5c1`。恢复到不同机器后应配置 `RF_HOME`，并调整实验脚本中记录的绝对根路径。

## 环境与数据

已验收环境包括 Python 3.11、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Lightning 2.6.6 和 editable RF-DETR 1.12.0.dev0。具体安装版本以 `environment/` 为准。安装工程后需按对应阶段说明选择入口；历史安装补丁和训练脚本不能在已有实验目录盲目重跑。

```bash
# 在准备好的 CUDA/PyTorch 环境中：
python -m pip install -e ./rf-detr
```

原始 COCO train2017/val2017 图像与标注未重复上传，需从 COCO 官方来源准备至 `datasets/coco/`。本项目已有验收数据为 train 118287 张、val 5000 张、80 个稀疏 COCO 类别 ID。数据、缓存、登录凭据和 SSH 密钥不属于此仓库备份。

## 已有结果与边界

完整 COCO val2017、maxDets=100，以下 AP 为百分制。协议不同的结果单列，不能混为同一消融比较。

| 协议 | 结果 |
|---|---|
| 官方初始 RF-DETR Small | AP 52.9224 |
| 10 轮全模型基线微调，seed42 | 最佳 regular AP 52.7681 |
| 4 轮全模型 MARN / query-only 配对 | 未建立 MARN 的正向机制收益，详见逐轮原始结果 |
| 冻结底座，各 1 轮，query-only / MARN / 检索监督 | regular AP 52.9337 / 52.9209 / 52.9177 |
| 修正监督后的全模型各 1 轮，query-only / 随机路由 / 在线 UAMR | regular AP 52.7754 / 52.7729 / 52.7955 |

最后一组在线路由相对 query-only 仅 +0.020051 AP 百分点，仍低于官方初始结果，没有达到追加种子复验门槛。单种子短筛选不能证明稳定收益或普遍无效。工程检查通过只证明实现运行，不等于精度收益。

## 来源和许可

底座来自 [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)，本实验起点 commit 为 `eca736acab9f6fe93f2cbc82a1d9f6edfb53f0ea`。请保留并遵守 `rf-detr/` 中的上游 LICENSE、NOTICE 和相关依赖/权重条款；本归档不更改其许可。COCO 数据的许可和使用条件由原始发布方规定。
