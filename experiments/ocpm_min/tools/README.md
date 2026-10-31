# tools

Scripts used for the runs recorded in `../results/RESULTS.md`. They are kept
here so the study can be replayed on a GPU host; nothing in `rf-detr/src` is
modified by any of them.

| Script | Purpose |
|---|---|
| `preflight.sh` | eight-point gate: no stale process, idle GPU, free disk, weights md5, dataset counts, imports/CUDA, git state, frozen protocol. Exits non-zero on failure and is called by `run_abc.sh`. |
| `run_abc.sh` | the controlled comparison A → B → C on the full dataset. Skip-aware and retries a condition once. `STEPS=2000` switches to an exact step budget. |
| `run_lowdata.sh` | the low-data protocol: A / B / D over a per-class image subset. `LR=1e-4` reproduces the headroom-creating setting; `RHO` / `BETA_MAX` / `QPG` add the ablation knobs. |
| `run_ablation.sh` | thin wrappers over `run_lowdata.sh` for the three ablations: `wide` (residual budget ×10), `cover` (every query), `lr` (lr 1e-4). |
| `watchdog.sh` | read-only health monitor: samples step counter, GPU and disk every 60 s, writes `health.json` and `events.log`, restarts a stalled runner at most twice. |
| `gen_coco_c.py` | writes the COCO-C style corrupted validation sets (5 corruptions, one severity per invocation). `train2017` and the annotations are symlinked from the clean dataset, only `val2017` is rewritten. |
| `eval_coco_c.py` | evaluates a saved checkpoint on a corrupted validation root without retraining. |
| `progress.py` | live terminal dashboard for a running job: condition, step, epoch, step rate, GPU, per-epoch AP. |

## Typical sequence

```bash
# host preparation is outside this repository (see the top-level README)

bash experiments/ocpm_min/tools/preflight.sh

# 3-epoch controlled comparison, the number that became the project baseline
bash experiments/ocpm_min/tools/run_abc.sh

# a cheaper 2000-step comparison
STEPS=2000 OUT_ROOT=/root/autodl-tmp/project_marn-detr/ocpm_runs_2k \
  bash experiments/ocpm_min/tools/run_abc.sh

# low-data protocol, first without headroom then with lr 1e-4
bash experiments/ocpm_min/tools/run_lowdata.sh
LR=1e-4 bash experiments/ocpm_min/tools/run_lowdata.sh

# ablations
bash experiments/ocpm_min/tools/run_ablation.sh wide
bash experiments/ocpm_min/tools/run_ablation.sh cover

# corruption sets, then evaluate any checkpoint on them
python experiments/ocpm_min/tools/gen_coco_c.py
python experiments/ocpm_min/tools/eval_coco_c.py --condition A \
  --ckpt <path>/A_last.ckpt --corruption gaussian_noise \
  --root /root/autodl-tmp/datasets/coco_c --output <out>
```

## Notes that cost time to learn

- `--limit-batches` caps **one epoch**, so an exact step budget needs
  `--epochs 1 --limit-batches N`. Passing both a step cap and `epochs=3`
  multiplies them.
- Copy files to the host and check the md5 before extracting; a silently failed
  transfer looks exactly like a patch that did not take effect.
- Do not run CPU-heavy generation or evaluation alongside training. Four
  generation workers plus an evaluation pushed load to 22 on a 16-vCPU box and
  measurably slowed the training run that was being measured.
- The health monitor must read the same run root as the active runner. Pointing
  it at an old directory makes it mistake a finished run for a stalled one and
  restart the live one.
- `imagecorruptions` imports `pkg_resources`, which setuptools 84 removed; pin
  `setuptools<81` in the environment that generates the corruption sets.
