# OCPM minimal comparable experiment harness

A self-contained harness that runs the MARN/OCPM ablation on COCO with RF-DETR
Small, compares the conditions under a frozen protocol, and records cost,
memory footprint and prototype-bank behaviour.

Nothing under `rf-detr/src` is modified: every intervention lives in this
directory and is installed at runtime (a monkeypatched reader for the random
read condition, callback-based probes, and CLI overrides on the shared config).

## Conditions

| Condition | Prototype bank | Read policy | Purpose |
|---|---|---|---|
| `A` | none | — | plain detector baseline |
| `B` | full OCPM | deterministic random valid slot | placebo: memory traffic with no retrieval quality |
| `C` | full OCPM | deterministic random valid slot | same path as `B` plus assertions that the new fields persist and that `write_quality` never enters maintenance ordering |
| `D` (CLI only) | full OCPM | repository similarity reader | the method's real read path: `--condition B --read-policy similarity` |

`--read-policy similarity` keeps the repository reader untouched; `random`
replaces `reader.forward` with a vectorised valid-slot draw so the placebo
costs a fraction of the learned reader.

## Protocol

The shared, frozen configuration is built by `common.build_train_config`: seed
42, batch 16, gradient accumulation 1, bf16, EMA 0.993, cosine schedule with
0.5-epoch warmup, `eval_interval=1`, `eval_max_dets=100`, full `val2017`
(5000 images) evaluation. Conditions differ only in the memory module.

`--epochs` and `--limit-batches` are independent: `--limit-batches` caps one
epoch, so a run of exactly N steps needs `--epochs 1 --limit-batches N`.

## Entry points

```bash
# one condition
python experiments/ocpm_min/run_condition.py \
  --condition B --output <dir> --read-policy similarity --epochs 1 --limit-batches 2000

# six-step smoke test with assertions on bank timing, checkpoint round trip and quality accounting
python experiments/ocpm_min/run_smoke.py --output <dir>

# pairwise comparison of finished conditions (checks weights, seed, step count, data order)
python experiments/ocpm_min/compare.py --runs <dir-with-A/B/C>
```

### Ablation switches

| Flag | Default | Meaning |
|---|---|---|
| `--dataset-dir` / `--expect-train-images` | COCO root / 118287 | run on a subset (used for the low-data protocol) |
| `--read-policy {random,similarity}` | `random` | read path |
| `--lr` / `--lr-encoder` | protocol values | learning-rate overrides |
| `--rho` / `--beta-max` | 0.1 / 0.5 | MGQR residual budget as a fraction of the query norm, and the gate ceiling |
| `--queries-per-group` | 30 | how many queries per group are refined (300 = every query) |

## Tools

| File | Purpose |
|---|---|
| `tools/preflight.sh` | eight-point gate (processes, GPU, disk, weights md5, dataset counts, imports, git state, frozen protocol) |
| `tools/run_abc.sh` | the controlled A → B → C comparison, skip-aware, one retry per condition |
| `tools/run_lowdata.sh` | the low-data protocol (A / B / D over an image subset) with lr and MGQR override knobs |
| `tools/run_ablation.sh` | the `wide` / `cover` / `lr` ablations |
| `tools/watchdog.sh` | read-only health monitor that restarts a stalled runner at most twice |
| `tools/gen_coco_c.py` | writes COCO-C style corrupted validation sets (5 corruptions × severity) reusing the clean `train2017` and annotations through symlinks |
| `tools/eval_coco_c.py` | evaluates a saved checkpoint on a corrupted validation root without retraining |
| `tools/progress.py` | live terminal dashboard for a running job (progress, step rate, GPU, per-epoch AP) |
| `results/RESULTS.md`, `results/recorded_results.json` | every measured value, with the provenance caveat below |

## Data availability

The raw artifacts of these runs (per-run `*_metrics.json`, `*_bank.json`,
`*_quality.json`, `*_resources.json`, the read-record logs, `monitor.csv` and the
checkpoints) lived on a rented GPU instance and were **not archived before that
instance was released**, so they are gone. `results/` transcribes every value that
was logged while the runs were live, and `tools/` contains the exact pipeline that
produced them, so the study can be replayed on a GPU host.

## Measured behaviour

Recorded on one RTX 5090 with the full protocol; numbers are single-seed and are
screening values, not claims of a stable improvement.

| Setting | Result |
|---|---|
| `A`, 3 epochs (22179 steps) | AP 0.5278 — reproduces the repository's historical 10-epoch baseline (0.5277) and the official pretrained weights (0.5292) |
| `A`, 2000 steps | AP 0.5281 — step count is not a variable here (0.0003 difference) |
| `A` vs `B`, 2000 steps | 0.5281 vs 0.5280 |
| `B` / `D`, 1500 steps, lr 1e-4 | 0.4859 / 0.4863 against `A` 0.4862 |
| wide residual budget (`rho=0.5`, `beta_max=1.0`) | `B_wide` 0.4875, `D_wide` 0.4829 — widening the bound by 10× does not change the outcome |
| COCO-C severity 5 (mean of 5 corruptions) | `A` 0.2333, `B` 0.2349, `D` 0.2358 |
| COCO-C severity 3 (mean of 5 corruptions) | `A` 0.3977, `B` 0.3968, `D` 0.3967 |

Cost, measured on the same batches: `A` 0.219 s/step, `B` 3.47 s/step (16×),
`D` 7.09–12 s/step (33–54×). Peak allocated memory is 8.78 GiB for `A` and
8.83–8.88 GiB with the bank, so the bank itself costs almost no memory; the
overhead is CPU time inside the per-candidate bookkeeping.

The read path was audited directly: the random placebo retrieves prototypes
with mean cosine −0.075 (unrelated), the similarity reader with mean 0.441,
and the two produce AP within 0.0008 of each other.

## Reproducing the environment

The runs used Python 3.11 with torch 2.7.1+cu128, torchvision 0.22.1+cu128,
pytorch-lightning 2.6.6, transformers 5.21.0, supervision 0.31.2, timm,
pycocotools, torchmetrics, faster-coco-eval, vernier, hotcoco and triton 3.3.1.
`tools/gen_coco_c.py` additionally needs `imagecorruptions`, `scikit-image` and
`setuptools<81` (newer setuptools removed `pkg_resources`, which
`imagecorruptions` imports).
