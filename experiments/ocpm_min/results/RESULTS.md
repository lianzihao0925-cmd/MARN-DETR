# Recorded results

> **Provenance warning.** These numbers were transcribed from the session log of the
> Remote-SSD instance (`qjAnnp`, `/root/autodl-tmp/project_marn-detr`) after that
> instance was released. The raw artifacts (`*_metrics.json`, `*_bank.json`,
> `*_quality.json`, `*_resources.json`, `read_records_*.jsonl`, `monitor.csv`,
> checkpoints, and the corrupted validation sets) were **not archived** before the
> release and are **lost**. What follows is a faithful transcription of the values
> that were printed and logged while the runs were live. The scripts that produced
> them live in `tools/`; re-running them on a GPU host regenerates the artifacts.

Protocol for every run below: RF-DETR Small, official pretrained weights
(md5 `fb37061c1af7bace359c91b723a8d5c1`), seed 42 unless stated, batch 16,
gradient accumulation 1, bf16, EMA 0.993, cosine schedule with 0.5-epoch warmup,
`eval_max_dets=100`, full `val2017` (5000 images) evaluation.

## 1. Controlled comparison on clean COCO (118,287 train images)

| Run | Steps | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| Official pretrained (reference) | 0 | 0.5292 | — | — | — | — | — |
| Historical 10-epoch baseline in the repo | 73,930 | 0.5277 | — | — | — | — | — |
| **A** (no OCPM) | 22,179 | **0.5278** | 0.7189 | 0.5692 | — | — | — |
| **A** (no OCPM) | 2,000 | **0.5281** | 0.7193 | 0.5700 | 0.2444 | 0.5082 | 0.7178 |
| **B** (OCPM, random read) | 2,000 | **0.5280** | 0.7191 | 0.5693 | 0.2439 | 0.5080 | 0.7182 |

- `B − A = −0.0001` AP at equal step count: no measurable effect.
- `A(2000) − A(22179) = +0.0003`: step count is not a variable in this protocol.
- A separate one-epoch A run (7,393 steps) scored AP 0.5288 / AP50 0.7196 / AP75 0.5691.

Protocol consistency across A and B: identical weight md5, identical
`train_config_sha256 = 096cfef40e685d13`, identical data-order fingerprint
`de8272bc9bfaf5dd`.

### Cost and memory (same 2,000 batches)

| | A | B |
|---|---|---|
| Training wall time | 541.7 s | 5,461.7 s (**10.1×**) |
| Seconds per batch | 0.219 | 2.685 |
| Peak allocated | 8.78 GiB | 8.83 GiB |
| Peak reserved | 10.16 GiB | 10.16 GiB |

The bank costs essentially no memory; the overhead is CPU time in the
per-candidate bookkeeping (`observe()` ≈ 2.4 s/step, `_record_quality_stats`
≈ 6,000 calls/step, `torch.isfinite` ≈ 20,000 calls/step).

### Bank behaviour after 2,000 steps (condition B)

| Metric | Value |
|---|---|
| Slots filled | 320 / 320 |
| Classes covered | 80 / 80 |
| Duplicate rate | 0 |
| Median prototype age | 392 steps |
| Cumulative fill / update / replace | 320 / 11,978 / **160,923** |
| Read events | 16,160 |
| Q_write mean (n = 1,287 samples) | 0.478 |

About 80 slots are replaced per step, so the bank carries almost no long-lived
information.

## 2. Low-data protocol (958 images, one image subset of 12 per class, 25 epochs = 1,500 steps)

### 2a. lr 1e-5 (no headroom)

| Arm | AP |
|---|---|
| A (no OCPM), seed 42 | 0.5222 |
| A, seed 43 | 0.5225 |
| B (random read), seed 42 | 0.5223 |
| B, seed 43 | 0.5220 |
| D (repository similarity read), seed 42 | 0.5220 |

Spread ≤ 0.0005. Training loss moved only 5.274 → 4.92, i.e. the model barely
left the pretrained optimum, so this setting creates no room for a mechanism to
recover anything.

### 2b. lr 1e-4 (3.6 AP of headroom created by the higher learning rate)

| Arm | AP | AP50 | AP75 | APs |
|---|---|---|---|---|
| A, seed 42 | **0.4862** | 0.6791 | 0.5195 | 0.2042 |
| A, seed 43 | **0.4865** | 0.6782 | 0.5202 | 0.2002 |
| B (random read), seed 42 | **0.4859** | 0.6789 | 0.5202 | 0.2014 |
| B, seed 43 | **0.4818** | — | — | 0.1990 |
| D (similarity read), seed 42 | **0.4863** | 0.6780 | 0.5204 | 0.1981 |
| B_wide (random, rho 0.5, beta_max 1.0) | **0.4875** | — | — | 0.2124 |
| D_wide (similarity, rho 0.5, beta_max 1.0) | **0.4829** | — | — | 0.2010 |

- `D − A = +0.0001`; `B − A = −0.0003`; widening the MGQR residual budget by 10×
  does not change the outcome (five arms span 0.0046 with no ordering).
- Seed spread: A = 0.0003, B = 0.0041. The mechanism adds variance (≈13×) without
  adding accuracy.
- Cost: A 0.215 s/step, B 3.47 s/step (16×), D 7.09–11.6 s/step (33–54×).

### 2c. What the read path actually retrieved

| Policy | Mean cosine(query, retrieved prototype) | Mean class compatibility |
|---|---|---|
| B (random valid slot) | **−0.075** | 0.012 |
| D (repository similarity) | **0.441** | 0.047 |

A genuinely unrelated retrieval and a genuinely matched retrieval differ by
0.0004 AP, so the content of the memory never reaches the prediction.

## 3. COCO-C (5 corruptions × severity 5 and 3, 5,000 images each)

Generated with `tools/gen_coco_c.py` (imagecorruptions, severity 5 and 3);
`train2017` and the annotations are symlinked from the clean dataset, only
`val2017` is replaced.

### 3a. Clean-trained checkpoint (condition A, 3 epochs, clean AP 0.5278), severity 5

| Corruption | AP | AP50 | AP75 | APs |
|---|---|---|---|---|
| gaussian_noise | 0.1200 | 0.1834 | 0.1252 | 0.0214 |
| motion_blur | 0.2116 | 0.3526 | 0.2169 | 0.0062 |
| contrast | 0.2566 | 0.3702 | 0.2765 | 0.0639 |
| jpeg_compression | 0.2956 | 0.4398 | 0.3128 | 0.0767 |
| saturate | 0.4522 | — | — | — |

### 3b. Same-source checkpoints (all from the lr 1e-4 low-data runs), severity 5

| Corruption | A | B (random) | D (similarity) | D − A |
|---|---|---|---|---|
| gaussian_noise | 0.0937 | 0.0968 | 0.0947 | +0.0009 |
| motion_blur | 0.1895 | 0.1864 | 0.1875 | −0.0020 |
| contrast | 0.2076 | 0.2188 | 0.2217 | +0.0141 |
| jpeg_compression | 0.2702 | 0.2657 | 0.2682 | −0.0020 |
| saturate | 0.4056 | 0.4070 | 0.4067 | +0.0011 |
| **mean** | **0.2333** | **0.2349** | **0.2358** | **+0.0024** |

### 3c. Same-source checkpoints, severity 3

| Corruption | A | B (random) | D (similarity) | D − A |
|---|---|---|---|---|
| gaussian_noise | 0.3392 | 0.3384 | 0.3357 | −0.0035 |
| motion_blur | 0.3100 | 0.3084 | 0.3093 | −0.0007 |
| contrast | 0.4445 | 0.4449 | 0.4453 | +0.0009 |
| jpeg_compression | 0.4190 | 0.4160 | 0.4174 | −0.0016 |
| saturate | 0.4758 | 0.4763 | 0.4755 | −0.0003 |
| **mean** | **0.3977** | **0.3968** | **0.3967** | **−0.0010** |

The severity-5 `contrast` cell (+0.0141) does **not** reproduce at severity 3
(+0.0009), and the placebo arm B also gains there, so it is a single-run
fluctuation rather than a property of the retrieval path.

## 4. Summary of the evidence matrix

| Setting | A vs D | D − A |
|---|---|---|
| Clean COCO, 22,179 steps | A 0.5278 | — |
| Clean COCO, 2,000 steps | A 0.5281 / B 0.5280 | −0.0001 (B) |
| Low data, lr 1e-5 | A 0.5222–0.5225 / B / D ≈ 0.522 | ≤0.0005 |
| Low data, lr 1e-4 | A 0.4862 / B 0.4859 / D 0.4863 | +0.0001 |
| Wide MGQR budget | B_wide 0.4875 / D_wide 0.4829 | −0.0033 |
| COCO-C severity 5 | mean 0.2333 / 0.2349 / 0.2358 | +0.0024 |
| COCO-C severity 3 | mean 0.3977 / 0.3968 / 0.3967 | −0.0010 |

Runs that were planned but never completed before the instance was released:
`D_s43` (stopped at step 749/1,500) and the coverage ablation `B_all`
(`queries_per_group=300`).

## 5. Reproducing

The scripts under `tools/` are the ones used for these runs:

```bash
# 1. build the environment (see the repository README for the pinned stack)
# 2. preflight: weights md5, dataset counts, imports, frozen config
bash tools/preflight.sh
# 3. controlled comparison on the full dataset
bash tools/run_abc.sh                 # A -> B -> C, 3 epochs by default
# 4. the low-data protocol on a per-class subset
bash tools/run_lowdata.sh             # A / B / D x seeds, 1,500 steps
# 5. ablations (wide residual budget, lr 1e-4, coverage)
bash tools/run_ablation.sh
# 6. corruption sets and their evaluation
python tools/gen_coco_c.py && python tools/eval_coco_c.py --help
```

`tools/watchdog.sh` is the health monitor used during the long runs: it is
read-only apart from `health.json` and `events.log`, and restarts a stalled
runner at most twice.
