"""Evaluate a saved checkpoint on COCO-C style corrupted validation sets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/project_marn-detr")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rf-detr" / "src"))

from experiments.ocpm_min import common  # noqa: E402


def build_module(condition: str, model_config, train_config, categories, quality_eta):
    from rfdetr.marn.lifecycle import MARNTrainingModule
    from rfdetr.training.module_model import RFDETRModelModule

    if condition == "A":
        return RFDETRModelModule(model_config, train_config, load_encoder_weights=False)
    module = MARNTrainingModule(
        model_config,
        train_config,
        class_ids=categories,
        marn_config=common.build_marn_config(quality_eta),
    )
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("A", "B"), required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/datasets/coco_c", help="corruption root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-eta", type=float, default=common.QUALITY_ETA)
    args = parser.parse_args()

    import torch

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset_dir = f"{args.root}/{args.corruption}"

    model_config = common.build_model_config()
    train_config = common.build_train_config(out, dataset_dir=dataset_dir)
    datamodule = common.build_datamodule(model_config, train_config)
    categories = common.class_ids(datamodule)
    module = build_module(args.condition, model_config, train_config, categories, args.quality_eta)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"missing keys: {len(missing)} e.g. {missing[:3]}", flush=True)
    if unexpected:
        print(f"unexpected keys: {len(unexpected)} e.g. {unexpected[:3]}", flush=True)
    module.eval()

    metrics = common.evaluate_coco(model_config, train_config, module, datamodule)
    result = {
        "condition": args.condition,
        "corruption": args.corruption,
        "ckpt": args.ckpt,
        "AP": metrics["AP"],
        "AP50": metrics.get("AP50"),
        "AP75": metrics.get("AP75"),
        "APs": metrics.get("APs"),
        "APm": metrics.get("APm"),
        "APl": metrics.get("APl"),
    }
    common.atomic_json(out / f"{args.corruption}.json", result)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
