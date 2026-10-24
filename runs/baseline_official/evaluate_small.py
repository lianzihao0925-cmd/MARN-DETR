import os

os.environ["RF_HOME"] = "/root/autodl-tmp/models/roboflow"
os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/cache/torch"
os.environ["XDG_CACHE_HOME"] = "/root/autodl-tmp/cache"

import json
from pathlib import Path
from rfdetr import RFDETRSmall


def main():
    output = Path(
        "/root/autodl-tmp/project_marn-detr/runs/baseline_official"
    )
    output.mkdir(parents=True, exist_ok=True)

    model = RFDETRSmall(
        pretrain_weights=(
            "/root/autodl-tmp/models/roboflow/rf-detr-small.pth"
        ),
        device="cuda",
    )

    metrics = model.evaluate(
        split="val",
        dataset_file="coco",
        dataset_dir="/root/autodl-tmp/project_marn-detr/datasets/coco",
        batch_size=2,
        num_workers=2,
        device="cuda",
        output_dir=str(output),
    )

    print("\n=== 完整验证结果 ===", flush=True)
    print(metrics, flush=True)
    with (output / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("A4_EVALUATION_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
