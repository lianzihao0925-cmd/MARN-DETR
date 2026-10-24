"""Read source only for C3 integration; no imports of torch/rfdetr, no GPU work.

Run from anywhere on AutoDL. Writes c3_context.zip beside this script. It
contains selected Python source and a SHA256 manifest, not datasets/weights.
The collector and zip may be deleted after their contents have been reviewed.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path("/root/autodl-tmp/project_marn-detr/rf-detr/src/rfdetr")


def collect(root, destination):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = set()
    for directory in ("datasets", "marn"):
        files.update((root / directory).rglob("*.py"))
    for name in (
        "config.py", "_namespace.py",
        "training/module_data.py", "training/module_model.py",
        "training/trainer.py", "training/param_groups.py",
        "training/callbacks/ema.py", "training/callbacks/best_model.py",
        "training/callbacks/coco_eval.py", "training/coco_map.py",
        "models/transformer.py", "models/lwdetr.py", "models/criterion.py",
        "models/weights.py",
    ):
        p = root / name
        if p.is_file():
            files.add(p)
    files.update(root.rglob("*ema*.py"))
    contents = {}
    for p in sorted(files):
        resolved = p.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Source symlink outside package: {p}")
        name = p.relative_to(root).as_posix()
        contents[name] = p.read_bytes()
    manifest = {
        "purpose": "C3: augmentation/object identity, EMA bank lifecycle, evaluation metrics",
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in contents.items()},
    }
    # Exclusive creation avoids silently replacing an earlier collection.
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr("rfdetr/" + name, data)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    return len(contents)


if __name__ == "__main__":
    destination = Path(__file__).resolve().with_name("c3_context.zip")
    count = collect(ROOT, destination)
    print("C3_CONTEXT_READY", destination)
    print("Python source files:", count)
    print("Archive bytes:", destination.stat().st_size)
    print("No training, dataset, model-weight or environment-variable collection performed.")
