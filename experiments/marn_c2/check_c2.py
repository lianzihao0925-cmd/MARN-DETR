"""Bounded real-model C2 check. Uses two training images, not a training benchmark."""
import os
os.environ.update({
    "RF_HOME": "/root/autodl-tmp/models/roboflow",
    "HF_HOME": "/root/autodl-tmp/cache/huggingface",
    "TORCH_HOME": "/root/autodl-tmp/cache/torch",
    "XDG_CACHE_HOME": "/root/autodl-tmp/cache",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
})

import copy
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from pytorch_lightning import Trainer, Callback, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pycocotools.coco import COCO
from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.marn.integration import MARNModelModule, MARNConfig, attach

ROOT = Path("/root/autodl-tmp/project_marn-detr")
OUT = Path(__file__).resolve().parent / "check_outputs"


class TinyTrainDataset(Dataset):
    def __init__(self):
        coco = COCO(str(ROOT / "datasets/coco/annotations/instances_train2017.json"))
        self.class_ids = sorted(coco.getCatIds())
        self.samples = []
        for image_id in sorted(coco.imgs):
            info = coco.imgs[image_id]
            anns = [a for a in coco.imgToAnns[image_id] if not a.get("iscrowd", 0)]
            if not anns:
                continue
            with Image.open(ROOT / "datasets/coco/train2017" / info["file_name"]) as img:
                img = img.convert("RGB")
                w, h = img.size
                tensor = TF.to_tensor(TF.resize(img, [512, 512]))
                tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)
            keep = (boxes[:, 2:] > boxes[:, :2]).all(-1)
            if not keep.any():
                continue
            boxes = boxes[keep]
            boxes = torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), -1)
            boxes /= torch.tensor([w, h, w, h])
            target = {
                "boxes": boxes,
                "labels": torch.tensor([a["category_id"] for a in anns])[keep],
                "marn_object_id": torch.tensor([a["id"] for a in anns])[keep],
                "image_id": torch.tensor([image_id]),
                "orig_size": torch.tensor([h, w]), "size": torch.tensor([512, 512]),
            }
            self.samples.append((tensor, target))
            if len(self.samples) == 2:
                break
        assert len(self.samples) == 2

    def __len__(self):
        return 12  # six batches / accumulation two = three updates at an epoch boundary

    def __getitem__(self, index):
        return copy.deepcopy(self.samples[index % 2])


def collate(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    else:
        assert a == b


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    seed_everything(42, workers=True)
    OUT.mkdir(exist_ok=True)
    ckpt = OUT / "c2_epoch_boundary.ckpt"
    if ckpt.exists():
        raise RuntimeError(f"Refusing to overwrite a previous native-boundary test: {ckpt}")
    mc = RFDETRSmallConfig(
        pretrain_weights="/root/autodl-tmp/models/roboflow/rf-detr-small.pth",
        num_classes=90, resolution=512, gradient_checkpointing=False, compile=False, cuda_graphs=False,
    )
    tc = TrainConfig(
        dataset_dir=str(ROOT / "datasets/coco"), dataset_file="coco", output_dir=str(OUT),
        batch_size=2, grad_accum_steps=2, epochs=2, multi_scale="off", amp_dtype="bf16",
        use_ema=False, lr=1e-5, lr_encoder=1.5e-5, warmup_epochs=0,
        num_workers=0, tensorboard=False, wandb=False, progress_bar=None,
    )
    data = TinyTrainDataset()
    loader = DataLoader(data, batch_size=2, shuffle=False, collate_fn=collate, num_workers=0)
    cfg = MARNConfig()
    plain = RFDETRModelModule(copy.deepcopy(mc), copy.deepcopy(tc)).cuda().train()
    images, targets = next(iter(loader))
    images = images.cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

    def evaluate_pair():
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = plain.model(images, targets)
            losses = plain.criterion(outputs, targets)
        return outputs, losses

    original = evaluate_pair()
    adapter = attach(plain, data.class_ids, cfg)
    adapter.enabled = False
    equal_tree(original, evaluate_pair())
    print("C2_DISABLED_FORWARD_LOSS_IDENTICAL", flush=True)
    adapter.enabled = True
    equal_tree(original, evaluate_pair())
    print("C2_EMPTY_BANK_FORWARD_LOSS_IDENTICAL", flush=True)
    # Populate a test-only bank from captured training features to exercise actual
    # retrieval at zero initialization. This is not a production bank commit.
    candidates = list(adapter.pending.values())
    assert candidates, "No reliable source candidates in the two training images"
    with torch.no_grad():
        for quality, label, feature, geometry, _ in candidates:
            slots = ((adapter.bank.classes == label) & ~adapter.bank.valid).nonzero().flatten()
            if slots.numel():
                slot = int(slots[0])
                adapter.bank.features[slot].copy_(feature)
                adapter.bank.geometry[slot].copy_(geometry)
                adapter.bank.quality[slot] = quality
                adapter.bank.valid[slot] = True
                adapter.bank.updated_step[slot] = 0
        adapter.committed_step.fill_(1)
    adapter.pending.clear()
    adapter.snapshot = None
    equal_tree(original, evaluate_pair())
    assert adapter.last_read["accepted"] > 0
    print("C2_NONEMPTY_ZERO_INIT_FORWARD_LOSS_IDENTICAL", flush=True)
    # This equivalence-only probe performed no optimizer update: discard its staging.
    adapter.pending.clear()
    adapter.capture = adapter.snapshot = None
    adapter.observed = False
    del original, adapter, plain
    torch.cuda.empty_cache()

    records = []

    class Audit(Callback):
        def on_train_start(self, trainer, module):
            assert module.training and module.model.training and module.criterion.training
            assert module.adapter.training
            ids = {id(p) for g in trainer.optimizers[0].param_groups for p in g["params"]}
            assert all(id(p) in ids for p in module.adapter.parameters() if p.requires_grad)
            self.previous_step = trainer.global_step
            self.saved_bank = {k: v.clone() for k, v in module.adapter.bank.state_dict().items()}
            self.backward_calls = 0

        def on_before_backward(self, trainer, module, loss):
            assert torch.isfinite(loss).all()
            self.backward_calls += 1
            # Every microbatch reads the same pre-update bank throughout accumulation.
            equal_tree(self.saved_bank, module.adapter.bank.state_dict())

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            a = module.adapter
            if trainer.global_step == self.previous_step:
                assert a.snapshot is not None
                equal_tree(self.saved_bank, a.bank.state_dict())
                return
            assert trainer.global_step == self.previous_step + 1
            assert int(a.committed_step) == trainer.global_step
            assert a.snapshot is None and not a.pending and a.capture is None
            r = dict(a.last_commit)
            print("C2_WINDOW", json.dumps({"batch_idx": batch_idx,
                  "epoch": trainer.current_epoch, "global_step": trainer.global_step,
                  "backward_calls": self.backward_calls,
                  "bank_microbatches": r["microbatches"]}), flush=True)
            assert r["microbatches"] == self.backward_calls == 2
            if r["eligible"]:
                assert r["unique_objects"] < r["eligible"], "Repeated groups/GT were not deduplicated"
            records.append(r)
            print("C2_COMMIT", json.dumps(r), flush=True)
            self.previous_step = trainer.global_step
            self.saved_bank = {k: v.clone() for k, v in a.bank.state_dict().items()}
            self.backward_calls = 0

        def on_train_epoch_end(self, trainer, module):
            # Native epoch-end callbacks will save before fit-loop teardown.
            if trainer.global_step == 3:
                assert trainer.current_epoch == 0
                trainer.should_stop = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(OUT), filename="c2_epoch_boundary", monitor=None,
        every_n_epochs=1, save_top_k=1, save_on_train_epoch_end=True,
        enable_version_counter=False, auto_insert_metric_name=False,
    )

    def trainer(audit, save=False):
        return Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=2,
                       accumulate_grad_batches=2, gradient_clip_val=0.1,
                       callbacks=[audit, checkpoint_callback] if save else [audit],
                       logger=False, enable_checkpointing=save,
                       enable_progress_bar=False, enable_model_summary=False,
                       limit_val_batches=0, num_sanity_val_steps=0, default_root_dir=str(OUT))

    model = MARNModelModule(copy.deepcopy(mc), copy.deepcopy(tc), class_ids=data.class_ids, marn_config=cfg)
    fit = trainer(Audit(), save=True)
    fit.fit(model, train_dataloaders=loader)
    assert int(model.adapter.bank.valid.sum()) > 0, "No reliable matches wrote to bank"
    assert model.adapter.refiner.out.weight.detach().abs().sum() > 0, "Real detection loss did not train MGQR"
    assert records[-1]["step"] == 3
    assert ckpt.is_file(), "Native epoch-end checkpoint was not written"
    boundary = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert boundary["epoch"] == 0 and boundary["global_step"] == 3
    print("C2_NATIVE_BOUNDARY_SAVED epoch=0 global_step=3", flush=True)
    saved = {k: v.detach().cpu().clone() for k, v in model.adapter.state_dict().items()}
    optimizer_saved = cpu_tree(fit.optimizers[0].state_dict())
    scheduler_saved = cpu_tree(fit.lr_scheduler_configs[0].scheduler.state_dict())
    equal_tree(optimizer_saved, cpu_tree(boundary["optimizer_states"][0]))
    equal_tree(scheduler_saved, cpu_tree(boundary["lr_schedulers"][0]))
    for key, value in saved.items():
        equal_tree(value, boundary["state_dict"]["model.transformer.decoder.marn." + key])
    del boundary
    # Lightning strategy teardown can move the model back to CPU after fit().
    # Standalone inference must explicitly restore the input/model device match.
    model.to(images.device).eval()
    bank_before = {k: v.clone() for k, v in model.adapter.bank.state_dict().items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.model(images)
    assert torch.isfinite(out["pred_logits"]).all() and torch.isfinite(out["pred_boxes"]).all()
    equal_tree(bank_before, model.adapter.bank.state_dict())
    print("C2_EVAL_BANK_FROZEN", flush=True)

    # Reload full training state; this tests boundary recovery, not exact sampler/RNG continuation.
    class RestoreAudit(Audit):
        def on_train_start(self, trainer, module):
            assert trainer.global_step == 3
            equal_tree(saved, {k: v.detach().cpu() for k, v in module.adapter.state_dict().items()})
            equal_tree(optimizer_saved, cpu_tree(trainer.optimizers[0].state_dict()))
            equal_tree(scheduler_saved, cpu_tree(trainer.lr_scheduler_configs[0].scheduler.state_dict()))
            super().on_train_start(trainer, module)
            print("C2_RESTORE_STATE_OK", flush=True)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if self.previous_step == 3 and self.backward_calls == 0:
                print(f"C2_RESUME_FIRST_BATCH epoch={trainer.current_epoch} batch_idx={batch_idx}", flush=True)
                assert trainer.current_epoch == 1 and batch_idx == 0

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, module, outputs, batch, batch_idx)
            if trainer.global_step == 4:
                trainer.should_stop = True

    del out, model, fit
    torch.cuda.empty_cache()
    restored = MARNModelModule(copy.deepcopy(mc), copy.deepcopy(tc), class_ids=data.class_ids, marn_config=cfg)
    restored.train()
    resume = trainer(RestoreAudit())
    resume.fit(restored, train_dataloaders=loader, ckpt_path=str(ckpt))
    assert int(restored.adapter.committed_step) == 4
    assert len(records) == 4 and records[-1]["microbatches"] == 2
    (OUT / "c2_result.json").write_text(json.dumps({"passed": True, "commits": records}, indent=2))
    print("C2_INTEGRATION_CHECK_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
