"""C2 single-GPU eager detection integration. No routing probes or EMA yet."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import PrototypeBank, PrototypeReader, BoundedMGQR


@dataclass(frozen=True)
class MARNConfig:
    layer: int = 0
    slots_per_class: int = 4
    queries_per_group: int = 30
    confidence_min: float = 0.2
    iou_min: float = 0.5
    similarity_update: float = 0.9
    feature_ema: float = 0.9
    warmup_steps: int = 1  # smoke-test setting; formal schedule is not yet frozen


def paired_iou(a, b):
    a0, a1 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b0, b1 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    inter = (torch.minimum(a1, b1) - torch.maximum(a0, b0)).clamp_min(0).prod(-1)
    return inter / (a[:, 2:].prod(-1) + b[:, 2:].prod(-1) - inter).clamp_min(1e-6)


class MARNAdapter(nn.Module):
    def __init__(self, dim, class_ids, num_queries, config=None):
        super().__init__()
        self.config = config or MARNConfig()
        self.num_queries = num_queries
        if not 0 < self.config.queries_per_group <= num_queries:
            raise ValueError("Invalid per-group query budget")
        self.bank = PrototypeBank(class_ids, dim, self.config.slots_per_class)
        self.reader = PrototypeReader(dim, class_ids)
        self.refiner = BoundedMGQR(dim)
        self.register_buffer("committed_step", torch.tensor(0, dtype=torch.long))
        self.enabled = True
        self.snapshot = None
        self.pending = {}
        self.capture = None
        self.last_commit = {}
        self.last_read = {}
        self.window_reads = 0
        self.observed = False
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {k: 0 for k in ("matches", "invalid", "low_confidence", "low_iou", "duplicates", "eligible")}

    def forward(self, raw, normalized, reference, heads, bbox_reparam):
        if not self.enabled:
            return raw
        if self.capture is not None:
            raise RuntimeError("Previous training capture was not consumed by criterion")
        step = int(self.committed_step)
        if self.training:
            if self.snapshot is None:
                self.snapshot = self.bank.snapshot(step)
                self.window_reads = 0
            snapshot = self.snapshot
            self.window_reads += 1
        else:
            if self.pending or self.snapshot is not None:
                raise RuntimeError("Evaluation must start at an optimizer boundary")
            snapshot = self.bank.snapshot(step)

        # Heads are supplied by the current LWDETR instance, never registered twice.
        # Class/geometry priors are observations; new key and residual modules learn.
        with torch.no_grad():
            cls_head, box_head = heads
            logits = cls_head(normalized.detach())
            delta = box_head(normalized.detach())
            if bbox_reparam:
                boxes = torch.cat((delta[..., :2] * reference[..., 2:] + reference[..., :2],
                                   delta[..., 2:].exp() * reference[..., 2:]), -1)
            else:
                boxes = (delta + reference).sigmoid()
        if self.training:
            self.capture = raw.detach()

        b, q, _ = raw.shape
        if q % self.num_queries:
            raise ValueError("Query count is not a multiple of official group size")
        groups = q // self.num_queries
        # Random routing is isolated from detector/dropout/DataLoader RNG.
        # Each repeated training group receives the same relative query selection.
        gen = torch.Generator(device=raw.device)
        gen.manual_seed(420001 + step * 1009 + (self.window_reads if self.training else 0))
        ranks = torch.rand(b, self.num_queries, generator=gen, device=raw.device)
        idx = ranks.topk(self.config.queries_per_group, dim=-1).indices
        one = torch.zeros(b, self.num_queries, dtype=torch.bool, device=raw.device)
        one.scatter_(1, idx, True)
        selected = one[:, None].expand(-1, groups, -1).reshape(b, q)
        if step < self.config.warmup_steps:
            selected = torch.zeros_like(selected)
        memory, summary, accepted = self.reader(raw, logits, boxes, selected, snapshot)
        self.last_read = {"step": step, "selected": int(selected.sum()),
                          "accepted": int(accepted.sum()), "groups": groups}
        return self.refiner(raw, memory, summary, accepted)

    @torch.no_grad()
    def observe(self, outputs, all_indices, suffixes, targets):
        if not self.enabled or not self.training:
            return
        if self.capture is None or self.snapshot is None:
            raise RuntimeError("Missing insertion-layer capture or step snapshot")
        self.observed = True
        index = suffixes.index(f"_{self.config.layer}")
        layer_outputs = outputs[index]
        indices = all_indices[index]
        raw = self.capture
        self.capture = None
        for b, ((src, dst), target) in enumerate(zip(indices, targets)):
            src, dst = src.to(raw.device), dst.to(raw.device)
            if "valid" in target:
                keep = target["valid"][dst].bool()
                src, dst = src[keep], dst[keep]
            if src.numel() == 0:
                continue
            if "marn_object_id" not in target:
                raise RuntimeError("COCO annotation IDs missing; use C2 checker dataset until formal transforms are wired")
            if len(target["marn_object_id"]) != len(target["labels"]):
                raise RuntimeError("Annotation IDs no longer align after transforms")
            labels = target["labels"][dst]
            features = raw[b, src].float()
            geometry = layer_outputs["pred_boxes"][b, src].float()
            conf = layer_outputs["pred_logits"][b, src, labels].float().sigmoid()
            iou = paired_iou(geometry, target["boxes"][dst].float())
            valid = (torch.isfinite(features).all(-1) & torch.isfinite(geometry).all(-1)
                     & (geometry[:, 2:] > 0).all(-1) & torch.isfinite(conf) & torch.isfinite(iou))
            self.stats["matches"] += src.numel()
            self.stats["invalid"] += int((~valid).sum())
            self.stats["low_confidence"] += int((valid & (conf < self.config.confidence_min)).sum())
            self.stats["low_iou"] += int((valid & (conf >= self.config.confidence_min) & (iou < self.config.iou_min)).sum())
            keep = valid & (conf >= self.config.confidence_min) & (iou >= self.config.iou_min)
            rows = keep.nonzero().flatten()
            quality = (conf * iou.clamp(0, 1)).sqrt()
            image_id = int(target["image_id"].reshape(-1)[0])
            # Batch transfer metadata; per-object CPU loop is a correctness-first writer.
            meta = torch.stack((target["marn_object_id"][dst[rows]], labels[rows], src[rows]), -1).cpu().tolist()
            quals = quality[rows].cpu().tolist()
            for r, (object_id, label, query_id), qual in zip(rows.tolist(), meta, quals):
                self.stats["eligible"] += 1
                key = (image_id, object_id)
                old = self.pending.get(key)
                if old is not None:
                    self.stats["duplicates"] += 1
                    if old[0] >= qual:
                        continue
                # Clone individual candidates; do not retain a full activation graph/storage.
                self.pending[key] = (qual, label, features[r].clone(), geometry[r].clone(), query_id)

    @torch.no_grad()
    def commit(self):
        if not self.enabled:
            return
        if not self.training or self.capture is not None or self.snapshot is None or not self.observed:
            raise RuntimeError("Commit requires a completed training loss/backward/optimizer window")
        bank = self.bank
        step = int(self.committed_step) + 1
        operations = {"fill": 0, "update": 0, "replace": 0}
        count = len(self.pending)
        # MARN_C3E_COMMIT_V1: discrete slot metadata crosses to the host once.
        # Keep every floating-point similarity/priority/update on its original
        # device with the original tensor shape and operation order. Candidates
        # remain sorted globally; only scalar decisions move to the host.
        classes = bank.classes.cpu().tolist()
        validity = bank.valid.cpu().tolist()
        slot_lists = {}
        for i, label in enumerate(classes):
            slot_lists.setdefault(label, []).append(i)
        slot_tensors = {}

        def argmax(values):
            # torch.argmax returns the first maximum, including the first NaN.
            for i, value in enumerate(values):
                if math.isnan(value):
                    return i
            return max(range(len(values)), key=values.__getitem__)

        for (image_id, object_id), (quality, label, feature, geometry, query_id) in sorted(self.pending.items()):
            indices = slot_lists.get(label, [])
            if not indices:
                raise ValueError(f"No bank slots for raw category {label}")
            if label not in slot_tensors:
                slot_tensors[label] = bank.classes.new_tensor(indices)
            slots = slot_tensors[label]
            valid_indices = [i for i in indices if validity[i]]
            slot, action = None, None
            if valid_indices:
                valid_slots = bank.classes.new_tensor(valid_indices)
                sim = F.cosine_similarity(bank.features[valid_slots].float(), feature[None], dim=-1)
                values = sim.cpu().tolist()
                best = argmax(values)
                if values[best] >= self.config.similarity_update:
                    slot, action = valid_indices[best], "update"
            if slot is None:
                empty = [i for i in indices if not validity[i]]
                if empty:
                    slot, action = empty[0], "fill"
                else:
                    normalized = F.normalize(bank.features[slots].float(), dim=-1)
                    sims = normalized @ normalized.T
                    sims.fill_diagonal_(-1)
                    redundancy = sims.max(-1).values.clamp_min(0)
                    age = (step - bank.updated_step[slots]).float() / self.reader.config.age_scale
                    priority = (1 - bank.quality[slots]) + 0.1 * age + 0.1 * redundancy
                    slot, action = indices[argmax(priority.cpu().tolist())], "replace"
            if action == "update":
                # EMA features/geometry/quality; provenance records latest contributor.
                momentum = self.config.feature_ema
                bank.features[slot].lerp_(feature, 1 - momentum)
                bank.geometry[slot].lerp_(geometry, 1 - momentum)
                bank.quality[slot].lerp_(torch.as_tensor(quality, device=feature.device), 1 - momentum)
            else:
                bank.features[slot].copy_(feature)
                bank.geometry[slot].copy_(geometry)
                bank.quality[slot] = quality
            bank.updated_step[slot] = step
            bank.valid[slot] = True
            validity[slot] = True
            bank.source_image[slot], bank.source_object[slot] = image_id, object_id
            operations[action] += 1
        self.committed_step.fill_(step)
        self.last_commit = {"step": step, "microbatches": self.window_reads,
                            "unique_objects": count, **self.stats, **operations,
                            "valid_slots": sum(validity),
                            "covered_classes": len({c for c, v in zip(classes, validity) if v})}
        self.pending.clear()
        self.snapshot = None
        self.observed = False
        self._reset_stats()


def attach(module, class_ids, config=None):
    config = config or MARNConfig()
    model = module.model
    decoder = model.transformer.decoder
    if config.layer != 0 or decoder.num_layers < 2:
        raise ValueError("C2 supports insertion after layer 0 with a decoder tail")
    if module.model_config.compile or module.model_config.cuda_graphs or module.model_config.gradient_checkpointing:
        raise ValueError("C2 requires eager execution without activation checkpointing")
    # MARN_C3B_EMA_GATE_V1: C2 stays guarded; C3B verifies its callback at fit start.
    if module.train_config.use_ema and getattr(module, "_marn_ema_policy", None) != "online_bank_exact_copy_v1":
        raise ValueError("MARN EMA requires the C3B MARNTrainingModule and MARNEMACallback")
    if model.segmentation_head is not None or getattr(model, "use_grouppose_keypoints", False):
        raise ValueError("C2 supports ordinary detection only")
    adapter = MARNAdapter(decoder.d_model, class_ids, model.num_queries, config)
    decoder.marn = adapter.to(next(model.parameters()).device)
    # Plain bound method; criterion neither owns nor serializes a second copy.
    module.criterion.marn_observer = adapter.observe
    return adapter


from rfdetr.training.module_model import RFDETRModelModule


class MARNModelModule(RFDETRModelModule):
    def __init__(self, model_config, train_config, *, class_ids, marn_config=None):
        super().__init__(model_config, train_config)
        attach(self, class_ids, marn_config)

    @property
    def adapter(self):
        return self.model.transformer.decoder.marn

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # Lightning executes training_step/backward in the closure. Commit only
        # after the actual optimizer call returns; accumulation-only batches do not call here.
        result = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.adapter.commit()
        return result

    def on_fit_start(self):
        if self.trainer.world_size != 1 or self.trainer.precision not in ("bf16-mixed", "32-true"):
            raise ValueError("C2 supports single-GPU BF16 or FP32 only (no skipped FP16 optimizer steps)")
        super().on_fit_start()

    def on_save_checkpoint(self, checkpoint):
        adapter = self.adapter
        if adapter.pending or adapter.snapshot is not None or adapter.capture is not None:
            raise RuntimeError("Save only after the optimizer and bank commit")
        super().on_save_checkpoint(checkpoint)
        checkpoint["marn_config"] = asdict(adapter.config)
        checkpoint["marn_class_ids"] = adapter.reader.class_ids.cpu().tolist()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("marn_config") != asdict(self.adapter.config):
            raise ValueError("MARN configuration differs from checkpoint")
        if checkpoint.get("marn_class_ids") != self.adapter.reader.class_ids.cpu().tolist():
            raise ValueError("MARN class IDs differ from checkpoint")
        super().on_load_checkpoint(checkpoint)
