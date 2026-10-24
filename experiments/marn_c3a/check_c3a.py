"""CPU-only bounded check of IDs through the actual baseline data pipeline."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import copy
import hashlib
import json
import random
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/project_marn-detr')
PACKAGE = ROOT / 'rf-detr/src/rfdetr'
BASELINE = ROOT / 'runs/baseline_10ep_seed42'
HERE = Path(__file__).resolve().parent
OUT = HERE / 'check_outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
REPORT = {'passed': False, 'stage': 'imports', 'scope': 'CPU data path only; no training or EMA test'}


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def main():
    import numpy as np
    import torch
    from PIL import Image
    from rfdetr.config import RFDETRSmallConfig, TrainConfig
    from rfdetr.datasets.coco import ConvertCoco
    from rfdetr.datasets._torchvision import crop
    from rfdetr.datasets.transforms import AlbumentationsWrapper, alb
    from rfdetr.marn.data import MARNDataModule

    torch.set_num_threads(4)

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), (a.keys(), b.keys())
            for key in a:
                exact(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                exact(x, y)
        else:
            assert a == b

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    REPORT['stage'] = 'synthetic identity and crop'
    converter = ConvertCoco()
    converter.marn_keep_object_ids = True
    image = Image.new('RGB', (100, 100))
    # Same category, distinct IDs, plus crowd / degenerate / out-of-bounds boxes.
    # Large int64 IDs catch accidental conversion to float32 in augmentation.
    ids = [2**40 + 11, 2**40 + 12]
    anns = [
        {'id': ids[0], 'bbox': [5, 5, 10, 10], 'category_id': 1, 'area': 100, 'iscrowd': 0},
        {'id': 203, 'bbox': [5, 5, 10, 10], 'category_id': 1, 'area': 100, 'iscrowd': 1},
        {'id': 204, 'bbox': [30, 30, 0, 10], 'category_id': 1, 'area': 0, 'iscrowd': 0},
        {'id': ids[1], 'bbox': [70, 70, 10, 10], 'category_id': 1, 'area': 100, 'iscrowd': 0},
        {'id': 205, 'bbox': [110, 110, 10, 10], 'category_id': 1, 'area': 100, 'iscrowd': 0},
    ]
    raw = {'image_id': 7, 'annotations': anns}
    _, target = converter(image, copy.deepcopy(raw))
    assert target['marn_object_id'].tolist() == ids
    _, kept = crop(image, copy.deepcopy(target), top=0, left=0, height=40, width=40)
    assert kept['marn_object_id'].tolist() == [ids[0]]
    exact(kept['boxes'], torch.tensor([[5., 5., 15., 15.]]))
    _, empty = crop(image, copy.deepcopy(target), top=40, left=40, height=10, width=10)
    assert empty['marn_object_id'].shape == (0,) and empty['boxes'].shape == (0, 4)
    _, empty_input = converter(image, {'image_id': 7, 'annotations': []})
    assert empty_input['marn_object_id'].shape == (0,)
    if alb is not None:
        for x0, y0, x1, y1, expected in [(0, 0, 40, 40, [ids[0]]), (40, 40, 50, 50, [])]:
            wrapper = AlbumentationsWrapper(alb.Crop(x_min=x0, y_min=y0, x_max=x1, y_max=y1, p=1.0))
            _, result = wrapper(image, copy.deepcopy(target))
            assert result['marn_object_id'].tolist() == expected
            assert len(result['boxes']) == len(expected)
        REPORT['albumentations_crop'] = 'passed'
    else:
        REPORT['albumentations_crop'] = 'not installed; active pipeline still tested below'
    converter.marn_keep_object_ids = False
    _, plain = converter(image, copy.deepcopy(raw))
    exact(plain, {k: v for k, v in target.items() if k != 'marn_object_id'})
    REPORT['synthetic_identity'] = 'passed'
    print('C3A_SYNTHETIC_IDENTITY_OK', flush=True)

    REPORT['stage'] = 'load saved baseline data configuration'
    config_path = BASELINE / 'resolved_train_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Saved baseline configuration required: {config_path}; do not guess defaults')
    saved = json.loads(config_path.read_text(encoding='utf-8'))
    config = dict(saved)
    # Only resource/output fields differ; preserve augmentation and target packing.
    resource_overrides = dict(num_workers=4, persistent_workers=False, pin_memory=False,
                              prefetch_factor=2, accelerator='cpu', devices=1,
                              output_dir=str(OUT))
    config.update(resource_overrides)
    tc = TrainConfig(**config)
    mc = RFDETRSmallConfig(pretrain_weights=None, num_classes=90, resolution=512, device='cpu')
    assert tc.batch_size == 16 and tc.grad_accum_steps == 1
    assert getattr(tc.multi_scale, 'value', tc.multi_scale) == 'off'
    assert tc.square_resize_div_64 and tc.dataset_file == 'coco'
    assert Path(tc.dataset_dir).resolve() == (ROOT / 'datasets/coco').resolve()
    REPORT['baseline_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    REPORT['resource_overrides'] = resource_overrides
    REPORT['pack_targets'] = tc.pack_targets
    (OUT / 'resolved_data_config.json').write_text(tc.model_dump_json(indent=2), encoding='utf-8')
    dm = MARNDataModule(mc, tc)
    dm.setup('fit')
    train, val = dm._dataset_train, dm._dataset_val
    assert len(train) == 118287 and len(val) == 5000
    assert train.prepare.marn_keep_object_ids
    assert not getattr(val.prepare, 'marn_keep_object_ids', False)
    REPORT['transform_repr'] = repr(train._transforms)
    print('C3A_BASELINE_DATA_CONFIG_LOADED', flush=True)

    def audit_target(t, coco):
        object_ids = t['marn_object_id']
        assert object_ids.dtype == torch.int64
        assert object_ids.ndim == 1 and len(object_ids) == len(t['labels']) == len(t['boxes'])
        image_id = int(t['image_id'].reshape(-1)[0])
        values = object_ids.tolist()
        assert len(values) == len(set(values)), (image_id, values)
        for object_id, label in zip(values, t['labels'].tolist()):
            ann = coco.anns[object_id]
            assert ann['image_id'] == image_id and ann['category_id'] == label
            assert not ann.get('iscrowd', 0)
        assert torch.isfinite(t['boxes']).all()
        return len(values)

    REPORT['stage'] = 'paired real-data augmentation equivalence'
    counts = {'train_pairs': 0, 'val_pairs': 0, 'objects_checked': 0}
    for split, dataset in [('train', train), ('val', val)]:
        # Shallow-copy the dataset to share its read-only COCO index, not duplicate it.
        # Copy each transform tree from one state, including Albumentations' own RNG.
        indices = list(range(8)) if split == 'val' else sorted(set(
            list(range(8)) + [int(i * (len(dataset) - 1) / 23) for i in range(24)]))
        for index in indices:
            for repeat in range(2 if split == 'train' else 1):
                plain_ds, marn_ds = copy.copy(dataset), copy.copy(dataset)
                plain_ds.prepare, marn_ds.prepare = copy.deepcopy(dataset.prepare), copy.deepcopy(dataset.prepare)
                plain_ds.prepare.marn_keep_object_ids = False
                marn_ds.prepare.marn_keep_object_ids = True
                plain_ds._transforms = copy.deepcopy(dataset._transforms)
                marn_ds._transforms = copy.deepcopy(dataset._transforms)
                value = 42000 + index * 2 + repeat
                seed(value)
                plain_image, plain_target = plain_ds[index]
                seed(value)
                marn_image, marn_target = marn_ds[index]
                exact(plain_image, marn_image)
                exact(plain_target, {k: v for k, v in marn_target.items() if k != 'marn_object_id'})
                counts['objects_checked'] += audit_target(marn_target, dataset.coco)
                counts[split + '_pairs'] += 1
        print('C3A_PAIRED_SPLIT_OK', split, counts[split + '_pairs'], flush=True)
    REPORT['paired_augmentation'] = counts

    REPORT['stage'] = 'native collate and CPU transfer'
    # Exercise actual collate/packing and the exact transfer method training uses.
    # First use known samples and compare every target tensor before/after packing.
    known = [train[i] for i in range(4)]
    before = [copy.deepcopy(t) for _, t in known]
    packed = dm._collate_fn_train(known)
    REPORT['known_collated_target_type'] = type(packed[1]).__name__
    _, unpacked = dm.transfer_batch_to_device(packed, torch.device('cpu'), 0)
    exact(before, unpacked)
    # Force a uniform target shape so a packer cannot hide a field-loss bug behind
    # the fallback it uses for ragged target shapes; also exercise an empty target.
    empty_target = copy.deepcopy(known[0][1])
    for key in ('boxes', 'labels', 'area', 'iscrowd', 'marn_object_id'):
        empty_target[key] = empty_target[key][:0]
    for sample, label in [(known[0], 'uniform'), ((known[0][0], empty_target), 'empty')]:
        batch = [copy.deepcopy(sample), copy.deepcopy(sample)]
        original_targets = [copy.deepcopy(t) for _, t in batch]
        packed_batch = dm._collate_fn_train(batch)
        _, transferred = dm.transfer_batch_to_device(packed_batch, torch.device('cpu'), 0)
        exact(original_targets, transferred)
        REPORT[label + '_collated_target_type'] = type(packed_batch[1]).__name__
    del known, before, packed, unpacked, batch, packed_batch, transferred
    print('C3A_COLLATE_TRANSFER_EXACT_OK', flush=True)

    REPORT['stage'] = 'official multi-worker train loader'
    seed(42)
    loader = dm.train_dataloader()
    assert loader.batch_size == 16 and loader.num_workers == 4
    observed, total_objects, target_types = 0, 0, set()
    iterator = iter(loader)
    try:
        for batch_index in range(6):
            batch = next(iterator)
            target_types.add(type(batch[1]).__name__)
            _, targets = dm.transfer_batch_to_device(batch, torch.device('cpu'), 0)
            assert len(targets) == 16
            for target in targets:
                total_objects += audit_target(target, train.coco)
                observed += 1
            print(f'C3A_LOADER_BATCH_OK batch={batch_index + 1}/6 images={observed}', flush=True)
    finally:
        # Bounded diagnostic exits without waiting to traverse an entire epoch.
        shutdown = getattr(iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()
    REPORT['loader'] = {'images': observed, 'objects_checked': total_objects,
                        'workers': loader.num_workers, 'batch_size': loader.batch_size,
                        'collated_target_types': sorted(target_types)}
    assert observed == 96
    assert not torch.cuda.is_initialized(), 'CPU-only check unexpectedly initialized CUDA'
    REPORT['cuda_initialized'] = False
    REPORT['passed'] = True
    REPORT['stage'] = 'complete'
    print('C3A_DATA_PIPELINE_CHECK_COMPLETED', flush=True)


def package_report():
    # Small source-only supplement needed to finish EMA / lightweight checkpoints.
    # The previous context collector did not include these imported utilities.
    extras = ['utilities/tensors.py', 'utilities/state_dict.py', 'training/checkpoint.py',
              'marn/data.py', 'datasets/coco.py']
    report_zip = HERE / ('c3a_report_' + OUT.name + '.zip')
    with zipfile.ZipFile(report_zip, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                z.write(path, 'report/' + path.name)
        for name in extras:
            path = PACKAGE / name
            if path.is_file() and path.resolve().is_relative_to(PACKAGE.resolve()):
                z.write(path, 'source/' + name)
    return report_zip


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    original_out, original_err = sys.stdout, sys.stderr
    started = time.monotonic()
    with (OUT / 'c3a_check.log').open('w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            main()
        except BaseException as exc:
            REPORT['error'] = repr(exc)
            traceback.print_exc()
        finally:
            REPORT['elapsed_seconds'] = time.monotonic() - started
            (OUT / 'c3a_result.json').write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding='utf-8')
            sys.stdout, sys.stderr = original_out, original_err
    print('C3A_REPORT_ZIP', package_report(), flush=True)
    sys.exit(0 if REPORT['passed'] else 1)
