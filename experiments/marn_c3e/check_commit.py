"""Exact old/candidate commit comparison, before installing any model source."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
SOURCE = Path('/root/autodl-tmp/project_marn-detr/rf-detr/src/rfdetr/marn/integration.py')
EXPECTED = 'ab5d6d955f2eb845528749be973a73ce2769dfd8908610b8c76eefcb3a7f0474'


def load(name, path):
    spec = importlib.util.spec_from_file_location('rfdetr.marn.' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare(reference, candidate):
    assert reference.last_commit == candidate.last_commit
    assert reference.stats == candidate.stats
    assert not reference.pending and not candidate.pending
    assert reference.snapshot is None and candidate.snapshot is None
    for name, tensor in reference.state_dict().items():
        actual = candidate.state_dict()[name]
        assert tensor.dtype == actual.dtype and tensor.shape == actual.shape, name
        torch.testing.assert_close(tensor, actual, rtol=0, atol=0, equal_nan=True, msg=name)


def prepare(adapter, pending, microbatches):
    adapter.train()
    adapter.snapshot = adapter.bank.snapshot(int(adapter.committed_step))
    adapter.capture = None
    adapter.observed = True
    adapter.window_reads = microbatches
    adapter.pending = {key: (q, label, f.clone(), g.clone(), query)
                       for key, (q, label, f, g, query) in pending.items()}
    adapter.stats = dict(matches=200, invalid=3, low_confidence=2, low_iou=5,
                         duplicates=100, eligible=190)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED, 'Original source differs'
    old = load('integration_commit_reference', SOURCE)
    new = load('integration_commit_candidate', HERE / 'integration_candidate.py')
    torch.set_num_threads(4)
    torch.manual_seed(62026)
    assert torch.cuda.is_available()
    report = {'passed': False, 'checks': [], 'candidate_sha256': hashlib.sha256(
        (HERE / 'integration_candidate.py').read_bytes()).hexdigest(), 'original_sha256': EXPECTED}
    ids = [1, 3, 7]
    for device, autocast in [('cpu', False), ('cuda', False), ('cuda', True)]:
        reference = old.MARNAdapter(64, ids, 300).to(device)
        candidate = new.MARNAdapter(64, ids, 300).to(device)
        candidate.load_state_dict(reference.state_dict(), strict=True)
        for window in range(20):
            pending = {}
            for item in range(48):
                label = ids[item % len(ids)]
                feature = torch.randn(64, device=device)
                # Guaranteed updates interleaved with novel candidates/fills/replacements.
                if window and item % 4 == 0:
                    indices = (reference.bank.classes == label) & reference.bank.valid
                    if bool(indices.any()):
                        feature = reference.bank.features[indices][0].clone()
                geometry = torch.rand(4, device=device).clamp_min(.01)
                pending[(item % 5 + 1, window * 1000 + item)] = (
                    .5 + (item % 10) / 20, label, feature.float(), geometry.float(), item)
            for adapter in (reference, candidate): prepare(adapter, pending, 1 + window % 3)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                reference.commit()
                candidate.commit()
            compare(reference, candidate)
        # Exact threshold/tie cases, plus deliberately malformed valid NaN slots:
        # the optimization must retain torch.argmax's first-NaN behavior as well.
        for variant in ('tie', 'threshold', 'nan_valid', 'nan_invalid', 'empty'):
            with torch.no_grad():
                reference.bank.valid.fill_(variant != 'nan_invalid')
                reference.bank.features.zero_()
                reference.bank.features[:, 0] = 1
                reference.bank.geometry.fill_(.2)
                reference.bank.quality.fill_(.8)
                reference.bank.updated_step.fill_(0)
                if variant.startswith('nan'):
                    reference.bank.features[0, 0] = float('nan')
            candidate.load_state_dict(reference.state_dict(), strict=True)
            pending = {}
            if variant != 'empty':
                for i, cosine in enumerate((.8999999, .9, .9000001, 0.0)):
                    feature = torch.zeros(64, device=device)
                    if variant == 'tie': feature[0] = 1
                    else:
                        feature[0] = cosine
                        feature[1] = (1 - cosine**2)**.5
                    pending[(1, i)] = (.8, 1, feature, torch.full((4,), .3, device=device), i)
            for adapter in (reference, candidate): prepare(adapter, pending, 2)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                reference.commit()
                candidate.commit()
            compare(reference, candidate)
        # A strict roundtrip has no new persistent fields or changes in schema.
        restored = new.MARNAdapter(64, ids, 300).to(device)
        restored.load_state_dict(candidate.state_dict(), strict=True)
        for key, value in candidate.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0, equal_nan=True)
        report['checks'].append(dict(device=device, autocast_bf16=autocast,
                                     random_windows=20, candidates_per_window=48,
                                     edge_cases=['tie', 'threshold', 'nan_valid', 'nan_invalid', 'empty'],
                                     state_and_statistics_exact=True, strict_roundtrip=True))
    report['passed'] = True
    (HERE / 'commit_check_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('C3E_COMMIT_EXACT_CHECK_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
