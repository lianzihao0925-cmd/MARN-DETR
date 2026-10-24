"""Read-only, bounded performance diagnosis derived from accepted C3C harness.

No model-source installation, checkpoint writes, or long training. Instrumented
timings are diagnostic and must not be used as the C3C throughput acceptance.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from functools import wraps

os.environ.update({
    'RF_HOME': '/root/autodl-tmp/models/roboflow',
    'HF_HOME': '/root/autodl-tmp/cache/huggingface',
    'TORCH_HOME': '/root/autodl-tmp/cache/torch',
    'XDG_CACHE_HOME': '/root/autodl-tmp/cache',
    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
    'TOKENIZERS_PARALLELISM': 'false',
})
HERE = Path(__file__).resolve().parent
HARNESS_SHA = '0a3513a5429ca74908131a5addcd9fc5512737951b0b40cddd88b298ee3b7c9e'


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Unexpected diagnostic harness anchor: {old!r}')
    return text.replace(old, new, 1)


def build_harness():
    data = (HERE / 'benchmark_c3c.py').read_bytes()
    if hashlib.sha256(data).hexdigest() != HARNESS_SHA:
        raise RuntimeError('Diagnostic harness hash differs; no training started')
    code = data.decode('utf-8')
    code = once(code, 'MEASURE = 50', 'MEASURE = 10')
    code = once(code, 'REPETITIONS = 3', 'REPETITIONS = 1')
    code = once(code, "order = ['baseline', 'marn'] if repetition % 2 else ['marn', 'baseline']", "order = ['marn']")
    code = once(code, 'callbacks=[ema, timer]', 'callbacks=[ema, timer, _probe_callback(OUT)]')
    start = code.index("            left, right = pair['baseline'], pair['marn']")
    end = code.index("        REPORT['measurements_completed'] = True", start)
    code = code[:start] + (
        "        REPORT['summary'] = {'diagnostic_only': True, 'throughput_acceptance': False}\n"
    ) + code[end:]
    code = once(code, "'benchmark_outputs'", "'profile_outputs'")
    code = code.replace('c3c_result.json', 'c3d_result.json')
    code = code.replace('c3c_benchmark.log', 'c3d_profile.log')
    code = code.replace('c3c_report_', 'c3d_report_')
    code = code.replace('C3C_', 'C3D_')
    code = once(code, "for name in ('benchmark_c3c.py', 'source_guard.json'):",
                "for name in ('profile_c3d.py', 'benchmark_c3c.py', 'source_guard.json'):")
    compile(code, str(HERE / 'benchmark_c3c.py'), 'exec')
    return code


def install_probe():
    import torch
    from pytorch_lightning import Callback
    from rfdetr.marn.integration import MARNAdapter
    from rfdetr.marn.core import PrototypeReader, BoundedMGQR
    from rfdetr.marn.lifecycle import MARNEMACallback

    class Probe(Callback):
        def __init__(self, folder):
            self.folder = folder
            self.enabled = False
            self.samples = {}
            self.originals = []
            self.profiler = None
            self.closed = False
            self.restored = False
            self.profile_finished = False

        def wrap(self, cls, name, label):
            old = getattr(cls, name)
            self.originals.append((cls, name, old))

            @wraps(old)
            def call(*args, **kwargs):
                if not self.enabled:
                    return old(*args, **kwargs)
                started = time.perf_counter()
                try:
                    with torch.profiler.record_function(label):
                        return old(*args, **kwargs)
                finally:
                    self.samples.setdefault(label, []).append(time.perf_counter() - started)
            setattr(cls, name, call)

        def on_fit_start(self, trainer, pl_module):
            for cls, name, label in (
                (MARNAdapter, 'forward', 'MARN.adapter_forward_inclusive'),
                (PrototypeReader, 'forward', 'MARN.reader'),
                (BoundedMGQR, 'forward', 'MARN.refiner'),
                (MARNAdapter, 'observe', 'MARN.observe'),
                (MARNAdapter, 'commit', 'MARN.commit'),
                (MARNEMACallback, '_update_ema_for_step', 'MARN.ema'),
            ):
                self.wrap(cls, name, label)
            # attach() saved a bound observer before these class wrappers existed.
            # Rebind only this in-memory criterion and restore its original hook.
            self.originals.append((pl_module.criterion, 'marn_observer', pl_module.criterion.marn_observer))
            pl_module.criterion.marn_observer = pl_module.adapter.observe
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=10, warmup=2, active=5, repeat=1),
                on_trace_ready=self.write_profile, record_shapes=False,
                profile_memory=False, with_stack=False,
            )
            self.profiler.start()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.enabled = batch_idx >= 10

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.profiler.step()

        def write_profile(self, prof):
            averages = prof.key_averages()
            (self.folder / 'profile_cpu.txt').write_text(
                averages.table(sort_by='self_cpu_time_total', row_limit=80), encoding='utf-8')
            (self.folder / 'profile_cuda.txt').write_text(
                averages.table(sort_by='self_cuda_time_total', row_limit=80), encoding='utf-8')
            rows = []
            for event in averages:
                rows.append(dict(name=event.key, calls=event.count,
                                 cpu_total_us=event.cpu_time_total,
                                 self_cpu_total_us=event.self_cpu_time_total,
                                 device_total_us=getattr(event, 'device_time_total', 0),
                                 self_device_total_us=getattr(event, 'self_device_time_total', 0)))
            (self.folder / 'profile_events.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            self.profile_finished = True

        def close(self):
            if self.closed:
                return
            self.closed = True
            self.enabled = False
            try:
                if self.profiler is not None:
                    self.profiler.stop()
            finally:
                for cls, name, old in reversed(self.originals):
                    setattr(cls, name, old)
                self.restored = True
                # Inclusive CPU wall time includes GPU waits at existing syncs.
                # Nested scopes overlap and must not be summed as percentages.
                summary = {name: dict(calls=len(values), total_seconds=sum(values),
                                      mean_ms=1000 * sum(values) / len(values))
                           for name, values in self.samples.items()}
                (self.folder / 'profile_scopes.json').write_text(json.dumps(dict(
                    profiled_batches=5, scoped_batches=10, scopes=summary,
                    profile_finished=self.profile_finished, hooks_restored=self.restored,
                    interpretation='Inclusive host wall times; nested scopes overlap; instrumented speed is not acceptance.'
                ), indent=2), encoding='utf-8')

        def on_train_end(self, trainer, pl_module):
            self.close()

        def on_exception(self, trainer, pl_module, exception):
            self.close()

    return Probe


def main():
    # Resolve and validate all text transformations before importing GPU modules.
    code = build_harness()
    probe = install_probe()
    namespace = dict(__name__='__main__', __file__=str(HERE / 'benchmark_c3c.py'),
                     _probe_callback=probe)
    exec(compile(code, namespace['__file__'], 'exec'), namespace)


if __name__ == '__main__':
    main()
