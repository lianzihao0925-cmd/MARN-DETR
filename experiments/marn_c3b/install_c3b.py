"""Version-guarded C3B installation; no torch import, model, or GPU work."""
import ast
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/project_marn-detr')
PACKAGE = ROOT / 'rf-detr/src/rfdetr'
HERE = Path(__file__).resolve().parent
MARKER = 'MARN_C3B_EMA_GATE_V1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patched(source):
    old = '''    if module.train_config.use_ema:
        raise ValueError("C2 short check uses use_ema=False; EMA bank policy must be wired before formal training")
'''
    new = '''    # MARN_C3B_EMA_GATE_V1: C2 stays guarded; C3B verifies its callback at fit start.
    if module.train_config.use_ema and getattr(module, "_marn_ema_policy", None) != "online_bank_exact_copy_v1":
        raise ValueError("MARN EMA requires the C3B MARNTrainingModule and MARNEMACallback")
'''
    if source.count(old) != 1 or MARKER in source:
        raise RuntimeError('Unexpected C2 EMA guard; no files changed')
    result = source.replace(old, new, 1)
    ast.parse(result)
    return result


def main():
    integration = PACKAGE / 'marn/integration.py'
    destination = PACKAGE / 'marn/lifecycle.py'
    if destination.exists() or MARKER in integration.read_text(encoding='utf-8'):
        raise RuntimeError('C3B already/partially installed. Do not reinstall; run check_c3b.py.')
    guard = json.loads((HERE / 'source_guard.json').read_text(encoding='utf-8'))
    for name, expected in guard.items():
        path = PACKAGE / name
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise RuntimeError(f'Source differs from reviewed context: {name}; no files changed')
    original = integration.read_bytes()
    changed = patched(original.decode('utf-8')).encode('utf-8')
    addition = (HERE / 'lifecycle.py').read_bytes()
    ast.parse(addition.decode('utf-8'))
    backup = ROOT / 'runs/source_backups' / ('marn_c3b_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(integration, backup / 'integration.py')
    (backup / 'manifest.json').write_text(json.dumps({
        'original_sha256': digest(original), 'patched_sha256': digest(changed),
        'new_file': str(destination), 'new_file_sha256': digest(addition),
    }, indent=2), encoding='utf-8')
    temp = integration.with_name(integration.name + '.marn_c3b.tmp')
    made_temp = made_addition = replaced = False
    try:
        with temp.open('xb') as f:
            made_temp = True
            f.write(changed)
        with destination.open('xb') as f:
            made_addition = True
            f.write(addition)
        os.replace(temp, integration)
        replaced = True
    except BaseException:
        if replaced:
            integration.write_bytes(original)
        if made_addition:
            destination.unlink()
        raise
    finally:
        if made_temp and temp.exists():
            temp.unlink()
    print('C3B_BACKUP', backup, flush=True)
    print('C3B_INSTALLED', flush=True)


if __name__ == '__main__':
    main()
