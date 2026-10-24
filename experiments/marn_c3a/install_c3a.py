"""Back up and install the opt-in COCO-ID seam. No model construction or GPU use."""
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
MARKER = 'MARN_C3A_OBJECT_IDS_V1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patched(source):
    tree = ast.parse(source)
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConvertCoco']
    fn, = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__']
    lines = source.splitlines(keepends=True)
    part = ''.join(lines[fn.lineno - 1:fn.end_lineno])
    anchor = '        target["labels"] = classes\n'
    if part.count(anchor) != 1 or MARKER in source:
        raise RuntimeError('Unexpected ConvertCoco anchor; no source was written')
    addition = '''
        # MARN_C3A_OBJECT_IDS_V1: opt in before workers start; use the very same keep mask.
        if getattr(self, "marn_keep_object_ids", False):
            object_ids = [obj["id"] for obj in anno]
            if any(type(value) is not int for value in object_ids):
                raise TypeError("MARN requires integer original COCO annotation IDs")
            if len(set(object_ids)) != len(object_ids):
                raise ValueError("Duplicate annotation IDs within one COCO image")
            target["marn_object_id"] = torch.as_tensor(object_ids, dtype=torch.int64)[keep]
'''
    part = part.replace(anchor, anchor + addition, 1)
    result = ''.join(lines[:fn.lineno - 1]) + part + ''.join(lines[fn.end_lineno:])
    ast.parse(result)
    return result


def main():
    manifest = json.loads((HERE / 'source_guard.json').read_text(encoding='utf-8'))
    coco = PACKAGE / 'datasets/coco.py'
    destination = PACKAGE / 'marn/data.py'
    original = coco.read_bytes()
    if MARKER.encode() in original or destination.exists():
        raise RuntimeError('C3A already or partially installed. Do not reinstall; run check_c3a.py.')
    for name, expected in manifest.items():
        path = PACKAGE / name
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise RuntimeError(f'Source differs from the supplied C3 context: {name}; no files changed')
    new_source = patched(original.decode('utf-8')).encode('utf-8')
    new_data = (HERE / 'data.py').read_bytes()
    ast.parse(new_data.decode('utf-8'))
    if not (PACKAGE / 'marn/integration.py').is_file():
        raise RuntimeError('Install requires the existing C2 integration')
    backup = ROOT / 'runs/source_backups' / ('marn_c3a_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(coco, backup / 'coco.py')
    record = {'original_sha256': digest(original), 'patched_sha256': digest(new_source),
              'new_file': str(destination), 'new_file_sha256': digest(new_data)}
    (backup / 'manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary = coco.with_name(coco.name + '.marn_c3a.tmp')
    created_data = False
    created_temporary = False
    replaced_coco = False
    try:
        with temporary.open('xb') as f:
            created_temporary = True
            f.write(new_source)
        with destination.open('xb') as f:
            created_data = True
            f.write(new_data)
        os.replace(temporary, coco)
        replaced_coco = True
    except BaseException:
        if replaced_coco:
            coco.write_bytes(original)
        if created_data:
            destination.unlink()
        raise
    finally:
        if created_temporary and temporary.exists():
            temporary.unlink()
    print('C3A_BACKUP', backup, flush=True)
    print('C3A_INSTALLED', flush=True)


if __name__ == '__main__':
    main()
