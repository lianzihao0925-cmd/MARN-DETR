"""Read-only inventory for the user-authorized MARN GitHub backup.

Run on AutoDL with its existing Python. Does not read credential contents,
change the project, start training, or upload anything.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules',
             '.mypy_cache', '.pytest_cache', '.ruff_cache', '.cache'}
WEIGHTS = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx', '.engine'}
SENSITIVE_NAMES = {'.env', '.netrc', '.git-credentials', 'authorized_keys',
                   'id_rsa', 'id_ed25519', 'autodl_task_key',
                   'autodl_upload_key', 'autodl_known_hosts'}


def classify(relative):
    """Return only a reason; never read excluded file contents."""
    parts = relative.parts
    name = relative.name.lower()
    if parts and parts[0].lower() in {'datasets', 'dataset'}:
        return 'dataset'
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return 'cache_or_vcs'
    if name in SENSITIVE_NAMES or name.startswith('.env.'):
        return 'credential_or_environment'
    if name.endswith(('.pem', '.key', '.p12', '.pfx')):
        return 'credential_or_environment'
    if name.endswith(('.pyc', '.pyo')) or (name.endswith('.lock') and
                                         name not in {'uv.lock', 'poetry.lock', 'pdm.lock', 'pipfile.lock'}):
        return 'generated'
    if name.startswith('ssh_auth_debug') or name.startswith('autodl_bridge'):
        return 'connection_diagnostic'
    return None


def iter_files(root):
    for base, dirs, names in os.walk(root, followlinks=False):
        parent = Path(base)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and
                         not (parent == root and d.lower() in {'datasets', 'dataset'}) and
                         not (parent / d).is_symlink())
        for name in sorted(names):
            path = parent / name
            yield path, path.relative_to(root)


def git_read(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip() if result.returncode == 0 else None


def inventory(root):
    records, excluded, totals = [], [], Counter()
    for path, relative in iter_files(root):
        reason = classify(relative)
        if path.is_symlink():
            reason = 'symlink_review'
        if reason:
            excluded.append({'path': relative.as_posix(), 'reason': reason})
            continue
        size = path.stat().st_size
        kind = 'artifact' if path.suffix.lower() in WEIGHTS or size >= 80 * 1024**2 else 'git'
        totals[kind + '_files'] += 1
        totals[kind + '_bytes'] += size
        records.append({'path': relative.as_posix(), 'bytes': size, 'kind': kind})
    official = Path('/root/autodl-tmp/models/roboflow/rf-detr-small.pth')
    if official.is_file() and not official.is_symlink():
        records.append({'path': 'pretrained/rf-detr-small.pth', 'source': str(official),
                        'bytes': official.stat().st_size, 'kind': 'artifact'})
        totals['artifact_files'] += 1
        totals['artifact_bytes'] += official.stat().st_size
    return {'schema': 'marn_backup_inventory_v1',
            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'project': str(root), 'repository': 'lianzihao0925-cmd/MARN-DETR',
            'git': {'branch': git_read(root / 'rf-detr', 'branch', '--show-current'),
                    'head': git_read(root / 'rf-detr', 'rev-parse', 'HEAD'),
                    'status': git_read(root / 'rf-detr', 'status', '--short'),
                    'upstream_url': git_read(root / 'rf-detr', 'config', '--get', 'remote.origin.url')},
            'totals': dict(totals), 'files': records, 'excluded': excluded,
            'dataset_policy': 'COCO images and annotations stay on server; publish setup instructions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path,
                        default=Path('/root/autodl-tmp/project_marn-detr'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve(strict=True)
    result = inventory(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'totals': result['totals'],
                      'weights': [x for x in result['files'] if x['kind'] == 'artifact']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
