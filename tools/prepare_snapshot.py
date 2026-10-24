"""Prepare a new Git snapshot and a deduplicated artifact manifest, without publishing.

The source project is read only. The output must be a new directory outside the
project. Run inventory_project.py first and review its inventory JSON.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from inventory_project import classify, WEIGHTS


PATTERNS = {
    'private_key': re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'aws_access_key': re.compile(rb'\bAKIA[A-Z0-9]{16}\b'),
    'credential_in_url': re.compile(rb'https?://[^\s/:@]{2,}:[^\s/@]{6,}@'),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-tag', required=True)
    args = parser.parse_args()
    data = json.loads(args.inventory.read_text(encoding='utf-8'))
    assert data['repository'] == 'lianzihao0925-cmd/MARN-DETR'
    root = Path(data['project']).resolve(strict=True)
    out = args.output.resolve()
    assert not out.is_relative_to(root), 'Staging must stay outside the original project'
    assert not out.exists(), 'Output already exists; inspect before reusing'
    out.mkdir(parents=True)
    artifacts, copies, source_manifest, findings = {}, {}, [], []
    for item in data['files']:
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert classify(relative) is None
        source = Path(item.get('source', root / relative))
        assert not source.is_symlink()
        if 'source' not in item:
            assert source.resolve().is_relative_to(root)
        else:
            assert str(source) == '/root/autodl-tmp/models/roboflow/rf-detr-small.pth'
        before = source.stat()
        assert before.st_size == item['bytes'], 'Source changed after inventory: ' + str(relative)
        sha = digest(source)
        if item['kind'] == 'artifact':
            # A single GitHub Release asset is limited to 2 GiB.
            if before.st_size >= 2 * 1024**3:
                raise ValueError('Artifact needs explicit split handling: ' + str(relative))
            asset = artifacts.setdefault(sha, {'asset_name': sha + source.suffix.lower(),
                                               'bytes': before.st_size, 'sha256': sha, 'paths': []})
            asset['paths'].append(relative.as_posix())
            copies.setdefault(sha, str(source))
        else:
            contents = source.read_bytes()
            for name, pattern in PATTERNS.items():
                if pattern.search(contents):
                    # Record filename and category only; never print matched secrets.
                    findings.append({'path': relative.as_posix(), 'category': name})
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            target.chmod(before.st_mode & 0o777)
            assert hashlib.sha256(contents).hexdigest() == sha
            source_manifest.append({'path': relative.as_posix(), 'bytes': len(contents), 'sha256': sha})
        after = source.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(relative)
    if findings:
        report = out.parent / (out.name + '_review.json')
        report.write_text(json.dumps(findings, indent=2), encoding='utf-8')
        raise RuntimeError('Credential-pattern review required; inspect ' + str(report))
    repository = data['repository']
    records = list(artifacts.values())
    for record in records:
        record['download_url'] = ('https://github.com/' + repository + '/releases/download/' +
                                  args.release_tag + '/' + record['asset_name'])
    manifest = {'schema': 'marn_github_artifacts_v1', 'repository': repository,
                'release_tag': args.release_tag,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'artifacts': records,
                'original_path_policy': 'Paths are relative to the AutoDL project; external official weight is under pretrained/.'}
    artifact_dir = out / 'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (artifact_dir / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2), encoding='utf-8')
    (artifact_dir / 'upstream.json').write_text(json.dumps(data['git'], indent=2), encoding='utf-8')
    # Absolute source paths belong only to the private staging area, never to Git.
    private = out.parent / (out.name + '_upload_sources.json')
    private.write_text(json.dumps(copies, indent=2), encoding='utf-8')
    print(json.dumps({'snapshot': str(out), 'git_files': len(source_manifest),
                      'artifact_paths': sum(len(x['paths']) for x in records),
                      'unique_artifacts': len(records),
                      'unique_artifact_bytes': sum(x['bytes'] for x in records),
                      'credential_pattern_findings': 0}))


if __name__ == '__main__':
    main()
