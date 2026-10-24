"""Restore Release artifacts to their recorded project paths and verify SHA256.

Existing matching files are reused. A different existing file is never replaced
unless --overwrite is explicitly supplied. Only standard Python is required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError('Invalid artifact destination: ' + relative)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/manifest.json'))
    parser.add_argument('--destination', type=Path, default=Path('.'))
    parser.add_argument('--match', help='Restore paths containing this text only')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    assert manifest['schema'] == 'marn_github_artifacts_v1'
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for artifact in manifest['artifacts']:
        paths = [p for p in artifact['paths'] if not args.match or args.match in p]
        if not paths:
            continue
        for relative in paths:
            path = inside(root, relative)
            if path.is_file() and path.stat().st_size == artifact['bytes'] and sha256(path) == artifact['sha256']:
                print('VERIFIED_EXISTING', relative, flush=True)
                continue
            if path.exists() and not args.overwrite:
                raise FileExistsError('Different existing file: ' + str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.download-part')
            if temporary.exists():
                raise FileExistsError('Partial download exists; inspect or remove: ' + str(temporary))
            try:
                digest = hashlib.sha256()
                size = 0
                req = urllib.request.Request(artifact['download_url'], headers={'User-Agent': 'MARN-DETR-restore'})
                with urllib.request.urlopen(req, timeout=120) as response, temporary.open('xb') as output:
                    while block := response.read(8 * 1024**2):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                if size != artifact['bytes'] or digest.hexdigest() != artifact['sha256']:
                    raise ValueError('Artifact size or SHA256 mismatch: ' + relative)
                os.replace(temporary, path)
                print('RESTORED_AND_VERIFIED', relative, flush=True)
            except BaseException:
                # Preserve partial data for diagnosis; never replace a valid target.
                raise


if __name__ == '__main__':
    main()
