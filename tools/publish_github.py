"""Publish the prepared snapshot and all recorded artifacts to the authorized repo.

Run only in the private AutoDL upload staging directory after official gh device
login. Resumes completed assets by name/digest, never force-pushes or clobbers
different assets, and does not modify/delete original project files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPOSITORY = 'lianzihao0925-cmd/MARN-DETR'


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging.resolve(strict=True)
    snapshot = staging / 'snapshot'
    gh = Path((staging / 'gh_path.txt').read_text()).resolve(strict=True)
    env = os.environ.copy()
    env.update(GH_CONFIG_DIR=str(staging / 'gh_auth'), GIT_TERMINAL_PROMPT='0',
               GH_PROMPT_DISABLED='1')
    manifest = json.loads((snapshot / 'artifacts/manifest.json').read_text())
    assert manifest['repository'] == REPOSITORY
    tag = manifest['release_tag']
    sources = json.loads((staging / 'snapshot_upload_sources.json').read_text())
    report_path = staging / 'publish_status.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'repository': REPOSITORY, 'release_tag': tag, 'stage': 'starting', 'uploaded': [],
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': False}
    started = time.monotonic()

    def save(stage=None):
        if stage:
            report['stage'] = stage
        report['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic(report_path, report)

    def run(command, *, cwd=None, allow_failure=False, timeout=180):
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, text=True)
        if result.returncode and not allow_failure:
            # Git/gh errors contain no token command-line arguments in this workflow.
            raise RuntimeError(str(command[0]) + ': ' + result.stderr[-3000:])
        return result

    def api(endpoint):
        return json.loads(run([gh, 'api', endpoint]).stdout)

    def git(*arguments, allow_failure=False):
        return run(['git', *arguments], cwd=snapshot, allow_failure=allow_failure)

    def release_assets():
        result = run([gh, 'api', '--paginate', '--slurp',
                      'repos/' + REPOSITORY + '/releases/' + str(report['release_id']) + '/assets?per_page=100'])
        return {asset['name']: asset for page in json.loads(result.stdout) for asset in page}

    def validate(asset, expected):
        assert asset['size'] == expected['bytes'], ('Wrong remote size', expected['asset_name'])
        value = asset.get('digest')
        assert value == 'sha256:' + expected['sha256'], ('Remote SHA256 unavailable or different', expected['asset_name'], value)
        return {'asset_name': expected['asset_name'], 'bytes': asset['size'],
                'sha256': expected['sha256'], 'github_digest': value,
                'asset_id': asset['id'], 'download_url': asset['browser_download_url'],
                'original_paths': expected['paths']}

    try:
        user = api('user')
        assert user['login'].lower() == REPOSITORY.split('/')[0].lower(), 'Authenticated GitHub account differs from requested owner'
        repo = api('repos/' + REPOSITORY)
        assert repo['full_name'].lower() == REPOSITORY.lower()
        assert repo.get('permissions', {}).get('push'), 'No repository push permission'
        report.update(authenticated_account=user['login'], public_repository=not repo['private'])
        if not (snapshot / '.git').exists():
            remote_refs = run(['git', 'ls-remote', 'https://github.com/' + REPOSITORY + '.git']).stdout
            assert not remote_refs.strip(), 'Repository no longer empty; inspect before import'
            git('init', '-b', 'main')
            git('config', 'user.name', user['login'])
            git('config', 'user.email', str(user['id']) + '+' + user['login'] + '@users.noreply.github.com')
            git('config', 'credential.https://github.com.helper', '')
            git('config', '--add', 'credential.https://github.com.helper', '!' + str(gh) + ' auth git-credential')
            git('remote', 'add', 'origin', 'https://github.com/' + REPOSITORY + '.git')
            git('add', '--force', '--all', '--', '.')
            staged = git('ls-files', '-z').stdout.split('\0')
            staged = [p for p in staged if p]
            assert len(staged) >= 970, len(staged)
            assert all(Path(name).suffix.lower() not in {'.pt', '.pth', '.ckpt', '.safetensors'} for name in staged)
            git('commit', '-m', 'Archive MARN-DETR source, experiment evidence and weight manifests')
            report['initial_commit'] = git('rev-parse', 'HEAD').stdout.strip()
            save('pushing_code')
        git('push', '--set-upstream', 'origin', 'main')
        report['code_commit'] = git('rev-parse', 'HEAD').stdout.strip()
        save('creating_draft_release')
        release = run([gh, 'api', 'repos/' + REPOSITORY + '/releases/tags/' + tag], allow_failure=True)
        if release.returncode:
            if git('rev-parse', '--verify', 'refs/tags/' + tag, allow_failure=True).returncode:
                git('tag', tag)
            git('push', 'origin', 'refs/tags/' + tag)
            notes = staging / 'release_notes.md'
            notes.write_text('''# AutoDL project and weight archive

Snapshot of the user-authorized MARN-DETR workspace, including baseline,
mechanism screening, frozen and online experiments, engineering checks, and
official Small initialization. See artifacts/manifest.json for all original
paths and SHA256 values; tools/restore_weights.py restores selected paths.

These are research artifacts, including negative results and short checks.
No stable accuracy gain, multi-seed result or SOTA is claimed. The upstream
RF-DETR license and the original weight terms remain applicable.
''')
            run([gh, 'release', 'create', tag, '--repo', REPOSITORY, '--draft',
                 '--verify-tag', '--title', 'MARN-DETR AutoDL archive 2026-10-24',
                 '--notes-file', notes])
            release_data = api('repos/' + REPOSITORY + '/releases/tags/' + tag)
        else:
            release_data = json.loads(release.stdout)
        report['release_id'] = release_data['id']
        report['release_url'] = 'https://github.com/' + REPOSITORY + '/releases/tag/' + tag
        save('uploading_artifacts')
        links = staging / 'asset_links'
        links.mkdir(exist_ok=True)
        remote = release_assets()
        pending, verified = [], []
        for artifact in manifest['artifacts']:
            if artifact['asset_name'] in remote:
                verified.append(validate(remote[artifact['asset_name']], artifact))
            else:
                pending.append(artifact)
        report['uploaded'] = verified
        save()

        def upload(artifact):
            source = Path(sources[artifact['sha256']]).resolve(strict=True)
            assert source.stat().st_size == artifact['bytes']
            h = hashlib.sha256()
            with source.open('rb') as stream:
                while block := stream.read(8 * 1024**2):
                    h.update(block)
            assert h.hexdigest() == artifact['sha256'], 'Artifact changed after snapshot preparation'
            link = links / artifact['asset_name']
            if not link.exists():
                link.symlink_to(source)
            assert link.resolve() == source
            # No --clobber: different existing assets are never removed.
            for attempt in range(3):
                result = run([gh, 'release', 'upload', tag, str(link), '--repo', REPOSITORY],
                             allow_failure=True, timeout=3600)
                if result.returncode == 0:
                    return artifact
                existing = release_assets()
                if artifact['asset_name'] in existing:
                    validate(existing[artifact['asset_name']], artifact)
                    return artifact
                if attempt == 2:
                    raise RuntimeError('Asset upload failed: ' + artifact['asset_name'] + ' ' + result.stderr[-1000:])
                time.sleep(3)

        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(upload, item): item for item in pending}
            for job in as_completed(jobs):
                artifact = job.result()
                actual = release_assets()[artifact['asset_name']]
                report['uploaded'].append(validate(actual, artifact))
                save()
                print('ARTIFACT_VERIFIED', len(report['uploaded']), '/', len(manifest['artifacts']),
                      artifact['asset_name'], flush=True)
        actual = release_assets()
        complete = [validate(actual[item['asset_name']], item) for item in manifest['artifacts']]
        assert len(complete) == len(manifest['artifacts'])
        atomic(snapshot / 'artifacts/published_assets.json', {
            'repository': REPOSITORY, 'release_tag': tag, 'assets': complete,
            'verified_by': 'GitHub asset size and server-reported SHA256 digest'})
        git('add', 'artifacts/published_assets.json')
        if git('diff', '--cached', '--quiet', allow_failure=True).returncode:
            git('commit', '-m', 'Record verified GitHub Release asset hashes and download mapping')
        git('push', 'origin', 'main')
        save('publishing_release')
        run([gh, 'release', 'edit', tag, '--repo', REPOSITORY, '--draft=false', '--latest=false'])
        released = api('repos/' + REPOSITORY + '/releases/tags/' + tag)
        assert released['draft'] is False
        remote_main = api('repos/' + REPOSITORY + '/git/ref/heads/main')['object']['sha']
        assert remote_main == git('rev-parse', 'HEAD').stdout.strip()
        report.update(passed=True, code_commit=remote_main, release_id=released['id'],
                      verified_assets=len(complete),
                      verified_bytes=sum(x['bytes'] for x in complete),
                      artifact_paths=sum(len(x['original_paths']) for x in complete),
                      session_wall_seconds=time.monotonic() - started)
        save('complete')
        print('GITHUB_ARCHIVE_COMPLETED', report['release_url'], flush=True)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error)[-3000:])
        save('failed')
        raise


if __name__ == '__main__':
    main()
