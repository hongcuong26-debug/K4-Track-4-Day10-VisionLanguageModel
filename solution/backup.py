"""Copy completed checkpoints and evidence into a dedicated Drive folder."""
import argparse
import json
import shutil
import time
from pathlib import Path


def backup_once(checkpoint_root, destination):
    checkpoint_root, destination = Path(checkpoint_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for experiment in ('e1', 'e2', 'e3', 'e4_data_0.1'):
        for source in sorted((checkpoint_root / experiment).glob('*')):
            marker = source / 'run.json'
            if not source.is_dir() or not marker.exists():
                continue
            try:
                metadata = json.loads(marker.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if metadata.get('completed_steps', 0) <= 0:
                continue
            target = destination / 'runs' / experiment / source.name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.copying')
            shutil.copytree(source, temporary, dirs_exist_ok=True)
            temporary.rename(target)
            copied.append(str(target))
        log = checkpoint_root / experiment / 'train.jsonl'
        if log.exists():
            target = destination / 'runs' / experiment / log.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, target)
    # Small audit files are refreshed; raw predictions remain private to Drive.
    for folder in ('report', 'work/validation', 'work/lab_logs'):
        if Path(folder).exists():
            shutil.copytree(folder, destination / folder, dirs_exist_ok=True)
    for source in Path('work').glob('*.json*'):
        if source.is_file():
            target = destination / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-root', required=True)
    ap.add_argument('--destination', required=True,
                    help='A new dedicated folder for this run; do not reuse a previous run folder')
    args = ap.parse_args()
    destination = Path(args.destination)
    if destination.exists():
        raise FileExistsError('Use a new backup folder so old checkpoints cannot mask new ones')
    while True:
        try:
            copied = backup_once(args.checkpoint_root, destination)
            state = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         destination=str(destination), new_checkpoints=copied, status='ok')
            print(json.dumps(state), flush=True)
            Path('work/backup_status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        except Exception as error:
            print(f'Backup error: {error}', flush=True)
            Path('work/backup_status.json').write_text(
                json.dumps(dict(status='error', error=str(error)), indent=2), encoding='utf-8')
        progress = Path('work/lab_progress.json')
        if progress.exists():
            try:
                status = json.loads(progress.read_text(encoding='utf-8'))['status']
                if status in ('complete', 'needs_threshold_review', 'failed'):
                    break
            except (ValueError, KeyError):
                pass
        time.sleep(60)


if __name__ == '__main__':
    main()
