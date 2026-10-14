"""Persist real result files in notebook output, without Drive filesystem access."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

RUNS = ('e1', 'e2', 'e3', 'e4_data_0.1')


def emit_snapshot(output_root):
    files = {}
    for pattern in ('report/evidence/*.json', 'work/*.json*', 'work/validation/*.json'):
        for path in Path('.').glob(pattern):
            if path.is_file() and path.suffix in ('.json', '.jsonl'):
                files[path.as_posix()] = path.read_bytes().decode('utf-8')
    for name in RUNS:
        for filename in ('train.jsonl', 'final/run.json'):
            path = Path(output_root) / name / filename
            if path.exists():
                files[f'work/runs/{name}/{filename}'] = path.read_bytes().decode('utf-8')
    for path in Path('report').glob('*.json'):
        files[path.as_posix()] = path.read_bytes().decode('utf-8')
    payload = json.dumps(dict(version=1, files=files), ensure_ascii=False).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(gzip.compress(payload, mtime=0)).decode('ascii')
    Path('work/portable_results.b64').write_text(encoded, encoding='ascii')
    print(f'BEGIN_DAY10_SNAPSHOT sha256={digest} bytes={len(payload)}', flush=True)
    print(encoded, flush=True)
    print('END_DAY10_SNAPSHOT', flush=True)
    return digest


def restore_snapshot(encoded, expected_sha256, output_root='work/runs'):
    payload = gzip.decompress(base64.b64decode(encoded.strip(), validate=True))
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Result snapshot checksum mismatch')
    data = json.loads(payload)
    if data['version'] != 1:
        raise ValueError('Unsupported snapshot version')
    root = Path.cwd().resolve()
    resolved = []
    for name, content in data['files'].items():
        path = Path(name)
        if '\\' in name or '..' in path.parts:
            raise ValueError('Snapshot path escapes workspace')
        if path.is_absolute() or not name.startswith(('work/', 'report/')):
            raise ValueError('Unexpected snapshot path')
        if path.suffix not in ('.json', '.jsonl') or not isinstance(content, str):
            raise ValueError('Only JSON result files can be restored')
        if name.startswith('work/runs/'):
            allowed_root = Path(output_root).resolve()
            path = allowed_root / path.relative_to('work/runs')
        else:
            allowed_root = root
        target = path.resolve()
        if not target.is_relative_to(allowed_root):
            raise ValueError('Snapshot path escapes workspace')
        resolved.append((target, content))
    # Validate every path before any write.
    for target, content in resolved:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode('utf-8'))
    return len(resolved)
