import base64
import gzip
import hashlib
import json
from pathlib import Path
import pytest
from solution.portable_results import emit_snapshot, restore_snapshot


def test_real_result_text_round_trip_without_model_weights(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path('work').mkdir()
    raw = '{"image_id":"ảnh-1","raw":"[]"}\r\n'.encode()
    Path('work/e1_test.jsonl').write_bytes(raw)
    checkpoint = Path('runs/e1/final')
    checkpoint.mkdir(parents=True)
    (checkpoint/'run.json').write_text('{"completed_steps":300}')
    (checkpoint/'adapter_model.safetensors').write_bytes(b'weights must be excluded')
    digest = emit_snapshot('runs')
    encoded = Path('work/portable_results.b64').read_text()
    assert 'BEGIN_DAY10_SNAPSHOT' in capsys.readouterr().out
    payload = json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert not any('safetensors' in key for key in payload['files'])
    restore = tmp_path/'restore'
    restore.mkdir()
    monkeypatch.chdir(restore)
    restore_snapshot(encoded, digest, output_root='restored-runs')
    assert Path('work/e1_test.jsonl').read_bytes() == raw
    assert json.loads(Path('restored-runs/e1/final/run.json').read_text())['completed_steps'] == 300


def test_restore_validates_all_paths_before_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = json.dumps({'version':1, 'files':{'work/valid.json':'{}', 'work/../../escape.json':'{}'}}).encode()
    encoded = base64.b64encode(gzip.compress(data)).decode()
    with pytest.raises(ValueError, match='escapes workspace'):
        restore_snapshot(encoded, hashlib.sha256(data).hexdigest())
    assert not Path('work/valid.json').exists()


def test_restore_rejects_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    encoded = base64.b64encode(gzip.compress(b'{"version":1,"files":{}}')).decode()
    with pytest.raises(ValueError, match='checksum'):
        restore_snapshot(encoded, '0'*64)


def test_restore_can_use_explicit_checkpoint_folder_outside_repo(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    repo.mkdir()
    monkeypatch.chdir(repo)
    payload = json.dumps({'version':1, 'files':{'work/runs/e1/final/run.json':'{"completed_steps":300}'}}).encode()
    encoded = base64.b64encode(gzip.compress(payload)).decode()
    checkpoint_root = tmp_path/'explicit-checkpoints'
    restore_snapshot(encoded, hashlib.sha256(payload).hexdigest(), checkpoint_root)
    assert json.loads((checkpoint_root/'e1/final/run.json').read_text())['completed_steps'] == 300
