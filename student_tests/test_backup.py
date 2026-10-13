import json
from solution.backup import backup_once


def test_backup_copies_only_completed_checkpoints_and_preserves_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    complete = tmp_path / 'runs/e1/step-100'
    complete.mkdir(parents=True)
    (complete / 'run.json').write_text(json.dumps({'completed_steps': 100}))
    (complete / 'adapter_model.safetensors').write_bytes(b'checkpoint fixture')
    incomplete = tmp_path / 'runs/e1/step-200'
    incomplete.mkdir()
    (incomplete / 'adapter_model.safetensors').write_bytes(b'not ready')
    (tmp_path / 'runs/e1/train.jsonl').write_text('{"step":100}\n')
    destination = tmp_path / 'drive/new-run'
    copied = backup_once(tmp_path / 'runs', destination)
    assert len(copied) == 1
    assert (destination / 'runs/e1/step-100/adapter_model.safetensors').read_bytes() == b'checkpoint fixture'
    assert (complete / 'adapter_model.safetensors').read_bytes() == b'checkpoint fixture'
    assert not (destination / 'runs/e1/step-200').exists()
    assert backup_once(tmp_path / 'runs', destination) == []


def test_backup_recovers_its_partial_copy_and_refreshes_audit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    complete = tmp_path / 'runs/e2/final'
    complete.mkdir(parents=True)
    (complete / 'run.json').write_text(json.dumps({'completed_steps': 300}))
    (complete / 'extra_projector.pt').write_bytes(b'complete weights')
    temporary = tmp_path / 'drive/new-run/runs/e2/final.copying'
    temporary.mkdir(parents=True)
    (temporary / 'extra_projector.pt').write_bytes(b'partial')
    evidence = tmp_path / 'report/evidence'
    evidence.mkdir(parents=True)
    (evidence / 'e2.json').write_text('{"score_version":1}')
    backup_once(tmp_path / 'runs', tmp_path / 'drive/new-run')
    assert not temporary.exists()
    assert (tmp_path / 'drive/new-run/runs/e2/final/extra_projector.pt').read_bytes() == b'complete weights'
    assert (tmp_path / 'drive/new-run/report/evidence/e2.json').read_text() == '{"score_version":1}'
