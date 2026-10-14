import json
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
import solution.run as runner


class Batch(dict):
    def to(self, device):
        return self


class Processor:
    tokenizer = SimpleNamespace(batch_decode=lambda tokens, **kwargs: [f'reply-{row[0].item()}' for row in tokens])

    def apply_chat_template(self, conversations, **kwargs):
        assert kwargs['processor_kwargs'] == {'padding': True}
        return Batch(input_ids=torch.tensor(conversations).reshape(-1, 1))


class Model:
    def __init__(self, maximum_batch):
        self.maximum_batch = maximum_batch
        self.config = SimpleNamespace(use_cache=False)

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        assert kwargs['do_sample'] is False
        if len(input_ids) > self.maximum_batch:
            raise torch.cuda.OutOfMemoryError('simulated allocation failure')
        return torch.cat((input_ids, input_ids+100), dim=1)


def mock_images(monkeypatch):
    monkeypatch.setattr(runner, 'open_image', lambda entry: int(entry['image_id']))
    monkeypatch.setattr(runner, 'messages', lambda image: image)
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)


def test_oom_retry_preserves_every_image_once_and_decodes_only_new_tokens(tmp_path, monkeypatch):
    mock_images(monkeypatch)
    entries = [{'image_id':str(i)} for i in range(5)]
    output = tmp_path/'predictions.jsonl'
    protocol = runner.predict(Model(2), Processor(), entries, output, batch_size=8)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [{'image_id':str(i), 'raw':f'reply-{i+100}'} for i in range(5)]
    assert protocol['requested_batch_size'] == 8
    assert protocol['effective_batch_size'] == 2
    assert protocol['actual_batch_sizes'] == [1, 2]


def test_oom_at_single_image_stops_without_inventing_a_prediction(tmp_path, monkeypatch):
    mock_images(monkeypatch)
    output = tmp_path/'predictions.jsonl'
    with pytest.raises(torch.cuda.OutOfMemoryError, match='batch size 1'):
        runner.predict(Model(0), Processor(), [{'image_id':'1'}], output, batch_size=1)
    assert output.read_text() == ''
