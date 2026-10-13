"""Tiny dense model verifies module routing and projector persistence on CPU."""
import copy
import torch
import pytest
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from solution.modeling import apply_setup, is_projector, load_checkpoint, save_checkpoint


def tiny_model():
    config = Qwen3VLConfig(text_config=dict(vocab_size=128, hidden_size=32,
        intermediate_size=64, num_hidden_layers=36, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, eos_token_id=2, pad_token_id=0),
        vision_config=dict(depth=24, hidden_size=32, intermediate_size=64,
            num_heads=4, out_hidden_size=32, num_position_embeddings=16,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
            deepstack_visual_indexes=[5, 11, 17]))
    return Qwen3VLForConditionalGeneration(config)


def setup(name):
    return dict(name=name, lora=dict(r=2, alpha=4, dropout=0.05,
        targets="llm+vit" if name == "E3" else "llm"), train_projector=name != "E1")


@pytest.mark.parametrize("name,expected", [("E1", 252), ("E2", 252), ("E3", 348)])
def test_lora_routing_and_frozen_blocks(name, expected):
    model, matched = apply_setup(tiny_model(), setup(name))
    assert len(matched) == expected
    assert not any(is_projector(n) for n in matched)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert "lora_" in n or (name != "E1" and is_projector(n))
            assert p.dtype == torch.float32
        if is_projector(n):
            assert p.requires_grad == (name != "E1")


def test_projector_saved_and_restored_and_missing_file_rejected(tmp_path):
    base = tiny_model()
    original = copy.deepcopy(base.state_dict())
    model, _ = apply_setup(base, setup("E2"))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if is_projector(name):
                p.add_(0.25)

    class Processor:
        def save_pretrained(self, directory):
            pass

    run = dict(config=setup("E2"), dataset_version="unit-test")
    save_checkpoint(model, Processor(), tmp_path, run)
    fresh = tiny_model()
    fresh.load_state_dict(original)
    restored, _ = load_checkpoint(fresh, tmp_path)
    target = dict(restored.named_parameters())
    for name, p in model.named_parameters():
        if is_projector(name):
            assert torch.equal(p, target[name])
    (tmp_path / "extra_projector.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing trained projector"):
        load_checkpoint(tiny_model(), tmp_path)
