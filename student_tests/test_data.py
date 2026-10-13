import pytest
import torch
from solution.data import encode_example, collate_fn


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        full = len(messages) == 2
        result = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5] if full else [1, 2, 3]]),
                      pixel_values=torch.ones(4, 8), image_grid_thw=torch.tensor([[1, 2, 2]]))
        result["attention_mask"] = torch.ones_like(result["input_ids"])
        result["mm_token_type_ids"] = torch.zeros_like(result["input_ids"])
        return result


def test_only_assistant_suffix_is_supervised_and_visual_tensors_concatenate():
    sample = encode_example(None, "[]", Processor())
    assert sample["labels"].tolist() == [-100, -100, -100, 4, 5]
    short = {k: (v[:4] if k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids") else v)
             for k, v in sample.items()}
    batch = collate_fn([sample, short], 0)
    assert batch["labels"][1, -1].item() == -100
    assert batch["attention_mask"][1, -1].item() == 0
    assert batch["pixel_values"].shape == (8, 8)
    assert batch["image_grid_thw"].shape == (2, 3)
    assert "mm_token_type_ids" in batch


def test_template_mismatch_is_rejected():
    class BadProcessor(Processor):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if len(messages) == 2:
                result["input_ids"][0, 0] = 99
            return result
    with pytest.raises(ValueError, match="prefix"):
        encode_example(None, "[]", BadProcessor())
