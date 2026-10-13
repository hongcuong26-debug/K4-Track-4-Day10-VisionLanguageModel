"""Multimodal encoding: only the assistant suffix contributes to loss."""
from pathlib import Path
from PIL import Image
import torch

from solution.grounding import PROMPT, objects_to_answer


def open_image(entry):
    path = Path(entry["image_path"])
    if not path.is_absolute():
        path = Path("data") / path
    with Image.open(path) as image:
        return image.convert("RGB")


def messages(image):
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]


def encode_example(image, answer, processor):
    user = messages(image)
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template(user, add_generation_prompt=True, **kwargs)
    full = processor.apply_chat_template(user + [{"role": "assistant", "content": answer}],
                                         add_generation_prompt=False, **kwargs)
    n = prompt["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n], prompt["input_ids"][0]):
        raise ValueError("Chat template prompt is not a prefix of the answer sequence")
    labels = full["input_ids"][0].clone()
    labels[:n] = -100
    if not (labels != -100).any():
        raise ValueError("Empty supervised assistant suffix")
    result = {k: full[k][0] for k in ("input_ids", "attention_mask")}
    result.update(labels=labels, pixel_values=full["pixel_values"],
                  image_grid_thw=full["image_grid_thw"])
    if "mm_token_type_ids" in full:
        result["mm_token_type_ids"] = full["mm_token_type_ids"][0]
    return result


def collate_fn(batch, pad_token_id):
    result = {}
    for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0),
                         ("labels", -100), ("mm_token_type_ids", 0)):
        if key in batch[0]:
            result[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=padding)
    for key in ("pixel_values", "image_grid_thw"):
        result[key] = torch.cat([b[key] for b in batch], dim=0)
    return result


class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, entries, processor):
        self.entries, self.processor = entries, processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        return encode_example(open_image(entry), objects_to_answer(entry["objects"]), self.processor)
