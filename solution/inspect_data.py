"""Audit splits, render GT samples and verify the real multimodal processor."""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import ImageDraw
from scripts.labcommon import read_jsonl
from solution.grounding import objects_to_answer, yolo_to_box


def audit():
    seen = set()
    stats = {}
    expected = {"train": (1422, 1074, 1750, 691), "val": (276, 66, 470, 103),
                "test": (311, 121, 299, 163)}
    image_ids = set()
    for split in expected:
        rows = read_jsonl(f"data/splits/{split}.jsonl")
        sessions = {r["session"] for r in rows}
        ids = {r["image_id"] for r in rows}
        if seen & sessions or image_ids & ids or len(ids) != len(rows):
            raise ValueError("Session overlap or duplicate image IDs detected")
        seen |= sessions
        image_ids |= ids
        counts = Counter(o["label"] for r in rows for o in r["objects"])
        actual = (len(rows), counts["pothole"], counts["crack"], counts["manhole"])
        if actual != expected[split]:
            raise ValueError(f"Unexpected {split} counts: {actual} != {expected[split]}")
        stats[split] = dict(n_images=len(rows), n_sessions=len(sessions), boxes=dict(counts))
    meta = json.loads(Path("data/splits/meta.json").read_text(encoding="utf-8"))
    if not meta["dataset_version"].startswith("d10-"):
        raise ValueError("Unexpected dataset version")
    stats["dataset_version"] = meta["dataset_version"]
    Path("work").mkdir(exist_ok=True)
    Path("work/data_audit.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def draw_boxes(image, boxes, color, scale=1000):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        bb = box.get("bbox_2d", box.get("bbox"))
        x1, y1, x2, y2 = bb
        coords = [x1/scale*image.width, y1/scale*image.height,
                  x2/scale*image.width, y2/scale*image.height]
        draw.rectangle(coords, outline=color, width=2)
        draw.text((coords[0], max(0, coords[1]-12)), box["label"], fill=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processor", action="store_true")
    args = ap.parse_args()
    audit()
    # Import torch only when images/encoding are requested.
    from solution.data import open_image
    rows = read_jsonl("data/splits/train.jsonl")
    output = Path("report/figures/ground_truth")
    output.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows[:6]):
        image = open_image(row)
        draw_boxes(image, [yolo_to_box(o["label"], o["yolo"]) for o in row["objects"]], "lime")
        image.save(output / f"gt_{i+1}.jpg")
    if args.processor:
        from transformers import AutoProcessor
        from solution.data import encode_example, collate_fn
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        sample = encode_example(open_image(rows[0]), objects_to_answer(rows[0]["objects"]), processor)
        merge = processor.image_processor.merge_size
        tokens = int(sample["image_grid_thw"].prod(dim=-1).sum()) // merge**2
        decoded = processor.tokenizer.decode(sample["labels"][sample["labels"] != -100])
        if not decoded.startswith(objects_to_answer(rows[0]["objects"])) or "<|im_end|>" not in decoded:
            raise ValueError("Assistant labels do not include target JSON and end-of-turn token")
        batch = collate_fn([sample, sample], processor.tokenizer.pad_token_id)
        if batch["image_grid_thw"].shape[0] != 2:
            raise ValueError("Bad multimodal collation")
        diagnostics = dict(vision_tokens=tokens, image_grid_thw=sample["image_grid_thw"].tolist(),
                           assistant_decoded=decoded, patch_size=processor.image_processor.patch_size)
        Path("work/encoding_audit.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(diagnostics)


if __name__ == "__main__":
    main()
