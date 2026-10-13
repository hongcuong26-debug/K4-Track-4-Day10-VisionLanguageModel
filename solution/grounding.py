"""Grounding labels and the verifiable reward, without GPU dependencies."""
import json
import math
from scripts.labcommon import CLASSES
from scripts.score import iou_xyxy, parse_output

PROMPT = (
    "Locate road damage in this image. Detect only objects of classes: "
    "pothole, crack, manhole. Output only a JSON list of objects with label "
    "and bbox_2d integer coordinates normalized on a 0-1000 scale: "
    '[{"label":"pothole","bbox_2d":[x1,y1,x2,y2]}]. '
    "If no damage is found, output []."
)


def yolo_to_box(label, yolo):
    if label not in CLASSES or len(yolo) != 4:
        raise ValueError("Unknown label or malformed YOLO box")
    xc, yc, w, h = yolo
    if not all(math.isfinite(v) for v in yolo) or w <= 0 or h <= 0:
        raise ValueError("YOLO coordinates must be finite with positive size")
    coords = [(xc-w/2)*1000, (yc-h/2)*1000, (xc+w/2)*1000, (yc+h/2)*1000]
    x1, y1, x2, y2 = [round(max(0, min(1000, v))) for v in coords]
    # Also handle a tiny box rounded onto the right/bottom boundary.
    if x2 <= x1:
        x1 = min(x1, 999)
        x2 = x1 + 1
    if y2 <= y1:
        y1 = min(y1, 999)
        y2 = y1 + 1
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects):
    boxes = [yolo_to_box(o["label"], o["yolo"]) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0],
                              b["label"], b["bbox_2d"][2], b["bbox_2d"][3]))
    return json.dumps(boxes, separators=(",", ":"))


def compute_reward(completion, gt_boxes):
    """GT uses scorer format {label, bbox}; coordinates are norm1000."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total = 0.0
    for label in {b["label"] for b in gt_boxes + pred}:
        gt = [b["bbox"] for b in gt_boxes if b["label"] == label]
        pr = [b["bbox"] for b in pred if b["label"] == label]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(gt)
                        for j, b in enumerate(pr)), reverse=True)
        used_gt, used_pr = set(), set()
        for iou, i, j in pairs:
            if iou < 0.5:
                break
            if i not in used_gt and j not in used_pr:
                total += iou
                used_gt.add(i)
                used_pr.add(j)
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)
