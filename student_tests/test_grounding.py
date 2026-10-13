import json
import pytest
from solution.grounding import compute_reward, objects_to_answer, yolo_to_box


def test_coordinate_roundtrip_and_edge():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4])["bbox_2d"] == [400, 300, 600, 700]
    assert yolo_to_box("crack", [1, 1, 0.00001, 0.00001])["bbox_2d"] == [999, 999, 1000, 1000]
    assert yolo_to_box("crack", [0.5, 0.5, 0, 0.2])["bbox_2d"] == [500, 400, 501, 600]
    with pytest.raises(ValueError):
        yolo_to_box("crack", [0.5, 0.5, -0.1, 0.2])


def test_ordering_is_independent_of_input_order():
    objects = [{"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
               {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    assert objects_to_answer(objects) == objects_to_answer(objects[::-1])
    assert json.loads(objects_to_answer(objects))[0]["label"] == "pothole"


GT = [{"label": "pothole", "bbox": [100, 100, 200, 200]}]
GOOD = '{"label":"pothole","bbox_2d":[100,100,200,200]}'


@pytest.mark.parametrize("completion,gt,expected", [
    (f"[{GOOD}]", GT, 1.0), ("not JSON", GT, 0.0),
    ('[{"label":"crack","bbox_2d":[100,100,200,200]}]', GT, 0.2),
    (f"[{GOOD},{GOOD}]", GT, 0.6), ("[]", [], 1.0),
    ("[]", GT, 0.2), ('[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]', GT, 0.2),
])
def test_reward(completion, gt, expected):
    assert compute_reward(completion, gt) == pytest.approx(expected)
