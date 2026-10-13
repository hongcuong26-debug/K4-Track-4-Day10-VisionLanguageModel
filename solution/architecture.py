"""Count original dense architecture without downloading weight tensors."""
import json
from pathlib import Path
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3VLForConditionalGeneration


def main():
    config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    # Tie outside Accelerate's parameter-registration hook: otherwise the hook
    # can wrap aliases in separate Parameters and double-count the lm_head.
    model.tie_weights()
    counts = dict(vision_backbone=0, projector=0, language_and_embedding=0, total=0)
    for name, param in model.named_parameters():
        n = param.numel()
        counts["total"] += n
        if "visual.merger" in name or "visual.deepstack_merger_list" in name:
            counts["projector"] += n
        elif "visual" in name:
            counts["vision_backbone"] += n
        else:
            counts["language_and_embedding"] += n
    counts.update(patch_size=config.vision_config.patch_size,
                  spatial_merge_size=config.vision_config.spatial_merge_size,
                  deepstack_visual_indexes=config.vision_config.deepstack_visual_indexes)
    Path("work").mkdir(exist_ok=True)
    Path("work/architecture.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
