"""Optional E5: GRPO from a completed SFT adapter, with verifiable IoU reward."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset, Image as DatasetImage
from transformers import TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes
from solution.grounding import PROMPT, compute_reward
from solution.modeling import load_base, load_checkpoint, save_checkpoint
from solution.run import predict, score


def iou_reward(completions, gt_boxes, **kwargs):
    rewards = []
    for completion, gt in zip(completions, gt_boxes):
        if isinstance(completion, list):
            content = completion[0]["content"]
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content)
        else:
            content = completion
        rewards.append(compute_reward(content, gt))
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-root", default="work/runs")
    args = ap.parse_args()
    set_seed(42)
    config = yaml.safe_load(Path("configs/e5.yaml").read_text(encoding="utf-8"))["grpo"]
    directory = Path(args.output_root)/"e5"
    if (directory / "final").exists():
        raise FileExistsError("E5 already completed")
    model, processor = load_base("Qwen/Qwen3-VL-4B-Instruct")
    model, sft_run = load_checkpoint(model, args.checkpoint, trainable=True)
    dataset_version = json.loads(Path("data/splits/meta.json").read_text(encoding="utf-8"))["dataset_version"]
    if dataset_version != sft_run["dataset_version"]:
        raise ValueError("Dataset version differs from SFT checkpoint")
    model.enable_input_require_grads()
    model.config.use_cache = False
    entries = read_jsonl("data/splits/train.jsonl")
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    dataset = Dataset.from_list([dict(prompt=prompt,
        image=str(Path("data")/e["image_path"]), gt_boxes=_gt_boxes(e)) for e in entries])
    dataset = dataset.cast_column("image", DatasetImage())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_config = dict(sft_run["config"], max_steps=config["max_steps"], lr=float(config["learning_rate"]))
    run = dict(config=run_config, grpo=config, dataset_version=dataset_version,
               sft_checkpoint=str(args.checkpoint), trainable_params=trainable,
               n_train=len(entries), completed_steps=0,
               train_ids_sha256=hashlib.sha256("\n".join(e["image_id"] for e in entries).encode()).hexdigest())
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    class AuditCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            with (directory / "train.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(dict(step=state.global_step, **(logs or {})))+"\n")

        def on_save(self, args, state, control, model=None, **kwargs):
            torch.cuda.synchronize()
            run.update(completed_steps=state.global_step,
                       peak_vram_gb=torch.cuda.max_memory_allocated()/1e9,
                       train_minutes=(time.perf_counter()-started)/60)
            save_checkpoint(model, processor, directory/f"checkpoint-{state.global_step}", run)

    training_args = GRPOConfig(output_dir=str(directory), num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"], max_steps=config["max_steps"],
        learning_rate=float(config["learning_rate"]), beta=config["beta"],
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, use_vllm=False,
        remove_unused_columns=False, report_to="none", save_steps=20, logging_steps=1,
        optim="adamw_torch", seed=42)
    def audited_reward(completions, gt_boxes, **kwargs):
        rewards = iou_reward(completions, gt_boxes, **kwargs)
        with (directory / "reward_audit.jsonl").open("a", encoding="utf-8") as file:
            for completion, reward in zip(completions, rewards):
                file.write(json.dumps(dict(completion=completion, reward=reward), ensure_ascii=False)+"\n")
        return rewards

    trainer = GRPOTrainer(model=model, processing_class=processor, reward_funcs=audited_reward,
                          args=training_args, train_dataset=dataset, callbacks=[AuditCallback()])
    trainer.train()
    torch.cuda.synchronize()
    run.update(completed_steps=trainer.state.global_step,
               peak_vram_gb=torch.cuda.max_memory_allocated()/1e9,
               train_minutes=(time.perf_counter()-started)/60)
    save_checkpoint(model, processor, directory/"final", run)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = load_base("Qwen/Qwen3-VL-4B-Instruct")
    model, run = load_checkpoint(model, directory/"final")
    pred = Path("work/e5_test.jsonl")
    predict(model, processor, read_jsonl("data/splits/test.jsonl"), pred)
    score(pred, "e5", "test", run)


if __name__ == "__main__":
    main()
