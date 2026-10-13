"""Run one experiment: python -m solution.run --config configs/e1.yaml --name e1."""
import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from transformers import get_cosine_schedule_with_warmup, set_seed

from scripts.labcommon import read_jsonl
from solution.data import RoadDataset, collate_fn, messages, open_image
from solution.modeling import apply_setup, load_base, load_checkpoint, save_checkpoint


def train(model, processor, config, entries, directory, dataset_version):
    generator = torch.Generator().manual_seed(config["seed"])
    loader = torch.utils.data.DataLoader(RoadDataset(entries, processor),
        batch_size=config["batch_size"], shuffle=True, generator=generator,
        collate_fn=partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("Trainable weights must be float32 for fp16 GradScaler")
    optimizer = torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=math.ceil(0.05*config["max_steps"]),
        num_training_steps=config["max_steps"])
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    run = {"config": config, "dataset_version": dataset_version,
           "trainable_params": sum(p.numel() for p in params),
           "n_train": len(entries), "train_ids_sha256": hashlib.sha256(
               "\n".join(e["image_id"] for e in entries).encode()).hexdigest(),
           "gpu": torch.cuda.get_device_name(0), "completed_steps": 0}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.jsonl").write_text("", encoding="utf-8")
    iterator = iter(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config["max_steps"]+1):
        loss_sum = 0.0
        for _ in range(config["grad_accum"]):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at optimizer step {step}")
            loss_sum += loss.detach().item()/config["grad_accum"]
            scaler.scale(loss/config["grad_accum"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Nonfinite gradient at step {step}; reduce LR")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        run.update(completed_steps=step, peak_vram_gb=torch.cuda.max_memory_allocated()/1e9,
                   train_minutes=(time.perf_counter()-started)/60)
        row = dict(step=step, loss=loss_sum, lr=scheduler.get_last_lr()[0],
                   grad_norm=float(grad_norm), peak_vram_gb=run["peak_vram_gb"],
                   elapsed_minutes=run["train_minutes"])
        with (directory / "train.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row)+"\n")
        if step == 1 or step % 10 == 0:
            print(row, flush=True)
        if step % 100 == 0 or step == config["max_steps"]:
            save_checkpoint(model, processor, directory / f"step-{step}", run)
    save_checkpoint(model, processor, directory / "final", run)
    return run


def predict(model, processor, entries, output, max_new_tokens=256, batch_size=4):
    """Left-padded image batches; keep untouched output for each image ID."""
    model.eval()
    model.config.use_cache = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for start in range(0, len(entries), batch_size):
            batch_entries = entries[start:start+batch_size]
            conversations = [messages(open_image(entry)) for entry in batch_entries]
            inputs = processor.apply_chat_template(conversations, tokenize=True,
                add_generation_prompt=True, processor_kwargs={"padding": True},
                return_dict=True, return_tensors="pt").to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raw_outputs = processor.tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for entry, raw in zip(batch_entries, raw_outputs):
                file.write(json.dumps({"image_id": entry["image_id"], "raw": raw}, ensure_ascii=False)+"\n")
            file.flush()
            print(f"inference {start+len(batch_entries)}/{len(entries)}", flush=True)


def score(pred, name, split, run):
    output = Path("report/evidence" if split == "test" else "work/validation") / f"{name}.json"
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred), "--name", name.upper(),
           "--split", split, "--out", str(output)]
    for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
        if key in run:
            cmd += ["--meta", f"{key}={run[key]}"]
    cmd += ["--meta", f"n_train={run.get('n_train', 0)}"]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-root", default="work/runs")
    ap.add_argument("--checkpoint", help="Inference only; load this saved final checkpoint")
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, help="Use 1 for T4 OOM, preserving effective batch")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--val-only", action="store_true", help="Train and validate, no test scoring")
    a = ap.parse_args()
    if Path(a.name).name != a.name or a.name in (".", ".."):
        ap.error("name must be a single filename component")
    if not 0 < a.train_fraction <= 1:
        ap.error("train-fraction must be in (0, 1]")
    config = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if config["name"] not in ("E0", "E1", "E2", "E3"):
        ap.error("Use E1/E2/E3 config as E4 base; E5 uses solution.grpo")
    if a.batch_size:
        effective = config["batch_size"]*config["grad_accum"]
        if a.batch_size <= 0 or effective % a.batch_size:
            ap.error("batch-size must divide effective batch")
        config.update(batch_size=a.batch_size, grad_accum=effective//a.batch_size)
    config["train_fraction"] = a.train_fraction
    set_seed(config.get("seed", 42))
    dataset_version = json.loads(Path("data/splits/meta.json").read_text(encoding="utf-8"))["dataset_version"]
    directory = Path(a.output_root)/a.name
    model, processor = load_base(config["model_id"])
    run = {"trainable_params": 0}
    if a.checkpoint:
        model, run = load_checkpoint(model, a.checkpoint)
        if run["dataset_version"] != dataset_version:
            raise ValueError("Checkpoint was trained on a different dataset version")
        if run["completed_steps"] != run["config"]["max_steps"]:
            raise ValueError("Checkpoint training is incomplete")
    elif config["name"] != "E0":
        if (directory / "final").exists():
            raise FileExistsError("Run already exists; use --checkpoint for inference or a new name")
        entries = sorted(read_jsonl("data/splits/train.jsonl"), key=lambda e: e["image_id"])
        random.Random(config["seed"]).shuffle(entries)
        entries = entries[:max(1, int(len(entries)*a.train_fraction))]
        model, matched = apply_setup(model, config)
        print(f"LoRA targets: {len(matched)}", flush=True)
        run = train(model, processor, config, entries, directory, dataset_version)
        # Evaluate a fresh model so missing projector persistence cannot go unnoticed.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, processor = load_base(config["model_id"])
        model, run = load_checkpoint(model, directory / "final")
    splits = ["val"] if a.val_only else (["val", "test"] if a.split == "test" else ["val"])
    for split in splits:
        entries = read_jsonl(f"data/splits/{split}.jsonl")
        pred = Path("work") / f"{a.name}_{split}.jsonl"
        predict(model, processor, entries, pred, config.get("max_new_tokens", 256))
        score(pred, a.name, split, run)


if __name__ == "__main__":
    main()
