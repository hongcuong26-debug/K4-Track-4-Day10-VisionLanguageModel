"""Measure worst-case E3 batch-2 training memory before the full experiments."""
import json
from pathlib import Path
import yaml
from transformers import set_seed
from scripts.labcommon import read_jsonl
from solution.modeling import apply_setup, load_base
from solution.run import train


def main():
    config = yaml.safe_load(Path("configs/e3.yaml").read_text(encoding="utf-8"))
    config.update(max_steps=1, batch_size=2, grad_accum=1, train_fraction=1.0)
    set_seed(config["seed"])
    entries = sorted(read_jsonl("data/splits/train.jsonl"),
                     key=lambda e: (-len(e["objects"]), e["image_id"]))[:2]
    version = json.loads(Path("data/splits/meta.json").read_text(encoding="utf-8"))["dataset_version"]
    model, processor = load_base(config["model_id"])
    model, matched = apply_setup(model, config)
    print(f"Preflight targets={len(matched)}, boxes={[len(e['objects']) for e in entries]}", flush=True)
    run = train(model, processor, config, entries, Path("work/preflight_e3_batch2"), version)
    run.update(purpose="Memory check only; not a completed E3 experiment", image_ids=[e["image_id"] for e in entries])
    Path("report/preflight_e3_batch2.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
