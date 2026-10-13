"""Clean E4 data-size ablation; select the base with validation F1 only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="work/runs")
    args = ap.parse_args()
    candidates = {}
    for name in ("e1", "e2", "e3"):
        candidates[name] = json.loads(Path(f"work/validation/{name}.json").read_text(encoding="utf-8"))
    best = max(candidates, key=lambda n: candidates[n]["micro"]["f1"])
    checkpoint = Path(args.output_root) / best / "final"
    run = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
    baseline = run["config"]
    if baseline["train_fraction"] != 1.0:
        raise ValueError("E4 base must use the complete train set")
    # Reuse the identical full-data predictions, avoiding an unnecessary retrain.
    cmd = [sys.executable, "scripts/score.py", "--pred", f"work/{best}_test.jsonl",
           "--name", "E4_DATA_1.0", "--out", "report/evidence/e4_data_1.0.json"]
    for key in ("trainable_params", "peak_vram_gb", "train_minutes", "n_train"):
        cmd += ["--meta", f"{key}={run[key]}"]
    subprocess.run(cmd, check=True)
    changed = dict(baseline, train_fraction=0.1)
    manifest = dict(base=best, selection_split="val", variable="train_fraction",
                    values=[0.1, 1.0], configs={"0.1": changed, "1.0": baseline},
                    baseline_checkpoint=str(checkpoint), baseline_predictions=f"work/{best}_test.jsonl")
    Path("work/e4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "solution.run", "--config", f"configs/{best}.yaml",
                    "--name", "e4_data_0.1", "--train-fraction", "0.1",
                    "--batch-size", str(baseline["batch_size"]), "--output-root", args.output_root], check=True)


if __name__ == "__main__":
    main()
