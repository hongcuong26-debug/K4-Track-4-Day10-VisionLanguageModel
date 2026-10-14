"""Complete E1–E4 after E0, recording progress and downloadable audit bundles."""
import argparse
import json
import subprocess
import sys
import time
import zipfile
import shutil
from pathlib import Path
from solution.portable_results import emit_snapshot


def completed_result(name, output_root):
    """Allow result-only recovery after a runtime reset; verify raw predictions."""
    from scripts.labcommon import read_jsonl
    from scripts.score import evaluate
    import yaml
    try:
        run = json.loads((Path(output_root)/name/'final/run.json').read_text(encoding='utf-8'))
        version = json.loads(Path('data/splits/meta.json').read_text(encoding='utf-8'))['dataset_version']
        if run['completed_steps'] != run['config']['max_steps'] or run['completed_steps'] != 300:
            return False
        if run['dataset_version'] != version:
            return False
        expected_config = yaml.safe_load(Path(f'configs/{name}.yaml').read_text(encoding='utf-8'))
        expected_config['train_fraction'] = 1.0
        if run['config'] != expected_config or run['n_train'] != len(read_jsonl('data/splits/train.jsonl')):
            return False
        if any(run[key] <= 0 for key in ('trainable_params', 'peak_vram_gb', 'train_minutes')):
            return False
        for split in ('val', 'test'):
            entries = read_jsonl(f'data/splits/{split}.jsonl')
            rows = read_jsonl(f'work/{name}_{split}.jsonl')
            predictions = {row['image_id']: row['raw'] for row in rows}
            if len(rows) != len(predictions) or set(predictions) != {e['image_id'] for e in entries}:
                return False
            path = Path(f'report/evidence/{name}.json' if split == 'test' else f'work/validation/{name}.json')
            evidence = json.loads(path.read_text(encoding='utf-8'))
            if evidence['score_version'] != 1 or evidence['split'] != split or evidence['dataset_version'] != version:
                return False
            if any(evidence['meta'][key] != run[key] for key in ('trainable_params', 'peak_vram_gb', 'train_minutes', 'n_train')):
                return False
            measured = evaluate(entries, predictions)
            if any(measured[key] != evidence[key] for key in ('micro', 'per_class', 'f1_ci95', 'parse_valid_rate', 'missing_predictions', 'n_images')):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def bundle(output_root):
    destination = Path("work/lab_artifacts.zip")
    temporary = destination.with_suffix(".partial.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in Path("report").rglob("*"):
            if path.is_file():
                archive.write(path, path.as_posix())
        for path in Path("work").glob("*.json*"):
            if path.is_file():
                archive.write(path, path.as_posix())
        for folder in ("work/validation", "work/lab_logs"):
            for path in Path(folder).rglob("*"):
                if path.is_file():
                    archive.write(path, path.as_posix())
        for name in ("e1", "e2", "e3", "e4_data_0.1"):
            for filename in ("train.jsonl", "final/run.json"):
                path = Path(output_root)/name/filename
                if path.exists():
                    archive.write(path, f"work/runs/{name}/{filename}")
        for name in ("REPORT.md", "CHECKLIST.md", "BAO_CAO_LOI_VA_LUU_Y.md", "lab_day10.ipynb"):
            archive.write(name, name)
    temporary.replace(destination)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="work/runs")
    ap.add_argument("--student-name", required=True)
    args = ap.parse_args()
    Path("work/lab_logs").mkdir(parents=True, exist_ok=True)
    progress_path = Path("work/lab_progress.json")
    progress = dict(status="running", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    completed=[], current="initial_checks")

    def update(stage, status="running"):
        progress.update(current=stage, status=status,
                        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(json.dumps(progress), flush=True)

    def execute(stage, arguments):
        update(stage)
        with Path(f"work/lab_logs/{stage}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    process.wait(timeout=45)
                    break
                except subprocess.TimeoutExpired:
                    rows = Path(args.output_root)/stage/'train.jsonl'
                    stage_log = Path(f"work/lab_logs/{stage}.log")
                    latest = rows.read_text(encoding='utf-8').splitlines()[-1] if rows.exists() and rows.stat().st_size else ''
                    tail = stage_log.read_text(encoding='utf-8').splitlines()[-1] if stage_log.stat().st_size else ''
                    print(f'PROGRESS {stage} {latest} {tail}', flush=True)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
        progress["completed"].append(stage)
        update(stage)
        bundle(args.output_root)
        emit_snapshot(args.output_root)

    try:
        if not Path('work/e0_test.jsonl').exists():
            shutil.copy2('report/predictions/e0_test.jsonl', 'work/e0_test.jsonl')
        e0 = json.loads(Path("report/evidence/e0.json").read_text(encoding="utf-8"))
        if e0["n_images"] != 311 or e0["missing_predictions"]:
            raise ValueError("Complete E0 test evidence is required first")
        bundle(args.output_root)
        emit_snapshot(args.output_root)
        for name in ("e1", "e2", "e3"):
            if completed_result(name, args.output_root):
                progress['completed'].append(name)
                update(name)
                print(f'RECOVERED {name}: complete training metadata and raw val/test scores verified', flush=True)
                continue
            checkpoint = Path(args.output_root)/name/"final"
            arguments = ["-m", "solution.run", "--config", f"configs/{name}.yaml", "--name", name,
                         "--output-root", args.output_root, "--batch-size", "2"]
            if checkpoint.exists():
                arguments += ["--checkpoint", str(checkpoint)]
            execute(name, arguments)
        execute("e4", ["-m", "solution.ablation", "--output-root", args.output_root])
        execute("paired_analysis", ["-m", "solution.analysis"])
        execute("report", ["-m", "solution.report", "--student-name", args.student_name,
                           "--output-root", args.output_root])
        execute("submission_check", ["scripts/check_submission.py"])
        e1 = json.loads(Path("report/evidence/e1.json").read_text(encoding="utf-8"))
        thresholds = dict(parse_valid_rate_at_least_095=e1["parse_valid_rate"] >= 0.95,
                          e1_f1_gain_at_least_010=e1["micro"]["f1"]-e0["micro"]["f1"] >= 0.10)
        Path("report/rubric_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        update("complete", "complete" if all(thresholds.values()) else "needs_threshold_review")
        bundle(args.output_root)
        emit_snapshot(args.output_root)
    except Exception as error:
        progress["error"] = str(error)
        update(progress["current"], "failed")
        bundle(args.output_root)
        emit_snapshot(args.output_root)
        raise


if __name__ == "__main__":
    main()
