"""Build a report draft and error illustrations from real predictions/evidence."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, evaluate, parse_output


def illustrations(entries, predictions, name):
    from solution.data import open_image
    from solution.inspect_data import draw_boxes
    ranked = []
    for entry in entries:
        pred = parse_output(predictions[entry["image_id"]])
        counts, _ = _match(_gt_boxes(entry), pred or [])
        tp, fp, fn = (sum(v[i] for v in counts.values()) for i in range(3))
        ranked.append((fp+fn, -tp, entry, pred or []))
    ranked.sort(key=lambda r: (r[0], r[1], r[2]["image_id"]))
    # Show both strongest matches and most severe errors; no random cherry picking.
    chosen = ranked[:3] + list(reversed(ranked[-3:]))
    out = Path("report/figures/predictions")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (errors, neg_tp, entry, pred) in enumerate(chosen):
        image = open_image(entry)
        draw_boxes(image, _gt_boxes(entry), "lime")
        draw_boxes(image, pred, "red")
        path = out / f"{name}_{i+1}.jpg"
        image.save(path)
        records.append(dict(path=str(path), image_id=entry["image_id"],
                            tp=-neg_tp, fp_plus_fn=errors, raw=predictions[entry["image_id"]]))
    (out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-name", required=True)
    ap.add_argument("--output-root", default="work/runs")
    ap.add_argument("--output", default="REPORT.md")
    args = ap.parse_args()
    files = [Path(f"report/evidence/e{i}.json") for i in range(4)]
    files += sorted(Path("report/evidence").glob("e4*.json"))
    if len(files) < 6:
        raise ValueError("Need E0–E3 and both E4 values before producing a report")
    if Path("report/evidence/e5.json").exists():
        files.append(Path("report/evidence/e5.json"))
    evidence = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}
    audit = json.loads(Path("work/data_audit.json").read_text(encoding="utf-8"))
    for name, ev in evidence.items():
        if ev["split"] != "test" or ev["n_images"] != 311 or ev["missing_predictions"]:
            raise ValueError(f"Incomplete test evidence: {name}")
        if ev["dataset_version"] != audit["dataset_version"]:
            raise ValueError("Mixed dataset versions")
    manifest = json.loads(Path("work/e4_manifest.json").read_text(encoding="utf-8"))
    architecture = json.loads(Path("work/architecture.json").read_text(encoding="utf-8"))
    best = manifest["base"]
    entries = read_jsonl("data/splits/test.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(f"work/{best}_test.jsonl")}
    figures = illustrations(entries, preds, best)
    e0_preds = {r["image_id"]: r["raw"] for r in read_jsonl("work/e0_test.jsonl")}
    wrong_scale = evaluate(entries, preds, coord="pixel")
    Path("work/wrong_scale.json").write_text(json.dumps(wrong_scale, indent=2), encoding="utf-8")
    lines = [f"Họ tên: {args.student_name}", f"`dataset_version`: {audit['dataset_version']}", "",
             "Báo cáo nháp sinh từ kết quả thật; cần tự kiểm tra nhận định và bổ sung quan sát trước khi nộp.", "",
             "## Bảng tổng hợp", "",
             "| Mã | Tham số train | VRAM GB | Phút | F1 | CI95 | FP/ảnh | pothole | crack | manhole |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for name, ev in evidence.items():
        meta = ev["meta"]
        ci = ev["f1_ci95"]
        values = [name, str(meta.get("trainable_params", 0)), str(meta.get("peak_vram_gb", "—")),
                  str(meta.get("train_minutes", "—")), str(ev["micro"]["f1"]), str(ci),
                  str(ev["fp_per_image"])] + [str(ev["per_class"][c]["f1"]) for c in CLASSES]
        lines.append("| " + " | ".join(values) + " |")
    best_ev = evidence[best]
    worst = min(CLASSES, key=lambda c: best_ev["per_class"][c]["f1"])
    e0_errors = Counter("parse_invalid" if parse_output(raw) is None else
                        "valid_empty" if not parse_output(raw) else "valid_nonempty" for raw in e0_preds.values())
    example_id = next((k for k, v in e0_preds.items() if parse_output(v) is None), next(iter(e0_preds)))
    questions = [
        ("Kiến trúc", "Luồng: ViT trích đặc trưng → merger/DeepStack ánh xạ sang chiều LLM → LLM sinh JSON autoregressive. "
         "E1 cập nhật LoRA LLM (252 module); E2 thêm toàn bộ merger và DeepStack float32; E3 thêm 96 module ViT. "
         f"Số tham số dense đếm trên model meta: {architecture}; không dùng numel của trọng số đã đóng gói 4-bit."),
        ("E0", f"parse_valid_rate = {evidence['e0']['parse_valid_rate']}. Thống kê output: {dict(e0_errors)}. "
         "Scorer chấp nhận markdown fence và trích JSON trong văn bản, nên lời dẫn chưa chắc là lỗi parse. "
         f"Ví dụ ảnh {example_id}:\n```text\n{e0_preds[example_id]}\n```\n"
         "Cần tự xem output và ảnh để xác định hai lỗi phổ biến nhất về nhãn/định vị; thống kê cú pháp không đủ để kết luận."),
        ("Thang 0–1000", f"Cùng dự đoán {best}, chấm đúng norm1000 F1={best_ev['micro']['f1']}, "
         f"cố ý chấm sai pixel F1={wrong_scale['micro']['f1']}. Không huấn luyện lại. "
         "x_pixel=x_norm×width/1000 và y_pixel=y_norm×height/1000; nhầm thang làm đổi vị trí/kích thước và giảm IoU."),
        ("Mask loss", "labels của prompt, token ảnh và padding bằng -100; chỉ suffix assistant và token kết thúc được giám sát. "
         "Model tự dịch labels một vị trí khi tính causal loss, nên không dịch thêm trong collator. "
         "Loss trên toàn chuỗi buộc học dự đoán cả token prompt/placeholder ảnh và làm lệch mục tiêu JSON; không phải hồi quy tái tạo pixel. "
         "Kiểm tra tiền tố chat template, decode labels và số token được lưu trong work/encoding_audit.json."),
        ("Projector", "Skip visual/lm_head giữ projector ngoài lượng tử hóa. Linear4bit là trọng số đóng gói, "
         "không trở thành trọng số dense có thể train chỉ bằng requires_grad=True. "
         "PEFT chuẩn bị k-bit rồi giữ phần thị giác đóng băng fp16, mở merger float32 và dùng autocast fp16/GradScaler. "
         "Float32 tăng độ ổn định lưu trọng số/gradient, không bảo đảm mọi phép tính dưới autocast đều float32. "
         "extra_projector.pt được lưu cùng adapter và nạp vào model mới trước suy luận."),
        ("Split theo phiên quay", "Các frame liền kề có vật thể và bối cảnh gần trùng nhau; random split có thể đánh giá trên frame gần ảnh train. "
         "Gom theo session giảm rò rỉ đó, nhưng không chứng minh mọi session độc lập hoàn toàn. "
         f"Thống kê thật, đã kiểm tra session không giao nhau: `{json.dumps(audit, ensure_ascii=False)}`."),
        ("So sánh E1/E2/E3", f"Cấu hình nền chọn bằng validation là {best}. Đối chiếu F1, CI95, VRAM và phút trong bảng. "
         "Cần tự chọn đánh đổi triển khai từ số đo này; F1 cao hơn không tự chứng minh lợi ích có ý nghĩa thống kê. "
         "Batch 1 × tích lũy 8 giữ effective batch 8 và dùng đồng nhất giữa các setup."),
        ("CI95", "Độ rộng CI95 đo được: " + "; ".join(f"{n}={ev['f1_ci95'][1]-ev['f1_ci95'][0]}" for n, ev in evidence.items()) +
         ". Scorer bootstrap theo ảnh; frame trong một session có tương quan nên CI có thể lạc quan. "
         "Hai CI overlap không chứng minh bằng nhau; cần paired bootstrap chênh lệch F1, tốt hơn là resample session, "
         "và nhiều seed train để kết luận mạnh hơn."),
        ("Lỗi theo lớp", f"Lớp có F1 thấp nhất ở {best}: {worst}, F1={best_ev['per_class'][worst]['f1']}. "
         "Xem 6 ảnh bên dưới để phân biệt bỏ sót, nhầm lớp và sai tọa độ. "
         "Box nhỏ và mất chi tiết sau resize/merging là giả thuyết cần đối chiếu ảnh, không phải nguyên nhân đã được chứng minh."),
        ("E4", f"Chỉ đổi train_fraction giữa 0.1 và 1.0 trên nền {best}; "
         f"manifest lưu hai config: {manifest['configs']}. Số bước, LR, rank, seed và effective batch giữ nguyên. "
         "10% được chọn bằng shuffle seed cố định; 100% tái dùng chính checkpoint/prediction nền. "
         "Đây là so sánh cùng ngân sách optimizer steps, khiến 10% được lặp nhiều epoch hơn. "
         "Đối chiếu CI95 trong bảng và tránh suy ra khác biệt có ý nghĩa chỉ từ overlap.")]
    for i, (title, answer) in enumerate(questions, 1):
        lines += ["", f"### Câu {i} — {title}", "", answer]
    lines += ["", "## Ảnh minh họa", "", "GT xanh lá; dự đoán đỏ. Ba ảnh ít lỗi nhất và ba ảnh nhiều lỗi nhất.", ""]
    for fig in figures:
        lines += [f"![{fig['image_id']}]({fig['path']})", f"TP={fig['tp']}; FP+FN={fig['fp_plus_fn']}", ""]
    if "e5" in evidence:
        e5 = evidence["e5"]
        raw_e5 = read_jsonl("work/e5_test.jsonl")
        parsed = [parse_output(row["raw"]) for row in raw_e5]
        empty = sum(p == [] for p in parsed)
        giant = sum(any((b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]) >= 900000
                        for b in (p or [])) for p in parsed)
        lines += ["## Bonus E5", "",
                  f"GRPO F1={e5['micro']['f1']}, CI95={e5['f1_ci95']}; "
                  f"ΔF1 với nền SFT {best}={e5['micro']['f1']-best_ev['micro']['f1']}. "
                  f"Output rỗng {empty}/311; output có box ≥90% diện tích {giant}/311. "
                  "Đây là chỉ báo reward hacking cần đối chiếu GT; output rỗng không mặc nhiên là gian lận. "
                  "Log train/reward nằm ở checkpoint Drive. Chưa thể suy ra khác biệt có ý nghĩa từ CI overlap.", ""]
    lines += ["## Nhìn lại", "", "Cần bổ sung nhận xét cá nhân về lỗi thực tế. "
              "Các CI này chưa bao gồm bất định do seed huấn luyện hoặc thay đổi cung đường."]
    Path(args.output).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Wrote {args.output}; review qualitative observations before submission")


if __name__ == "__main__":
    main()
