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
        draw_boxes(image, _gt_boxes(entry), "lime", width=4)
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
    stats_path = Path("report/statistical_analysis.json")
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    quality_path = Path("report/label_quality.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    best = manifest["base"]
    entries = read_jsonl("data/splits/test.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(f"work/{best}_test.jsonl")}
    figures = illustrations(entries, preds, best)
    e0_preds = {r["image_id"]: r["raw"] for r in read_jsonl("work/e0_test.jsonl")}
    wrong_scale = evaluate(entries, preds, coord="pixel")
    Path("work/wrong_scale.json").write_text(json.dumps(wrong_scale, indent=2), encoding="utf-8")
    lines = [f"Họ tên: {args.student_name}", f"`dataset_version`: {audit['dataset_version']}", "",
             "Bảng số liệu đọc trực tiếp từ evidence do scorer của đề sinh; nhận định phân biệt số đo và giả thuyết.", "",
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
    e0_fn = evidence['e0']['micro']['fn']
    e0_fp = evidence['e0']['micro']['fp']
    e0_worst = max(CLASSES, key=lambda c: evidence['e0']['per_class'][c]['fn'])
    comparisons = []
    for first, second in (("e1", "e2"), ("e2", "e3")):
        a, b = evidence[first], evidence[second]
        comparisons.append(f"{second}−{first}: ΔF1={b['micro']['f1']-a['micro']['f1']:.6f}, "
            f"ΔVRAM={b['meta']['peak_vram_gb']-a['meta']['peak_vram_gb']:.6f} GB, "
            f"Δphút={b['meta']['train_minutes']-a['meta']['train_minutes']:.6f}")
    low, full = evidence['e4_data_0.1'], evidence['e4_data_1.0']
    overlap = max(low['f1_ci95'][0], full['f1_ci95'][0]) <= min(low['f1_ci95'][1], full['f1_ci95'][1])
    questions = [
        ("Kiến trúc", "Luồng: ViT trích đặc trưng → merger/DeepStack ánh xạ sang chiều LLM → LLM sinh JSON autoregressive. "
         "E1 cập nhật LoRA LLM (252 module); E2 thêm toàn bộ merger và DeepStack float32; E3 thêm 96 module ViT. "
         f"Số tham số dense đếm trên model meta: {architecture}; không dùng numel của trọng số đã đóng gói 4-bit."),
        ("E0", f"parse_valid_rate = {evidence['e0']['parse_valid_rate']}. Thống kê output: {dict(e0_errors)}. "
         "Scorer chấp nhận markdown fence và trích JSON trong văn bản, nên lời dẫn chưa chắc là lỗi parse. "
         f"Ví dụ ảnh {example_id}:\n```text\n{e0_preds[example_id]}\n```\n"
         f"Hai loại lỗi kiểm chứng bằng matching của scorer: bỏ sót {e0_fn} box GT và sinh {e0_fp} box "
         f"không khớp đúng lớp/IoU≥0.5. Lớp bị bỏ sót nhiều nhất theo số đếm là {e0_worst}, "
         f"FN={evidence['e0']['per_class'][e0_worst]['fn']}. Một FP có thể là sai lớp, sai vị trí hoặc box thừa; "
         "không đồng nhất tất cả FP với hallucination. "
         "Hai ví dụ đã đối chiếu trực quan: vlcsnap-2025-02-19-17h03m28s017 có 5 pothole GT "
         "nhưng 0 TP, 5 FP, 5 FN do lệch box/nhầm lớp; vlcsnap-2025-02-19-17h03m29s526 "
         "có 3 pothole và 1 crack GT nhưng sinh 4 manhole, trong đó một box bị lặp. "
         "Chi tiết và sáu ảnh E0 tại [E0_FINDINGS.md](report/E0_FINDINGS.md)."),
        ("Thang 0–1000", f"Cùng dự đoán {best}, chấm đúng norm1000 F1={best_ev['micro']['f1']}, "
         f"cố ý chấm sai pixel F1={wrong_scale['micro']['f1']}. Không huấn luyện lại. "
         "x_pixel=x_norm×width/1000 và y_pixel=y_norm×height/1000; nhầm thang làm đổi vị trí/kích thước và giảm IoU. "
         "Ví dụ x_norm=750 tương ứng x_pixel=480 trên ảnh rộng 640 pixel. "
         "Với box không cần clip hoặc sửa cạnh suy biến, sai số làm tròn một góc tối đa 0.5 đơn vị norm, "
         "tức 0.32 pixel ngang và 0.18 pixel dọc "
         "trên ảnh 640×360; nhầm cả thang lớn hơn nhiều sai số lượng tử tọa độ này."),
        ("Mask loss", "labels của prompt, token ảnh và padding bằng -100; chỉ suffix assistant và token kết thúc được giám sát. "
         "Model tự dịch labels một vị trí khi tính causal loss, nên không dịch thêm trong collator. "
         "Loss trên toàn chuỗi buộc học dự đoán cả token prompt/placeholder ảnh và làm lệch mục tiêu JSON; không phải hồi quy tái tạo pixel. "
         "Mask nhãn không detach đặc trưng ảnh hoặc chặn gradient qua attention: khi mở LoRA ViT ở E3, "
         "loss assistant vẫn truyền gradient về khối thị giác qua đặc trưng dùng để sinh đáp án. "
         "Kiểm tra tiền tố chat template, decode labels và số token được lưu trong work/encoding_audit.json."),
        ("Projector", "Skip list thực tế gồm visual, model.visual và lm_head để giữ projector ngoài lượng tử hóa "
         "trên Transformers 5.19. Linear4bit là trọng số đóng gói, "
         "không trở thành trọng số dense có thể train chỉ bằng requires_grad=True. "
         "PEFT chuẩn bị k-bit rồi giữ phần thị giác đóng băng fp16, mở merger float32 và dùng autocast fp16/GradScaler. "
         "Float32 tăng độ ổn định lưu trọng số/gradient, không bảo đảm mọi phép tính dưới autocast đều float32. "
         "extra_projector.pt được lưu cùng adapter và nạp vào model mới trước suy luận."),
        ("Split theo phiên quay", "Các frame liền kề có vật thể và bối cảnh gần trùng nhau; random split có thể đánh giá trên frame gần ảnh train. "
         "Gom theo session giảm rò rỉ đó, nhưng không chứng minh mọi session độc lập hoàn toàn. "
         f"Thống kê thật, đã kiểm tra session không giao nhau: `{json.dumps(audit, ensure_ascii=False)}`."),
        ("So sánh E1/E2/E3", f"Cấu hình nền chọn bằng validation là {best}. Đối chiếu F1, CI95, VRAM và phút trong bảng. "
         + "; ".join(comparisons) + ". F1 cao hơn không tự chứng minh lợi ích có ý nghĩa thống kê. "
         "Batch 2 × tích lũy 4 giữ effective batch 8, đúng cấu hình đề và dùng đồng nhất giữa các setup. "
         "VRAM là max_memory_allocated chia 1e9, không phải bộ nhớ reserved hay tổng nvidia-smi. "
         "Thời gian trong bảng đo phần train, không tính tải model và suy luận; checkpoint nằm trong vòng train. "
         "Lượt thử batch 1 đã dừng do kiểm tra box zero-size quá chặt; lượt thử batch 2 cũng bị ngắt "
         "khi Colab gửi interrupt cả nhóm subprocess. Các lượt này không được tính là E1 hoàn chỉnh. "
         "Sau khi runtime cũ reset, E0 đã lưu được dùng lại; lượt E1 trong bảng được chạy lại từ model gốc."),
        ("CI95", "Độ rộng CI95 đo được: " + "; ".join(f"{n}={ev['f1_ci95'][1]-ev['f1_ci95'][0]}" for n, ev in evidence.items()) +
         ". Scorer bootstrap theo ảnh; frame trong một session có tương quan nên CI có thể lạc quan. "
         "Hai CI overlap không chứng minh bằng nhau; cần paired bootstrap chênh lệch F1, tốt hơn là resample session, "
         "và nhiều seed train để kết luận mạnh hơn. Kết quả paired bootstrap theo ảnh và session ở bảng bổ sung bên dưới. "
         f"Tập test chỉ có {audit['test']['n_sessions']} session, nên cluster CI cũng có độ phân giải rất hạn chế."),
        ("Lỗi theo lớp", f"Lớp có F1 thấp nhất ở {best}: {worst}, F1={best_ev['per_class'][worst]['f1']}. "
         "Xem 6 ảnh bên dưới để phân biệt bỏ sót, nhầm lớp và sai tọa độ. "
         f"Diện tích box trung vị (% khung hình) đo từ GT test: `{stats.get('median_box_area_percent', {})}`. "
         "Box nhỏ và mất chi tiết sau resize/merging là giả thuyết cần đối chiếu ảnh, không phải nguyên nhân đã được chứng minh."),
        ("E4", f"Chỉ đổi train_fraction giữa 0.1 và 1.0 trên nền {best}; "
         f"manifest lưu hai config: {manifest['configs']}. Số bước, LR, rank, seed và effective batch giữ nguyên. "
         "10% được chọn bằng shuffle seed cố định; 100% tái dùng chính checkpoint/prediction nền. "
         "Đây là so sánh cùng ngân sách optimizer steps, khiến 10% được lặp nhiều epoch hơn. "
         "300 bước × effective batch 8 là 2400 lượt ảnh, xấp xỉ 1.69 epoch với 1422 ảnh "
         "và 16.90 epoch với 142 ảnh. Vì vậy không diễn giải đây là so sánh cùng số epoch. "
         f"F1(10%)={low['micro']['f1']}, F1(100%)={full['micro']['f1']}; "
         f"ΔF1(10%−100%)={low['micro']['f1']-full['micro']['f1']}; CI95 riêng có overlap={overlap}. "
         "Không suy ra khác biệt có ý nghĩa chỉ từ overlap; dùng paired CI và ghi rõ đây là một seed.")]
    for i, (title, answer) in enumerate(questions, 1):
        lines += ["", f"### Câu {i} — {title}", "", answer]
    if stats:
        lines += ["", "## Paired bootstrap bổ sung", "",
                  "2.000 lượt, seed=0; cùng một tập ảnh/session được lấy mẫu cho cả hai model. "
                  "Đây là phân tích bổ sung, không thay CI95 do scorer của đề sinh.", "",
                  "| Chênh lệch | ΔF1 | CI95 theo ảnh | CI95 theo session |",
                  "|---|---:|---|---|"]
        for pair, values in stats.get('paired_f1_differences', {}).items():
            lines.append(f"| {pair} | {values['delta_f1']:.6f} | {values['image_ci95']} | {values['session_ci95']} |")
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
    lines += ["## Nhìn lại", "", "Các lỗi đã gặp trong quá trình chạy: torchao cũ làm PEFT không tạo được LoRA; "
              "skip module tên ngắn không giữ được ViT ngoài 4-bit; đếm tham số meta trong ngữ cảnh Accelerate "
              "có thể đếm đôi embedding/lm_head; log subprocess mặc định không hiện đầy đủ trong cell. "
              f"Audit chất lượng nhãn: `{quality}`. Có một box crack zero-height trong train; "
              "hàm encode sửa về cạnh tối thiểu 1 đơn vị thay vì xóa nhãn hoặc đổi split. "
              "Những lỗi này được sửa trước khi thu evidence. "
              "Một seed và ba session test giới hạn khả năng suy rộng; CI chưa bao gồm bất định do seed "
              "huấn luyện hoặc chuyển sang cung đường mới."]
    Path(args.output).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Wrote {args.output}; review qualitative observations before submission")


if __name__ == "__main__":
    main()
