# Phân tích E1 đã chạy thật

E1 huấn luyện đủ 300 bước trên 1422 ảnh train; dùng batch 2, accumulation 4,
LoRA rank 16/alpha 32 trên LLM. ViT và projector đóng băng.
Nguồn số đo: [evidence](evidence/e1.json), [metadata](training/e1_run.json),
[nhật ký 300 bước](training/e1_train.jsonl), [dự đoán test](predictions/e1_test.jsonl).

F1 test tăng từ 0.10091743119266056 (E0) lên 0.2601908065915004 (E1).
JSON hợp lệ tăng từ 275/311 lên 306/311 ảnh; năm ảnh E1 còn lỗi định dạng.
E1 đạt cả hai ngưỡng rubric: valid ≥ 0.95 và tăng F1 ≥ 0.10.
Đây chưa phải kết luận cho E2, E3 hoặc E4.

E1 có TP=150, FP=420, FN=433. Số FP tăng so với E0 (245), đồng thời TP
tăng mạnh (44 → 150); vì vậy không nên diễn giải F1 tăng thành mọi loại lỗi đều giảm.
F1 manhole=0.5469387755102041, crack=0.18636995827538247,
pothole=0.1693121693121693. Crack chiếm 353/420 FP của E1.

Bootstrap ghép cặp 2000 lần, seed 0, dùng cùng các ảnh được lấy mẫu cho E0 và E1:
CI95 của ΔF1 là [0.12171894566445739, 0.19842672051152616].
Bootstrap theo session cho [0.14296049378193737, 0.4081632653061225].
Test chỉ có ba session nên CI theo session rất rời rạc và chưa phản ánh đủ
biến thiên giữa các cung đường mới. Xem [số đo bootstrap](e1_analysis.json).

## Sáu ảnh kiểm tra

Box xanh lá là nhãn gốc, đỏ là dự đoán. Chọn tất định ba ảnh ít lỗi nhất
và ba ảnh nhiều lỗi nhất, với thứ tự hòa theo ID; không dùng các ảnh này
để ước tính tỷ lệ thành công của toàn bộ tập test.

| Ảnh | TP | FP | FN | Quan sát |
|---|---:|---:|---:|---|
| [e1_1](figures/predictions/e1_1.jpg) | 2 | 0 | 0 | Hai nắp cống được định vị đúng. |
| [e1_2](figures/predictions/e1_2.jpg) | 2 | 0 | 0 | Hai nắp cống khi xe tới gần hơn. |
| [e1_3](figures/predictions/e1_3.jpg) | 2 | 0 | 0 | Hai nắp cống gần camera; box vẫn khớp IoU ≥ 0.5. |
| [e1_4](figures/predictions/e1_4.jpg) | 1 | 5 | 6 | Dự đoán toàn crack, bỏ sót cả bốn pothole; nhiều box lệch vùng GT. |
| [e1_5](figures/predictions/e1_5.jpg) | 0 | 9 | 1 | Chín box crack dọc mép vỉa hè, không khớp box crack GT duy nhất. |
| [e1_6](figures/predictions/e1_6.jpg) | 1 | 6 | 4 | Gán pothole/crack cùng một vùng ở hai cặp box; chỉ một pothole khớp. |

Ba ảnh đúng là các frame gần nhau trong cùng một session; hai ảnh lỗi e1_4/e1_6
cũng là frame gần nhau. Các ví dụ này minh họa trực tiếp tương quan video:
không thể coi mỗi ảnh là một tình huống độc lập hoàn toàn.
ID ảnh và JSON sinh ra đầy đủ ở [manifest](figures/predictions/e1_manifest.json).

## Tài nguyên

33030144 tham số trainable, peak allocated VRAM 6.92527104 GB,
53.60908415115 phút cho vòng train. Đây là bộ nhớ tensor do PyTorch cấp phát,
không phải toàn bộ VRAM hiển thị trong nvidia-smi; thời gian không gồm tải model,
suy luận val/test hay các lượt thử đã bị gián đoạn.

Kiểm tra hiện tại: scorer gốc chấm lại val/test khớp evidence;
`scripts/`, `tests/`, `configs/` không có diff so với đề.
Checker còn 4 FAIL do E2/E3/E4 chưa hoàn tất và REPORT cuối còn placeholder.
