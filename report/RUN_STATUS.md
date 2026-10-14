# Nhật ký chạy thật trên Colab T4

Notebook: https://colab.research.google.com/drive/16oFysX3Xsyf-6nn93L04yczaPDRwxdZ0

- Dataset: `d10-21e4070301`; 1422/276/311 ảnh train/val/test.
- Runtime: Tesla T4, PyTorch 2.11.0+cu130, Python 3.13; thư viện còn lại theo requirements của đề.
- E0 đã hoàn tất val/test. `e0.json` được sao chép nguyên các trường/số từ đầu ra scorer hiển thị ở ô 7. Bản dự đoán thô test đã được chuyển nguyên byte về máy bằng gzip/base64 trong đầu ra notebook do tải tệp Colab không hoạt động: 52466 byte, SHA256 `0934507f8dd8cb65bd81925eebfd02e82a92f5902465828782f2b65ce59cf764`; kiểm tra hash thành công. Dự đoán lưu ở `work/e0_test.jsonl` (không commit).
- E1 thử nghiệm batch 1 dừng sau 11 bước vì một box crack có chiều cao YOLO bằng 0. Nhật ký được giữ trong `e1_batch1_failed_trial`; đây không phải thí nghiệm E1 hoàn chỉnh.
- Sau commit `19a531c`, kiểm tra E3 batch 2 trên hai ảnh có 13 và 11 box thành công: 348 targets LoRA, 148426752 tham số trainable, loss 0.904615581035614, gradient norm 2.8389151096343994, peak allocated VRAM 7.276648448 GB. Chỉ chạy 1 bước để kiểm tra bộ nhớ, không dùng làm evidence E3.
- Pipeline bắt buộc E1–E4 khởi động lại lúc `2026-10-13T21:45:37Z`, PID 18771. E1/E2/E3 chạy batch 2 × accumulation 4, 300 optimizer steps. E4 chọn nền bằng validation rồi chỉ đổi train_fraction.
- Lượt PID 18771 bị ngắt khi dừng ô monitor của notebook; Colab gửi SIGINT cả nhóm tiến trình. Có checkpoint bước 100 nhưng chưa lưu optimizer/RNG, nên không dùng để giả vờ resume chính xác. Nhật ký và checkpoint được giữ trong `e1_batch2_interrupted_trial`. Pipeline mới PID 25005 được chạy với `start_new_session=True`, bắt đầu lại E1 từ model gốc. Đã mở hộp thoại mount Drive và đang chờ xác nhận quyền truy cập để sao lưu checkpoint.

Chưa đủ điều kiện nộp cho đến khi có E1–E4 hoàn chỉnh, báo cáo/ảnh dự đoán thật và checker đạt 0 FAIL.

## Kiểm tra lại ngày 14/10/2026

`check_submission.py` trên repo hiện trả về **5 FAIL, 0 WARN**: thiếu evidence E1,
E2, E3, E4 và REPORT còn placeholder. E0, dataset_version, 20 commit và danh sách
file được track đều đạt kiểm tra.

Kết nối lại Colab đã cấp runtime T4 mới. Kiểm tra trong kernel xác nhận thư mục
repository và `/content/day10-vlm/runs` đều không còn; không có file evidence
hay `lab_progress.json` của phiên trước trên runtime mới. Vì vậy không thể xác
nhận hoặc lấy lại E1–E4 từ phiên cũ. E0 và sáu ảnh đã được lưu về máy/GitHub nên
vẫn còn. Quyền kết nối Drive chưa được xác nhận; phải thiết lập sao lưu trước
khi chạy lại các lượt dài để tránh mất checkpoint lần nữa.

## Lượt hiện tại: E1 đã hoàn thành

Pipeline khởi động từ model gốc lúc `2026-10-14T04:18:27Z`, trên T4 mới,
source `aca2e5e`, batch 2 × accumulation 4, 300 optimizer steps. Không mount Drive;
sau mỗi chặng đã hoàn tất, pipeline xuất gzip/base64 chứa JSON kết quả vào output
notebook đang lưu trên Drive. Đây là bản sao dự đoán/metadata/log, không phải trọng số.

E1 hoàn thành train và suy luận val/test trước `2026-10-14T05:51:48Z`.
Đã lưu snapshot lên GitHub cá nhân, pull về máy và xác minh SHA256 payload:
`534aaf7399072d40924e4b258dcf900d79a50bcbf0cd32ada4e0a0e33dc9a20e`.
`solution.complete_lab.completed_result('e1', 'work/runs')` trả về `True`:
đủ 300 bước, config khớp đề, đầy đủ ID val/test không trùng, metadata khớp,
chấm lại bằng scorer gốc khớp các metric/CI95.

- Tham số trainable: 33030144; peak allocated VRAM: 6.92527104 GB.
- Thời gian vòng train: 53.60908415115 phút; không gồm tải model và suy luận.
- Test: 311 ảnh; parse valid 0.9839228295819936; F1 0.2601908065915004.
- CI95 F1: [0.22383985441310283, 0.2981029810298103].
- F1 tăng 0.15927337539883984 so với E0, đạt ngưỡng ≥ 0.10.
- F1 pothole/crack/manhole: 0.1693121693121693 / 0.18636995827538247 / 0.5469387755102041.

Dự đoán thô và nhật ký E1 hiện được lưu lâu dài dưới `report/predictions/`,
`report/training/`, `report/validation/` và `report/snapshots/e1.b64`.
E2 đang huấn luyện. E2/E3/E4 và REPORT cuối cùng còn chờ kết quả thật;
bài chưa đủ điều kiện nộp.
