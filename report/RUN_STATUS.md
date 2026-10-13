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
