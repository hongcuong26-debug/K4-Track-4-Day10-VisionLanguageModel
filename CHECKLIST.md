# Tiến độ Lab Day 10

- [x] Đọc README, GUIDE, RUBRIC, cấu hình và công cụ chấm.
- [x] Giữ nguyên `scripts/`, `tests/`, `configs/`.
- [x] Viết pipeline QLoRA E0–E3; mask loss; lưu/nạp projector riêng.
- [x] Viết E4 data_size 10%/100%, chọn nền bằng validation.
- [x] Viết reward IoU và kiểm thử độc lập ngoài `tests/`.
- [x] Chuẩn bị notebook Colab và công cụ lập báo cáo từ số liệu thật.
- [x] Kết nối Colab T4; dùng tải Kaggle công khai, không cần Secrets; checkpoint ở runtime.
- [x] Tải dữ liệu; xác nhận split/session, `dataset_version`, 6 ảnh GT.
- [x] Đo token ảnh và decode assistant labels bằng processor thật: 220 token, grid [1,22,40], patch 16.
- [x] Chạy E0 đủ 311 prediction test; lưu raw, xác nhận SHA256, chấm lại và vẽ 6 ảnh E0.
- [x] E1 đủ 300 bước, 276 prediction val và 311 prediction test; chấm lại bằng scorer gốc.
- [ ] Chạy E2, E3 đủ 300 bước và đủ 311 prediction test.
- [ ] Chạy E4 và đối chiếu manifest chỉ đổi `train_fraction`.
- [ ] Sinh REPORT.md, 6 ảnh dự đoán; tự viết phân tích hai lỗi E0 và nhận xét cá nhân.
- [x] E1 valid = 0.9839228295819936; F1(E1) − F1(E0) = 0.15927337539883984, đạt hai ngưỡng rubric.
- [x] Tối thiểu 5 commit theo tiến độ thực, không đưa dataset/trọng số/secret vào Git.
- [ ] `check_submission.py`: 0 FAIL; kiểm tra thêm ngưỡng rubric vì checker chưa kiểm hết.
- [ ] (Tùy chọn) Chạy GRPO E5, kiểm tra reward hacking, so sánh F1/CI95 với SFT.

Đã tải dữ liệu d10-21e4070301 và kiểm tra processor trên T4. REPORT.md vẫn là mẫu
cho đến khi notebook có đủ kết quả thật. Không dùng số minh họa của đề làm số đo.
