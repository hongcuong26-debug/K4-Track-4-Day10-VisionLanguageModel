# Chạy bài trên Colab T4

1. Mở Google Colab → Upload notebook → chọn `lab_day10.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Thêm Colab Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`, cho notebook quyền truy cập.
4. Chạy cell đầu; khi được hỏi, tải lên `lab_day10_colab_bundle.zip` đi kèm.
   Cell clone repo cá nhân rồi bổ sung mã từ bundle nếu GitHub chưa có solution.
5. Chạy các cell theo thứ tự, cho phép mount Drive. Kiểm tra `STUDENT_NAME` và
   `OUTPUT_ROOT`; thư mục này lưu adapter/projector/log riêng tư.
6. E1–E3 chạy 300 bước/lượt. E4 data_size chọn nền bằng validation và chạy 10%
   train; 100% tái dùng kết quả nền. E5 mặc định tắt, có thể bật sau E4.
7. Xem báo cáo nháp, tự phân tích lỗi E0/ảnh và nhận xét cá nhân. Tải
   `day10_results.zip` về, giải nén vào repo máy tính, cập nhật CHECKLIST và commit
   kết quả thật. Chạy `python scripts/check_submission.py` trước khi đẩy GitHub.

Không có khóa Kaggle: tải zip dataset, đưa vào runtime, thay cell prepare bằng
`run('scripts/prepare_data.py', '--zip', '/content/dataset.zip')`.

Nếu Colab ngắt: adapter cuối lượt nằm ở Drive. Sau khi dựng lại môi trường/dữ
liệu, bật cell `RERUN_INFERENCE` cho E1/E2/E3 đã hoàn tất. Checkpoint giữa lượt
chưa lưu optimizer/RNG, nên không hỗ trợ resume huấn luyện chính xác.

Số đo chưa tồn tại ở bản chuẩn bị: `REPORT.md` giữ mẫu, `report/evidence/` sẽ
được tạo khi chạy. Không thay bằng số tham khảo trong đề. Mã không bảo đảm trước
các ngưỡng F1 hoặc đủ VRAM cho mọi phiên T4; kết quả phải được kiểm tra thực tế.

Kiểm thử CPU, với Python đã có thư viện:

```bash
python -m pytest -q tests student_tests
```

Các test của học viên ở `student_tests/`. Không sửa `scripts/`, `tests/`,
`configs/`. Sau khi điền báo cáo, test gốc kiểm mẫu REPORT có thể fail vì nó
yêu cầu marker; điều này khác checker bài nộp yêu cầu xóa marker.
