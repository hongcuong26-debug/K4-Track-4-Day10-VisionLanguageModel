# Bản sao kết quả Colab

Các tệp `.b64` là gzip/base64 của JSON chứa dự đoán thô, evidence,
metadata và nhật ký huấn luyện đã chạy thật. Không chứa dữ liệu ảnh,
khóa API hay trọng số. Dùng để khôi phục kết quả khi runtime Colab bị reset.

| Tệp | SHA256 của JSON sau giải nén |
|---|---|
| e1.b64 | 534aaf7399072d40924e4b258dcf900d79a50bcbf0cd32ada4e0a0e33dc9a20e |

Khôi phục từ thư mục gốc repository bằng `solution.portable_results.restore_snapshot`.
Hàm kiểm tra hash và đường dẫn trước khi ghi; checkpoint trọng số không được khôi phục.

`solution.complete_lab` tự khôi phục snapshot mới nhất trong `manifest.json`
khi runtime mới chưa có metadata E1. Sau đó chấm lại dự đoán thô và kiểm tra
config/metadata trước khi bỏ qua các lượt đã hoàn tất. Không resume train giữa chừng.
