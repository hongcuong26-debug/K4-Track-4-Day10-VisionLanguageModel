# Bản sao kết quả Colab

Các tệp `.b64` là gzip/base64 của JSON chứa dự đoán thô, evidence,
metadata và nhật ký huấn luyện đã chạy thật. Không chứa dữ liệu ảnh,
khóa API hay trọng số. Dùng để khôi phục kết quả khi runtime Colab bị reset.

| Tệp | SHA256 của JSON sau giải nén |
|---|---|
| e1.b64 | 534aaf7399072d40924e4b258dcf900d79a50bcbf0cd32ada4e0a0e33dc9a20e |

Khôi phục từ thư mục gốc repository bằng `solution.portable_results.restore_snapshot`.
Hàm kiểm tra hash và đường dẫn trước khi ghi; checkpoint trọng số không được khôi phục.
