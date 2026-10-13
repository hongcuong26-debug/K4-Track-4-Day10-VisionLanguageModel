# Các bẫy kỹ thuật và giới hạn kiểm chứng

1. **LoRA ViT thiếu `attn.proj`.** Chỉ thêm `qkv`, `linear_fc1`, `linear_fc2`
   được 72 module ViT. Mã dùng regex đủ 4 module/block × 24 block = 96,
   tránh áp LoRA nhầm lên merger. Tổng E1/E2 252; E3 348.
2. **Patch size.** Config chính thức Qwen3-VL-4B có patch_size=16,
   spatial_merge_size=2. Số token phụ thuộc resize thật của processor:
   `sum(prod(image_grid_thw))/merge_size**2`; với resize 640×352 sẽ là 220.
   Không ép ảnh 640×360 thành patch 14 để khớp giải thích trong đề.
3. **PEFT và dtype.** `prepare_model_for_kbit_training` có thể đưa tham số dense
   về fp32. Mã đưa phần thị giác đóng băng và embedding/lm_head về fp16,
   giữ adapter trainable fp32, mở merger/DeepStack fp32. Autocast fp16 vẫn
   có thể tính linear ở fp16; trọng số fp32 không loại bỏ mọi nguy cơ overflow.
4. **Projector ngoài adapter.** Lưu `extra_projector.pt` ở mọi checkpoint.
   Suy luận nạp model mới, kiểm tra đủ key/shape và file projector; thiếu thì dừng.
5. **Mask loss.** Prompt phải là tiền tố chuỗi đầy đủ; labels=-100 cho prompt,
   ảnh và padding; giữ JSON và im_end. Nối pixel_values/image_grid_thw, pad
   mm_token_type_ids nếu processor trả trường này. Model tự shift causal labels.
6. **T4.** Notebook dùng batch 1/grad_accum 8 đồng nhất các setup để giữ effective
   batch 8. Đây là lựa chọn triển khai cần ghi trong báo cáo. Nếu OOM vẫn xảy ra,
   không tự giảm bước hoặc đổi hyperparameter rồi gọi kết quả là cấu hình chuẩn.
7. **Checkpoint.** Adapter/projector được ghi trực tiếp vào Drive mỗi 100 bước
   và cuối lượt. Chưa có resume optimizer/RNG giữa chừng: checkpoint phục vụ
   khôi phục trọng số/suy luận; muốn đủ 300 bước sau ngắt cần chạy lại lượt mới.
   `train.jsonl` ghi bước, loss, LR, gradient norm, VRAM, thời gian thật.
8. **Parse hợp lệ ≠ đúng.** Scorer chấp nhận markdown fence và trích JSON trong
   văn bản; `parse_valid_rate` cao chưa chứng minh box chính xác hay JSON thuần.
9. **CI và selection.** Chọn nền E4 bằng val, giữ test để báo cáo. Bootstrap của
   đề lấy mẫu theo ảnh; frame cùng session tương quan nên CI có thể lạc quan.
   CI overlap không chứng minh hai model tương đương; paired/session bootstrap
   phù hợp hơn nếu cần kết luận thống kê mạnh.
10. **Kiểm tra bài nộp.** Checker chưa kiểm ngưỡng F1 cải thiện hoặc số ảnh minh
    họa. Test gốc yêu cầu REPORT còn marker `<điền` vì kiểm tra mẫu đề, còn
    checker yêu cầu bỏ marker. Chạy test gốc trước khi sinh báo cáo hoàn chỉnh;
    sau khi điền REPORT, riêng test kiểm tra mẫu có thể thất bại. Không sửa test.
11. **Reward.** Công thức của đề cho JSON hợp lệ nhưng không khớp reward=0.2,
    kể cả `[]` trên ảnh có GT. Giant box/box lặp không lấy được reward hoàn hảo
    nhờ threshold/one-to-one/denominator, nhưng vẫn cần theo dõi hành vi thực.

Nguồn đối chiếu:

- [Config model](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json)
- [Qwen3-VL Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [GRPO docs đúng tag v1.9.2](https://github.com/huggingface/trl/blob/v1.9.2/docs/source/grpo_trainer.md)

12. **Colab thực tế (14/10/2026).** T4 dùng torch 2.11.0+cu130/Python 3.13,
    có torchao 0.10.0 cài sẵn. PEFT 0.21.2 từ chối torchao <0.16.0 ngay khi
    tạo LoRA dù bài này dùng BitsAndBytes. Gỡ riêng torchao trong runtime đã
    giải quyết xung đột; 83 test pass, 1 skip trước khi tải dữ liệu.
13. **Log subprocess trong notebook.** Dùng Popen với stdout=PIPE/stderr=STDOUT
    rồi print từng dòng để nhìn thấy loss, tiến độ và traceback trong output cell.
    subprocess.run mặc định có thể đưa log về kernel console thay vì output cell.
14. **Skip modules trên Transformers 5.19.0.** Chạy model thật cho thấy chỉ
    `["visual", "lm_head"]` vẫn lượng tử hóa `model.visual.blocks.0.attn.qkv`.
    Hàm should_convert_module kiểm tra tên đầy đủ hoặc tiền tố đầy đủ; phải thêm
    `"model.visual"` vào skip list. Mã kiểm tra mọi module vision sau khi nạp,
    dừng ngay khi phát hiện Linear4bit để tránh train E2/E3 trên projector sai kiểu.

Đã xác nhận dataset và mask bằng processor thật trên Colab T4: grid [1,22,40],
220 token thị giác, assistant JSON + im_end. Kiểm thử CPU không thay thế kết quả
huấn luyện/suy luận đủ tập test hoặc bằng chứng GRPO.
