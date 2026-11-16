# Đối chiếu bài báo — cập nhật 2026-10-01

Nguồn: [arXiv:2406.17716v3](https://arxiv.org/html/2406.17716v3), Section 5 và 6.2.
Các mục REPORTED trong phiên bản tài liệu cũ chưa có căn cứ đã được thay thế.

| Mục | Giá trị công bố | Vị trí |
|---|---|---|
| Train / validation / test | 8012 / 1000 / 1000 | Table 9 |
| Chiều biểu diễn NLIMoE | 1024 | Section 5, pooling |
| Pooling NLIMoE | Mean, gồm special tokens | Eq. 2 |
| Expert | Linear d→d, Dropout, ReLU | Eq. 7 |
| Router | Linear có bias + softmax | Eq. 3 |
| Complexity | Linear d→1 có bias + sigmoid | Eq. 4 |
| Ngưỡng dynamic | 0.1 + 0.1 × C | Eq. 5, Section 6.2 |
| Chọn expert | P > ngưỡng | Eq. 6 |
| Aggregation | Weighted sum chuẩn hóa trên expert được chọn | Eq. 9 |
| Classification | LayerNorm rồi Linear d→3 | Eq. 10 |
| Dynamic experts | 7 | Section 6.2 |
| Top-K ViANLI | N=5, K=4 | Section 6.2, Table 12 |
| Loss | CE + 0.001 × entropy/N + 0.01 × balance | Eq. 12–16 |
| Max length | 256 | Section 6.2 |
| Learning rate | 1e-5 | Section 6.2 |
| Batch size / epochs | 16 / 7 | Section 6.2 |
| Evaluation frequency | 400 | Section 6.2 |
| Weight decay / Adam epsilon | 0 / 1e-8 | Section 6.2 |
| Dropout | 0.4 | Section 6.2 |

## Chưa được công bố hoặc chưa xác minh

- Exact pretrained checkpoint NLIMoE: dùng `xlm-roberta-large` là suy luận
  từ d=1024, không phải tên checkpoint được trích trực tiếp.
- AdamW, no early stopping, seed 42 và chọn best validation accuracy
  là lựa chọn triển khai cho các chi tiết NOT_REPORTED.
- Evaluation frequency được hiểu là số optimizer step; validation cuối run
  được thêm để đảm bảo luôn có checkpoint.
- Phạm vi của dropout 0.4 và scheduler, clipping không được công bố. Các cách hiểu:
  - v3 (`configs/v3/`): dropout 0.4 cả trong encoder (hidden/attention) và
    expert, LR cố định, không warmup. Seed 42 trên A100: B1 và B2 đoán một
    nhãn cho mọi câu (val acc 0.334, macro-F1 0.167), B2 test acc 0.337.
  - v4 (`configs/v4/`): dropout 0.4 chỉ ở expert và head, encoder giữ mặc
    định 0.1; LR tuyến tính với 10% warmup. B0 học bình thường (val 0.457
    ở 59% quá trình). B1, B2 đạt val ~0.43 ở bước 400 rồi sụp từ bước 800
    (val 0.333, macro-F1 0.167, cùng định tuyến cho mọi câu khi đánh giá).
  - v5 (`configs/`, mặc định): như v4 cộng gradient clipping max_norm 1.0
    (mặc định của Hugging Face Trainer; bài báo không nêu).
- Dense head dùng CLS/Dropout/Linear là giả định giữ từ dự án, không tự coi
  là head tác giả.
- Backprop qua mask cứng không được mô tả. Bản equations dùng hard mask nên
  complexity gate không nhận gradient; không thêm straight-through ngầm.
- Khi không chọn expert, MoE vector bằng zero để tránh NaN; đây là quy ước
  số học ngoài công thức chưa xác định mẫu số bằng 0 của bài báo.
- Công thức phụ trợ được giữ cho Top-K; chi tiết thay đổi loss riêng cho
  biến thể này chưa được xác minh từ code tác giả.

Không đánh dấu baseline là frozen hay reproduction chính xác 100% khi những
chi tiết này chưa được giải quyết. Các file .config.json cạnh checkpoint
lưu cấu hình thực tế từng run, bao gồm override CLI.
