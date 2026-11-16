# Phân tích Sai lệch Kết quả (Discrepancy Analysis)

Tài liệu này dùng để điều tra, phân tích nguyên nhân nếu kết quả tái hiện (Reproduced) có độ chênh lệch so với kết quả bài báo công bố (Published).

---

## 1. So sánh Tổng quan

| Mô hình | Kết quả Công bố (Published) | Kết quả Tái hiện (Reproduced Mean) | Độ lệch ($\Delta = \text{Rep} - \text{Pub}$) |
|---|---:|---:|---:|
| **XLM-R Dense** | - | - | - |
| **NLIMoE Top-K** | - | - | - |
| **NLIMoE Static Threshold** | - | - | - |
| **NLIMoE Dynamic Threshold** | ~47% | - | - |

---

## 2. Danh mục Kiểm tra Nguyên nhân Tiềm năng (Checklist)

### 2.1 Dữ liệu & Tiền xử lý (Data & Tokenization)
- [ ] Phiên bản dataset trên Hugging Face có trùng khớp với bản tác giả dùng lúc viết báo?
- [ ] Label mapping có đúng thứ tự không (`0: entailment, 1: neutral, 2: contradiction`)?
- [ ] Định dạng cặp câu: `<s> P </s></s> H </s>` hay `<s> P </s> H </s>`?
- [ ] Max sequence length có gây cắt cụt (truncation) thông tin quan trọng ở premise/hypothesis dài không?

### 2.2 Kiến trúc Mô hình (Model Architecture)
- [ ] Checkpoint backbone: `xlm-roberta-base` hay `xlm-roberta-large`?
- [ ] Pooling: Dùng vector `[CLS]` token hay Mean pooling qua toàn bộ chuỗi?
- [ ] Cấu trúc Expert: Kích thước tầng ẩn trung gian và hàm kích hoạt (GELU vs ReLU)?
- [ ] Router: Tuyến tính hay phi tuyến?

### 2.3 Quá trình Huấn luyện (Optimization & Training)
- [ ] Learning rate cho backbone và router/experts.
- [ ] Batch size thực tế và Gradient accumulation steps.
- [ ] Warmup ratio và Learning rate scheduler.
- [ ] Số epochs và cơ chế Early Stopping.

### 2.4 Cơ chế Định tuyến MoE (MoE Specifics)
- [ ] Softmax temperature trong router.
- [ ] Trọng số cân bằng tải (Load balancing auxiliary loss coefficient).
- [ ] Ngưỡng kích hoạt static threshold $\tau$ được chọn như thế nào.

---

## 3. Kết luận Điều tra

```text
Likely causes (Nguyên nhân khả dĩ nhất):
1. 
2. 
3. 

Unrecoverable details (Thông tin paper không công bố và không thể khôi phục):
1. 
2. 

Confidence in reproduction (Mức độ tin cậy của bản tái hiện):
[ LOW / MEDIUM / HIGH ]
```
