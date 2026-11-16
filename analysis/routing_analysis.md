# Chẩn đoán và Phân tích Định tuyến MoE (Routing Analysis)

Tài liệu này theo dõi các chỉ số hành vi định tuyến của Router.

---

## 1. Mức độ Sử dụng Expert (Expert Utilization / Usage)

| Expert ID | Tỷ lệ Mẫu Kích hoạt (%) | Trọng số Trung bình (Avg Weight) | Tình trạng Hoạt động |
|---|---:|---:|---|
| **Expert 0** | - | - | [ Active / Underutilized / Collapsed ] |
| **Expert 1** | - | - | [ Active / Underutilized / Collapsed ] |
| **Expert 2** | - | - | [ Active / Underutilized / Collapsed ] |
| **Expert 3** | - | - | [ Active / Underutilized / Collapsed ] |

- **Phát hiện Sập nhánh (Collapse Detection)**: Cảnh báo nếu một expert chiếm >60% tải hoặc một expert bị bỏ rơi (<5% tải).

---

## 2. Số lượng Expert Kích hoạt Trung bình trên mỗi Mẫu (Average Experts / Sample)

- **Giá trị đo được**: `-` experts / sample
- **Đánh giá**:
  - Với Top-1: luôn là 1.0.
  - Với Top-2: luôn là 2.0.
  - Với Threshold Routing: biến thiên theo độ phức tạp của câu.

---

## 3. Độ hỗn loạn Định tuyến (Router Entropy)

$$\mathcal{H}(p) = - \sum_{i=1}^N p_i \log p_i$$

- **Entropy Trung bình**: `-`
- **Ý nghĩa**:
  - Entropy cao $\approx \log(N)$: Router phân bổ đều cho các expert, chưa học được sự chuyên biệt hóa.
  - Entropy thấp: Router rất tự tin chọn 1 hoặc vài expert cụ thể.

---

## 4. Phân bố Expert theo Nhãn NLI (Expert Distribution per Class)

| Nhãn Gold | Expert 0 (%) | Expert 1 (%) | Expert 2 (%) | Expert 3 (%) |
|---|---:|---:|---:|---:|
| **Entailment** | - | - | - | - |
| **Contradiction** | - | - | - | - |
| **Neutral** | - | - | - | - |
