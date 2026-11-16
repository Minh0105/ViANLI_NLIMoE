# Phân tích Lỗi Dự đoán (Error Analysis)

Tài liệu này theo dõi và phân loại tối thiểu 100 mẫu dự đoán sai của mô hình baseline trên tập kiểm thử (Test Set).

---

## 1. Bảng Phân loại Loại Lỗi (Error Taxonomy)

| Phân loại lỗi (Error Type) | Số lượng mẫu (Count) | Tỷ lệ (%) | Ví dụ ID tiêu biểu | Mô tả bản chất lỗi |
|---|---:|---:|---|---|
| **Negation** | - | - | | Bỏ sót từ phủ định (*không, chẳng, chưa*) hoặc bị bẫy phủ định kép |
| **Numerical / Quantity** | - | - | | Sai lệch về so sánh số lượng, mốc số (*ít nhất, tối đa, khoảng*) |
| **Temporal** | - | - | | Sai lệch thứ tự trước sau, thời gian diễn ra sự kiện |
| **Lexical** | - | - | | Đồng nghĩa, trái nghĩa, sắc thái từ vựng tiếng Việt |
| **Entity / Reference** | - | - | | Nhầm lẫn danh từ riêng, đại từ chỉ người/vật |
| **World Knowledge** | - | - | | Đòi hỏi tri thức thực tế ngoài câu (*Hà Nội là thủ đô của VN*) |
| **High Lexical Overlap** | - | - | | Premise và Hypothesis giống nhau nhiều từ nhưng ý nghĩa khác nhau |
| **Ambiguous / Annotation**| - | - | | Câu mơ hồ, đa nghĩa hoặc nhãn gán gốc gây tranh cãi |
| **Other** | - | - | | Các trường hợp phức tạp khác |
| **Tổng cộng** | **>=100** | **100%** | | |

---

## 2. Chi tiết Một số Mẫu Lỗi Điển hình

### Mẫu 1: Negation Failure
- **Sample ID**: 
- **Premise**: 
- **Hypothesis**: 
- **Gold Label**: 
- **Predicted Label**: 
- **Phân tích**: 

### Mẫu 2: Numerical Failure
- **Sample ID**: 
- **Premise**: 
- **Hypothesis**: 
- **Gold Label**: 
- **Predicted Label**: 
- **Phân tích**: 
