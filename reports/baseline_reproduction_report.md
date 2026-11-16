> CẬP NHẬT 2026-10-01: Baseline chưa frozen. Kiến trúc/cấu hình đã chỉnh theo
> arXiv v3; xem analysis/paper_extraction.md để biết các giả định còn lại.
> Chưa có kết quả huấn luyện đầy đủ với cấu hình mới. Phần dưới là khung báo
> cáo lịch sử; không coi các tuyên bố freeze là kết quả đã đạt.

# Báo cáo Tổng kết Tái hiện Baseline NLIMoE trên ViANLI
*(Baseline Reproduction Report)*

---

## 1. Mục tiêu Nghiên cứu (Objective)
Tái hiện trung thực mô hình đường cơ sở (Baseline) XLM-RoBERTa Dense và NLIMoE từ bài báo *"A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial Natural Language Inference in Vietnamese"* trên tập dữ liệu `uitnlp/ViANLI`. Kết quả baseline được đóng băng nhằm làm mốc so sánh khoa học, công bằng với mô hình đề xuất **Lightweight MoE**.

---

## 2. Mô hình Gốc của Bài báo (Paper Baseline Overview)
- **Backbone**: `xlm-roberta-base`
- **Kiến trúc MoE**: NLIMoE với Router phân luồng đến các nhánh chuyên gia (Experts).
- **Cơ chế Routing**: Top-K, Static Threshold, Dynamic Threshold.

---

## 3. Thiết lập Tái hiện (Reproduction Setup)
- **Môi trường phần cứng**: GPU NVIDIA GeForce RTX 5080 Laptop GPU (16GB VRAM), OS Windows 11.
- **Thư viện chính**: PyTorch, Hugging Face Transformers, Datasets, Scikit-learn.
- **Seeds cố định**: 42, 123, 2026.
- **Tập dữ liệu**: `uitnlp/ViANLI` (8,012 Train / 1,000 Val / 1,000 Test).

---

## 4. Kết quả Thực nghiệm Tổng hợp (Experimental Results)

| Mô hình / Biến thể | Published | Reproduced Mean | Std | Delta ($\Delta$) | Val Acc | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **XLM-R Dense (B0)** | | | | | | |
| **NLIMoE Top-1 (B1a)** | | | | | | |
| **NLIMoE Top-2 (B1b)** | | | | | | |
| **NLIMoE Static Threshold (B1c)**| | | | | | |
| **NLIMoE Dynamic Threshold (B1d)**| ~47% | | | | | |

---

## 5. Phân tích Hành vi Định tuyến (Routing Behavior)
- **Mức độ sử dụng chuyên gia (Utilization)**:
- **Hiện tượng sập nhánh (Collapse)**:
- **Số lượng chuyên gia trung bình kích hoạt**:
- **Router Entropy**:

---

## 6. Phân tích Sai lệch (Discrepancy Analysis)
- Đánh giá sự khác biệt giữa điểm số tái hiện và công bố.

---

## 7. Giới hạn Nghiên cứu (Limitations)
- Các chi tiết bài báo không công bố (`NOT_REPORTED`).

---

## 8. Đóng băng Baseline (Baseline Recommendation & Freeze)
- **Mã định danh**: `BASELINE_V1_FROZEN`
- **Git Commit Hash**: 
- **Cấu hình chuẩn hóa**: `configs/nlimoe_dynamic.yaml`
- **Quyết định hiện tại**: CHƯA ĐÓNG BĂNG; cần xác minh chi tiết chưa công bố và chạy lại cấu hình mới.
