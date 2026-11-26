# Kết quả v5 (3 seed) — cập nhật 2026-11-25

Cấu hình v5: encoder dropout 0,1, lr tuyến tính với 10% warmup, gradient clipping 1,0
(xem [paper_extraction.md](paper_extraction.md)). Tập kiểm thử ViANLI 1.000 câu; seed 42 / 123 / 2026;
A100 40 GB, bf16, batch 16.

## Mã thực nghiệm

Báo cáo dùng mã E/A; tên trong code (cột cuối) giữ nguyên để hàng đợi nhận ra các lần chạy đã xong.

| Mã | Mô hình | Vai trò | Tên trong code |
|---|---|---|---|
| E0 | XLM-R Large | Baseline | B0 |
| E1 | NLIMoE (bài báo) | Baseline | B1 |
| E2 | E1 + STE | Đề xuất chính | B2 |
| E3a / E3b | E2 + bottleneck r = 256 / 384 | Lightweight | B3 / B3_r384 |
| E4a / E4b | E2 + phạt số chuyên gia λ = 0,02 / 0,1 | Lightweight | B4_c002 / B4_c01 |
| E5a / E5b | XLM-R Base / + MoE + STE | Lightweight | B0_base / B2_base |
| E6 | E5b chưng cất từ E2 | Lightweight | B5_kd |
| A1 / A2 | Top-K / bỏ L_bal | Ablation | A_topk / A_nobal |
| A3a / A3b | STE T = 0,05 / 0,2 | Ablation | B2_T005 / B2_T02 |

Các bảng dưới viết theo tên trong code: B0 = E0, B1 = E1, B2 = E2, B3 = E3a, A_topk = A1, A_nobal = A2.

## Accuracy và macro-F1 (trung bình ± độ lệch chuẩn mẫu)

| Biến thể | Accuracy (%) | Macro-F1 (%) | Chuyên gia/câu | Accuracy từng seed |
|---|---|---|---|---|
| B0 · XLM-R Large | 45,8 ± 4,1 | 40,0 ± 7,2 | 1 | 49,1 / 47,0 / 41,2 |
| B1 · NLIMoE | 44,1 ± 1,9 | 43,5 ± 1,3 | 2,7 / 7 | 46,3 / 42,6 / 43,4 |
| B2 · + STE | 45,8 ± 2,2 | 45,6 ± 2,0 | 4,0 / 7 | 47,6 / 43,3 / 46,4 |
| B3 · + bottleneck r = 256 | 41,0 ± 6,4 | 38,1 ± 9,3 | 2,8 / 7 | 40,6 / 47,5 / 34,8 |
| A_topk · Top-K (N = 5, K = 4) | 44,5 ± 0,2 | 43,2 ± 1,0 | 4 / 5 | 44,3 / 44,4 / 44,7 |
| A_nobal · B2 bỏ L_bal | 44,1 ± 2,2 | 42,3 ± 2,9 | 1,8 / 7 | 45,6 / 41,6 / 45,0 |

Bài báo: XLM-R Large 45,5%, NLIMoE 47,3% (một lần chạy).

Sụp một phần: B0 seed 123 / 2026 gần như không dự đoán nhãn mâu thuẫn (4% / 0% số câu);
B3 seed 2026 gần như không dự đoán nhãn suy ra (1%); A_nobal seed 42 dùng 1 chuyên gia cho mọi câu.

## McNemar chính xác, gộp 3 seed (3.000 cặp)

| So sánh | Chênh (điểm) | n₀₁ / n₁₀ | p |
|---|---|---|---|
| B2 vs B1 | +1,7 | 337 / 287 | 0,0497 |
| B2 vs B0 | 0,0 | 497 / 497 | 1,00 |
| B1 vs B0 | −1,7 | 452 / 502 | 0,11 |
| B3 vs B2 | −4,8 | 480 / 624 | < 0,001 |
| A_topk vs B1 | +0,4 | 410 / 399 | 0,73 |
| A_nobal vs B2 | −1,7 | 353 / 404 | 0,07 |

B2 hơn B1 ở cả 3 seed (+1,3 / +0,7 / +3,0). Ensemble trung bình xác suất 3 seed: B0 49,2%, B2 47,9%, B1 45,6%.

## Gradient của nhánh độ phức tạp

Ghi ‖∂L/∂w_c‖ mỗi 20 bước (175 lần × 3 seed):

| | Số lần bằng 0 | Lớn nhất | C trung bình đầu → cuối huấn luyện |
|---|---|---|---|
| B1 | 525 / 525 | 0 | 0,53 → 0,43 · 0,33 → 0,35 · 0,41 → 0,57 |
| B2 | 0 / 525 | 0,72 | 0,53 → 0,25 · 0,33 → 0,28 · 0,41 → 0,26 |

C của B1 trôi theo hướng khác nhau giữa các seed (do h thay đổi, w_c đứng yên); C của B2 giảm nhất quán.

## C có phản ánh độ khó không? (Spearman, tập kiểm thử)

ρ_s(C, đoán sai), theo seed 42 / 123 / 2026:

| | ρ_s(C, sai) | p | Accuracy theo 3 nhóm C (thấp / giữa / cao) | Chuyên gia/câu theo nhóm C |
|---|---|---|---|---|
| B1 s42 | +0,074 | 0,018 | 50,9 / 42,6 / 45,3 | 2,07 / 2,21 / 2,02 |
| B1 s123 | −0,020 | 0,52 | 39,8 / 45,9 / 42,0 | 4,54 / 3,76 / 3,05 |
| B1 s2026 | −0,087 | 0,006 | 40,4 / 38,1 / 51,7 | 1,69 / 2,41 / 2,11 |
| B2 s42 | −0,072 | 0,022 | 45,5 / 45,0 / 52,3 | 4,37 / 4,00 / 3,09 |
| B2 s123 | −0,125 | < 0,001 | 39,8 / 38,4 / 51,7 | 4,59 / 4,57 / 3,33 |
| B2 s2026 | −0,074 | 0,019 | 43,1 / 46,2 / 49,8 | 4,71 / 4,13 / 3,40 |

- B1: dấu của tương quan đổi giữa các seed, phù hợp với việc w_c không được huấn luyện.
- B2: tương quan âm, có ý nghĩa ở cả 3 seed. Câu có C cao được đoán đúng nhiều hơn và dùng ít chuyên gia hơn
  (ρ = 0,1 + 0,1·C cao hơn). Như vậy, với công thức của bài báo, C học được đóng vai trò điểm "dễ"
  chứ không phải "khó": câu khó nhận C thấp, ngưỡng thấp và nhiều chuyên gia hơn.
- Giả thuyết H1 trên slide ("C tương quan dương với tỷ lệ đoán sai") không đúng về dấu; điều được xác nhận là
  C của B2 có quan hệ đơn điệu, nhất quán với độ khó, còn C của B1 thì không. Độ lớn tương quan nhỏ (|ρ_s| ≤ 0,13).

## Thí nghiệm bổ sung (chạy 2026-11-25/26 trên L4, cùng cấu hình v5)

Accuracy và macro-F1 trên tập kiểm thử, trung bình ± độ lệch chuẩn qua seed 42 / 123 / 2026:

| Mã | Mô hình | Accuracy (%) | Macro-F1 (%) | Val (%) | Chuyên gia/câu | Từng seed |
|---|---|---|---|---|---|---|
| E2 | tham chiếu (T = 0,1) | 45,8 ± 2,2 | 45,6 ± 2,0 | 46,4 | 4,02 | 47,6 / 43,3 / 46,4 |
| E3b | bottleneck r = 384 | 45,7 ± 2,1 | 44,7 ± 2,1 | 47,2 | 3,65 | 46,9 / 47,0 / 43,3 |
| E4a | phạt chi phí λ = 0,02 | 45,2 ± 0,7 | 41,9 ± 2,9 | 46,0 | 1,79 | 44,8 / 44,7 / 46,0 |
| E4b | phạt chi phí λ = 0,1 | 44,7 ± 0,7 | 43,5 ± 2,1 | 47,2 | 1,00 | 44,2 / 44,4 / 45,5 |
| E5a | XLM-R Base | 43,1 ± 0,9 | 36,7 ± 1,3 | 46,0 | 1 | 42,4 / 42,7 / 44,1 |
| E5b | XLM-R Base + MoE + STE | 44,5 ± 0,6 | 39,6 ± 3,3 | 45,1 | 3,15 | 43,9 / 44,5 / 45,0 |
| E6 | E5b chưng cất từ E2 | 44,3 ± 1,0 | 42,9 ± 2,6 | 45,6 | 2,67 | 43,7 / 43,7 / 45,5 |
| A3a | E2, T = 0,05 | 46,0 ± 0,9 | 45,9 ± 0,7 | 46,4 | 4,25 | 44,9 / 46,6 / 46,4 |
| A3b | E2, T = 0,2 | 46,0 ± 2,1 | 44,4 ± 1,9 | 46,6 | 3,50 | 44,1 / 48,3 / 45,7 |

McNemar chính xác, gộp 3 seed (3.000 cặp); "thắng" là số seed mà biến thể thứ nhất có accuracy cao hơn:

| So sánh | Chênh | n₀₁ / n₁₀ | p | Thắng |
|---|---|---|---|---|
| E3b vs E2 | 0,0 | 398 / 399 | 1,00 | 1/3 |
| E3b vs E3a | +4,8 | 610 / 467 | < 0,001 | 2/3 |
| E4a vs E2 | −0,6 | 450 / 468 | 0,57 | 1/3 |
| E4b vs E2 | −1,1 | 332 / 364 | 0,24 | 1/3 |
| E5b vs E5a | +1,4 | 283 / 241 | 0,07 | 3/3 |
| E5b vs E1 | +0,4 | 502 / 491 | 0,75 | 2/3 |
| E5b vs E0 | −1,3 | 251 / 290 | 0,10 | 1/3 |
| E6 vs E5b | −0,2 | 375 / 380 | 0,88 | 1/3 |
| E6 vs E0 | −1,5 | 405 / 449 | 0,14 | 1/3 |
| A3a vs E2 | +0,2 | 361 / 355 | 0,85 | 1/3 |
| A3b vs E2 | +0,3 | 390 / 382 | 0,80 | 1/3 |

Ensemble trung bình xác suất 3 seed: E0 49,2%, A3b 49,0%, E2 47,9%, E3b 47,1%, A3a 46,6%, E6 45,1%, E5b 44,5%, E4a 44,5%, E5a 43,4%.

Nhận xét:
- Nhiệt độ STE: T = 0,05 / 0,1 / 0,2 cho accuracy 46,0 / 45,8 / 46,0, không khác biệt có ý nghĩa; STE không nhạy với T trong khoảng này. Giữ T = 0,1.
- Bottleneck: r = 384 (−24,9% tham số khối chuyên gia) giữ nguyên accuracy của E2; r = 256 thì không.
- Phạt chi phí: λ = 0,02 giảm số chuyên gia từ 4,02 xuống 1,79 mỗi câu (−55%), accuracy giảm 0,6 điểm, không có ý nghĩa thống kê. λ = 0,1 đẩy mọi câu về 1 chuyên gia, tức MoE suy biến thành một chuyên gia duy nhất. Macro-F1 của E4a thấp hơn (seed 2026 chỉ dự đoán mâu thuẫn ở 4% số câu).
- XLM-R Base: MoE + STE thêm 1,4 điểm so với Base thuần (thắng 3/3 seed, p = 0,07) và ngang NLIMoE trên Large (E1, p = 0,75); kém E0 1,3 điểm (p = 0,10).
- Chưng cất: không thay đổi accuracy so với E5b, nhưng macro-F1 tăng 3,3 điểm và dự đoán cân bằng hơn giữa 3 nhãn (E5a, E5b seed 2026 gần như không dự đoán mâu thuẫn, dưới 0,5% số câu; E6 seed 2026 dự đoán mâu thuẫn 37%).
- Chi phí: phần encoder của XLM-R Base có khoảng 85 triệu tham số ngoài embedding, so với khoảng 303 triệu của Large, nên số phép nhân-cộng mỗi câu ít hơn khoảng 3,5 lần (ước tính theo tham số, chưa đo độ trễ; xem scripts/efficiency.py).
- Thời gian huấn luyện trên L4 sau tối ưu (batch 16 × 1, checkpoint ghi một lần, AdamW fused): khoảng 13 phút mỗi lần chạy Large, khoảng 15 phút mỗi lần chạy Base khi 3 job Base chạy cùng lúc.
