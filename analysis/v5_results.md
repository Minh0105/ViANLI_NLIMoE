# Kết quả v5 (3 seed) — cập nhật 2026-11-27

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
- Chi phí: phần encoder của XLM-R Base có khoảng 85 triệu tham số ngoài embedding, so với khoảng 303 triệu của Large, nên số phép nhân-cộng mỗi câu ít hơn khoảng 3,5 lần; số đo thật ở mục "Hiệu năng suy luận" bên dưới.
- Thời gian huấn luyện trên L4 sau tối ưu (batch 16 × 1, checkpoint ghi một lần, AdamW fused): khoảng 13 phút mỗi lần chạy Large, khoảng 15 phút mỗi lần chạy Base khi 3 job Base chạy cùng lúc.

## Thí nghiệm P2 của REVISION_PLAN (chạy 2026-11-26/27 trên L4, cùng cấu hình v5)

| Mã | Mô hình | Tên trong code | Accuracy (%) | Macro-F1 (%) | Val (%) | Chuyên gia/câu | Từng seed |
|---|---|---|---|---|---|---|---|
| A4 | XLM-R Large + mean pooling + LN | A_mean_ln | 45,7 ± 1,2 | 45,2 ± 0,6 | 46,5 | — | 44,7 / 45,4 / 47,1 |
| A5 | 1 chuyên gia, không định tuyến | A_1exp | 45,7 ± 0,6 | 45,5 ± 0,7 | 47,6 | 1,00 | 45,0 / 46,2 / 45,9 |
| A6 | E2, empty_selection_policy = zero_safe | B2_safe | 45,8 ± 1,3 | 44,7 ± 1,7 | 47,7 | 3,62 | 44,3 / 46,6 / 46,5 |
| E3c | residual bottleneck r = 256 | B3_res256 | 45,1 ± 1,4 | 44,3 ± 1,3 | 46,8 | 4,22 | 45,0 / 43,7 / 46,5 |
| E7 | E2 + trọng số lớp [1; 1; 1,2] | B2_cw | 45,7 ± 1,0 | 45,2 ± 0,5 | 47,1 | 4,37 | 46,8 / 44,9 / 45,4 |
| E8a | E5b chưng cất từ ensemble E0 + E2 | B5_kd_ens | 45,6 ± 0,2 | 45,2 ± 0,4 | 45,9 | 2,75 | 45,6 / 45,4 / 45,7 |
| E8b | E2 chưng cất từ ensemble E0 + E2 | B2_kd_ens | **47,3 ± 0,8** | **47,0 ± 0,5** | 48,2 | 2,00 | 48,1 / 46,6 / 47,3 |

Recall nhãn mâu thuẫn (trung bình 3 seed): E2 38,5%, E7 32,1%, E8b 35,6%, A4 40,6%, A5 38,2%.
Trên tập kiểm thử không câu nào rơi vào trường hợp không chọn chuyên gia (empty_selection_pct = 0 ở mọi lần chạy MoE).

## Kiểm định chính thức: bootstrap theo câu (scripts/analyze_results.py)

Bootstrap 10.000 lần trên 1.000 câu, giữ các seed của một câu cùng nhau; p hai phía từ hoán vị đổi dấu.
McNemar ensemble: McNemar chính xác giữa hai ensemble trung bình xác suất 3 seed. McNemar gộp 3.000 cặp
(các mục ở trên) chỉ còn là số tham khảo vì các cặp cùng câu không độc lập.

| So sánh | Chênh | CI 95% | p bootstrap | Thắng | Ensemble A / B | p McNemar ensemble |
|---|---|---|---|---|---|---|
| E1 vs E0 | −1,7 | [−4,0; +0,7] | 0,17 | 1/3 | 49,2 / 45,6 | 0,02 |
| E2 vs E1 | +1,7 | [+0,0; +3,3] | **0,05** | 3/3 | 45,6 / 47,9 | 0,07 |
| E2 vs E0 | 0,0 | [−2,4; +2,5] | 1,00 | 1/3 | 49,2 / 47,9 | 0,45 |
| E3a vs E2 | −4,8 | [−6,8; −2,8] | < 0,001 | 1/3 | 47,9 / 46,4 | 0,32 |
| E3b vs E2 | 0,0 | [−1,9; +1,8] | 1,00 | 1/3 | 47,9 / 47,1 | 0,60 |
| E3c vs E3a | +4,1 | [+1,6; +6,5] | < 0,001 | 2/3 | 46,4 / 45,9 | 0,80 |
| E3c vs E2 | −0,7 | [−2,6; +1,3] | 0,51 | 2/3 | 47,9 / 45,9 | 0,16 |
| E4a vs E2 | −0,6 | [−2,7; +1,5] | 0,59 | 1/3 | 47,9 / 44,5 | 0,04 |
| E4b vs E2 | −1,1 | [−2,7; +0,5] | 0,21 | 1/3 | 47,9 / 48,1 | 0,94 |
| E5b vs E5a | +1,4 | [−0,1; +3,0] | 0,09 | 3/3 | 43,4 / 44,5 | 0,44 |
| E5b vs E1 | +0,4 | [−2,0; +2,8] | 0,79 | 2/3 | 45,6 / 44,5 | 0,53 |
| E5b vs E0 | −1,3 | [−2,9; +0,3] | 0,12 | 1/3 | 49,2 / 44,5 | 0,00 |
| E6 vs E5b | −0,2 | [−2,1; +1,8] | 0,89 | 1/3 | 44,5 / 45,1 | 0,72 |
| E6 vs E0 | −1,5 | [−3,6; +0,6] | 0,18 | 1/3 | 49,2 / 45,1 | 0,01 |
| A1 vs E1 | +0,4 | [−1,4; +2,1] | 0,71 | 2/3 | 45,6 / 45,5 | 1,00 |
| A2 vs E2 | −1,7 | [−3,6; +0,2] | 0,09 | 0/3 | 47,9 / 46,7 | 0,43 |
| A3a vs E2 | +0,2 | [−1,4; +1,8] | 0,84 | 1/3 | 47,9 / 46,6 | 0,28 |
| A3b vs E2 | +0,3 | [−1,5; +2,0] | 0,80 | 1/3 | 47,9 / 49,0 | 0,46 |
| A4 vs E0 | 0,0 | [−2,5; +2,4] | 1,00 | 1/3 | 49,2 / 47,0 | 0,16 |
| A5 vs A4 | 0,0 | [−1,8; +1,7] | 1,00 | 2/3 | 47,0 / 46,5 | 0,75 |
| A5 vs E1 | +1,6 | [+0,0; +3,2] | 0,05 | 2/3 | 45,6 / 46,5 | 0,50 |
| A6 vs E2 | 0,0 | [−1,6; +1,6] | 1,00 | 2/3 | 47,9 / 47,9 | 1,00 |
| E7 vs E2 | −0,1 | [−2,0; +1,8] | 0,97 | 1/3 | 47,9 / 47,9 | 1,00 |
| E8a vs E6 | +1,3 | [−0,3; +2,9] | 0,14 | 3/3 | 45,1 / 46,1 | 0,42 |
| E8b vs E2 | +1,6 | [−0,1; +3,3] | 0,07 | 3/3 | 47,9 / 48,6 | 0,61 |

"Thắng": số seed mà biến thể thứ nhất có accuracy cao hơn. Cột ensemble: A là biến thể đứng sau "vs", B là biến thể đứng trước.

Nhận xét:
- STE (E2 vs E1) có ý nghĩa ở mức biên theo bootstrap (p = 0,05) và McNemar ensemble (p = 0,07); con số pooled 0,0497 trước đây không bị thổi phồng nhiều.
- Đối chứng: A4 (pooling + LN, không MoE) và A5 (một chuyên gia luôn bật) đều đạt 45,7%, ngang E2 (45,8%). Phần lớn lợi thế macro-F1 của E2 so với E0 (45,6 so với 40,0) đến từ pooling + LN (A4: 45,2). Trên ViANLI, khối 7 chuyên gia có định tuyến chưa cho thêm accuracy so với một chuyên gia; NLIMoE gốc còn kém A5 1,6 điểm (p = 0,05). Đóng góp của ngưỡng học được là điều khiển được chi phí (E4a, E8b), không phải accuracy.
- Residual bottleneck (E3c) chữa sụp của E3a với cùng số tham số (+4,1 điểm, p < 0,001) và không kém E2 có ý nghĩa (p = 0,51).
- Trọng số lớp (E7) không cải thiện nhãn mâu thuẫn: recall C giảm từ 38,5% xuống 32,1%.
- Chưng cất ensemble: E8b là mô hình đơn tốt nhất (47,3 ± 0,8%, macro-F1 47,0%), thắng E2 ở 3/3 seed nhưng p = 0,07, và dùng 2,0 chuyên gia mỗi câu so với 4,0 của E2. E8a là mô hình Base tốt nhất và ổn định nhất (độ lệch chuẩn 0,2). Chưa có kiểm định E8b vs E0 / E1.
- Ensemble của E4a thấp hơn E2 (44,5 so với 47,9, p = 0,04) dù trung bình từng seed không khác biệt; có thể các seed E4a sai giống nhau hơn nên ensemble ít lợi hơn (chưa kiểm tra).

## Gradient của nhánh độ phức tạp, mọi biến thể

- ‖∂L/∂w_c‖ = 0 ở 175/175 lần đo mỗi seed của E1, A1 (Top-K) và A5 (một chuyên gia), như dự đoán.
- Mọi biến thể có STE: 0 lần bằng 0, trừ E4b có 1 lần ở seed 42 và seed 123 (khi mọi câu đã về 1 chuyên gia).
- Gai gradient: giá trị lớn nhất 2,0·10⁹ / 2,7·10⁸ / 1,2·10¹⁰ ở E4a, 5,1·10⁸ ở E4b seed 42 và 5,2·10⁹ ở E3c seed 42, trong khi trung vị chỉ 0,01–0,08. Khớp với phép chia cho clamp_min(1e-9) khi một câu huấn luyện không chọn chuyên gia nào (chính sách "zero"); gradient clipping 1,0 giữ huấn luyện ổn định. A6 (zero_safe) có giá trị lớn nhất 0,39–0,52, accuracy ngang E2.
- C của E4a, E4b tăng lên 0,97–1,00 ở cuối huấn luyện (ngưỡng ρ lên 0,2), đúng hướng phạt chi phí đẩy.

## Phân tích chéo 63 lần chạy

- Độ đồng thuận trung bình giữa hai lần chạy: 62,6%; oracle (ít nhất một lần đúng): 96,8%.
- 32 câu không lần chạy nào đúng (3,2%): 14 suy ra, 13 mâu thuẫn, 5 trung lập; đa số bị gần như mọi lần chạy đoán cùng một nhãn sai. Danh sách: reports/generated/unanimously_wrong.csv.
- 30,5% câu đúng ở dưới 20% số lần chạy.
- Bỏ phiếu đa số 63 lần chạy: 48,5%; trung bình xác suất: 48,4%; accuracy trung bình mỗi lần chạy: 45,0%.
- Recall gộp mọi lần chạy: suy ra 59,0%, trung lập 45,9%, mâu thuẫn 30,2%.
- Lần chạy gần như bỏ hẳn một nhãn (dưới 5% số câu): 6/63 (E0 × 2, E3a, E4a, E5a, E5b).
- Bảng gán kiểu bẫy cho 120 lỗi của E2 seed 42: reports/generated/error_annotation_sheet.csv (cột trap_type chưa điền).

## Hiệu năng suy luận (scripts/efficiency.py, checkpoint seed 42, 2026-11-27, GPU L4 không chạy job khác)

| Mã | GMACs/câu | GPU bf16 (ms/câu) | MoE sparse / dense (ms) | MoE % thời gian | Bộ nhớ GPU đỉnh (MB) | int8 (MB) | CPU int8 (ms/câu) | Acc bf16 → int8 (%) |
|---|---|---|---|---|---|---|---|---|
| E0 | 18,47 | 2,35 | — | — | 2.891 | 1.269 | 42,7 | 49,0 → 43,8 |
| E1 | 18,47 | 2,47 | 0,106 / 0,059 | 4,3 | 2.907 | 1.276 | 42,1 | 46,5 → 46,0 |
| E2 | 18,47 | 2,48 | 0,106 / 0,057 | 4,3 | 2.908 | 1.276 | 42,0 | 47,8 → 42,4 |
| E3a | 18,47 | 2,53 | 0,133 / 0,080 | 5,3 | 2.894 | 1.273 | 42,3 | 40,5 → 40,8 |
| E3b | 18,47 | 2,63 | 0,134 / 0,080 | 5,1 | 2.902 | 1.274 | 47,8 | 46,8 → 41,1 |
| E4a | 18,47 | 2,47 | 0,077 / 0,058 | 3,1 | 2.908 | 1.276 | 42,1 | 44,8 → 43,7 |
| E4b | 18,47 | 2,45 | 0,057 / 0,057 | 2,3 | 2.908 | 1.276 | 50,0 | 44,4 → 44,8 |
| E5a | 5,23 | 1,12 | — | — | 1.382 | 816 | 14,8 | 42,2 → 40,6 |
| E5b | 5,23 | 1,22 | 0,108 / 0,059 | 8,8 | 1.389 | 820 | 15,2 | 44,0 → 39,1 |
| E6 | 5,23 | 1,24 | 0,111 / 0,058 | 8,9 | 1.389 | 820 | 14,9 | 43,8 → 37,5 |

- Base so với Large: ít phép tính hơn 3,5 lần, nhanh hơn khoảng 2,1 lần trên GPU và 2,8 lần trên CPU int8, nhẹ hơn khoảng 1,5 lần ở int8, bộ nhớ GPU khi suy luận khoảng một nửa.
- Khối MoE chỉ chiếm 2–5% thời gian của mô hình Large (khoảng 9% với Base). Định tuyến thưa chậm hơn chạy dense cả 7 chuyên gia trên GPU, trừ khi còn khoảng 1 chuyên gia; thời gian sparse giảm theo số chuyên gia (0,106 → 0,077 → 0,057 ms).
- Lượng tử hoá int8 động làm giảm 5–6 điểm ở E0, E2, E3b, E5b, E6 nhưng gần như không ảnh hưởng E1, E3a, E4a, E4b. E3c, E8a, E8b chưa được đo.
