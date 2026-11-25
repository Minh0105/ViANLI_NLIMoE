# Kết quả v5 (3 seed) — cập nhật 2026-11-25

Cấu hình v5: encoder dropout 0,1, lr tuyến tính với 10% warmup, gradient clipping 1,0
(xem [paper_extraction.md](paper_extraction.md)). Tập kiểm thử ViANLI 1.000 câu; seed 42 / 123 / 2026;
A100 40 GB, bf16, batch 16.

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

## Thí nghiệm bổ sung (đang chờ chạy)

| Biến thể | Config | Thay đổi so với gốc |
|---|---|---|
| B2_T005, B2_T02 | nlimoe_ste_t005 / t02 | STE T = 0,05 / 0,2 (T = 0,1 trước đây chưa được dò) |
| B4_c002, B4_c01 | nlimoe_ste_cost002 / cost01 | B2 + λ · số chuyên gia mỗi câu, λ = 0,02 / 0,1 |
| B3_r384 | nlimoe_ste_bottleneck384 | B3 với r = 384 (−24,9% tham số khối chuyên gia) |
