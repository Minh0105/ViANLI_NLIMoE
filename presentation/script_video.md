# Script thuyết trình video proposal · CS2311.11

**Đề tài:** Mixture-of-Experts nhẹ với ngưỡng định tuyến học được cho suy luận ngôn ngữ đối kháng tiếng Việt

- **Tổng thời lượng dự kiến:** khoảng 15 phút cho 31 slide, nói khoảng 130 từ/phút.
- **Bản rút gọn 10 phút:** bỏ phần đánh dấu *(có thể lược)*. Slide 8–10 và 30–31 chỉ cần lướt qua.
- **Ký hiệu:** `[...]` là chỗ cần thay bằng thông tin của nhóm. Chữ trong ngoặc vuông in nghiêng là hành động (*[chỉ vào bảng]*), không đọc thành tiếng.
- **Mẹo đọc số:** "45,8" đọc là "bốn mươi lăm phẩy tám phần trăm"; "p = 0,05" đọc là "p bằng không phẩy không năm".

---

## Phần 1 · Mở đầu và bài toán (khoảng 2 phút)

### Slide 1 · Trang bìa (15 giây)
Xin chào thầy cô và các bạn. Nhóm [XX] gồm [tên các thành viên] xin trình bày proposal đồ án cuối kỳ môn CS2311.11 với đề tài: **Mixture-of-Experts nhẹ với ngưỡng định tuyến học được cho suy luận ngôn ngữ đối kháng tiếng Việt.**

### Slide 2 · Bài toán (40 giây)
Bài toán của nhóm là **suy luận ngôn ngữ tự nhiên**, viết tắt là NLI. Mô hình nhận hai câu: một **tiền đề** và một **giả thuyết**, rồi trả lời giả thuyết có suy ra được từ tiền đề hay không.

*[chỉ vào ví dụ]* Tiền đề nói: trong 157 trụ sở có 15 bệnh viện công và 35 trường học công.
- "Có 35 trường học công": **suy ra** được.
- "Các trụ sở đều ở TP.HCM": tiền đề không nói, nên là **trung lập**.
- "Có 15 trường học công": **mâu thuẫn**, vì 15 là số bệnh viện chứ không phải số trường.

Câu cuối là mẫu thật trong dữ liệu. Nó cho thấy dữ liệu cố tình đặt bẫy: chỉ đổi một con số, các từ gần như trùng hết.

### Slide 3 · Bộ dữ liệu ViANLI (35 giây)
Nhóm dùng **ViANLI**, bộ NLI đối kháng tiếng Việt đầu tiên, gồm hơn 10 nghìn cặp câu: 8.012 câu huấn luyện, 1.000 câu kiểm định và 1.000 câu kiểm thử.

"Đối kháng" nghĩa là người viết giả thuyết cố tình tìm câu làm mô hình mạnh nhất đoán sai, giống quy trình của ANLI tiếng Anh. Vì vậy mô hình tốt nhất cũng chỉ đạt **khoảng 46%**, hơn đoán ngẫu nhiên 33% khoảng 13 điểm. Đây là một bộ dữ liệu rất khó.

---

## Phần 2 · Tổng quan nghiên cứu (khoảng 2 phút)

### Slide 4 · Tổng quan ngoài nước (45 giây)
Về tình hình ngoài nước, nhóm chia thành năm hướng.
- **Dữ liệu NLI.** Các bộ lớn như SNLI, MNLI, XNLI có lối tắt thống kê: mô hình chỉ đọc giả thuyết hoặc đếm từ trùng vẫn đoán đúng khá nhiều.
- **NLI đối kháng.** ANLI ra đời để chặn các lối tắt đó, nhưng chủ yếu là tiếng Anh.
- **Mô hình đa ngữ.** XLM-R mạnh, nhưng khi tinh chỉnh, kết quả dao động lớn theo seed.
- **Mixture-of-Experts thưa.** MoE như Switch hay Mixtral tăng dung lượng mà giữ chi phí, nhưng mỗi đầu vào luôn dùng một số chuyên gia cố định.
- **Định tuyến động.** Hướng mới năm 2024 cho số chuyên gia thay đổi theo độ khó, nhưng mới thử trên mô hình sinh văn bản.

### Slide 5 · Tổng quan trong nước (40 giây)
Trong nước đã có PhoBERT năm 2020, bộ ViNLI năm 2022, và năm 2024 nhóm tác giả ViANLI đề xuất **NLIMoE**, đạt 47,3% so với 45,5% của XLM-R Large.

Tuy vậy nhóm thấy bốn điểm còn tồn tại:
- Kết quả chỉ từ **một lần chạy**.
- Cơ chế ngưỡng động **không nhận gradient**, sẽ chứng minh ở slide 12.
- Chưa có phiên bản MoE nhẹ.
- Bảy chi tiết cài đặt không được công bố.

### Slide 6 · Nghiên cứu có trước (30 giây)
Về công bố liên quan, [đọc thông tin của nhóm, hoặc: "nhóm chưa có công bố liên quan"].

Để chuẩn bị cho đề tài, nhóm đã:
- Cài đặt lại NLIMoE đúng theo công thức của bài báo, có 20 kiểm tra tự động.
- Khắc phục hiện tượng mô hình sụp khi huấn luyện qua ba phiên bản cấu hình.
- Chạy tổng cộng **42 lần**: 14 cấu hình, mỗi cấu hình 3 seed, trên Google Colab.

---

## Phần 3 · Paper baseline và phép tính (khoảng 3 phút)

### Slide 7 · NLIMoE (45 giây)
Đây là kiến trúc baseline NLIMoE, gồm năm bước.
1. XLM-R Large đọc cặp câu và lấy trung bình các vector token thành một vector câu h, 1024 chiều.
2. Router tính xác suất P cho 7 chuyên gia.
3. Một nhánh nhỏ ước lượng **độ phức tạp C** của câu, từ đó đặt **ngưỡng ρ = 0,1 + 0,1·C**.
4. Chuyên gia nào có xác suất vượt ngưỡng thì được chọn, và đầu ra của chúng được gộp theo trọng số.
5. Lớp phân loại cho ra ba nhãn.

Hàm mất mát gồm cross-entropy, cộng một số hạng entropy và một số hạng cân bằng tải. Ý tưởng là câu khác nhau dùng số chuyên gia khác nhau.

### Slide 8 · Bước 1–2 (30 giây) *(có thể lược)*
Để dễ theo dõi, nhóm tính tay một ví dụ nhỏ.
- **Mean pooling:** ba token thật cộng lại rồi chia ba, ra h = (2, 2). Token đệm có trọng số 0 nên không ảnh hưởng.
- **Router:** bảy điểm số đi qua softmax thành bảy xác suất, từ 0,35 giảm dần tới 0,03, tổng bằng 1.

### Slide 9 · Bước 3–4 (35 giây) *(có thể lược)*
- **Bước 3:** với u = 0,12, sigmoid cho C = 0,53, nên ngưỡng ρ = 0,153.
- **Bước 4:** so từng xác suất với 0,153. Chỉ chuyên gia 1 (0,35) và chuyên gia 2 (0,25) vượt ngưỡng. Chuẩn hoá lại, trọng số là 0,583 và 0,417.

Câu này dùng 2 chuyên gia. Lưu ý: C càng lớn thì ngưỡng càng cao, nên càng **ít** chuyên gia được chọn.

### Slide 10 · Bước 5 và loss (30 giây) *(có thể lược)*
Lớp phân loại cho xác suất nhãn mâu thuẫn là 0,785. Đây là nhãn đúng, nên cross-entropy bằng 0,242. Cộng thêm entropy và cân bằng tải với hệ số rất nhỏ, tổng loss là 0,262, trong đó cross-entropy chiếm 92%.

### Slide 11 · Kết quả công bố (35 giây)
Bài báo báo cáo NLIMoE hơn XLM-R Large **1,8 điểm**. Nhóm tính sai số chuẩn của accuracy trên 1.000 câu: khoảng **1,6 điểm**. Như vậy mức chênh chỉ khoảng 1,1 lần sai số chuẩn, lại từ một lần chạy. Điều đó chưa đủ để kết luận, và cũng là lý do nhóm đánh giá trên nhiều seed.

---

## Phần 4 · Hạn chế và tính mới (khoảng 2 phút 30 giây)

### Slide 12 · Lỗ hổng gradient (50 giây)
Đây là phát hiện chính của nhóm. *[chỉ vào chuỗi]* Để học được trọng số w_c của nhánh độ phức tạp, gradient phải đi qua chuỗi: w_c → C → ρ → mặt nạ m → loss.

Nhưng mặt nạ m = 1 khi P > ρ là một **hàm bậc thang**. Dịch ngưỡng một chút thì m vẫn là 0 hoặc 1, nên đạo hàm của m theo ρ **bằng 0** hầu như ở mọi nơi. Nhân với 0 thì cả tích bằng 0.

Hệ quả: w_c không bao giờ được cập nhật. Ngưỡng "động" thực chất là một hàm ngẫu nhiên từ lúc khởi tạo, không phản ánh độ khó. Nhóm đo thực tế: gradient bằng 0 ở **cả 525 trên 525 lần đo**.

### Slide 13 · Các hạn chế khác (30 giây) *(có thể lược)*
Ngoài ra:
- Khối 7 chuyên gia có **7,35 triệu tham số** nhưng chưa ai thử chuyên gia nhỏ hơn.
- Chưa có phân tích chuyên gia nào xử lý kiểu bẫy nào.
- Có 7 chi tiết cài đặt không được công bố. Nhóm ghi từng giả định vào cấu hình để người khác chạy lại được.

### Slide 14 · Tính khoa học và tính mới (40 giây)
Tóm lại, vấn đề khoa học là: **làm sao để ngưỡng định tuyến nhận được gradient mà không thay đổi cách mô hình dự đoán?**

Hướng giải quyết của nhóm:
- Dùng **Straight-Through Estimator** cho ngưỡng.
- Thử chuyên gia bottleneck nhỏ hơn.
- Đo gradient trực tiếp.
- Đánh giá trên 3 seed với kiểm định McNemar.

Về khoa học, nhóm chứng minh và sửa một lỗi gradient. Về thực tiễn, nhóm cung cấp mã nguồn mở và một mô hình NLI tiếng Việt ổn định hơn.

### Slide 15 · Câu hỏi nghiên cứu (20 giây)
Câu hỏi nghiên cứu của nhóm: **nếu ngưỡng định tuyến thật sự học được độ khó, MoE có thể nhỏ đi một nửa mà vẫn giữ độ chính xác không?**

Nhóm trả lời bằng ba cải tiến: ngưỡng học được bằng STE; chuyên gia bottleneck; phạt chi phí và backbone nhỏ. Tất cả được đánh giá trên 3 seed với kiểm định McNemar.

### Slide 16 · Mục tiêu, đối tượng, phạm vi (30 giây)
Nhóm đặt bốn mục tiêu đo được:
- **MT1:** tái hiện hai baseline với 3 seed.
- **MT2:** ngưỡng nhận gradient khác 0, accuracy không thấp hơn NLIMoE.
- **MT3:** giảm khoảng một nửa tham số khối chuyên gia, accuracy giảm không quá 1 điểm.
- **MT4:** phân tích định tuyến và ít nhất 100 lỗi.

Đến nay MT1 và MT2 đã đạt, MT3 đạt một phần, MT4 đang thực hiện. Phạm vi là bộ ViANLI, không huấn luyện mô hình ngôn ngữ lớn.

---

## Phần 5 · Phương pháp đề xuất (khoảng 2 phút 30 giây)

### Slide 17 · Cải tiến 1: STE (45 giây)
Ý tưởng của STE: **lượt xuôi giữ nguyên, lượt ngược đổi cách tính đạo hàm.**
- **Lượt xuôi**, khi dự đoán: vẫn dùng mặt nạ cứng như bài báo, nên hành vi suy luận không đổi.
- **Lượt ngược**, khi học: mượn đạo hàm của một mặt nạ mềm, là sigmoid của (P − ρ) chia nhiệt độ T. Hàm này trơn, nên đạo hàm khác 0, nhất là khi P gần ngưỡng.

Trong code chỉ là một dòng: mặt nạ cứng cộng mặt nạ mềm trừ mặt nạ mềm đã detach. Lượt xuôi hai phần mềm triệt tiêu, lượt ngược gradient đi qua phần mềm.

### Slide 18 · Ví dụ tính gradient (40 giây) *(có thể lược)*
Dùng lại ví dụ cũ với ngưỡng 0,153. Mặt nạ cứng cho đạo hàm bằng 0 ở cả bảy chuyên gia. Với STE, đạo hàm khác 0, lớn nhất ở chuyên gia 3, chuyên gia có xác suất 0,15, sát ngưỡng nhất.

Ý nghĩa: nếu bật chuyên gia 3 làm loss giảm, gradient sẽ hạ ngưỡng xuống để chuyên gia đó được chọn. Trên dữ liệu thật, mô hình E2 thực sự hạ C trung bình từ khoảng 0,4–0,5 xuống 0,25 ở cả ba seed, tức là **ngưỡng đã học**.

### Slide 19 · Cải tiến 2: bottleneck (30 giây)
Mỗi chuyên gia gốc là một lớp 1024 → 1024, khoảng 1,05 triệu tham số. Nhóm thay bằng chuyên gia bottleneck: nén xuống 256 chiều rồi mở lại 1024, tương tự adapter. Mỗi chuyên gia còn khoảng 0,53 triệu tham số, giảm **49,9%**. Nhóm thử thêm r = 384, giảm khoảng 25%.

### Slide 20 · Nội dung bước 1–3 (30 giây)
Quy trình gồm sáu bước, mỗi bước chỉ đổi một thành phần.
- **Bước 1:** tái hiện E0 và E1.
- **Bước 2:** thêm STE thành E2, và dò nhiệt độ T.
- **Bước 3:** thay chuyên gia bằng bottleneck.

Các siêu tham số giữ đúng như bài báo: 7 epoch, batch 16, learning rate 1e-5. Nhóm chỉ thêm warmup và gradient clipping để chống sụp.

### Slide 21 · Nội dung bước 4–6 (30 giây)
- **Bước 4, ablation:** thay ngưỡng bằng top-K, và bỏ số hạng cân bằng tải.
- **Bước 5:** kiểm định McNemar, tương quan Spearman giữa C và việc đoán sai, cùng phân tích lỗi.
- **Bước 6, lightweight:** phạt số chuyên gia (E4), đổi sang backbone XLM-R Base (E5), và chưng cất tri thức từ E2 (E6).

Tài nguyên tính toán là Google Colab với GPU A100 và L4.

### Slide 22 · Thiết kế thí nghiệm (25 giây)
Tổng hợp lại: **E0 và E1 là hai baseline** của bài báo. E2 là đề xuất chính. E3 đến E6 là các thực nghiệm lightweight. A1 đến A3 là ablation. Tất cả chạy 3 seed, chọn checkpoint trên tập kiểm định, và tập kiểm thử chỉ chạy một lần.

---

## Phần 6 · Kết quả (khoảng 3 phút)

### Slide 23 · Kết quả chính (45 giây)
*[chỉ vào dòng E1]* Khi chạy lại 3 seed, NLIMoE gốc đạt **44,1%**, thấp hơn con số 47,3% của bài báo. Điều này cho thấy con số công bố nằm trong khoảng dao động giữa các seed.

*[chỉ vào dòng E2]* Thêm STE, E2 đạt **45,8%**, hơn E1 **1,7 điểm** và hơn ở cả 3 seed. Macro-F1 cao nhất bảng, 45,6%.

E0 có cùng accuracy nhưng độ lệch chuẩn rất lớn, 4,1 điểm, và 2 trên 3 seed gần như bỏ hẳn nhãn mâu thuẫn. Ablation cho thấy:
- Top-K không tốt hơn.
- Bỏ cân bằng tải làm có seed chỉ dùng 1 chuyên gia.
- Đổi nhiệt độ T không ảnh hưởng.

### Slide 24 · Kết quả lightweight (45 giây)
Với hướng lightweight:
- **Bottleneck r = 256** bị giảm mạnh và không ổn định, nên đây là kết quả âm.
- **r = 384** giữ nguyên accuracy với 25% ít tham số chuyên gia hơn.
- **Phạt chi phí λ = 0,02** giảm số chuyên gia mỗi câu từ 4 xuống 1,8, tức là **bớt 55%**, accuracy chỉ giảm 0,6 điểm. Đây là điều chỉ làm được nhờ STE: không có STE, mức phạt này không có gradient.
- **Đổi sang XLM-R Base** với MoE và STE đạt 44,5%, **ngang NLIMoE trên Large**, trong khi encoder ít phép tính hơn khoảng 3,5 lần.
- **Chưng cất** không tăng accuracy nhưng tăng macro-F1 3,3 điểm.

### Slide 25 · Kiểm định thống kê (45 giây)
Nhóm dùng kiểm định McNemar trên 3.000 cặp dự đoán gộp từ 3 seed.
- **E2 so với E1:** p = 0,05, có ý nghĩa ở mức biên, và thắng 3 trên 3 seed.
- **E4a so với E2:** p = 0,57, tức bớt 55% chuyên gia không làm giảm accuracy một cách có ý nghĩa.
- **E5b so với E1:** p = 0,75, xác nhận Base cộng MoE ngang NLIMoE trên Large.

Về gradient: E1 bằng 0 ở 525 trên 525 lần đo, E2 khác 0 ở 525 trên 525 lần đo. Tương quan giữa C và việc đoán sai của E2 có cùng dấu và có ý nghĩa ở cả 3 seed. Với E1, dấu tương quan đổi giữa các seed, đúng như dự đoán khi ngưỡng không học.

### Slide 26 · Quan sát về dữ liệu (45 giây)
Vì sao mọi mô hình đơn dừng ở khoảng 46%? Nhóm phân tích 42 lần chạy và thấy bốn điều:
- **Nhãn mâu thuẫn rất khó:** recall chỉ 27,9%, và tập train cũng ít câu mâu thuẫn nhất.
- **Mô hình bất ổn:** hai lần chạy bất kỳ chỉ cho cùng dự đoán 60,7% số câu. Nhưng 96,4% số câu được ít nhất một lần chạy đoán đúng.
- **36 câu không lần chạy nào đoán đúng:** có thể rất khó, hoặc nhãn có vấn đề.
- **Dữ liệu ít lối tắt:** từ "không" phân bố gần đều giữa ba nhãn.

Kết luận: 46% **chưa phải trần của dữ liệu**. Bỏ phiếu của 42 lần chạy đạt 48,7%, và ensemble 3 seed của E0 đạt 49,2%.

---

## Phần 7 · Kế hoạch, đóng góp, kết luận (khoảng 2 phút)

### Slide 27 · Kế hoạch thời gian (30 giây)
Kế hoạch gồm sáu giai đoạn.
- Đến 26/11, nhóm đã xong tái hiện, đề xuất chính và các thực nghiệm lightweight.
- Từ nay tới hết 30/11: đo độ trễ thật và lượng tử hoá int8, hoàn thiện video.
- Tháng 12: phân tích lỗi và 36 câu khó, rồi thử chưng cất ensemble và tinh chỉnh trung gian trên ViNLI.
- Giai đoạn cuối: chạy 3 seed cho cấu hình mới, viết báo cáo.

Nhóm gặp người hướng dẫn [lịch gặp].

### Slide 28 · Đóng góp (30 giây)
Đề tài có ba đóng góp:
1. Một bản **tái hiện NLIMoE có kiểm định**: 42 lần chạy, kiểm định McNemar, mã nguồn công khai.
2. **Ngưỡng định tuyến học được:** chứng minh và sửa chỗ gradient bị chặn, tăng 1,7 điểm so với NLIMoE gốc.
3. **MoE nhẹ hơn cho NLI tiếng Việt:** bớt 55% chuyên gia mà không mất accuracy có ý nghĩa, và mô hình Base ngang NLIMoE trên Large.

### Slide 29 · Kết luận và hướng phát triển (50 giây)
Thông điệp chính của nhóm: **khi ngưỡng định tuyến học được, chi phí của MoE trở thành thứ điều khiển được.**

Nhóm đã giải quyết ba vấn đề:
- Ngưỡng không học thì nay đã nhận gradient.
- Chi phí cố định thì nay giảm được 55% chuyên gia.
- Kết quả một lần chạy thì nay có 3 seed và kiểm định.

Giới hạn còn lại không nằm ở kiến trúc mà ở nhãn mâu thuẫn và độ bất ổn khi tinh chỉnh.

Hướng tiếp theo:
- **Trước báo cáo cuối kỳ:** phân tích 36 câu khó, tăng trọng số nhãn mâu thuẫn, đo độ trễ thật.
- **Để vượt 47% bằng một mô hình:** chưng cất từ ensemble đã đạt 48–49%.
- **Hướng tới bài báo:** đưa chuyên gia dạng LoRA vào bên trong encoder, nơi chiếm 99% chi phí tính toán, và mở rộng sang XNLI và ANLI.

### Slide 30–31 · Tài liệu tham khảo (10 giây)
Đây là 22 tài liệu tham khảo theo định dạng IEEE, trong đó các tài liệu in đậm năm là công trình trong 5 năm gần đây.

Cảm ơn thầy cô và các bạn đã lắng nghe. Nhóm rất mong nhận được góp ý.

---

## Câu hỏi có thể được hỏi và gợi ý trả lời

**1. "Sao không dùng Gumbel-softmax thay STE?"**
Gumbel-softmax thêm nhiễu ngẫu nhiên vào lượt xuôi, nên hành vi suy luận khác bài báo. STE giữ nguyên lượt xuôi, nên so sánh với baseline công bằng hơn và chỉ thêm một dòng code. Gumbel-softmax là hướng có thể thử thêm.

**2. "1,7 điểm có đáng kể không?"**
Mức này có ý nghĩa thống kê ở mức biên (p = 0,05) và lặp lại ở cả 3 seed. Nhưng điểm quan trọng hơn là cơ chế: không có STE, ngưỡng không học được, nên các hướng như phạt chi phí cũng không làm được.

**3. "Vì sao kết quả E1 thấp hơn bài báo?"**
Bài báo chạy một lần. Độ lệch chuẩn giữa các seed của nhóm là khoảng 2 điểm, nên 47,3% vẫn nằm trong khoảng dao động. Ngoài ra có 7 chi tiết cài đặt không được công bố.

**4. "C học được là độ khó hay độ dễ?"**
Theo công thức ρ = 0,1 + 0,1·C, C cao làm ngưỡng cao và dùng ít chuyên gia. Mô hình học cho câu dễ có C cao, nên thực tế C đóng vai trò điểm "dễ". Câu đoán sai có C thấp hơn và dùng nhiều chuyên gia hơn (4,4–4,7 so với 3,1–3,4).

**5. "Lightweight thật sự nhẹ được bao nhiêu?"**
Khối MoE chỉ chiếm khoảng 1% chi phí, nên bớt chuyên gia chủ yếu chứng minh khả năng điều khiển chi phí. Phần nhẹ thật đến từ backbone Base: encoder ít phép tính hơn khoảng 3,5 lần (ước tính theo tham số). Độ trễ thật sẽ được đo trước báo cáo cuối kỳ.
