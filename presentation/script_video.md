# Script thuyết trình video proposal · CS2311.11

**Đề tài:** Mixture-of-Experts nhẹ với ngưỡng định tuyến học được cho suy luận ngôn ngữ đối kháng tiếng Việt

- **Bản slide:** version 49 (27/11/2026), 31 slide; số liệu từ 63 lần chạy (21 cấu hình × 3 seed).
- **Tổng thời lượng dự kiến:** khoảng 2.950 tiếng, đọc khoảng 180 tiếng/phút, tức khoảng 16–17 phút.
- **Bản rút gọn khoảng 14–15 phút:** bỏ phần đánh dấu *(có thể lược)*; slide 30–31 chỉ lướt qua. Nếu cần ngắn hơn nữa, rút slide 4–5 và 20–21 còn một câu mỗi slide.
- **Ký hiệu:** `[...]` là chỗ cần thay bằng thông tin của nhóm. Chữ nghiêng trong ngoặc vuông là hành động (*[chỉ vào bảng]*), không đọc thành tiếng.
- **Mẹo đọc số:** "45,8" đọc là "bốn mươi lăm phẩy tám phần trăm"; "p = 0,05" đọc là "p bằng không phẩy không năm".
- **Mạch chính cần giữ:** NLIMoE có lỗi gradient → STE sửa được → nhờ đó điều khiển được chi phí → chưng cất ensemble cho mô hình tốt nhất; đồng thời nói thẳng rằng định tuyến chưa thêm accuracy.

---

## Phần 1 · Mở đầu và bài toán (khoảng 1 phút 30 giây)

### Slide 1 · Trang bìa (15 giây)
Xin chào thầy cô và các bạn. Nhóm [XX] gồm [tên các thành viên] xin trình bày proposal đồ án cuối kỳ môn CS2311.11 với đề tài: **Mixture-of-Experts nhẹ với ngưỡng định tuyến học được cho suy luận ngôn ngữ đối kháng tiếng Việt.**

### Slide 2 · Bài toán (40 giây)
Bài toán của nhóm là **suy luận ngôn ngữ tự nhiên**, viết tắt là NLI. Mô hình nhận hai câu, một **tiền đề** và một **giả thuyết**, rồi cho biết giả thuyết có suy ra được từ tiền đề hay không.

*[chỉ vào ví dụ]* Tiền đề nói: trong 157 trụ sở có 15 bệnh viện công và 35 trường học công.
- "Có 35 trường học công": **suy ra** được.
- "Các trụ sở đều ở TP.HCM": tiền đề không nói, nên là **trung lập**.
- "Có 15 trường học công": **mâu thuẫn**, vì 15 là số bệnh viện chứ không phải số trường.

Câu cuối là mẫu thật trong dữ liệu: chỉ tráo một con số, còn các từ gần như trùng hết. Đó là kiểu bẫy mà bộ dữ liệu này cố tình đặt ra.

### Slide 3 · Bộ dữ liệu ViANLI (35 giây)
Nhóm dùng **ViANLI**, bộ NLI đối kháng tiếng Việt đầu tiên, gồm hơn 10 nghìn cặp câu: 8.012 câu huấn luyện, 1.000 câu kiểm định và 1.000 câu kiểm thử.

"Đối kháng" nghĩa là người viết cố tình tìm giả thuyết làm mô hình mạnh nhất đoán sai, giống quy trình của ANLI tiếng Anh. Vì vậy các mô hình tốt nhất chỉ đạt **khoảng 46%**, hơn đoán ngẫu nhiên 33% khoảng 13 điểm.

---

## Phần 2 · Tổng quan nghiên cứu (khoảng 2 phút)

### Slide 4 · Tổng quan ngoài nước (45 giây)
Ngoài nước, nhóm chia thành năm hướng.
- **Dữ liệu NLI:** SNLI, MNLI, XNLI rất lớn nhưng có lối tắt thống kê; mô hình chỉ đọc giả thuyết hoặc đếm từ trùng vẫn đoán đúng khá nhiều.
- **NLI đối kháng:** ANLI ra đời để chặn các lối tắt đó, nhưng chủ yếu là tiếng Anh.
- **Mô hình đa ngữ:** XLM-R mạnh, nhưng khi tinh chỉnh thì kết quả dao động lớn theo seed.
- **MoE thưa:** Switch, Mixtral tăng dung lượng mà giữ chi phí, nhưng mỗi đầu vào luôn dùng một số chuyên gia cố định.
- **Định tuyến động:** hướng mới năm 2024 cho số chuyên gia thay đổi theo đầu vào, nhưng mới thử trên mô hình sinh văn bản.

### Slide 5 · Tổng quan trong nước (40 giây)
Trong nước đã có PhoBERT năm 2020, bộ ViNLI năm 2022, và năm 2024 nhóm tác giả ViANLI đề xuất **NLIMoE**, đạt 47,3% so với 45,5% của XLM-R Large.

Nhóm thấy bốn điểm còn tồn tại:
- Kết quả chỉ từ **một lần chạy**, không có kiểm định.
- Ngưỡng động **không nhận gradient**; nhóm sẽ chứng minh ở slide 12.
- Chưa có phiên bản MoE nhẹ.
- Bảy chi tiết cài đặt không được công bố.

### Slide 6 · Nghiên cứu có trước (30 giây)
Về công bố liên quan, [đọc thông tin của nhóm, hoặc: "nhóm chưa có công bố liên quan"].

Để chuẩn bị cho đề tài, nhóm đã:
- Cài đặt lại NLIMoE đúng công thức bài báo, kèm 20 kiểm tra tự động.
- Khắc phục hiện tượng mô hình sụp khi huấn luyện qua ba phiên bản cấu hình.
- Chạy tổng cộng **63 lần**: 21 cấu hình, mỗi cấu hình 3 seed, trên Google Colab.
- Viết công cụ ghi gradient và thống kê định tuyến cho từng câu.

---

## Phần 3 · Paper baseline và phép tính (khoảng 3 phút)

### Slide 7 · NLIMoE (45 giây)
Đây là baseline NLIMoE, gồm năm bước.
1. XLM-R Large đọc cặp câu; lấy trung bình các vector token thành vector câu h, 1024 chiều.
2. Router tính xác suất P cho 7 chuyên gia.
3. Một nhánh nhỏ ước lượng **độ phức tạp C** của câu, rồi đặt **ngưỡng ρ = 0,1 + 0,1·C**.
4. Chuyên gia nào có xác suất vượt ngưỡng thì được chọn, đầu ra của chúng được gộp theo trọng số.
5. Lớp phân loại cho ra ba nhãn.

Hàm mất mát gồm cross-entropy, cộng số hạng entropy và số hạng cân bằng tải. Ý tưởng là mỗi câu dùng một số chuyên gia khác nhau.

### Slide 8 · Bước 1–2 (30 giây) *(có thể lược)*
Để dễ theo dõi, nhóm tính tay một ví dụ nhỏ.
- **Mean pooling:** ba token thật cộng lại rồi chia ba, ra h = (2, 2). Token đệm có trọng số 0 nên không ảnh hưởng.
- **Router:** bảy điểm số qua softmax thành bảy xác suất, từ 0,35 giảm dần tới 0,03, tổng bằng 1.

### Slide 9 · Bước 3–4 (35 giây) *(có thể lược)*
- **Bước 3:** với u = 0,12, sigmoid cho C = 0,53, nên ngưỡng ρ = 0,153.
- **Bước 4:** so từng xác suất với 0,153. Chỉ chuyên gia 1 (0,35) và chuyên gia 2 (0,25) vượt ngưỡng. Chuẩn hoá lại, trọng số là 0,583 và 0,417.

Câu này dùng 2 chuyên gia. Lưu ý: C càng lớn thì ngưỡng càng cao, nên càng **ít** chuyên gia được chọn.

### Slide 10 · Bước 5 và loss (30 giây) *(có thể lược)*
Lớp phân loại cho xác suất nhãn mâu thuẫn là 0,785. Đây là nhãn đúng, nên cross-entropy bằng 0,242. Cộng entropy và cân bằng tải với hệ số rất nhỏ, tổng loss là 0,262, trong đó cross-entropy chiếm 92%.

### Slide 11 · Kết quả công bố (35 giây)
Bài báo báo cáo NLIMoE hơn XLM-R Large **1,8 điểm**. Nhóm tính sai số chuẩn của accuracy trên 1.000 câu: khoảng **1,6 điểm**. Mức chênh chỉ khoảng 1,1 lần sai số chuẩn, lại từ một lần chạy. Như vậy chưa đủ để kết luận, và đó là lý do nhóm đánh giá trên nhiều seed.

---

## Phần 4 · Hạn chế và tính mới (khoảng 2 phút 30 giây)

### Slide 12 · Lỗ hổng gradient (50 giây)
Đây là phát hiện chính của nhóm. *[chỉ vào chuỗi]* Để học trọng số w_c của nhánh độ phức tạp, gradient phải đi qua chuỗi: w_c → C → ρ → mặt nạ m → loss.

Nhưng mặt nạ m = 1 khi P > ρ là một **hàm bậc thang**. Dịch ngưỡng một chút thì m vẫn là 0 hoặc 1, nên đạo hàm của m theo ρ **bằng 0** hầu như ở mọi nơi. Nhân với 0 thì cả tích bằng 0.

Hệ quả: w_c không bao giờ được cập nhật, nên ngưỡng "động" thực chất chỉ là một hàm ngẫu nhiên từ lúc khởi tạo. Nhóm đo thực tế: gradient bằng 0 ở **cả 525 trên 525 lần đo**.

### Slide 13 · Các hạn chế khác (30 giây) *(có thể lược)*
Ngoài ra:
- Khối 7 chuyên gia có **7,35 triệu tham số**, nhưng chưa ai thử chuyên gia nhỏ hơn.
- Chưa có phân tích chuyên gia nào xử lý kiểu bẫy nào.
- Bảy chi tiết cài đặt không được công bố; nhóm ghi từng giả định vào cấu hình để người khác chạy lại được.

### Slide 14 · Tính khoa học và tính mới (40 giây)
Vấn đề khoa học là: **làm sao để ngưỡng định tuyến nhận được gradient mà không thay đổi cách mô hình dự đoán?**

Hướng giải quyết của nhóm:
- Dùng **Straight-Through Estimator** cho ngưỡng.
- Thử chuyên gia bottleneck nhỏ hơn.
- Đo gradient trực tiếp.
- Đánh giá trên 3 seed với **bootstrap theo câu**, kèm các mô hình đối chứng để biết lợi ích thật đến từ đâu.

Về khoa học, nhóm chứng minh và sửa một lỗi thiết kế. Về thực tiễn, nhóm công khai mã nguồn và một mô hình NLI tiếng Việt ổn định hơn.

### Slide 15 · Câu hỏi nghiên cứu (20 giây)
Câu hỏi nghiên cứu: **nếu ngưỡng định tuyến thật sự học được độ khó, MoE có thể nhỏ đi một nửa mà vẫn giữ độ chính xác không?**

Nhóm trả lời bằng ba cải tiến: ngưỡng học được bằng STE; chuyên gia bottleneck; phạt chi phí và backbone nhỏ. Tất cả đánh giá trên 3 seed với kiểm định bootstrap.

### Slide 16 · Mục tiêu, đối tượng, phạm vi (30 giây)
Nhóm đặt bốn mục tiêu đo được:
- **MT1:** tái hiện hai baseline với 3 seed.
- **MT2:** ngưỡng nhận gradient khác 0, accuracy không thấp hơn NLIMoE.
- **MT3:** giảm một nửa tham số khối chuyên gia, accuracy giảm không quá 1 điểm.
- **MT4:** phân tích định tuyến và ít nhất 100 lỗi.

Đến nay MT1, MT2 đã đạt. MT3 cũng đạt nhờ chuyên gia bottleneck có kết nối tắt: giảm 49,9% tham số, accuracy chỉ giảm 0,7 điểm. MT4 đang gán nhãn 120 lỗi. Phạm vi là bộ ViANLI, không huấn luyện mô hình ngôn ngữ lớn.

---

## Phần 5 · Phương pháp đề xuất (khoảng 2 phút 30 giây)

### Slide 17 · Cải tiến 1: STE (45 giây)
Ý tưởng của STE: **lượt xuôi giữ nguyên, lượt ngược đổi cách tính đạo hàm.**
- **Lượt xuôi**, khi dự đoán: vẫn dùng mặt nạ cứng như bài báo, nên hành vi suy luận không đổi.
- **Lượt ngược**, khi học: mượn đạo hàm của một mặt nạ mềm, là sigmoid của (P − ρ) chia nhiệt độ T. Hàm này trơn, nên đạo hàm khác 0, nhất là khi P gần ngưỡng.

Trong code chỉ là một dòng: mặt nạ cứng cộng mặt nạ mềm, trừ mặt nạ mềm đã detach. Ở lượt xuôi hai phần mềm triệt tiêu nhau; ở lượt ngược gradient đi qua phần mềm.

### Slide 18 · Ví dụ tính gradient (40 giây) *(có thể lược)*
Dùng lại ví dụ cũ với ngưỡng 0,153. Mặt nạ cứng cho đạo hàm bằng 0 ở cả bảy chuyên gia. Với STE, đạo hàm khác 0, lớn nhất ở chuyên gia 3, chuyên gia có xác suất 0,15, sát ngưỡng nhất.

Ý nghĩa: nếu bật chuyên gia 3 làm loss giảm, gradient sẽ hạ ngưỡng để chuyên gia đó được chọn. Trên dữ liệu thật, E2 hạ C trung bình từ khoảng 0,4–0,5 xuống 0,25 ở cả ba seed, tức là **ngưỡng đã học**.

### Slide 19 · Cải tiến 2: bottleneck (30 giây)
Mỗi chuyên gia gốc là một lớp 1024 → 1024, khoảng 1,05 triệu tham số. Nhóm thay bằng chuyên gia bottleneck: nén xuống 256 chiều rồi mở lại 1024, giống adapter. Mỗi chuyên gia còn khoảng 0,53 triệu tham số, giảm **49,9%**.

Thực nghiệm cho thấy cần thêm **kết nối tắt**: cộng lại đầu vào h vào đầu ra, để chuyên gia không làm mất thông tin. Nhóm cũng thử r = 384, giảm khoảng 25%.

### Slide 20 · Nội dung bước 1–3 (30 giây)
Quy trình gồm sáu bước; mỗi bước chỉ đổi một thành phần.
- **Bước 1:** tái hiện E0 và E1.
- **Bước 2:** thêm STE thành E2, dò nhiệt độ T.
- **Bước 3:** thay chuyên gia bằng bottleneck, có và không có kết nối tắt.

Siêu tham số giữ như bài báo: 7 epoch, batch 16, learning rate 1e-5. Nhóm chỉ thêm warmup và gradient clipping để chống sụp.

### Slide 21 · Nội dung bước 4–6 (30 giây)
- **Bước 4, ablation:** thay ngưỡng bằng top-K, và bỏ số hạng cân bằng tải.
- **Bước 5, kiểm định và phân tích:** bootstrap theo câu, tương quan giữa C và việc đoán sai, các mô hình đối chứng, và gán nhãn lỗi theo kiểu bẫy.
- **Bước 6, lightweight:** phạt số chuyên gia (E4), đổi sang XLM-R Base (E5), chưng cất tri thức từ E2 và từ ensemble (E6, E8), và đo độ trễ thật.

Tài nguyên tính toán là Google Colab với GPU A100 và L4.

### Slide 22 · Thiết kế thí nghiệm (30 giây)
Tổng hợp lại: **E0 và E1 là hai baseline** của bài báo. E2 là đề xuất chính. E3 đến E6 là các thực nghiệm lightweight. E7 thử tăng trọng số nhãn mâu thuẫn, E8 chưng cất từ ensemble. A1 đến A3 là ablation. A4 đến A6 là **đối chứng**: chúng tách riêng phần đóng góp của pooling, của định tuyến và của cách xử lý khi không chọn chuyên gia nào.

Tổng cộng 21 cấu hình, mỗi cấu hình 3 seed, tức 63 lần chạy, tất cả đã xong.

---

## Phần 6 · Kết quả (khoảng 3 phút 30 giây)

### Slide 23 · Kết quả chính (50 giây)
*[chỉ vào dòng E1]* Khi chạy lại 3 seed, NLIMoE gốc đạt **44,1%**, thấp hơn con số 47,3% của bài báo. Con số công bố nằm trong khoảng dao động giữa các seed.

*[chỉ vào dòng E2]* Thêm STE, E2 đạt **45,8%**, hơn E1 **1,7 điểm** và hơn ở cả 3 seed.

*[chỉ vào A4, A5]* Nhưng nhóm muốn trả lời trung thực: lợi ích thật đến từ đâu? Hai đối chứng cho câu trả lời:
- **A4** chỉ thêm pooling trung bình và LayerNorm vào XLM-R, không có MoE: đạt 45,7%, macro-F1 ngang E2. Vậy lợi thế macro-F1 của E2 so với E0 chủ yếu đến từ pooling.
- **A5** dùng đúng một chuyên gia, không định tuyến: cũng 45,7%, và hơn NLIMoE gốc 1,6 điểm.

Nghĩa là trên ViANLI, bảy chuyên gia có định tuyến **chưa cho thêm accuracy**. Giá trị của STE nằm ở chỗ khác, nhóm trình bày ở slide sau.

Các dòng còn lại: tăng trọng số nhãn mâu thuẫn (E7) không giúp; Top-K không tốt hơn; bỏ cân bằng tải làm có seed chỉ dùng 1 chuyên gia; đổi nhiệt độ T không ảnh hưởng; và A6 bỏ được các gai gradient mà accuracy không đổi.

### Slide 24 · Kết quả lightweight và chưng cất (50 giây)
Đây là nơi STE phát huy tác dụng: khi ngưỡng học được, **số chuyên gia trở thành thứ tối ưu được**.
- **Phạt chi phí λ = 0,02** giảm số chuyên gia mỗi câu từ 4 xuống 1,8, tức **bớt 55%**, accuracy chỉ giảm 0,6 điểm. Không có STE, mức phạt này không có gradient.
- **Bottleneck r = 256** bị sụp ở một seed. Thêm kết nối tắt (E3c) thì hết sụp, tăng 4,1 điểm, với cùng số tham số.
- **XLM-R Base** nhanh hơn Large khoảng **2,1 lần trên GPU** và **2,8 lần trên CPU int8**, theo đo thực tế.

*[chỉ vào dòng cuối]* Kết quả tốt nhất là **E8b**: E2 học lại từ ensemble 6 mô hình E0 và E2. E8b đạt **47,3%**, macro-F1 47,0%, và chỉ dùng **2 chuyên gia mỗi câu**, một nửa so với E2. Với Base, E8a đạt 45,6% và ổn định nhất, độ lệch chuẩn chỉ 0,2.

Một điều đáng chú ý: khối MoE chỉ chiếm 2 đến 5% thời gian suy luận. Muốn nhẹ thật thì phải thu nhỏ backbone.

### Slide 25 · Kiểm định thống kê (45 giây)
Ba seed của một mô hình được chấm trên cùng 1.000 câu, nên không thể coi 3.000 dự đoán là độc lập. Nhóm dùng **bootstrap theo câu**: lấy mẫu lại các câu, giữ nguyên 3 seed của mỗi câu.
- **E2 so với E1:** +1,7 điểm, p = 0,05, thắng 3 trên 3 seed. STE có ích ở mức biên.
- **E8b so với E2:** +1,6 điểm, p = 0,07, thắng 3 trên 3 seed. Gần có ý nghĩa.
- **E3c so với E3a:** +4,1 điểm, p nhỏ hơn 0,001. Kết nối tắt chữa được sụp.
- **E4a so với E2:** p = 0,59. Bớt 55% chuyên gia không làm giảm accuracy một cách có ý nghĩa.

Về gradient: E1 bằng 0 ở cả 525 lần đo, mọi biến thể có STE đều khác 0. Với E2, câu khó dùng 4,4 đến 4,7 chuyên gia, câu dễ chỉ 3,1 đến 3,4.

### Slide 26 · Quan sát về dữ liệu (45 giây)
Vì sao mô hình đơn chỉ dừng quanh 45 đến 47%? Nhóm phân tích 63 lần chạy và thấy bốn điều:
- **Nhãn mâu thuẫn rất khó:** recall chỉ khoảng 30%, và tập train cũng ít câu mâu thuẫn nhất.
- **Mô hình bất ổn:** hai lần chạy bất kỳ chỉ cho cùng dự đoán 62,6% số câu. Nhưng 96,8% số câu được ít nhất một lần chạy đoán đúng.
- **32 câu không lần chạy nào đoán đúng:** có thể rất khó, hoặc nhãn có vấn đề. Nhóm đang gán kiểu bẫy cho các câu này.
- **Dữ liệu ít lối tắt:** từ "không" phân bố gần đều giữa ba nhãn.

Kết luận: đây **chưa phải trần của dữ liệu**. Bỏ phiếu 63 lần chạy đạt 48,5%, ensemble E0 đạt 49,2%, và chưng cất ensemble đã đưa một mô hình đơn lên 47,3%.

---

## Phần 7 · Kế hoạch, đóng góp, kết luận (khoảng 2 phút)

### Slide 27 · Kế hoạch thời gian (30 giây)
Kế hoạch gồm sáu giai đoạn.
- Đến 27/11, nhóm đã xong 63 lần chạy, đo độ trễ, int8 và kiểm định bootstrap.
- Từ nay tới hết 30/11: hoàn thiện video và nộp proposal.
- Tháng 12: gán kiểu bẫy cho 32 câu khó và 120 lỗi; thử tinh chỉnh trung gian trên ViNLI và chưng cất vào XLM-R Base.
- Giai đoạn cuối: chạy 3 seed cho cấu hình mới, viết báo cáo.

Nhóm gặp người hướng dẫn [lịch gặp].

### Slide 28 · Đóng góp (30 giây)
Đề tài có ba đóng góp:
1. Một bản **tái hiện NLIMoE có kiểm định**: 63 lần chạy, bootstrap theo câu, các đối chứng tách phần đóng góp, mã nguồn công khai.
2. **Ngưỡng định tuyến học được:** chứng minh và sửa chỗ gradient bị chặn, tăng 1,7 điểm so với NLIMoE gốc, và câu khó dùng nhiều chuyên gia hơn.
3. **MoE nhẹ hơn cho NLI tiếng Việt:** bớt 55% chuyên gia mà không mất accuracy có ý nghĩa; chưng cất ensemble đạt 47,3% với 2 chuyên gia mỗi câu.

### Slide 29 · Kết luận và hướng phát triển (50 giây)
Thông điệp chính của nhóm: **khi ngưỡng định tuyến học được, chi phí của MoE trở thành thứ điều khiển được.**

Nhóm đã giải quyết ba vấn đề:
- Ngưỡng không học thì nay đã nhận gradient.
- Chi phí cố định thì nay giảm được 55% chuyên gia, và mô hình tốt nhất chỉ dùng 2 chuyên gia.
- Kết quả một lần chạy thì nay có 63 lần chạy, kiểm định và đối chứng.

Nhóm cũng nói rõ giới hạn: định tuyến chưa thêm accuracy so với một chuyên gia; lợi ích chính là chi phí. Phần còn lại nằm ở nhãn mâu thuẫn và độ bất ổn khi tinh chỉnh.

Hướng tiếp theo:
- **Trước báo cáo cuối kỳ:** gán kiểu bẫy cho câu khó và lỗi, kiểm định thêm cho E8b.
- **Thu hẹp khoảng cách Base và Large:** tinh chỉnh trung gian trên ViNLI, chưng cất ensemble vào Base.
- **Hướng tới bài báo:** đưa chuyên gia dạng LoRA vào bên trong encoder, nơi chiếm phần lớn chi phí, và mở rộng sang XNLI và ANLI.

### Slide 30–31 · Tài liệu tham khảo (10 giây)
Đây là 22 tài liệu tham khảo theo định dạng IEEE.

Cảm ơn thầy cô và các bạn đã lắng nghe. Nhóm rất mong nhận được góp ý.

---

## Câu hỏi có thể được hỏi và gợi ý trả lời

**1. "Sao không dùng Gumbel-softmax thay STE?"**
Gumbel-softmax thêm nhiễu ngẫu nhiên vào lượt xuôi, nên hành vi suy luận khác bài báo. STE giữ nguyên lượt xuôi, nên so sánh với baseline công bằng hơn và chỉ thêm một dòng code. Gumbel-softmax là hướng có thể thử thêm.

**2. "1,7 điểm có đáng kể không?"**
Theo bootstrap theo câu, mức này có ý nghĩa ở mức biên (p = 0,05) và lặp lại ở cả 3 seed. Nhưng điểm quan trọng hơn là cơ chế: không có STE, ngưỡng không học được, nên phạt chi phí cũng không làm được.

**3. "Vì sao kết quả E1 thấp hơn bài báo?"**
Bài báo chạy một lần. Độ lệch chuẩn giữa các seed của nhóm khoảng 2 điểm, nên 47,3% vẫn nằm trong khoảng dao động. Ngoài ra có 7 chi tiết cài đặt không được công bố.

**4. "C học được là độ khó hay độ dễ?"**
Theo công thức ρ = 0,1 + 0,1·C, C cao làm ngưỡng cao và dùng ít chuyên gia. Mô hình học cho câu dễ có C cao, nên thực tế C đóng vai trò điểm "dễ". Câu đoán sai có C thấp hơn và dùng nhiều chuyên gia hơn (4,4–4,7 so với 3,1–3,4).

**5. "Lightweight thật sự nhẹ được bao nhiêu?"**
Nhóm đã đo trên GPU L4 và CPU. Khối MoE chỉ chiếm 2–5% thời gian suy luận của Large, nên bớt chuyên gia chủ yếu chứng minh khả năng điều khiển chi phí. Phần nhẹ thật đến từ backbone Base: ít phép tính hơn 3,5 lần, nhanh hơn 2,1 lần trên GPU và 2,8 lần trên CPU int8, mô hình int8 nhỏ hơn khoảng 1,5 lần.

**6. "Một chuyên gia đã đạt 45,7% thì cần MoE làm gì?"**
Đúng là trên ViANLI, định tuyến chưa thêm accuracy; nhóm báo cáo điều này qua đối chứng A4 và A5. Giá trị của MoE với ngưỡng học được là chi phí điều khiển được theo từng câu: có thể giảm số chuyên gia bằng phạt chi phí, và mô hình tốt nhất E8b chỉ dùng 2 chuyên gia. Hướng tiếp theo là đưa chuyên gia vào trong encoder, nơi việc bớt tính toán có tác động lớn hơn nhiều.

**7. "47,3% của E8b là nhờ MoE hay nhờ chưng cất?"**
Nhiều khả năng phần lớn là nhờ chưng cất: thầy là trung bình của 6 mô hình E0 và E2, mà riêng ensemble 3 seed của E0 đã đạt 49,2%, của E2 đạt 47,9%. Để tách bạch, nhóm sẽ chưng cất cùng thầy vào mô hình một chuyên gia (A5) trước báo cáo cuối kỳ. Điều chắc chắn hiện nay là E8b đạt mức đó với một nửa số chuyên gia của E2.

**8. "Vì sao chuyên gia bottleneck r = 256 lại sụp?"**
Nhóm cho rằng khi không có kết nối tắt, đầu ra của khối MoE bị ép qua 256 chiều, nên một phần thông tin của câu bị mất và huấn luyện dễ lệch về một nhãn. Thêm đường tắt h + bottleneck(h) giữ lại thông tin gốc; E3c hết sụp với cùng số tham số.
