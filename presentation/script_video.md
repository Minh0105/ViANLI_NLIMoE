# Script thuyết trình video proposal · CS2311.11

**Đề tài:** Mixture-of-Experts nhẹ với ngưỡng định tuyến học được cho suy luận ngôn ngữ đối kháng tiếng Việt

- **Bản slide:** version 49, 31 slide. Script đi theo đúng thứ tự trên mỗi slide: tiêu đề, rồi các ô hoặc dòng bảng từ trên xuống, cuối cùng là dòng kết luận.
- **Tổng thời lượng dự kiến:** khoảng 32 phút, đọc khoảng 180 tiếng/phút. Thời gian ghi cạnh mỗi slide được tính theo số tiếng, nên hãy bấm giờ khi tập.
- **Bản rút gọn khoảng 28 phút:** bỏ các slide đánh dấu *(có thể lược)*. Muốn ngắn hơn nữa, ở slide 23–25 chỉ đọc các dòng in đậm.
- **Ký hiệu:** `[...]` là chỗ nhóm tự điền. Chữ nghiêng trong ngoặc vuông là hành động (*[chỉ vào bảng]*), không đọc thành tiếng.
- **Đọc số:** "45,8" đọc là "bốn mươi lăm phẩy tám phần trăm"; "p = 0,05" đọc là "p bằng không phẩy không năm"; "E2" đọc là "e hai".
- **Mạch chính:** NLIMoE có lỗi gradient → STE sửa được → nhờ đó điều khiển được chi phí → chưng cất ensemble cho mô hình tốt nhất. Song song, nói thẳng rằng định tuyến chưa thêm accuracy.

---

## Phần 1 · Mở đầu và bài toán (khoảng 2 phút 50 giây)

### Slide 1 · Trang bìa (45 giây)
Xin chào thầy cô và các bạn. Nhóm [XX], gồm [tên các thành viên], dưới sự hướng dẫn của [tên giảng viên], xin trình bày proposal đồ án cuối kỳ môn CS2311.11.

Đề tài của nhóm là: **Mixture-of-Experts nhẹ với ngưỡng định tuyến học được cho suy luận ngôn ngữ đối kháng tiếng Việt**. Tên tiếng Anh là Lightweight Mixture-of-Experts with Learnable Routing Thresholds for Vietnamese Adversarial Natural Language Inference.

Tên đề tài có ba ý. "Mixture-of-Experts" là kiến trúc gồm nhiều mạng con gọi là chuyên gia, và một bộ định tuyến chọn chuyên gia cho từng đầu vào. "Ngưỡng định tuyến học được" là đóng góp chính của nhóm. Còn "nhẹ" là mục tiêu giảm chi phí của khối chuyên gia.

### Slide 2 · Bài toán (1 phút 10 giây)
*[chỉ vào tiêu đề]* Bài toán của nhóm là **suy luận ngôn ngữ tự nhiên**, viết tắt là NLI. Mô hình nhận hai câu: một **tiền đề** và một **giả thuyết**, rồi trả lời giả thuyết có suy ra được từ tiền đề hay không.

*[chỉ vào ô tiền đề]* Đây là một tiền đề thật trong dữ liệu: "Trong 157 trụ sở có 107 toà nhà của sở ngành, quận huyện và các đơn vị trực thuộc; 15 bệnh viện công và 35 trường học công."

*[chỉ lần lượt ba ô]* Có ba nhãn:
- **Suy ra**: "Có 35 trường học công trong số 157 trụ sở". Tiền đề nói đúng điều này.
- **Trung lập**: "Các trụ sở này đều nằm ở TP.HCM". Tiền đề không nói gì về địa điểm, nên không khẳng định cũng không phủ định được.
- **Mâu thuẫn**: "Có 15 trường học công trong số 157 trụ sở". Con số 15 là số bệnh viện chứ không phải số trường.

Câu mâu thuẫn là mẫu thật trong dữ liệu; hai câu còn lại do nhóm viết để minh hoạ. Câu này cho thấy kiểu bẫy của dữ liệu: giả thuyết giữ gần như toàn bộ từ của tiền đề, chỉ tráo một con số. Một mô hình chỉ dựa vào việc đếm từ trùng sẽ đoán là "suy ra" và đoán sai.

### Slide 3 · Bộ dữ liệu ViANLI (55 giây)
*[chỉ vào tiêu đề]* Nhóm dùng bộ **ViANLI**, một bộ dữ liệu được viết ra để đánh lừa mô hình mạnh nhất.

*[chỉ lần lượt ba con số]*
- **10.012** cặp câu. Đây là bộ NLI đối kháng tiếng Việt đầu tiên.
- Chia thành **8.012** câu huấn luyện, 1.000 câu kiểm định và 1.000 câu kiểm thử; ba nhãn gần cân bằng.
- Mô hình tốt nhất chỉ đạt **khoảng 46%**, trong khi đoán ngẫu nhiên đã được 33%. Tức là chỉ hơn ngẫu nhiên khoảng 13 điểm.

*[chỉ vào ô "Cách tạo"]* Lý do dữ liệu khó như vậy là cách tạo: người viết giả thuyết nhằm làm mô hình đoán sai, giống quy trình của bộ ANLI tiếng Anh. Mẫu chỉ được giữ khi mô hình đoán sai và người kiểm tra đồng ý với nhãn. Vì vậy các mẹo thống kê như trùng từ hay từ phủ định không còn hiệu quả.

Nhóm cũng tự đo: một cặp câu dài trung bình khoảng 60 subword, dài nhất 164, nên giới hạn 256 token không cắt mẫu nào.

---

## Phần 2 · Tổng quan nghiên cứu (khoảng 3 phút 5 giây)

### Slide 4 · Tổng quan ngoài nước (1 phút 25 giây)
*[chỉ vào tiêu đề]* Ở ngoài nước, NLI đối kháng và Mixture-of-Experts phát triển song song. Nhóm tóm tắt thành năm hướng trong bảng, mỗi hướng gồm công trình, kết quả chính và điều còn tồn tại.

*[đi lần lượt từng dòng]*
- **Dữ liệu NLI:** SNLI, MNLI và XNLI là các bộ chuẩn lớn; XNLI mở rộng sang 15 ngôn ngữ, có tiếng Việt. Nhưng chúng có lối tắt thống kê: mô hình chỉ đọc giả thuyết vẫn đoán đúng khá nhiều, hoặc dựa vào việc trùng từ.
- **NLI đối kháng:** ANLI đưa người viết và mô hình vào cùng một vòng lặp, mô hình mạnh nhất bị đánh lừa có chủ đích. Hạn chế là chủ yếu tiếng Anh.
- **Mô hình đa ngữ:** XLM-R được tiền huấn luyện trên 100 ngôn ngữ và dẫn đầu XNLI khi công bố. Nhưng khi tinh chỉnh, kết quả dao động mạnh theo seed.
- **MoE thưa:** các công trình như GShard, Switch, Mixtral tăng dung lượng mô hình mà giữ chi phí tính toán, nhờ chỉ chạy K chuyên gia. Nhưng K cố định cho mọi đầu vào.
- **Định tuyến động:** Huang và cộng sự, và DeepSeekMoE, cho số chuyên gia thay đổi theo đầu vào. Tuy vậy mới thử trên mô hình ngôn ngữ lớn sinh văn bản, chưa trên NLI tiếng Việt.

*[chỉ vào dòng nhận xét]* Nhận xét của nhóm: MoE với số chuyên gia thay đổi là hướng mới, xuất hiện từ 2024, nhưng **cách học ngưỡng** và việc đánh giá trên dữ liệu đối kháng nhỏ chưa được nghiên cứu kỹ.

### Slide 5 · Tổng quan trong nước (55 giây)
*[chỉ vào tiêu đề]* Ở trong nước, tiếng Việt đã có dữ liệu NLI, nhưng MoE mới có một công trình.

*[chỉ lần lượt ba ô]*
- Năm 2020, **PhoBERT** là mô hình tiền huấn luyện đơn ngữ đầu tiên cho tiếng Việt.
- Năm 2022, **ViNLI** là bộ NLI miền mở tiếng Việt, công bố ở COLING, nhưng dữ liệu không đối kháng.
- Năm 2024, cùng với ViANLI, nhóm tác giả đề xuất **NLIMoE**, đạt 47,3% so với 45,5% của XLM-R Large. Đây là baseline của đề tài.

*[chỉ vào ô tối màu]* Khi đánh giá công trình này, nhóm thấy bốn điểm còn tồn tại:
- NLIMoE chỉ báo cáo **một lần chạy**, không có độ lệch chuẩn hay kiểm định thống kê.
- Ngưỡng động dùng mặt nạ cứng, nên nhánh độ phức tạp **không nhận gradient**. Nhóm sẽ chứng minh ở slide 12.
- Chưa có phiên bản MoE nhẹ hay phân tích định tuyến cho NLI tiếng Việt.
- Có 7 chi tiết cài đặt không được công bố, gây khó tái hiện.

### Slide 6 · Nghiên cứu có trước (50 giây)
*[chỉ vào ô bên trái]* Về công bố liên quan, [đọc thông tin của nhóm, hoặc: "nhóm chưa có công bố liên quan đến đề tài"].

*[chỉ vào ô bên phải]* Để chuẩn bị cho đề tài, nhóm đã làm bốn việc:
- Cài đặt NLIMoE đúng theo các phương trình 2 đến 16 của bài báo, kèm 20 kiểm tra tự động để đối chiếu với công thức.
- Phát hiện và khắc phục hiện tượng mô hình sụp khi huấn luyện, qua ba phiên bản cấu hình, từ v3 đến v5.
- Chạy **63 lần**: 21 cấu hình, mỗi cấu hình 3 seed, trên Colab với GPU A100 và L4, mỗi GPU chạy song song 3 job.
- Viết công cụ ghi gradient của nhánh độ phức tạp và thống kê định tuyến theo từng câu.

Mã nguồn và cấu hình được quản lý trên GitHub; mọi giả định cho chi tiết bài báo không công bố đều được ghi lại.

---

## Phần 3 · Paper baseline và phép tính (khoảng 4 phút 35 giây)

### Slide 7 · NLIMoE (1 phút 25 giây)
*[chỉ vào tiêu đề]* Đây là baseline NLIMoE. Ý tưởng chính: số chuyên gia được dùng thay đổi theo từng câu.

*[chỉ vào dãy năm ô]* Mô hình gồm năm bước: XLM-R Large, router, ngưỡng động, chọn và gộp chuyên gia, và phân loại. *[chỉ vào bảng]* Bảng bên dưới ghi phép tính của từng bước.
1. **Bước 1:** h là trung bình các vector token ở tầng cuối của XLM-R. Token đệm có trọng số 0 nên bị loại. Kết quả là một vector 1024 chiều đại diện cho cả cặp câu.
2. **Bước 2:** router nhân h với ma trận W_r rồi qua softmax, cho xác suất Pᵢ của 7 chuyên gia, tổng bằng 1.
3. **Bước 3:** một nhánh nhỏ tính độ khó C bằng sigmoid, rồi đặt **ngưỡng ρ = 0,1 + 0,1·C**. Vì C nằm giữa 0 và 1 nên ρ nằm giữa 0,1 và 0,2.
4. **Bước 4:** chuyên gia i được chọn khi Pᵢ lớn hơn ρ; mặt nạ mᵢ bằng 1 hoặc 0. Đầu ra z là trung bình có trọng số của các chuyên gia được chọn. Mỗi chuyên gia là Linear 1024 → 1024, rồi Dropout, rồi ReLU.
5. **Bước 5:** z qua LayerNorm và lớp tuyến tính, rồi softmax ra ba nhãn.

*[chỉ vào dòng L]* Hàm mất mát gồm cross-entropy, cộng 0,001 lần entropy của router và 0,01 lần loss cân bằng tải. Loss cân bằng tải ngăn mọi câu dồn vào một vài chuyên gia.

Cấu hình huấn luyện theo bài báo: 7 epoch, batch 16, learning rate 1e-5, dropout 0,4.

### Slide 8 · Bước 1–2 (45 giây) *(có thể lược)*
*[chỉ vào tiêu đề]* Để dễ theo dõi, nhóm tính tay một ví dụ nhỏ, đi từ cặp câu đến xác suất chọn từng chuyên gia.

*[chỉ vào ô bên trái]* **Bước 1, mean pooling.** Thu nhỏ còn 2 chiều: có ba token thật (1, 2), (3, 0), (2, 4), và một token đệm (5, 5). Mặt nạ là 1, 1, 1, 0. Cộng ba token thật được (6, 6), chia 3 được h = (2, 2). Token đệm có hệ số 0 nên không ảnh hưởng.

*[chỉ vào ô bên phải]* **Bước 2, router.** Giả sử router cho bảy điểm số từ 2,45 giảm dần tới 0. Lấy mũ e, được 11,59; 8,25; ... tổng là 33,08. Chia từng giá trị cho tổng, ta được P = 0,35; 0,25; 0,15; 0,10; 0,07; 0,05; 0,03. Tổng bằng 1; điểm càng lớn thì xác suất càng lớn.

### Slide 9 · Bước 3–4 (50 giây) *(có thể lược)*
*[chỉ vào ô bước 3]* **Bước 3, ngưỡng động.** Giả sử w_c·h + b_c = 0,12. Sigmoid của 0,12 là 0,530, nên ngưỡng ρ = 0,1 + 0,1 × 0,530 = 0,153.

*[chỉ vào bảng]* **Bước 4, chọn chuyên gia.** So từng xác suất với 0,153. Chỉ chuyên gia 1 (0,35) và chuyên gia 2 (0,25) vượt ngưỡng, nên mặt nạ là 1, 1, rồi toàn 0. Chuyên gia 3 có 0,15, chỉ thấp hơn ngưỡng một chút, nên không được chọn. Chuẩn hoá lại: tổng xác suất được chọn là 0,60; trọng số là 0,35 / 0,60 = 0,583 và 0,25 / 0,60 = 0,417.

*[chỉ vào ô tối màu]* Vậy z = 0,583 lần đầu ra chuyên gia 1, cộng 0,417 lần đầu ra chuyên gia 2. Câu này dùng 2 chuyên gia. Lưu ý quan trọng: C lớn thì ngưỡng cao, nên **ít** chuyên gia vượt ngưỡng.

### Slide 10 · Bước 5 và loss (40 giây) *(có thể lược)*
*[chỉ vào ô bước 5]* **Bước 5.** Lớp phân loại cho logit (−0,6; 0,1; 1,8). Qua softmax được xác suất 0,071; 0,143; và **0,785** cho nhãn mâu thuẫn. Nhãn đúng của ví dụ là mâu thuẫn, nên mô hình đoán đúng.

*[chỉ lần lượt các dòng bảng]*
- **Cross-entropy:** −ln 0,785 = 0,242.
- **Entropy của router:** 1,670 chia 7 bằng 0,239. Giá trị lớn nhất có thể là ln 7, khoảng 1,946.
- **Loss cân bằng tải:** nếu tải chia đều thì bằng 2,0; nếu mọi câu dồn vào chuyên gia 1 và 2 thì lên 7,0.
- **Tổng:** 0,242 + 0,001 × 0,239 + 0,01 × 2,0 = 0,262.

Cross-entropy chiếm 92% tổng loss; hai số hạng phụ chỉ đóng vai trò điều chỉnh nhẹ.

### Slide 11 · Kết quả công bố (55 giây)
*[chỉ vào tiêu đề]* Theo bài báo, NLIMoE hơn XLM-R Large 1,8 điểm trên tập kiểm thử.

*[chỉ vào ba ô số]* XLM-R Large đạt **45,5%**, NLIMoE đạt **47,3%**. Nhưng ô thứ ba là sai số chuẩn do nhóm tính: **±1,6 điểm**.

*[chỉ vào công thức bên trái]* Cách tính: mỗi câu kiểm thử là một lần thử đúng hoặc sai, nên sai số chuẩn của accuracy là căn của p nhân (1 − p) chia n. Với p khoảng 0,46 và n = 1.000 câu, ta được khoảng 0,016, tức 1,6 điểm phần trăm.

*[chỉ vào công thức bên phải]* Mức chênh là 47,3 − 45,5 = 1,8 điểm, tức chỉ khoảng **1,1 lần** sai số chuẩn.

*[chỉ vào dòng kết luận]* Hơn nữa, bài báo chỉ báo cáo một lần chạy, mà tinh chỉnh mô hình tiền huấn luyện dao động đáng kể theo seed. Vì vậy mức chênh này chưa đủ để kết luận, và cần đánh giá trên nhiều seed. Đây là lý do nhóm chạy mỗi cấu hình 3 lần.

---

## Phần 4 · Hạn chế và tính mới (khoảng 4 phút 30 giây)

### Slide 12 · Lỗ hổng gradient (1 phút 20 giây)
*[chỉ vào tiêu đề]* Đây là hạn chế chính của baseline và cũng là phát hiện chính của nhóm: **mặt nạ cứng chặn gradient, nên ngưỡng "động" không học được.**

*[chỉ vào chuỗi trên cùng]* Để học trọng số w_c của nhánh độ phức tạp, gradient phải đi ngược theo chuỗi: w_c → C → ρ → m → L. Theo quy tắc dây chuyền, đạo hàm ∂L/∂w_c là tích của bốn thừa số.

*[chỉ lần lượt bốn thừa số]*
- ∂L/∂m: loss thay đổi khi bật hoặc tắt một chuyên gia. Thừa số này khác 0.
- *[chỉ vào ô viền cam]* ∂m/∂ρ: mặt nạ m = 1 khi P > ρ là một **hàm bậc thang**. Dịch ngưỡng một chút thì m vẫn là 0 hoặc 1, nên đạo hàm **bằng 0** ở mọi điểm, trừ đúng điểm P = ρ, nơi hàm không khả vi.
- ∂ρ/∂C bằng 0,1, vì ρ = 0,1 + 0,1·C.
- ∂C/∂w_c bằng C(1 − C) nhân h, là đạo hàm của sigmoid.

*[chỉ vào dòng kết quả]* Nhân với 0 thì cả tích bằng 0.

*[chỉ vào đoạn hệ quả]* Hệ quả là w_c giữ nguyên giá trị khởi tạo suốt quá trình huấn luyện. Ngưỡng "động" thực chất chỉ là một hàm ngẫu nhiên của câu, không phản ánh độ khó. Bài báo không mô tả cách xử lý chỗ này. Router vẫn học bình thường, vì P xuất hiện trực tiếp trong tổng có trọng số; chỉ nhánh độ khó bị cắt.

Nhóm kiểm chứng bằng đo đạc: chuẩn gradient của w_c bằng 0 ở **cả 525 trên 525 lần đo**, tức 175 lần đo mỗi seed nhân 3 seed.

### Slide 13 · Các hạn chế khác (35 giây) *(có thể lược)*
*[chỉ vào tiêu đề]* Ngoài lỗi gradient, baseline còn ba hạn chế.

*[chỉ lần lượt ba ô]*
- **7,35 triệu tham số** cho khối chuyên gia: 7 chuyên gia, mỗi chuyên gia 1024 × 1024 trọng số cộng 1024 bias. Bài báo chưa thử chuyên gia nhỏ hơn.
- **Dấu hỏi:** chưa có phân tích chuyên gia nào xử lý kiểu bẫy nào, như phủ định, số lượng hay thời gian.
- **7 chi tiết không công bố:** checkpoint, optimizer, scheduler, seed, cách xử lý khi không chuyên gia nào được chọn, và một số chi tiết khác.

*[chỉ vào dòng cuối]* Với các chi tiết thiếu, nhóm chọn một giả định, ghi vào cấu hình và tài liệu tái hiện, để người khác chạy lại được.

### Slide 14 · Tính khoa học và tính mới (1 phút)
*[chỉ vào tiêu đề]* Tóm lại, đề tài sửa một lỗi gradient trong cơ chế định tuyến và đánh giá lại cho tin cậy. Slide này có bốn ô.

*[ô xám]* **Hạn chế hiện thời:** ngưỡng "động" của NLIMoE không học được; kết quả công bố từ một lần chạy; khối chuyên gia 7,35 triệu tham số.

*[ô cam]* **Vấn đề khoa học:** làm sao để ngưỡng định tuyến nhận gradient **mà không đổi lượt xuôi**, tức không đổi cách mô hình dự đoán? Về tính cấp thiết: NLIMoE là MoE duy nhất cho NLI tiếng Việt, nên cần được kiểm chứng trước khi phát triển tiếp.

*[ô xanh]* **Hướng giải quyết mới:** dùng Straight-Through Estimator cho ngưỡng; thử chuyên gia bottleneck; đo gradient trực tiếp; đánh giá 3 seed với **bootstrap theo câu** và McNemar; và thêm các mô hình đối chứng để tách riêng phần đóng góp của từng thành phần.

*[ô tối]* **Lợi ích:** về khoa học, chứng minh và khắc phục lỗi gradient, cùng một quy trình tái hiện có kiểm định. Về thực tiễn, mã nguồn mở và một mô hình NLI tiếng Việt ổn định hơn.

### Slide 15 · Câu hỏi nghiên cứu (35 giây)
*[đọc chậm câu hỏi]* Câu hỏi nghiên cứu của nhóm là: **nếu ngưỡng định tuyến thật sự học được độ khó, MoE có thể nhỏ đi một nửa mà vẫn giữ độ chính xác không?**

*[chỉ lần lượt bốn mục]* Nhóm trả lời bằng ba cải tiến và một cách đánh giá:
- Cải tiến 1: ngưỡng học được bằng STE.
- Cải tiến 2: chuyên gia bottleneck.
- Cải tiến 3: phạt chi phí và backbone nhỏ.
- Đánh giá: 3 seed và kiểm định bootstrap.

Câu hỏi được tách thành hai giả thuyết kiểm định được: H1, ngưỡng học được và có quan hệ với độ khó; H2, khối chuyên gia nhỏ đi một nửa mà accuracy không giảm có ý nghĩa.

### Slide 16 · Mục tiêu, đối tượng, phạm vi (55 giây)
*[chỉ vào ô mục tiêu]* Nhóm đặt bốn mục tiêu đo được:
- **MT1:** tái hiện XLM-R Large và NLIMoE với 3 seed.
- **MT2:** ngưỡng học được, tức chuẩn gradient của w_c lớn hơn 0, và accuracy không thấp hơn NLIMoE.
- **MT3:** giảm 49,9% tham số khối chuyên gia, accuracy giảm không quá 1 điểm.
- **MT4:** phân tích định tuyến và ít nhất 100 lỗi theo kiểu bẫy.

*[chỉ vào ô đối tượng]* Đối tượng nghiên cứu là khối MoE trên XLM-R Large, cơ chế định tuyến bằng ngưỡng, và bài toán NLI đối kháng tiếng Việt.

*[chỉ vào ô phạm vi]* Phạm vi gồm bộ ViANLI, cấu hình huấn luyện theo bài báo, GPU A100 và L4 trên Colab; không huấn luyện mô hình ngôn ngữ lớn.

*[chỉ vào dòng khả năng giải quyết]* Đến nay mã nguồn đã hoàn chỉnh và 63 lần chạy đã xong. MT1, MT2 đã đạt. MT3 cũng đạt nhờ chuyên gia bottleneck có kết nối tắt: giảm 49,9% tham số, accuracy chỉ giảm 0,7 điểm. MT4 đang gán nhãn 120 lỗi.

---

## Phần 5 · Phương pháp đề xuất (khoảng 6 phút 15 giây)

### Slide 17 · Cải tiến 1: STE (1 phút 20 giây)
*[chỉ vào tiêu đề]* Cải tiến đầu tiên là ngưỡng học được. Ý tưởng gói trong một câu: **lượt xuôi giữ mặt nạ cứng, lượt ngược mượn đạo hàm mềm.**

*[chỉ vào ô bên trái]* **Lượt xuôi**, khi dự đoán: vẫn dùng mᵢ = 1 khi Pᵢ > ρ, giống hệt bài báo. Hành vi suy luận không đổi.

*[chỉ vào ô viền cam]* **Lượt ngược**, khi học: nhóm dùng một mặt nạ mềm, là sigmoid của (Pᵢ − ρ) chia nhiệt độ T. Hàm này trơn, nên đạo hàm theo ρ bằng −m̃(1 − m̃) chia T, khác 0, và lớn nhất khi P gần ngưỡng. Nhờ vậy gradient tới được C và w_c.

*[chỉ vào ô tối màu]* Trong code, hai lượt được ghép bằng một công thức: mặt nạ STE bằng mặt nạ cứng cộng mặt nạ mềm, trừ mặt nạ mềm đã dừng gradient. Ở lượt xuôi, hai phần mềm triệt tiêu nhau nên giá trị đúng bằng mặt nạ cứng. Ở lượt ngược, phần dừng gradient coi như hằng số, nên đạo hàm lấy từ mặt nạ mềm. Trong PyTorch chỉ là một dòng với hàm detach.

*[chỉ vào dòng kiểm chứng]* Nhóm kiểm chứng bằng ba cách: chuẩn gradient của w_c phải khác 0; C phải có quan hệ nhất quán với việc đoán sai qua 3 seed, đo bằng Spearman; và dò nhiệt độ T với 0,05, 0,1 và 0,2 trên tập kiểm định.

Nhóm chọn STE thay vì Gumbel-softmax vì STE không thêm nhiễu vào lượt xuôi, nên so sánh với baseline công bằng hơn.

### Slide 18 · Ví dụ tính gradient (55 giây) *(có thể lược)*
*[chỉ vào tiêu đề]* Dùng lại ví dụ cũ với ρ = 0,153 và T = 0,1.

*[chỉ vào bảng, từ trên xuống]* Dòng xác suất và dòng mặt nạ cứng giống slide 9. Dòng thứ ba: đạo hàm của mặt nạ cứng theo ρ bằng **0 ở cả bảy chuyên gia**. Dòng thứ tư là mặt nạ mềm. Dòng cuối là đạo hàm với STE: **khác 0 ở cả bảy chuyên gia**, lớn nhất là −2,50 ở chuyên gia 3, vì xác suất 0,15 của nó nằm sát ngưỡng nhất.

*[chỉ vào ô tối màu]* Ghép theo quy tắc dây chuyền cho chuyên gia 3: −2,50 nhân 0,1 nhân 0,249 ra −0,062 lần ∂L/∂m₃ lần h. Giá trị này khác 0, nên w_c được cập nhật.

*[chỉ vào dòng đọc kết quả]* Ý nghĩa: nếu bật chuyên gia 3 làm loss giảm, gradient sẽ hạ ngưỡng, tức hạ C, để chuyên gia đó được chọn. Trên dữ liệu thật, E2 thực sự hạ C trung bình từ 0,41–0,53 xuống 0,25–0,28 ở cả ba seed. Nghĩa là **ngưỡng đã học**.

### Slide 19 · Cải tiến 2: bottleneck (50 giây)
*[chỉ vào tiêu đề]* Cải tiến thứ hai là chuyên gia bottleneck: nén xuống 256 chiều rồi mở lại 1024.

*[chỉ vào ô bên trái]* Chuyên gia gốc là một lớp 1024 → 1024, khoảng **1,05 triệu** tham số; 7 chuyên gia là 7.347.200 tham số.

*[chỉ vào ô bên phải]* Chuyên gia đề xuất đi qua 1024 → 256, ReLU, rồi 256 → 1024. Mỗi chuyên gia còn khoảng **0,53 triệu** tham số; 7 chuyên gia còn 3.678.976, tức **giảm 49,9%**.

*[chỉ vào hai dòng cuối]* Khi suy luận, nhóm dùng sparse dispatch: chỉ tính các chuyên gia được chọn. Cơ sở của ý tưởng là adapter của Houlsby và cộng sự: một bottleneck nhỏ đủ để thích nghi Transformer lớn. Nhóm còn thử bản có kết nối tắt, cộng thêm chính đầu vào h vào đầu ra, để không làm mất thông tin. Kết quả ở slide 24 sẽ cho thấy kết nối tắt này rất quan trọng.

### Slide 20 · Nội dung bước 1–3 (1 phút)
*[chỉ vào tiêu đề]* Phương pháp gồm sáu bước, nguyên tắc là tái hiện baseline rồi thay từng thành phần một.

*[đi lần lượt ba dòng]*
- **Bước 1, tái hiện:** mục tiêu là số liệu E0 và E1 tự chạy, có độ lệch chuẩn. Nhóm cài đúng phương trình của bài báo và ghi giả định cho chi tiết thiếu. Đã xong.
- **Bước 2, ngưỡng học được:** mục tiêu là gradient khác 0 và accuracy không thấp hơn E1. Dùng STE với T = 0,1, dò thêm 0,05 và 0,2. Đã xong.
- **Bước 3, MoE nhẹ:** mục tiêu là giảm 49,9% tham số, accuracy không giảm quá 1 điểm. E3a bị sụp một seed; E3b với r = 384 và E3c có kết nối tắt thì giữ được accuracy.

*[chỉ vào ba ô dưới]* Siêu tham số giữ cố định như bài báo: 7 epoch, batch 16, learning rate 1e-5, độ dài tối đa 256, dropout 0,4. Nhóm chỉ thêm, ở cấu hình v5: lịch learning rate tuyến tính với 10% warmup và gradient clipping 1,0 để chống sụp. Checkpoint được chọn theo accuracy kiểm định, đánh giá mỗi 400 bước.

### Slide 21 · Nội dung bước 4–6 (1 phút 5 giây)
*[chỉ vào tiêu đề]* Ba bước tiếp theo là ablation, kiểm định thống kê và hướng mở rộng.

*[đi lần lượt ba dòng]*
- **Bước 4, ablation:** để biết đóng góp của ngưỡng động và của loss cân bằng tải, nhóm thay ngưỡng bằng top-K với 4 trên 5 chuyên gia, và bỏ loss cân bằng tải khỏi E2. Đã xong.
- **Bước 5, kiểm định và phân tích:** bootstrap theo câu và McNemar, tương quan Spearman giữa C và đúng sai, các mô hình đối chứng A4 đến A6, và gán nhãn 120 lỗi. Phần kiểm định và đối chứng đã xong; phân tích lỗi đang làm.
- **Bước 6, lightweight:** phạt số chuyên gia (E4), cần STE mới làm được; đổi sang XLM-R Base (E5); chưng cất từ E2 và từ ensemble E0 + E2 (E6, E8). Tất cả đã chạy 3 seed và đã đo độ trễ.

*[chỉ vào hai ô dưới]* Về tài nguyên: [đơn vị phối hợp nếu có]; tính toán trên Google Colab Pro+ với GPU A100 40 GB và L4 24 GB; lưu trữ trên Google Drive và GitHub. Kết quả sẽ công bố gồm mã nguồn, cấu hình và log trên GitHub, cùng báo cáo đồ án.

### Slide 22 · Thiết kế thí nghiệm (1 phút 5 giây)
*[chỉ vào tiêu đề]* Tổng cộng có hai baseline, tám nhóm thực nghiệm mới và sáu ablation.

*[đi từ trên xuống bảng]*
- **E0 và E1** là hai baseline: XLM-R Large và NLIMoE của bài báo.
- **E2** là đề xuất chính: E1 cộng ngưỡng học được bằng STE.
- **E3 đến E6** là các thực nghiệm lightweight: chuyên gia bottleneck với ba biến thể a, b, c; phạt số chuyên gia với hai mức λ; backbone XLM-R Base; và chưng cất từ E2.
- **E7** thêm trọng số 1,2 cho nhãn mâu thuẫn, nhằm cải thiện nhãn khó nhất.
- **E8** chưng cất từ ensemble E0 + E2, vào Base và vào Large, nhằm cải thiện accuracy.
- **A1 đến A3** là ablation: Top-K, bỏ loss cân bằng tải, và dò nhiệt độ T.
- **A4 đến A6** là **đối chứng**: XLM-R với pooling trung bình và LayerNorm; một chuyên gia không định tuyến; và E2 với cách xử lý an toàn khi không chọn chuyên gia nào. Ba đối chứng này giúp tách riêng phần đóng góp của pooling, của định tuyến và của chi tiết cài đặt.

*[chỉ vào dòng nguyên tắc]* Mỗi thực nghiệm chỉ đổi một thành phần. 21 cấu hình nhân 3 seed là 63 lần chạy, tất cả đã xong.

---

## Phần 6 · Kết quả (khoảng 6 phút 45 giây)

### Slide 23 · Kết quả chính (1 phút 40 giây)
*[chỉ vào tiêu đề]* Kết quả đầu tiên nói thẳng thông điệp: **E2 hơn NLIMoE 1,7 điểm, nhưng chỉ ngang một chuyên gia.**

*[chỉ vào hai dòng baseline]*
- E0, XLM-R Large: 45,8%, nhưng độ lệch chuẩn tới 4,1 điểm, và 2 trên 3 seed gần như bỏ hẳn nhãn mâu thuẫn, nên macro-F1 chỉ 40,0%.
- E1, NLIMoE tự chạy: **44,1%**, thấp hơn con số 47,3% của bài báo. Con số công bố nằm trong khoảng dao động giữa các seed.

*[chỉ vào dòng E2]* Thêm STE, **E2 đạt 45,8%, macro-F1 45,6%**, dùng trung bình 4 trên 7 chuyên gia, và hơn E1 ở cả 3 seed.

*[chỉ vào A4, A5]* Nhưng nhóm muốn trả lời trung thực: lợi ích thật đến từ đâu? Hai đối chứng cho câu trả lời.
- **A4** chỉ thêm pooling trung bình và LayerNorm vào XLM-R, không có MoE: 45,7%, macro-F1 45,2%. Vậy phần lớn lợi thế macro-F1 của E2 so với E0 đến từ pooling.
- **A5** dùng đúng một chuyên gia, không định tuyến: cũng 45,7%, và hơn NLIMoE gốc 1,6 điểm.

*[chỉ vào các dòng còn lại]*
- E7, thêm trọng số cho nhãn mâu thuẫn: accuracy không đổi, mà recall nhãn mâu thuẫn còn giảm từ 38,5 xuống 32,1.
- A1, Top-K: ngang E1 nhưng dùng nhiều chuyên gia hơn.
- A2, bỏ loss cân bằng tải: seed 42 dồn mọi câu vào 1 chuyên gia.
- A3, đổi nhiệt độ T: 46,0% ở cả hai mức, tức không nhạy với T.
- A6, xử lý an toàn khi không chọn chuyên gia nào: hết gai gradient, accuracy không đổi.

*[chỉ vào dòng đọc kết quả]* Kết luận của slide: STE sửa được NLIMoE, nhưng trên ViANLI, bảy chuyên gia có định tuyến **chưa hơn một chuyên gia** về accuracy. Giá trị của STE nằm ở chỗ khác, ở slide tiếp theo.

### Slide 24 · Kết quả lightweight và chưng cất (1 phút 55 giây)
*[chỉ vào tiêu đề]* Tiêu đề slide này là kết quả tốt nhất: **chưng cất ensemble đạt 47,3% với nửa số chuyên gia.** Cột cuối so mọi dòng với E2, vốn đạt 45,8% với 4 chuyên gia.

*[chỉ vào ba dòng E3]* Về chuyên gia bottleneck:
- E3a, r = 256: giảm 50% tham số nhưng 1 seed bị sụp, trung bình chỉ 41,0%.
- E3b, r = 384: giảm 25% tham số, accuracy 45,7%, ngang E2.
- **E3c**, r = 256 có kết nối tắt: vẫn giảm 50% tham số, hết sụp, đạt 45,1%, cao hơn E3a 4,1 điểm.

*[chỉ vào hai dòng E4]* Về phạt chi phí, đây là nơi STE phát huy tác dụng: khi ngưỡng học được, **số chuyên gia trở thành thứ tối ưu được**. Với λ = 0,02, số chuyên gia giảm từ 4 xuống **1,8**, tức bớt 55%, accuracy 45,2%. Với λ = 0,1, mọi câu chỉ còn 1 chuyên gia. Không có STE, mức phạt này không có gradient.

*[chỉ vào ba dòng E5, E6]* Về backbone nhỏ: XLM-R Base nhanh hơn Large khoảng **2,1 lần trên GPU** và **2,8 lần trên CPU int8**. E5a là Base thuần, 43,1%; thêm MoE và STE thành E5b, 44,5%; chưng cất từ E2 thành E6, 44,3% nhưng macro-F1 tăng lên 42,9.

*[chỉ vào hai dòng E8]*
- **E8a**: Base chưng cất từ ensemble E0 + E2, đạt 45,6%, độ lệch chuẩn chỉ 0,2. Đây là mô hình Base tốt nhất và ổn định nhất.
- **E8b**: E2 chưng cất từ cùng ensemble, đạt **47,3%**, macro-F1 47,0%, và chỉ dùng **2,0 chuyên gia mỗi câu**, một nửa so với E2. Đây là mô hình đơn tốt nhất của nhóm, ngang con số bài báo công bố nhưng đo trên 3 seed.

*[chỉ vào dòng đo trên L4]* Một quan sát quan trọng từ đo thực tế: Large mất 2,3–2,6 ms mỗi câu, Base 1,1–1,2 ms. Khối MoE chỉ chiếm 2 đến 5% thời gian của Large. Vì vậy, muốn nhẹ thật thì phải thu nhỏ backbone; bớt chuyên gia chủ yếu chứng minh khả năng điều khiển chi phí.

### Slide 25 · Kiểm định thống kê (1 phút 55 giây)
*[chỉ vào tiêu đề]* Kiểm định bằng bootstrap theo câu cho thấy STE và chưng cất ensemble có lợi, ở mức biên.

Vì sao dùng bootstrap theo câu? Ba seed của một mô hình được chấm trên cùng 1.000 câu, nên 3.000 dự đoán không độc lập; gộp lại thì p-value nhỏ hơn thực tế. Nhóm lấy câu làm đơn vị: mỗi câu có tỷ lệ seed đoán đúng; lấy mẫu lại 1.000 câu 10.000 lần để có khoảng tin cậy 95%, và tính p bằng hoán vị đổi dấu.

*[đi lần lượt các dòng]*
- **E2 so với E1:** +1,7 điểm, khoảng tin cậy từ 0 đến 3,3, p = 0,05, thắng 3 trên 3 seed. STE có ích, ở mức biên.
- **E8b so với E2:** +1,6 điểm, khoảng tin cậy từ −0,1 đến 3,3, p = 0,07, thắng 3 trên 3 seed. Gần có ý nghĩa.
- **A5 so với E1:** +1,6 điểm, p = 0,05. Một chuyên gia không định tuyến còn hơn NLIMoE gốc.
- **A4 so với E0:** chênh 0, p = 1,00. Pooling và LayerNorm không đổi accuracy, nhưng tăng macro-F1 5,2 điểm.
- **E3c so với E3a:** +4,1 điểm, p nhỏ hơn 0,001. Kết nối tắt chữa được hiện tượng sụp.
- **E4a so với E2:** −0,6 điểm, p = 0,59. Bớt 55% chuyên gia không làm giảm accuracy một cách có ý nghĩa.
- **E5b so với E5a:** +1,4, p = 0,09. MoE và STE giúp backbone nhỏ, nhưng chưa chắc chắn.
- **E8a so với E6:** +1,3, p = 0,14. Thầy là ensemble tốt hơn thầy là E2.
- A3, A6 và E7 so với E2: chênh không quá 0,3 điểm, không phát hiện khác biệt.

*[chỉ vào dòng gradient]* Về gradient: E1 bằng 0 ở cả 525 lần đo, còn mọi biến thể có STE đều khác 0. Với E2, câu có C thấp, tức câu khó, dùng 4,4 đến 4,7 chuyên gia; câu có C cao chỉ dùng 3,1 đến 3,4. Ngưỡng đã phân bổ chuyên gia theo độ khó.

### Slide 26 · Quan sát về dữ liệu (1 phút 15 giây)
*[chỉ vào tiêu đề]* Từ 63 lần chạy, nhóm muốn hiểu vì sao mô hình đơn dừng quanh 45 đến 47% trên ViANLI. Có bốn quan sát.

*[ô cam]* **30,2%** là recall của nhãn mâu thuẫn, so với 59,0% của nhãn suy ra và 45,9% của nhãn trung lập. Tập train cũng ít câu mâu thuẫn nhất, 30,9%, trong khi tập test là 33,3%. Có 6 trên 63 lần chạy gần như bỏ hẳn một nhãn.

*[ô xanh]* **62,6%** là tỷ lệ hai lần chạy bất kỳ cho cùng dự đoán. Mô hình rất bất ổn giữa các seed, nhưng 96,8% số câu được ít nhất một lần chạy đoán đúng.

*[ô xám thứ nhất]* **3,2%**, tức 32 câu, không lần chạy nào đoán đúng: có thể khó thật hoặc nhãn có vấn đề. Ngoài ra 30,5% số câu chỉ đúng ở dưới 20% số lần chạy.

*[ô xám thứ hai]* **97%** tiền đề của tập test cũng có trong tập train; khác biệt nằm ở giả thuyết. Từ "không" phân bố gần đều ba nhãn, 35, 32 và 32%, nên dữ liệu ít lối tắt từ vựng.

*[chỉ vào ô kết luận]* Kết luận: giới hạn đến từ nhãn mâu thuẫn khó và độ bất ổn giữa các lần chạy, **chưa phải trần của dữ liệu**. Bỏ phiếu 63 lần chạy đạt 48,5%, ensemble của E0 đạt 49,2%, và chưng cất ensemble đã đưa một mô hình đơn lên 47,3%.

---

## Phần 7 · Kế hoạch, đóng góp, kết luận (khoảng 3 phút 40 giây)

### Slide 27 · Kế hoạch thời gian (1 phút 10 giây)
*[chỉ vào tiêu đề]* Kế hoạch gồm sáu giai đoạn, xen kẽ thí nghiệm, xêmina và viết bản thảo.

*[đi từ trên xuống bảng]*
- **09 đến 22/11**, đã xong: tái hiện E0, E1; đề xuất E2; khắc phục hiện tượng sụp; viết đề cương.
- **23 đến 27/11**, đã xong: E3 đến E8, ablation và đối chứng A1 đến A6, tổng 63 lần chạy; làm slide.
- **27 đến 30/11**: đo độ trễ GPU, CPU và int8, kiểm định bootstrap, đều đã xong; hoàn thiện video và nộp proposal trước ngày 01/12.
- **01 đến 14/12**: gán kiểu bẫy cho 32 câu khó và 120 lỗi; đo int8 cho E8; xêmina 1 trình bày kết quả sơ bộ.
- **15 đến 28/12**: tinh chỉnh trung gian trên ViNLI, chưng cất vào XLM-R Base; viết bản thảo chương 1 đến 3.
- **Từ 29/12 đến hạn nộp**: chạy 3 seed cho cấu hình mới, đóng gói mã và log; xêmina 2, viết chương 4, 5 và nộp báo cáo.

*[chỉ vào hai ô dưới]* Nhóm gặp người hướng dẫn [lịch gặp] và gửi báo cáo tiến độ 2 tuần một lần. Để kết hợp với việc học, GPU chạy nền vào buổi tối và cuối tuần, còn phân tích và viết làm vào giờ không có lớp.

### Slide 28 · Đóng góp (40 giây)
Đề tài có ba đóng góp dự kiến.

*[cột 01]* **Một bản tái hiện NLIMoE có kiểm định:** 63 lần chạy, 21 cấu hình nhân 3 seed, kiểm định bootstrap theo câu, các đối chứng để tách phần đóng góp, và mã nguồn công khai. Bản tái hiện cho thấy mức hơn 1,8 điểm của bài báo không được xác nhận.

*[cột 02]* **Ngưỡng định tuyến học được:** chứng minh và sửa chỗ gradient bị chặn; tăng 1,7 điểm so với NLIMoE gốc, p = 0,05; và câu khó được dùng nhiều chuyên gia hơn.

*[cột 03]* **MoE nhẹ hơn cho NLI tiếng Việt:** bớt 55% chuyên gia mà không mất accuracy có ý nghĩa; chưng cất ensemble đạt 47,3% với 2 chuyên gia mỗi câu.

### Slide 29 · Kết luận và hướng phát triển (1 phút 30 giây)
*[chỉ vào tiêu đề]* Thông điệp chính của nhóm: **ngưỡng định tuyến đã học được, nên MoE điều khiển được chi phí.**

*[đi lần lượt ba dòng bảng]*
- Vấn đề thứ nhất, **ngưỡng động không học được**. Giải pháp là Straight-Through Estimator. Kết quả: gradient từ 0 thành khác 0 ở 525 trên 525 lần đo; tăng 1,7 điểm, p = 0,05, thắng 3 trên 3 seed; câu khó dùng nhiều chuyên gia hơn.
- Vấn đề thứ hai, **chi phí MoE cố định**. Giải pháp là phạt số chuyên gia, backbone Base và chưng cất ensemble. Kết quả: bớt 55% chuyên gia với p = 0,59; Base nhanh hơn 2,1 lần trên GPU; và mô hình tốt nhất đạt 47,3% với 2 chuyên gia mỗi câu.
- Vấn đề thứ ba, **kết quả công bố từ một lần chạy, chưa có đối chứng**. Giải pháp là 63 lần chạy, bootstrap theo câu, đối chứng A4 và A5. Kết quả: hết sụp, và một kết luận trung thực: **định tuyến chưa thêm accuracy** so với một chuyên gia, lợi ích chính là chi phí.

*[chỉ lần lượt ba ô dưới]* Hướng phát triển:
- **Trước báo cáo cuối kỳ:** gán kiểu bẫy cho 32 câu khó và 120 lỗi; kiểm định thêm E8b so với E0 và E1; đo int8 cho E8.
- **Thu hẹp khoảng cách Base và Large:** tinh chỉnh trung gian trên ViNLI; chưng cất ensemble vào Base, hiện E8a đạt 45,6%.
- **Hướng tới bài báo:** đưa chuyên gia dạng LoRA vào bên trong encoder và định tuyến theo token, để phần tiết kiệm nằm ở chỗ tốn chi phí nhất; mở rộng sang XNLI và ANLI.

Các hướng này là kế hoạch, chưa phải kết quả.

### Slide 30–31 · Tài liệu tham khảo (20 giây)
*[lướt qua hai slide]* Đây là 22 tài liệu tham khảo theo định dạng IEEE, gồm các bộ dữ liệu NLI, các công trình về Mixture-of-Experts, XLM-R, STE, adapter và các phương pháp kiểm định.

Cảm ơn thầy cô và các bạn đã lắng nghe. Nhóm rất mong nhận được góp ý của thầy cô.

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
