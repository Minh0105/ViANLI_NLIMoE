# Khảo sát & Phân tích Dữ liệu ViANLI (Data Analysis)

Bộ dữ liệu: `uitnlp/ViANLI`  
Mô hình Tokenizer sử dụng đo lường: `xlm-roberta-base`

---

## 1. Thống kê Kích thước Phân chia (Splits)

| Split | Số lượng mẫu (Samples) | Tỷ lệ (%) | Ghi chú |
|---|---:|---:|---|
| **Train** | 8,012 | 80.0% | Huấn luyện mô hình |
| **Validation** | 1,000 | 10.0% | Chọn checkpoint & dò ngưỡng routing |
| **Test** | 1,000 | 10.0% | Đánh giá cuối cùng (độc lập, không tuning) |
| **Tổng cộng** | **10,012** | **100%** | |

---

## 2. Phân bố Nhãn (Class Distribution)

| Split | Entailment | Contradiction | Neutral | Tổng |
|---|---:|---:|---:|---:|
| **Train** | 2,615 | 2,473 | 2,924 | 8,012 |
| **Validation** | 334 | 333 | 333 | 1,000 |
| **Test** | 334 | 333 | 333 | 1,000 |

- **Nhận xét cân bằng dữ liệu**: Các lớp trong 3 split được phân bổ tương đối đồng đều theo chuẩn NLI.

---

## 3. Thống kê Chiều dài Văn bản (Sequence Lengths)

| Split / Đại lượng | Premise Mean (Từ) | Hyp Mean (Từ) | Subword Mean | Subword P95 | Subword Max |
|---|---:|---:|---:|---:|---:|
| **Train** | 29.0 | 16.7 | 59.5 | 95.0 | 144 |
| **Validation** | 28.8 | 17.4 | 59.9 | 94.0 | 148 |
| **Test** | 29.0 | 17.3 | 60.3 | 95.0 | 164 |

- **Max Length khuyến nghị**: `256` tokens bao phủ hơn 95% độ dài chuỗi mà không lãng phí tài nguyên tính toán.

---

## 4. 30 Mẫu Đối kháng Trích xuất Thủ công (Sample Adversarial Inspection)

| STT | Premise | Hypothesis | Gold Label |
|---|---|---|---|
| 1 | Nổi danh với những mẫu xe cơ bắp thuộc các dòng Touring, Street nhưng Harley-Davidson (H-D) đến trước 2020 chưa có một sản phẩm nào thuộc dòng Adventure, chuyên trị các địa hình off-road cực đoan. | Mẫu xe Harley-Davidson có cơ bắp. | `contradiction` |
| 2 | "Run for Pride" là cơ hội để các runner thể hiện sự hưởng ứng với một xã hội đa màu sắc, nơi những người thuộc cộng đồng LGBT (người đồng tính, song tính và chuyển giới) được sống với giới tính thật, được là chính mình, được yêu và bày tỏ cảm xúc. | Người chuyển giới là thành viên của cộng đồng LGBT. | `entailment` |
| 3 | Trước đó ít ngày, Youtuber Mỹ thậm chí khẳng định sẽ giành chiến thắng knock-out trước Mayweather. | Người Mỹ luôn vô cùng khiêm tốn. | `contradiction` |
| 4 | Đây là khoản viện trợ không hoàn lại của Nhật Bản cho Việt Nam trong buổi lễ ký kết công hàm trao đổi giữa Đại sứ quán Nhật và Bộ Tài nguyên và Môi trường, chiều 23/4. | Việt Nam không cần trả lại số tiền viện trợ cho Đất nước mặt trời mọc. | `entailment` |
| 5 | - Người đến quán bánh canh cá lóc O Thanh, hai chi nhánh 287/5 Nguyễn Đình Chiểu và 289 Nguyễn Đình Chiểu, phường 5, quận 3, từ ngày 12/5 đến sáng 20/5. | - Bánh canh cá lóc O Thanh lúc nào cũng vắng khách nhưng số người lui đến không nhiều. | `contradiction` |
| 6 | "Bệnh nhân 3298" là nam, 25 tuổi, địa chỉ tại quận Liên Chiểu, TP Đà Nẵng, là F1 của "bệnh nhân 3219". | Bệnh nhân 3298" và "bệnh nhân 3219" có địa chỉ tại quận Liên Chiểu, TP Đà Nẵng. | `neutral` |
| 7 | Việc nâng cấp đường Phạm Văn Đồng và một phần đường Nguyễn Tất Thành (qua TP Pleiku) được khởi công năm 2020, với chiều dài 4 km, trong đó hơn 2,5 được thảm nhựa dày 5 cm. | Chính quyền phê duyệt cho việc thi công con đường Phạm Văn Đồng vào năm 2020. | `neutral` |
| 8 | Siêu mẫu Mỹ cuối tuần qua cầm cờ Palestine, hòa vào đám đông hàng nghìn người ở Brooklyn, New York, hô hào khẩu hiệu ủng hộ người Palestine. | Siêu mẫu Mỹ cầm cờ Palestine hòa vào đám động chỉ với 1 nghìn người . | `contradiction` |
| 9 | "Đạo luật Goo Hara" về quyền thừa kế tài sản được đưa ra thảo luận từ cuối năm 2019 nhưng bị bác bỏ. | Mỗi đạo luật cần được thảo luận trước khi quyết định có ban hành hay không. | `entailment` |
| 10 | Một con trăn quý hiếm được phát hiện nằm trên cây bàng trong sân trường THPT Nguyễn Trãi, TP Hội An. | Trăn quý hiếm chỉ được phát hiện ở Hội An. | `neutral` |
| 11 | Clare Mehta gọi giai đoạn 30-45 là "tuổi thực sự trưởng thành". | 30-45 là lúc mọi người có bắt đầu hiểu và chăm lo cho cuộc sống của "sự trưởng thành". | `neutral` |
| 12 | Atletico lỡ cơ hội nới cách biệt từ đỉnh bảng sau trận hòa không bàn thắng trên sân Camp Nou. | Atletico đang đứng vị trí thứ 2 trên bảng xếp hạng. | `contradiction` |
| 13 | Tuy nhiên, khi thị trường vượt 1.265 điểm, áp lực bán lấn át bên mua. | Áp lực đáp ứng thị hiếu người mua sắm giúp cho nhiều sản phẩm được tạo ra trên thị trường. | `neutral` |
| 14 | Ingenuity cũng từng gặp trục trặc trong lúc chuẩn bị cho chuyến bay đầu tiên. | Ingenuity không phải luôn hoàn hảo trong mỗi chuyến bay. | `entailment` |
| 15 | Cho hình bình hành ABCD, điểm E là trung điểm cạnh AB như hình dưới đây. | E là điểm nằm chính giữa hình bình hành ABCD. | `contradiction` |
| 16 | "Đây là một trận đấu tuyệt vời, với những diễn biến căng thẳng và hấp dẫn. | Trận đấu diễn ra rất nhanh với những diễn biến căng thẳng và hấp dẫn. | `neutral` |
| 17 | Mặt trống trang trí hình mặt trời 14 tia, tượng trưng cho dương, giữa các tia trang trí hoa văn lông đuôi chim công hình tam giác, tượng trưng cho âm. | Mặt trời 14 tia thể hiện cho âm do mặt trời tỏa ra hơi nóng làm tan chảy thế lực ánh sáng. | `contradiction` |
| 18 | Theo Công an quận Hoàng Mai, Đông vừa đóng vai người môi giới vừa đóng vai khách mua dâm. | Đông là người khờ khạo ngốc nghếch. | `contradiction` |
| 19 | Với chiều dài khoảng 13 m, mực khổng lồ là một trong những động vật không xương sống lớn nhất trên Trái Đất. | Mực khổng lồ là loài thân mềm dài nhất trên địa cầu. | `entailment` |
| 20 | Hàng chục nghìn căn nhà với giá trị khoảng 19,3 tỷ USD có thể được người Hong Kong bán ra trong năm nay để di cư sang Anh. | Số lượng nhà được rao bán qua hãng giao dịch bất động sản lớn nhất Hong Kong tăng nhiều so với năm ngoái. | `neutral` |
| 21 | Giá vàng thế giới giảm sáng nay do lãi suất trái phiếu Mỹ và đồng bạc xanh diễn biến tích cực. | Lãi suất trái phiếu Mỹ và đồng bạc xanh chỉ ảnh hưởng đến giá vàng thế giới vào buổi sáng. | `neutral` |
| 22 | Tại lễ phát động được tổ chức trực tuyến trên toàn quốc, Chủ tịch Mặt trận Tổ quốc Việt Nam, ông Đỗ Văn Chiến kêu gọi các tổ chức, cá nhân và người nước ngoài đang làm việc, sinh sống ở Việt Nam... nêu cao ý thức tự giác, tinh thần đoàn kết, trách nhiệm với cộng đồng, chấp hành nghiêm quy định trong phòng, chống Covid-19. | Những công dân ngoại quốc tại hoạt động trong nước cũng được kêu gọi ý thức phòng, chống Covid-19. | `entailment` |
| 23 | Mike McCoy cùng chú chó Jake đang đi dạo quanh khu phố ở Holiday (bang Florida) thì con cá sấu xuất hiện, kéo tuột con chó xuống nước, hôm 11/5. | Cá sấu là động vật ăn cỏ và chỉ sống ở dưới nước. | `contradiction` |
| 24 | Doanh số điện thoại thông minh của hãng đã đi xuống sau khi Mỹ hạn chế hãng mua phần cứng, phần mềm và dịch vụ từ các nhà cung cấp Mỹ mà không có sự chấp thuận của chính phủ. | Điện thoại thông minh của hãng tuy đã đi xuống do Mỹ hạn chế nhưng đã bán được rất nhiều cho chính phủ. | `neutral` |
| 25 | Tháng 11/2020, công ty cũng chia cổ tức tiền mặt 20%. | Vào những tháng lẻ công ty sẽ chia cổ tức tiền mặt 20%. | `neutral` |
| 26 | Trước đó, Tesla cũng đã chấp nhận Bitcoin là đồng tiền số có thể dùng để mua xe điện của hãng. | Ngoài Bitcoin thì Tesla cũng chấp nhận Doge là đồng tiền số có thể dùng để mua xe điện của hãng. | `neutral` |
| 27 | Hôm 18/3, Mark Zuckerberg có buổi phỏng vấn với nhà đầu tư mạo hiểm Josh Constine qua ứng dụng Clubhouse. | Mark Zuckerberg nhận duy nhất phỏng vấn với nhà đầu tư mạo hiểm Josh Constine. | `neutral` |
| 28 | Trong 157 trụ sở có 107 toà nhà của sở ngành, quận huyện và các đơn vị trực thuộc; 15 bệnh viện công và 35 trường học công. | Có 15 trường học công trong số 157 trụ sở. | `contradiction` |
| 29 | Huyện Lạng Giang ở phía đông bắc của tỉnh Bắc Giang, diện tích hơn 24.000 ha, gồm 21 xã và 2 thị trấn, dân số hơn 207.000. | Khu vực phía đông bắc có diện tích đất canh tác rộng lớn. | `neutral` |
| 30 | "Sau đó, hàng ngày tôi đều ghé qua thăm thầy. | Tôi thường ghé qua thăm thầy. | `contradiction` |
