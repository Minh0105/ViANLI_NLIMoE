# Kiểm chứng prototype Lightweight MoE

Ngày: 2026-10-01 (Asia/Bangkok).

## Phạm vi

Bản thử nghiệm trong `configs/lightweight_moe_experimental.yaml`, chưa phải
baseline tái hiện bài báo và chưa có điểm benchmark ViANLI mới.

## Những thay đổi đã thực hiện

- Sanity check đánh giá chính tập train nhỏ; đạt 95% mới thành công, không
  đánh giá test và không thêm kết quả sanity vào sổ benchmark chung.
- Dynamic routing có lựa chọn straight-through sigmoid estimator: forward
  giữ mask cứng, backward lấy gradient xấp xỉ từ sigmoid. Mặc định hard cũ
  được giữ trong module; script huấn luyện phát hiện head thiếu gradient.
- Expert có đường dispatch sparse khi suy luận; huấn luyện vẫn dense.
- Bản thử nghiệm dùng expert hidden dimension 256 thay cho 1536.
- Sweep khôi phục cấu trúc dynamic nếu checkpoint có head dynamic, nạp strict
  rồi override sang static để đánh giá. Cấu hình phải khớp checkpoint.
- Accuracy và macro-F1 validation được lấy từ cùng epoch tốt nhất.
- Early stopping được triển khai; warmup MoE đọc từ cấu hình. Gradient
  accumulation khác 1 bị từ chối rõ ràng vì chưa triển khai.
- Tên các run thử nghiệm có timestamp để tránh ghi đè kết quả cũ.

## Kết quả kiểm chứng offline

Lệnh: `.venv/Scripts/python.exe -m unittest discover -s tests -v`

6/6 tests passed:

1. Head ngưỡng có gradient khác 0, tham số được cập nhật; MoE ghi nhớ được
   ít nhất 95% trên 32 vector giả lập sau 120 bước, loss cuối <25% loss đầu.
2. Straight-through giữ nguyên mask và trọng số forward so với hard routing
   khi cùng tham số.
3. Sparse expert dispatch cho kết quả khớp dense sau mask; với 6 mẫu chỉ chọn
   expert 0/1, mỗi expert nhận 3 mẫu và expert 2/3 không được gọi.
4. Các routing mode có trọng số hữu hạn, tổng mỗi hàng bằng 1 và fallback
   ít nhất một expert khi ngưỡng quá cao.
5. Sanity train/validation loader dùng đúng cùng 32 mẫu train; test không đổi.
6. Tích hợp với XLM-R khởi tạo ngẫu nhiên kích thước nhỏ: backward, checkpoint
   strict round-trip, parity dense/sparse và static override đều hoạt động.

Đã kiểm tra cú pháp các file Python trong src/scripts/tests.

| Đại lượng | MLP cũ | Prototype |
|---|---:|---:|
| Số tham số của 4 expert | 4,743,180 | 790,540 |
| Kích thước lớp ẩn expert | 1536 | 256 |

Giảm 83.33% tham số **phần expert**, không phải toàn mô hình. Không suy ra
tốc độ thực tế từ số tham số hoặc số expert được chọn.

## Giới hạn và việc cần làm tiếp

Cache Hugging Face mặc định tại máy hiện không có XLM-R hoặc ViANLI. Không
chạy pretrained XLM-R trên dữ liệu ViANLI trong đợt kiểm chứng này.
Các vector giả lập và encoder nhỏ chỉ kiểm tra cơ chế kỹ thuật, không kiểm
tra khả năng hiểu tiếng Việt hoặc robustness.

Cần chạy sanity thật, train/val/test đầy đủ, nhiều seed, và benchmark latency,
throughput, VRAM toàn mô hình. Sparse dispatch có overhead và có thể chậm
hơn dense trên GPU/batch nhỏ. Cần kiểm tra tỷ lệ truncation ở max_length=128;
khi so sánh kiến trúc, giữ preprocessing và ngân sách huấn luyện tương đương.

Nguồn đối chiếu: https://arxiv.org/html/2406.17716v3 (Section 5). Bài báo mô
tả mean pooling, complexity gate tuyến tính sigmoid, ngưỡng static + gamma*C,
expert FFN d→d và classification layer riêng. Module hiện có CLS mặc định,
MLP head dự đoán ngưỡng và expert xuất trực tiếp 3 logits. Không đánh dấu các
chi tiết này là đã tái hiện đúng. Bài báo nêu binary selection; cách truyền
gradient qua phép chọn cần được xác minh thêm từ mã tác giả trước khi freeze.
