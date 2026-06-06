# CSC4005 Lab 7 Report – Compression: KD + Quantization Trade-offs

## 1. Thông tin

- Họ tên: Nguyễn Hoàng Anh
- Mã sinh viên: 1771040002
- Lớp: KHMT 1701
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom09.git
- Kỹ thuật chọn: Quantization
- Link W&B nếu dùng KD: 
- Link model nếu không commit trực tiếp: 

## 2. Mô tả baseline model

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset 5 lớp |
| Số lớp | 5 (classroom, computerroom, library, corridor, office) |
| Số mẫu | 789 |
| Baseline model | Vision Transformer (ViT-B/16) fine-tuned |
| Baseline format | ONNX (exported từ Lab 6) |
| Baseline checkpoint/ONNX | models/vit_smartcampus.onnx |
| Baseline model size | 327.72 MB |

## 3. Kỹ thuật nén đã chọn

### Quantization

| Thông tin | Giá trị |
|---|---|
| Loại quantization | Dynamic |
| Input model | models/vit_smartcampus.onnx |
| Output model | models/vit_smartcampus_dynamic_int8.onnx |
| Dạng dữ liệu sau nén | INT8 (QInt8) |
| Công cụ | onnxruntime.quantization.quantize_dynamic |

Mô tả ngắn:

```text
Dynamic quantization được áp dụng lên model ONNX baseline (ViT-B/16).
Phương pháp này chuyển trọng số từ FP32 sang INT8 tại thời điểm inference,
giúp giảm kích thước model mà không cần calibration data.
Trọng số được quantize, activation vẫn ở FP32 nhưng tính toán dùng kernel INT8 khi có thể.
```

## 4. Kết quả đánh giá

| Model | Accuracy | Macro-F1 | Model size (MB) |
|---|---:|---:|---:|
| Baseline (FP32) | 0.9835 | 0.9790 | 327.72 |
| Quantized (INT8) | 0.9772 | 0.9689 | 83.24 |

Nhận xét:

- Accuracy giảm 0.63 điểm phần trăm (98.35% → 97.72%).
- Macro-F1 giảm 1.01 điểm phần trăm (97.90% → 96.89%).
- Mức giảm này **hoàn toàn chấp nhận được** trong bài toán Smart Campus vì:
  - Accuracy vẫn trên 97%, đủ tin cậy cho phân loại cảnh.
  - Trade-off chưa đến 1% accuracy để đổi lấy 74.6% giảm kích thước model.

## 5. Kết quả benchmark

| Model | Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 77.93 | 81.64 | 12.83 | 327.72 |
| Quantized | 1 | 46.64 | 50.88 | 21.44 | 83.24 |
| Baseline | 4 | 324.77 | 342.50 | 12.32 | 327.72 |
| Quantized | 4 | 183.59 | 199.15 | 21.79 | 83.24 |
| Baseline | 8 | 676.94 | 742.35 | 11.82 | 327.72 |
| Quantized | 8 | 398.60 | 462.81 | 20.07 | 83.24 |

## 6. Bảng trade-off

| Model | Accuracy | Macro-F1 | Mean latency @bs=1 | Throughput @bs=1 | Size (MB) | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 0.9835 | 0.9790 | 77.93 ms | 12.83 img/s | 327.72 | Model gốc, accuracy cao nhất |
| Quantized INT8 | 0.9772 | 0.9689 | 46.64 ms | 21.44 img/s | 83.24 | Nhẹ hơn 74.6%, nhanh hơn 40%, accuracy giảm <1% |

## 7. Phân tích

1. **Mô hình sau nén nhỏ hơn bao nhiêu phần trăm?**
   - Giảm 74.6% (327.72 MB → 83.24 MB).

2. **Latency giảm hay tăng?**
   - Giảm 40.15% ở batch size 1 (77.93 ms → 46.64 ms).
   - Giảm 43.5% ở batch size 4.
   - Giảm 41.1% ở batch size 8.

3. **Throughput thay đổi thế nào?**
   - Tăng 67.1% ở batch size 1 (12.83 → 21.44 img/s).
   - Cải thiện đáng kể ở mọi batch size.

4. **Accuracy/F1 giảm nhiều không?**
   - Accuracy giảm 0.64%, Macro-F1 giảm 1.03%. Mức giảm rất nhỏ.

5. **Nếu triển khai trên CPU hoặc edge device, bạn có chọn compressed model không?**
   - **Có.** Model quantized INT8 là lựa chọn tốt nhất cho triển khai CPU:
     - Nhẹ hơn 4x → dễ deploy trên thiết bị hạn chế bộ nhớ.
     - Nhanh hơn 40% → đáp ứng yêu cầu real-time tốt hơn.
     - Accuracy gần như không đổi → chất lượng phân loại vẫn đáng tin cậy.

6. **Nếu không chọn, lý do là gì?**
   - Không áp dụng. Trong trường hợp này, trade-off hoàn toàn có lợi cho quantized model.

## 8. Khi nào chọn KD, khi nào chọn Quantization?

- **Quantization phù hợp khi:**
  - Đã có model train tốt (ONNX/PyTorch) và muốn giảm kích thước/tăng tốc nhanh.
  - Không muốn train lại.
  - Triển khai trên CPU, cần giảm memory footprint.
  - Chấp nhận kiểm tra lại accuracy sau nén (thường giảm rất ít với dynamic quantization).

- **KD phù hợp khi:**
  - Teacher model quá lớn, cần student architecture nhỏ hơn hẳn (ví dụ MobileNet).
  - Muốn model chạy trên edge device cực kỳ hạn chế tài nguyên.
  - Sẵn sàng train lại và có đủ dữ liệu.
  - Muốn throughput cải thiện nhiều hơn nữa (student nhỏ hơn cả quantized ViT).

- **Nếu được làm lại cho Smart Campus:**
  - Quantization là lựa chọn thực tế nhất: nhanh triển khai, kết quả tốt ngay, không cần train lại.
  - KD có giá trị nếu cần deploy trên camera edge với RAM rất hạn chế (<50MB).

## 9. Kết luận

- Đã áp dụng **ONNX Dynamic Quantization (FP32 → INT8)** cho ViT-B/16 baseline.
- Model giảm **74.6% kích thước** (327.72 → 83.24 MB).
- Latency giảm **~40%** trên CPU, throughput tăng **~67%**.
- Accuracy chỉ giảm **0.64%**, Macro-F1 giảm **1.03%** – mức giảm không đáng kể.
- **Trade-off quan trọng nhất:** đổi <1% accuracy lấy 4x nhỏ hơn và 1.7x nhanh hơn.
- **Bài học:** Dynamic quantization là kỹ thuật compression "low-hanging fruit" – dễ áp dụng, hiệu quả cao, rủi ro thấp. Nên luôn thử trước khi xem xét các phương pháp phức tạp hơn.
