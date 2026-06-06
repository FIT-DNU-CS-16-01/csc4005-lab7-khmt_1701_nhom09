# CSC4005 Lab 7 Report – Compression: KD + Quantization Trade-offs

## 1. Thông tin

- Họ tên:
- Mã sinh viên:
- Lớp:
- Link GitHub repo:
- Kỹ thuật chọn: Quantization / Knowledge Distillation / Cả hai
- Link W&B nếu dùng KD:
- Link model nếu không commit trực tiếp:

## 2. Mô tả baseline model

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset |
| Số lớp | 5 |
| Baseline model | ... |
| Baseline format | PyTorch / ONNX |
| Baseline checkpoint/ONNX | ... |
| Baseline model size | ... MB |

## 3. Kỹ thuật nén đã chọn

### Nếu chọn Quantization

| Thông tin | Giá trị |
|---|---|
| Loại quantization | Dynamic / Static |
| Input model | ... |
| Output model | ... |
| Dạng dữ liệu sau nén | INT8 / khác |
| Công cụ | onnxruntime.quantization |

Mô tả ngắn:

```text
...
```

### Nếu chọn Knowledge Distillation

| Thông tin | Giá trị |
|---|---|
| Teacher model | ... |
| Student model | ... |
| alpha | ... |
| temperature | ... |
| epochs | ... |
| batch size | ... |
| optimizer | ... |

Công thức loss sử dụng:

```text
loss = alpha * CE(student_logits, labels) + (1 - alpha) * KD_loss(student_logits, teacher_logits, T)
```

## 4. Kết quả đánh giá

| Model | Accuracy | Macro-F1 | Model size (MB) |
|---|---:|---:|---:|
| Baseline | ... | ... | ... |
| Compressed | ... | ... | ... |

Nhận xét:

- Accuracy giảm bao nhiêu?
- Macro-F1 giảm bao nhiêu?
- Mức giảm này có chấp nhận được không? Vì sao?

## 5. Kết quả benchmark

| Model | Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | ... | ... | ... | ... |
| Compressed | 1 | ... | ... | ... | ... |
| Baseline | 4 | ... | ... | ... | ... |
| Compressed | 4 | ... | ... | ... | ... |
| Baseline | 8 | ... | ... | ... | ... |
| Compressed | 8 | ... | ... | ... | ... |

## 6. Bảng trade-off

| Model | Accuracy | Macro-F1 | Mean latency @bs=1 | Throughput @bs=1 | Size | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| Baseline | ... | ... | ... | ... | ... | ... |
| Compressed | ... | ... | ... | ... | ... | ... |

## 7. Phân tích

Trả lời:

1. Mô hình sau nén nhỏ hơn bao nhiêu phần trăm?
2. Latency giảm hay tăng?
3. Throughput thay đổi thế nào?
4. Accuracy/F1 giảm nhiều không?
5. Nếu triển khai trên CPU hoặc edge device, bạn có chọn compressed model không?
6. Nếu không chọn, lý do là gì?

## 8. Khi nào chọn KD, khi nào chọn Quantization?

Viết nhận xét ngắn:

- Khi nào quantization phù hợp?
- Khi nào KD phù hợp?
- Nếu được làm lại, bạn sẽ chọn kỹ thuật nào cho hệ thống Smart Campus?

## 9. Kết luận

Tóm tắt 5–8 dòng:

- Kỹ thuật nén đã dùng;
- Kết quả chính;
- Trade-off quan trọng nhất;
- Bài học rút ra.
