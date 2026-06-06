# Lab 7 Guide – Compression: KD + Quantization Trade-offs

## 1. Tinh thần của lab

Lab này không chỉ yêu cầu sinh viên “làm cho model nhỏ hơn”. Mục tiêu là trả lời:

```text
Nén mô hình có đáng không?
```

Muốn trả lời, cần đo đủ:

```text
accuracy
macro-F1
latency
throughput
model size
```

## 2. Hai hướng làm bài

Sinh viên chọn ít nhất một trong hai hướng:

### Option A – Quantization

Phù hợp nếu đã có file ONNX từ Lab 6.

Ưu điểm:

- nhanh triển khai;
- ít phải train lại;
- dễ đo model size và latency.

Nhược điểm:

- có thể làm giảm accuracy;
- không phải model nào cũng nhanh hơn rõ rệt.

### Option B – Knowledge Distillation

Phù hợp nếu muốn train một student model nhỏ hơn.

Ưu điểm:

- student model có thể nhỏ và nhanh hơn nhiều;
- phù hợp khi muốn triển khai trên thiết bị hạn chế.

Nhược điểm:

- cần train lại;
- cần hiểu loss KD;
- tốn thời gian hơn quantization.

## 3. Artefact cần có

```text
eval_before.json
eval_after.json
benchmark_before.csv
benchmark_after.csv
tradeoff_table.csv
tradeoff_table.md
báo cáo phân tích
```
