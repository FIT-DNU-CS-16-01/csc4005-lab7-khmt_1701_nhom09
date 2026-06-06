# Quantization Guide

## 1. Ý tưởng

Quantization giảm độ chính xác số học của trọng số hoặc activation, ví dụ từ FP32 xuống INT8. Mục tiêu là giảm kích thước model và tăng tốc inference.

## 2. Dynamic quantization

Trong scaffold này, cách đơn giản nhất là ONNX dynamic quantization:

```bash
python -m src.quantize_onnx \
  --input_onnx models/vit_smartcampus.onnx \
  --output_onnx models/vit_smartcampus_dynamic_int8.onnx
```

## 3. Điều cần kiểm tra

Sau quantization, sinh viên phải đo lại:

```text
accuracy
macro-F1
latency
throughput
model size
```

Không được kết luận chỉ dựa vào file nhỏ hơn.
