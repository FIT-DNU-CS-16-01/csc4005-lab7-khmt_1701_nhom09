# Trade-off Analysis Guide

## 1. Bảng tối thiểu cần có

| Model | Accuracy | Macro-F1 | Latency | Throughput | Size |
|---|---:|---:|---:|---:|---:|
| Baseline | ... | ... | ... | ... | ... |
| Compressed | ... | ... | ... | ... | ... |

## 2. Câu hỏi cần trả lời

1. Model nhỏ hơn bao nhiêu phần trăm?
2. Latency giảm bao nhiêu phần trăm?
3. Accuracy giảm bao nhiêu?
4. Throughput tăng hay giảm?
5. Trade-off này có chấp nhận được không?

## 3. Không nên kết luận kiểu

```text
Model sau nén tốt hơn vì nhẹ hơn.
```

Cần nói rõ:

```text
Model sau nén nhẹ hơn X%, nhanh hơn Y%, nhưng macro-F1 giảm Z điểm.
```
