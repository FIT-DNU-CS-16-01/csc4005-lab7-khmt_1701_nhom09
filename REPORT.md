# Báo cáo CSC4005 Lab 7 - Nén mô hình: KD + đánh đổi khi lượng tử hóa

## 1. Thông tin

- Thành viên nhóm:

| Họ tên | Mã sinh viên | Lớp | Phụ trách |
|---|---|---|---|
| Lưu Thanh Tùng | 1771040029 | KHMT-1701 | Hướng B (chưng cất tri thức) và tổng hợp báo cáo |
| Nguyễn Hoàng Anh | 1771040002 | KHMT-1701 | Hướng A (lượng tử hóa) |

- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom09.git
- Kỹ thuật đã thực hiện:
  - Hướng A: lượng tử hóa
  - Hướng B: chưng cất tri thức (bổ sung góc nhìn, không thay thế Hướng A)
- Link W&B project (Hướng B): https://wandb.ai/thanhtung-contact-official-/csc4005-lab7-compression
- Link W&B run được chọn: https://wandb.ai/thanhtung-contact-official-/csc4005-lab7-compression/runs/a6j6bg2h

## 2. Mô tả mô hình gốc

| Nội dung | Giá trị |
|---|---|
| Bài toán | Phân loại cảnh Smart Campus |
| Dataset | MIT Indoor Scenes 67, tập con 5 lớp |
| Số lớp | 5 (classroom, computerroom, library, corridor, office) |
| Số mẫu | 789 |
| Mô hình gốc | Vision Transformer (ViT-B/16) fine-tuned |
| Định dạng mô hình gốc | ONNX |
| File mô hình gốc | models/vit_smartcampus.onnx |
| Kích thước mô hình gốc | 327.72 MB |

## 3. Hướng A - Lượng tử hóa

### 3.1 Cấu hình và artefact

| Thông tin | Giá trị |
|---|---|
| Loại lượng tử hóa | Dynamic |
| Input model | models/vit_smartcampus.onnx |
| Output model | models/vit_smartcampus_dynamic_int8.onnx |
| Công cụ | onnxruntime.quantization.quantize_dynamic |
| Eval baseline | outputs/eval_baseline_onnx.json |
| Eval quantized | outputs/eval_quantized_onnx.json |
| Benchmark | outputs/benchmark_quantization.csv |
| Trade-off | outputs/tradeoff_table.csv, outputs/tradeoff_table.md |

### 3.2 Kết quả Hướng A

| Model | Accuracy | Macro-F1 | Mean latency @bs=1 (ms) | Throughput @bs=1 (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9835 | 0.9790 | 77.93 | 12.83 | 327.72 |
| Quantized INT8 | 0.9772 | 0.9689 | 46.64 | 21.44 | 83.24 |

Nhận xét nhanh Hướng A:

- Accuracy giảm 0.64%.
- Macro-F1 giảm 1.03%.
- Latency giảm 40.15%.
- Throughput tăng 67.09%.
- Kích thước mô hình giảm 74.60%.

## 4. Hướng B - Chưng cất tri thức (bổ sung)

### 4.1 Cấu hình và artefact

| Thông tin | Giá trị |
|---|---|
| Teacher checkpoint | checkpoints/teacher_vit_best_model.pt |
| Student model | MobileNetV2 |
| Không gian tìm kiếm KD | alpha in {0.2, 0.5, 0.8}; temperature in {1.0, 2.0, 4.0}; epochs = 50 |
| Batch size train | 16 |
| Kết quả grid | outputs/kd_grid_search_full.csv, outputs/kd_grid_search_full.json |
| Cấu hình được chọn | A50_T20_E50 (alpha=0.5, temperature=2.0, epochs=50) |
| Student checkpoint tốt nhất sau grid | outputs/A50_T20_E50/student_best.pt |
| Student checkpoint (thực thi Hướng B trước đó) | outputs/kd_mobilenet_student/student_best.pt |
| Student ONNX đã chọn | models/student_mobilenet_kd_A50_T20_E50.onnx |
| Eval student PyTorch (đã chọn) | outputs/eval_kd_student_A50_T20_E50.json |
| Eval student ONNX (đã chọn) | outputs/eval_kd_student_onnx_A50_T20_E50.json |
| Benchmark (đã chọn) | outputs/benchmark_kd_A50_T20_E50.csv |
| Trade-off (đã chọn) | outputs/tradeoff_table_kd_A50_T20_E50.csv, outputs/tradeoff_table_kd_A50_T20_E50.md |

### 4.2 Kết quả full-grid (9 cấu hình)

| Run | Alpha | Temperature | Epochs | Best val macro-F1 |
|---|---:|---:|---:|---:|
| A50_T20_E50 | 0.5 | 2.0 | 50 | 0.9521 |
| A80_T40_E50 | 0.8 | 4.0 | 50 | 0.9506 |
| A50_T40_E50 | 0.5 | 4.0 | 50 | 0.9431 |
| A80_T10_E50 | 0.8 | 1.0 | 50 | 0.9429 |
| A20_T40_E50 | 0.2 | 4.0 | 50 | 0.9411 |
| A80_T20_E50 | 0.8 | 2.0 | 50 | 0.9337 |
| A20_T20_E50 | 0.2 | 2.0 | 50 | 0.9264 |
| A50_T10_E50 | 0.5 | 1.0 | 50 | 0.9204 |
| A20_T10_E50 | 0.2 | 1.0 | 50 | 0.9033 |

Nhận xét từ grid search:

- Chọn cấu hình tốt nhất theo tiêu chí best val macro-F1: A50_T20_E50 (0.9521).
- Độ nhạy tham số lớn: chênh lệch giữa cấu hình tốt nhất và thấp nhất là 0.0488 macro-F1 (4.88 điểm phần trăm).
- Temperature cao hơn có xu hướng hữu ích trên bộ dữ liệu này: T=4.0 cho trung bình kết quả cao, nhưng điểm cao nhất toàn bộ lại nằm ở T=2.0.

### 4.3 Tại sao chọn A50_T20_E50? (cái được, cái mất)

Tham số được chọn:

- Alpha = 0.5 (cân bằng giữa hard label và soft label)
- Temperature = 2.0 (làm mềm phân bố teacher vừa đủ)
- Epochs = 50

Lý do chọn:

- Đây là cấu hình đạt best val macro-F1 cao nhất trong 9 cấu hình đã chạy (0.9521).
- Alpha=0.5 tránh 2 cực đoan:
  - Nếu alpha thấp (0.2), mô hình học theo teacher quá nhiều, dễ bị hút theo sai lệch teacher trên dữ liệu nhỏ.
  - Nếu alpha cao (0.8), mô hình nghiêng về hard label nhiều hơn, lợi ích KD có thể giảm.
- Temperature=2.0 tạo mức độ "làm mềm" vừa đủ để student học quan hệ giữa các lớp mà không làm loãng thông tin quá mức.

Cái được:

- Đạt chất lượng val macro-F1 cao nhất trong tập thực nghiệm.
- Vẫn giữ được lợi thế hệ thống của Hướng KD (student nhỏ, nhẹ, nhanh trên CPU).

Cái mất / đánh đổi:

- Chi phí huấn luyện tăng rõ ràng do phải sweep 9 cấu hình x 50 epochs.
- Chọn theo val macro-F1 tối ưu có nguy cơ overfit vào tập val; cần kiểm tra thêm trên tập test/production để xác nhận độ bền vững.
- Nếu ưu tiên tính ổn định hơn điểm tối đa, có thể cân nhắc A80_T40_E50 (0.9506) vì kết quả rất sát nút, nhưng quyết định hiện tại ưu tiên điểm cao nhất.

### 4.4 Kết quả Hướng B với cấu hình được chọn (A50_T20_E50)

| Model | Accuracy | Macro-F1 | Mean latency @bs=1 (ms) | Throughput @bs=1 (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9835 | 0.9790 | 194.56 | 5.14 | 327.72 |
| KD Student A50_T20_E50 (ONNX) | 0.9861 | 0.9819 | 5.33 | 187.66 | 8.48 |

Nhận xét nhanh Hướng B:

- Accuracy tăng 0.26%.
- Macro-F1 tăng 0.30%.
- Latency giảm 97.26%.
- Throughput tăng 3551.10%.
- Kích thước mô hình giảm 97.41%.

Lưu ý benchmark Hướng B:

- Benchmark Hướng B được chạy với warmup=5 và repeat=10 để đảm bảo hoàn tất ổn định trong môi trường hiện tại.
- Vì điều kiện benchmark khác Hướng A, cần ưu tiên so sánh nội bộ trong từng hướng và dùng bảng tổng hợp để kết luận theo bối cảnh triển khai.

## 5. Bảng tổng hợp hai hướng

| Hướng | Accuracy sau nén | Macro-F1 sau nén | Latency @bs=1 (ms) | Throughput @bs=1 (img/s) | Size sau nén (MB) |
|---|---:|---:|---:|---:|---:|
| A - Lượng tử hóa | 0.9772 | 0.9689 | 46.64 | 21.44 | 83.24 |
| B - KD (A50_T20_E50) | 0.9861 | 0.9819 | 5.33 | 187.66 | 8.48 |

Nhận xét tổng hợp:

- Hướng A dễ triển khai nhanh, không cần train lại, trade-off tốt và ổn định.
- Hướng B là góc nhìn bổ sung giá trị cao, cho mô hình gọn hơn và nhanh hơn đáng kể trong khi độ chính xác vẫn giữ tốt.
- Hai hướng bổ trợ cho nhau: Hướng A phù hợp khi cần nhanh; Hướng B phù hợp khi ưu tiên tối ưu mạnh cho edge/CPU.