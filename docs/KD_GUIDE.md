# Knowledge Distillation Guide

## 1. Ý tưởng

Knowledge Distillation dùng một mô hình lớn, gọi là teacher, để hướng dẫn một mô hình nhỏ hơn, gọi là student.

Trong lab này:

```text
Teacher: ViT đã fine-tune
Student: MobileNetV2 hoặc ResNet18
```

## 2. Loss gợi ý

```text
loss = alpha * CE(student_logits, labels)
     + (1 - alpha) * KL(student_logits/T, teacher_logits/T) * T^2
```

Trong đó:

- `T` là temperature;
- `alpha` điều chỉnh cân bằng giữa label thật và soft label từ teacher.

## 3. Sinh viên cần tự hoàn thiện

File `src/kd_train_student.py` có một số phần TODO:

```text
TODO: build dataloader
TODO: load teacher checkpoint
TODO: compute KD loss
TODO: save student checkpoint
TODO: log W&B
```
