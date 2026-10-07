# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Minh Thắng  **MSSV**: 2A202602706  **Ngày**: 2025
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 (suggested=256, dùng tier default=1024) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |
| LORA Rank | 16, Alpha 32 |

**Template có giữ khối `<think>` không?** Không — Model Qwen3.5-4B không tạo think block trong inference.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.5%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (3 dòng đầu):
```
<|im_start|>assistant
<think>
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Chỉ phần assistant response được tính loss — câu hỏi và system prompt không nằm trong supervised span.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3449.9 |
| (b) base + optimized prompt | **0.760** | 0.791 | **1.000** | 1043.0 |
| (c) LoRA fine-tune | **0.970** | 0.678 | 1.000 | 1543.3 |

**(b) có thật sự mạnh hơn (a) không?** Có — target tăng từ 0% lên 76%, format từ 0% lên 100%.

Bạn có sửa `OPTIMIZED_PROMPT` không? Không — dùng prompt mặc định từ lab.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable_params | LR | train loss | **target (NB5)** | VRAM GB |
|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6272 | **0.970** | 8.78 |
| `attn_only` | q,v | matched(283) | 32,456,704 | 1e-4 | 0.5384 | **0.970** | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | **3.86** |

---

**4.1 — `attn_only` HOÀ `correct` (0.970 = 0.970). Thứ tự theo target GIỐNG thứ tự theo train loss.**

Điều này nói rằng: với cùng ngân sách tham số (~32M), cả vị trí text-linear lẫn attention-only(q,v) đạt kết quả tương đương. **Rank (số lượng tham số) quan trọng hơn vị trí gắn adapter** — đó là kết luận quan trọng nhất từ lab này.

Tuy nhiên, `attn_only` có train loss THẤP hơn (0.538 vs 0.627) nhưng target BẰNG nhau. Điều này cho thấy train loss thấp không đảm bảo generalization tốt hơn.

---

**4.2 — `wrong_lr` dùng LR=1e-5 (scale của full-FT). Đường loss gần như phẳng: 2.163 → 1.570 sau 30 steps.**

Nếu chỉ nhìn loss mà không biết LR, ta sẽ kết luận sai: model "học được" vì loss giảm. Nhưng kết quả target=0.000 cho thấy model thực sự không học được gì — nó chỉ记住了 một vài pattern ngẫu nhiên. LR quá thấp cho LoRA khiến adapter không cập nhật đủ nhanh để thích ứng với task mới.

---

**4.3 — `qlora` tiết kiệm 56% VRAM (3.86GB vs 8.78GB), nhưng target giảm từ 0.970 xuống 0.940 (-3%).**

Số đo của tôi ủng hộ khuyến nghị "không dùng QLoRA cho Qwen3.5": quantization error làm giảm chất lượng adapter. Trade-off VRAM không đáng — chỉ tiết kiệm ~5GB nhưng mất 3% accuracy.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.210` · `regression Δ = -0.113` · `valid_trace_rate = 0.0`

**Diễn giải:**

Lab này cho kết quả FAILED, và đây là phát hiện có giá trị khoa học. Fine-tune thắng baseline (b) với +21% target accuracy, nhưng lại làm tổn hại general capability: regression score giảm 0.113 (vượt tolerance ±0.020).

Nguyên nhân: model được train THUẦN trên dữ liệu CSKH, không có "replay data" — 1-5% dữ liệu phổ thông để giữ knowledge gốc (theo deck §6.3). Điều này xảy ra vì:
1. Dataset quá nhỏ (250 mẫu)
2. Task CSKH quá narrow
3. Adapter "overfits" vào domain mới, overwrite knowledge cũ

**Bài học**: Fine-tuning không phải luôn tốt. Nếu không có chiến lược retain knowledge, model sẽ "quên" những gì đã biết.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt máy xay sinh tố... | 1.0 | ✓ | ✓ | ✅ FT thắng |
| 2 | Chào shop, mình đặt nồi chiên không dầu... | 1.0 | ✓ | ✓ | ✅ FT thắng |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt... | 0.75 | ✓ | thiếu sentiment | ❌ **FT thua** |
| 4 | Shop ơi, mình đặt nồi chiên không dầu... | 0.75 | ✓ | thiếu sentiment | ❌ **FT thua** |
| 5 | Shop ơi, mình đặt áo khoác gió... | 0.75 | ✓ | thiếu sentiment | ❌ **FT thua** |

**Mẫu chung ở các ca FT thua:** Tất cả đều thiếu trường `sentiment` trong JSON output. Model bị "lazy" — chỉ predict 3/4 fields đúng khi text quá ngắn hoặc sentiment không rõ ràng.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):**

Không nên deploy bản fine-tune này trong production. Mặc dù target accuracy tăng +21% so với optimized prompt, regression capability giảm đáng kể — đây là trade-off không chấp nhận được cho một hệ thống CSKH thực tế.

Đòn bẩy thật sự trong lab này là **chất lượng dữ liệu và chiến lược training**. Cụ thể:
1. **Mask design** quyết định model học đúng thứ — mask sai thì mọi số sau đều vô nghĩa
2. **Learning rate** phải được scale đúng cho LoRA — LR của full-FT sẽ thất bại hoàn toàn
3. **Vị trí gắn adapter** ít quan trọng hơn số lượng tham số trainable

Để cải thiện, cần thêm replay data (1-5% dữ liệu phổ thông) như deck §6.3 khuyến nghị, và có thể thử train lâu hơn với early stopping dựa trên regression score thay vì chỉ target loss.

**Ba điều tôi học được** (cụ thể, không generic):

1. **Train loss thấp không đồng nghĩa với performance tốt** — `attn_only` có loss=0.538 (thấp hơn `correct` 0.627) nhưng target bằng nhau. Model có thể overfit vào training data mà không generalize tốt hơn.

2. **QLoRA có quantization error đáng kể trên Qwen3.5** — Tiết kiệm 56% VRAM nhưng mất 3% accuracy. Với T4 16GB, không có lý do dùng QLoRA thay vì fp16 LoRA.

3. **Fine-tuning có thể làm hỏng model** — Regression giảm -11.3% cho thấy model "quên" knowledge cũ. Đây là catastrophiic forgetting, và giải pháp là thêm replay data trước khi train.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Thêm 5% replay data (dữ liệu phổ thông) vào training set để giảm catastrophic forgetting
2. Train với rank cao hơn (r=32 hoặc r=64) để tăng capacity của adapter
3. Thử dataset lớn hơn (500-1000 mẫu) để model có đủ data học generalization

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
