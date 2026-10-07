# Lab 21 — Evaluation Report

**Họ tên**: Bùi Gia Chính  **MSSV**: 2A202602693  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 (16 GB, ~14.6 GB khả dụng)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định của lab, không đổi) |
| Train / val | 225 / 25 (seed 42, `data/split/`) |
| `max_length` | 1024 (tier T4 mặc định) — p95 đo được chỉ là 98 token *(results/token_stats.json)*. **Lý do giữ 1024 thay vì 256 theo gợi ý p95**: corpus triage có câu trả lời ngắn (JSON 4 khoá) nên p95 tự nhiên thấp; giữ nguyên `max_length=1024` của tier để không phải tinh chỉnh lại batch/VRAM đã đo cho T4, và vì 1024 vẫn an toàn VRAM trên T4 (không đánh đổi gì để giữ dư địa cho câu dài hơn nếu đổi dataset sau). |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs (mặc định `EPOCHS_DEFAULT`) → 30 optimizer steps (225 mẫu train, effective batch 16) — cả 4 run (`correct` + 3 đối chứng NB4) đều dùng đúng 30 step |

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json: `"verdict": "reasoning preserved — safe to train on traces"`)*. Qwen3.5 không xoá khối `<think>...</think>` khi render qua `apply_chat_template`, nên dữ liệu suy luận (nếu có) truyền được tới loss.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss (`results/mask_proof.json → supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần KHÔNG được tính loss (hệ thống + câu hỏi của khách, `masked_preview`):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3139.8 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 960.9 |
| (c) LoRA fine-tune | 0.970 | 0.456 | 1.000 | 1347.3 |

*(nguồn: `results/baselines_frozen.json` cho (a)/(b), `results/verdict.json` cho (c); `optimized_prompt_sha=719e74d3b6232053` khớp SHA của `OPTIMIZED_PROMPT` gốc trong `labkit/generate.py` — không sửa)*

**(b) có thật sự mạnh hơn (a) không?** **Có, rất rõ.** Prompt naive ("Phân loại ticket sau.") khiến base model không biết schema JSON nên trả lời bằng văn xuôi tiếng Việt — target=0.000, format=0.000, và latency cao gấp 3.3x (3139.8ms) vì model sinh ra câu trả lời dài dòng không có điểm dừng EOS rõ ràng. Prompt tối ưu (có schema + enum + ví dụ few-shot) đưa target lên 0.765 và format lên 1.000, đồng thời latency giảm xuống 960.9ms vì model dừng đúng chỗ. Tôi **không sửa** `OPTIMIZED_PROMPT` — giữ nguyên bản gốc của lab vì nó đã đủ mạnh để tạo ra một đối thủ thật (yêu cầu rubric 3.1), và sửa yếu đi để fine-tune "thắng" dễ hơn là gian lận theo đúng cảnh báo của lab.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6271 | **0.970** | 393.5 | 8.78 |
| `attn_only` | q,v (matched) | 283 | 32,456,704 | 1e-4 | 0.5381 | **0.970** | 259.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 380.6 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 (4-bit) | 1e-4 | 0.7058 | **0.940** | 453.2 | 3.86 |

*(nguồn: `results/runs.csv` cho train loss/VRAM/giây, `results/autopsy.json` cho cột target/format/latency đo lại trên NB5 §4. `attn_only` khớp ngân sách tham số của `correct` với sai lệch 32,456,704 vs 32,464,896 = 0.025% < 5%, qua được `make verify`.)*

> format của cả 4 run đều = 1.000 trừ `wrong_lr` = 0.000.

Trả lời ba câu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Trên target, `attn_only` và `correct` **hoà tuyệt đối** (0.970 = 0.970, format 1.0 = 1.0) — không có bất kỳ chênh lệch nào dù khác hẳn vị trí gắn adapter (chỉ q,v vs toàn bộ 12 lớp linear). Thứ tự này **không khớp** với train loss: `attn_only` có loss thấp hơn (0.5381 so với 0.6271 của `correct`), và còn train nhanh hơn (259s vs 393.5s) cũng như suy luận nhanh hơn (859.9ms vs 1347.3ms theo `autopsy.json`). Nếu xếp hạng bằng loss như "Lỗi #3" mô tả, tôi sẽ kết luận sai rằng `attn_only` là lựa chọn *tốt hơn* `correct`, trong khi thực ra cả hai cùng đạt trần điểm trên một bài toán 4-trường JSON đơn giản — ở đây *rank* (r=283 bù đủ ngân sách tham số) đã thay được *vị trí gắn adapter* hoàn toàn, mâu thuẫn với giả thuyết của deck §11.2 rằng full-placement phải thắng attention-only ở cùng ngân sách. Kết luận hợp lý nhất là: với corpus nhỏ (225 mẫu) và task đơn giản (phân loại 4 trường có từ vựng đóng), bài toán "dễ" đến mức cả hai cấu hình đều bão hoà ở target=0.97 — đây chính là phát hiện đáng giá nhất của lab này với tôi (đúng như rubric 2.4 gợi ý), không phải việc một cấu hình thắng cấu hình khác.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
`wrong_lr` dùng LR=1e-5 (thang full-FT) thay vì 1e-4 (thang LoRA 10x). Loss giảm rất chậm và không đều: 2.163 → 2.066 → 1.606 → 1.326 → 1.141 → 1.119 (final_loss 1.5702 theo mean của toàn run), so với `correct` giảm dốc và mượt từ 2.163 xuống 0.0257 ở bước cuối (final_loss 0.6271). Nếu chỉ nhìn đường loss mà không biết LR, tôi dễ kết luận sai rằng "model cần train lâu hơn" hoặc "dữ liệu khó học" — cả hai đều sai; vấn đề thực là bước cập nhật trọng số quá nhỏ để 30 step có thể hội tụ với một adapter LoRA (deck §11.3: LoRA cần LR lớn hơn full-FT ~10x vì chỉ một phần nhỏ tham số được cập nhật). Hệ quả đo được là target=0.000 và format=0.000 tuyệt đối — model hầu như không học được format JSON trong 30 step với LR này.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` dùng 3.86GB VRAM đỉnh so với 8.78GB của `correct` — tiết kiệm 4.92GB (≈56%). Giá phải trả: train chậm hơn 60s (453.2s vs 393.5s, +15%, do chi phí dequant khi forward/backward qua trọng số 4-bit), final_loss cao hơn (0.7058 vs 0.6271), target thấp hơn nhẹ (0.940 vs 0.970, −0.03), và latency suy luận cao hơn 27% (1716.3ms vs 1347.3ms). Số đo của tôi **ủng hộ một phần** khuyến nghị của vendor: trên Qwen3.5-4B với T4 có sẵn 14.6GB, 8.78GB của bf16-LoRA đã vừa thoải mái, nên đánh đổi tốc độ+độ chính xác để tiết kiệm VRAM không cần thiết là phi lý — khuyến nghị "không QLoRA" đúng *trong tình huống này*. Nhưng tôi sẽ không kết luận QLoRA luôn tệ: nếu VRAM mới là ràng buộc thật (model lớn hơn hoặc GPU nhỏ hơn), mức giảm target chỉ 0.03 có thể là cái giá hợp lý để đổi lấy việc *chạy được* thay vì OOM.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.336` · `valid_trace_rate = 0.00`

Diễn giải: Bản fine-tune `correct` **thắng rõ** baseline (b) trên target (0.970 vs 0.765, Δ=+0.205) — điều kiện thứ nhất của cổng được thoả. Nhưng nó đánh đổi bằng một cú sụt **regression** rất nặng, từ 0.791 (cả (a) và (b)) xuống 0.456, Δ=−0.336 — vượt xa ngưỡng chịu đựng ±0.020 của `regression_gate()`, nên verdict là **FAILED** dù target thắng đậm. Nguyên nhân kỹ thuật gần như chắc chắn là **catastrophic forgetting**: 225 mẫu train toàn bộ là ticket CSKH → JSON, không có một dòng dữ liệu "phổ thông" nào được trộn vào, nên 30 step cập nhật 32.46M tham số (toàn bộ 12 lớp linear của decoder) đã kéo model lệch khỏi khả năng trả lời 15 câu hỏi kiến thức/chỉ dẫn chung trong tập regression — đúng như deck §6.3 cảnh báo và gợi ý khắc phục bằng cách trộn 1–5% dữ liệu phổ thông vào tập train. `valid_trace_rate=0.00` là kỳ vọng, không phải lỗi: corpus triage không có khối `<think>` trong câu trả lời (NB1 đã cảnh báo `MASK_MODE` khác `assistant-only` sẽ vô tác dụng trên corpus này), nên không có "reasoning trace" nào để đo. Nói cách khác: pipeline đúng, phép so sánh công bằng, nhưng *lựa chọn huấn luyện cụ thể này* (full-linear, không replay data, 1 epoch×2 trên corpus hẹp) sẽ không nên deploy nguyên trạng — cần thêm dữ liệu replay trước khi fine-tune này được coi là an toàn để thay thế prompt (b).

---

## 6. Định tính — bắt buộc có cả ca THUA

> Lưu ý: `notebooks/05_evaluate_and_verdict.py` chỉ log dự đoán của fine-tune (`preds_ft`)
> vào `results/qualitative.json`, không log từng dự đoán của baseline (b) — NB5 chỉ giữ
> điểm tổng hợp của (b). Vì vậy bảng dưới so **nhãn đúng** với **(c) fine-tune**, đúng với
> những gì `results/qualitative.json` thực chứa, thay vì bịa thêm cột (b) không có nguồn.

| # | Ticket (rút gọn) | Nhãn đúng (intent/urgency/...) | (c) fine-tune dự đoán | ft_score | Nhận xét |
|---|---|---|---|---|---|
| i=3 | "...bình giữ nhiệt...Chưa thấy tiền." | intent=hoan_tien, **urgency=thap**, sentiment=tich_cuc | intent=hoan_tien, **urgency=trung_binh**, sentiment=tich_cuc | 0.75 | ❌ **FT thua** — sai `urgency` |
| i=5 | "...nồi chiên không dầu...Thiếu phụ kiện." | intent=san_pham_loi, **urgency=thap**, sentiment=trung_tinh | intent=san_pham_loi, **urgency=trung_binh**, sentiment=trung_tinh | 0.75 | ❌ **FT thua** — sai `urgency` |
| i=12 | "...áo khoác gió...Bị lỗi." | intent=san_pham_loi, **urgency=thap**, sentiment=tich_cuc | intent=san_pham_loi, **urgency=trung_binh**, sentiment=tich_cuc | 0.75 | ❌ **FT thua** — sai `urgency` |
| i=47 | "...ốp lưng điện thoại...Shipper không gọi." | intent=van_chuyen, urgency=thap, sentiment=tich_cuc | khớp 4/4 | 1.00 | ✅ FT thắng |
| i=48 | "...ốp lưng điện thoại...Giá bao nhiêu." | intent=hoi_thong_tin, urgency=trung_binh, sentiment=trung_tinh | khớp 4/4 | 1.00 | ✅ FT thắng |
| i=49 | "...ốp lưng điện thoại...Sai màu." | intent=san_pham_loi, urgency=trung_binh, sentiment=trung_tinh | khớp 4/4 | 1.00 | ✅ FT thắng |

*(nguồn: `results/qualitative.json` cho `ft_score`/dự đoán, đối chiếu nhãn đúng với `data/eval_target.jsonl` theo đúng chỉ số `i`)*

**Có mẫu chung nào ở các ca FT thua không? Có — rất rõ.** Cả 3 ca thua đều sai **đúng một field, luôn là `urgency`**, và luôn theo **cùng một hướng**: nhãn đúng là `thap` (thấp) nhưng model luôn đoán `trung_binh` (trung bình). Cả 3 ticket đều có giọng điệu khá bình thản ("Khi nào tiện", không có từ ngữ thể hiện gấp gáp), nhưng cũng không có cụm "gấp/khẩn/sớm nhé" như các ticket `urgency=cao`. Giả thuyết của tôi: với chỉ 225 mẫu train, model học được ranh giới `cao` khá rõ (có từ khóa tường minh như "gấp/khẩn") nhưng ranh giới giữa `thap` và `trung_binh` mờ hơn khi ticket chỉ có "Khi nào tiện" — một cụm trung tính. Đáng chú ý: đây **không** phải lỗi do mất cân bằng lớp — tôi kiểm tra `data/train_seed.jsonl` thì `thap` (97 mẫu) thực ra là nhãn **phổ biến nhất**, nhiều hơn cả `trung_binh` (74 mẫu), nên model lệch về `trung_binh` không thể giải thích bằng "đoán theo lớp đa số". Nhiều khả năng hơn là ranh giới ngữ nghĩa thật giữa hai nhãn này mờ trong chính cách gán nhãn gốc của corpus (nhãn phụ thuộc vào ngữ cảnh khó đoán từ một câu ngắn), và 225 mẫu không đủ để model học được ranh giới đó một cách ổn định. Đây là lỗi dữ liệu/quy mô corpus, không phải lỗi pipeline hay mask.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không nên deploy** bản fine-tune `correct` ở trạng thái hiện tại. Nó thắng rất rõ baseline (b) trên đúng task được huấn luyện (target 0.970 vs 0.765, format hoàn hảo, latency nhanh hơn 1347ms vs mốc chờ đợi của task phức tạp hơn), nhưng đánh đổi bằng việc mất gần 34 điểm phần trăm năng lực tổng quát (regression 0.791→0.456) — vượt xa ngưỡng ±0.02 cho phép. Một hệ thống CSKH thật sự vẫn cần trả lời được câu hỏi ngoài kịch bản triage, nên cổng hồi quy FAILED ở đây phản ánh đúng một rủi ro deploy thật, không phải một con số kỹ thuật vô hại. Đòn bẩy thật sự trong lab này, theo những gì tôi đo được, **không phải** là vị trí gắn adapter — `attn_only` (chỉ q,v, r=283 matched) đạt đúng target=0.970 như `correct` (toàn bộ 12 lớp linear, r=16), hoà tuyệt đối trên cả format và gần như không khác biệt gì về chất lượng, trong khi train nhanh hơn 34% và suy luận nhanh hơn 36%. Đòn bẩy rõ ràng nhất là **learning rate đúng thang** (`wrong_lr` ở LR=1e-5 thất bại hoàn toàn, target=0.000) và **chất lượng/đa dạng dữ liệu train** (thiếu replay data gây catastrophic forgetting; ranh giới nhãn `thap` vs `trung_binh` không đủ rõ với 225 mẫu). Loss mask (mục 1-2) đúng là điều kiện cần — không có nó thì mọi số liệu sau đều vô nghĩa — nhưng trên corpus này nó không phải điều *phân biệt* giữa các run, vì cả 4 run NB3/NB4 đều dùng đúng mask đã chứng minh ở NB1.

**Ba điều tôi học được** (cụ thể, không generic):
1. "So sánh công bằng" khó hơn tôi nghĩ: so `q,v @ r=16` với `all-linear @ r=16` sẽ là so *ngân sách tham số*, không phải so *vị trí* — phải dùng `matched_rank()` để ép cùng ngân sách (32,456,704 vs 32,464,896, lệch 0.025%) mới đo được đúng câu hỏi "vị trí có quan trọng không". Khi làm đúng, câu trả lời trên corpus này hoá ra là "không nhiều" — ngược với trực giác ban đầu của tôi rằng full-placement chắc phải thắng.
2. Một adapter có loss huấn luyện thấp hơn (`attn_only` 0.5381 vs `correct` 0.6271) không đồng nghĩa với target-score cao hơn — ở đây cả hai hoà ở 0.970. Nếu tôi chỉ nhìn `runs.csv` (NB4) mà không chạy NB5 để chấm lại trên target thật, tôi đã xếp hạng sai thứ tự các run chỉ vì tin vào loss như một proxy.
3. Một fine-tune thắng rõ ràng trên chính task được train (target Δ=+0.205) vẫn có thể là một quyết định xấu để deploy, nếu nó kéo tụt năng lực khác đủ mạnh (regression Δ=−0.336). "PASS/FAIL" của cổng hồi quy không phải là thắng/thua đơn giản — tôi phải nhìn cả hai con số cùng lúc, không chỉ con số mà fine-tune thắng.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 5% dữ liệu phổ thông (Q&A chung, không liên quan triage) vào tập train để kiểm tra giả thuyết regression ở Mục 5, rồi chạy lại `correct` để xem cổng hồi quy có PASS không — đây là thực nghiệm trực tiếp nhất để xác nhận nguyên nhân catastrophic forgetting mà report này mới suy luận từ số liệu, chưa kiểm chứng bằng một run đối chứng thật.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap** (+3đ). `results/merge_check.json`: before_merge=0.97, after_merge=0.97, delta=0.0 (ngưỡng cho phép 0.01) — merge không làm tụt điểm. Hoán đổi 2 adapter (`correct`, `attn_only`) trên cùng một base model đang nạp trong VRAM, cả hai cho dự đoán đúng trên cùng một ticket thử (`{"intent": "van_chuyen", "urgency": "thap", "product": "tai nghe bluetooth", ...}`). Lưu ý kỹ thuật: `PeftModel.load_adapter()` lần thứ hai trên một model đã `device_map="auto"` kích hoạt lại `dispatch_model` của accelerate và yêu cầu `offload_folder` dù VRAM còn dư — phải truyền tham số này mới qua được.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
