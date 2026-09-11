# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu) 
> *GPT-4o ở temp thấp (0.0, 0.5) luôn chọn cùng chủ đề Sơn Đoòng, rất ổn định; temp cao hơn (1.0) mới đổi sang cà phê — cho thấy temp thấp = deterministic. GPT-4o-mini thì mỗi lần ra một chủ đề khác (xe ôm, phở, cà phê trứng, đường sắt) ngay cả ở temp 0.0, cho thấy mini đa dạng hơn nhưng kém nhất quán hơn. Latency và độ dài response cũng giảm khi temp tăng, vì model "quyết định" nhanh hơn thay vì luôn chọn token xác suất cao nhất.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Em chọn temperature = 0 vì khi temp = 0 trả về hang Sơn Đông; Khi em chọn temperature cao hơn (như 0.5, 1.0, 1.5) trả về chủ đề khác - điều này gây nguy hiểm vì có thể bịa thông tin

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng token: 10.000 $\times$ 3 $\times$ 350 = 10.5M token <br>
> GPT-4o cost: 10.5 $\times$ 10 = $105/ngay

> GPT-4o-moni: 10.5 $\times$ 0.6 = $6.3/ngay   

> GPT-4o đắt hơn 16.67 lần

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> System prompt thay đổi hoàn toàn phong cách phản hồi dù câu hỏi giống nhau. Với prompt "giáo viên tiểu học", model dùng ví dụ gần gũi ("cuốn sổ cái", "mắt xích trong sợi dây chuyền"), từ ngữ đơn giản, câu ngắn dễ hiểu cho trẻ em. Với prompt "chuyên gia tài chính", model sẽ dùng thuật ngữ chuyên ngành (distributed ledger, consensus mechanism, immutability), câu dài và phức tạp hơn. System prompt định hình persona, mức độ chuyên sâu, từ vựng và phong cách trả lời của model mà không cần thay đổi câu hỏi đầu vào.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Khoảng 25–35%. Tiktoken đếm ~150 token, ước lượng số từ / 0.75 ra ~111 token. Tokenizer BPE của OpenAI huấn luyện chủ yếu trên tiếng Anh, nên từ tiếng Anh thường là 1 token ("hello"), còn từ tiếng Việt có dấu ("tưởng", "khổng") bị tách thành 2–3 sub-token vì ít gặp trong tập huấn luyện.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming và non-streaming khác nhau ở cách xử lý dữ liệu.
> Streaming phù hợp khi cần phản hồi ngay lập tức — ví dụ chatbot cần hiện chữ từng từ để user không phải chờ, hệ thống phát hiện gian lận cần xử lý giao dịch ngay khi xảy ra, hay dữ liệu cảm biến IoT đến liên tục không có điểm dừng.
> Non-streaming phù hợp khi không cần kết quả tức thì — ví dụ tổng hợp báo cáo tài chính cuối tháng, phân tích dữ liệu lớn tìm xu hướng, hay khi phép tính phức tạp cần chạy xong hết mới có kết quả chính xác. Triển khai cũng đơn giản và rẻ hơn streaming.
> Nói ngắn gọn: cần nhanh, dữ liệu chảy liên tục → streaming. Không gấp, cần chính xác, dữ liệu có sẵn → non-streaming.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff lợi thế gì?* <br> 
> Mỗi lần retry chờ lâu gấp đôi (0.1s → 0.2s → 0.4s), giúp giãn dần lượng request gửi lên server, cho server thời gian hồi phục. <br>
> Hàng nghìn client retry delay cố định thì sao? <br>
> Tất cả cùng retry sau đúng 1 giây → tạo "thundering herd" — một đợt sóng request đồng loạt đập vào server, server lại quá tải, lại fail, lại retry cùng lúc — vòng lặp không bao giờ dứt. Exponential backoff (thường kèm thêm jitter ngẫu nhiên) phân tán các lần retry ra các thời điểm khác nhau, tránh hiệu ứng này.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Chọn persona: Trợ lý lập trình Python. <br>
> System prompt: "Bạn là trợ lý lập trình Python chuyên nghiệp. Trả lời bằng tiếng Việt, ngắn gọn, kèm code ví dụ. Không giải thích dài dòng trừ khi được hỏi thêm." <br>
> Vì sao yêu cầu "ngắn gọn, kèm code ví dụ"? <br>
> Lập trình viên cần code chạy được ngay, không cần đọc nhiều đoạn lý thuyết. Đồng thời giảm token output → giảm chi phí API và latency. <br>
> Vì sao chỉ định "trả lời bằng tiếng Việt"? <br>
> Không chỉ định thì model hay tự chuyển sang tiếng Anh khi gặp câu hỏi về code. Chỉ định rõ ngôn ngữ giúp output nhất quán cho người dùng Việt.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất?
> History chỉ giữ 3 lượt (6 message) — model quên hoàn toàn nội dung trước đó, nên cuộc hội thoại dài sẽ mất ngữ cảnh.
> Đề xuất cải thiện?
> Thêm bộ nhớ dài hạn bằng cách tóm tắt history cũ trước khi cắt. Trước khi history = history[-6:], gọi API tóm tắt các message bị cắt thành 1 đoạn ngắn, lưu vào một biến summary, rồi chèn summary đó vào system prompt mỗi lượt. Model vẫn giữ được bối cảnh tổng thể mà không tốn nhiều token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026