# Báo cáo Cá nhân (Reflection)
**Họ và tên:** Phạm Hoàng Kim Liên
**Vai trò:** LLM Consensus & Judging System
**Nhóm:** 1

## 1. Engineering Contribution
- Xây dựng module `llm_judge.py` xử lý hai mô hình giám khảo khác nhau (GPT-4o và Claude 3.5).
- Phát triển logic tính `Agreement Rate` (Độ đồng thuận chéo) và quy tắc phạt chênh lệch điểm (nếu điểm lệch lớn hơn 1 điểm). Chuyển logic điểm số này thành API đánh giá trung bình.

## 2. Technical Depth
- **Cohen's Kappa & Consensus Rate:** Ứng dụng bản sao logic Cohen's Kappa để đánh giá tính cực đoan của một mô hình Language Model trong việc đánh giá LLM khác (LLM-as-a-judge). 
- **Cost Efficiency:** Giải thích tại sao việc dùng nhiều model giá rẻ (như GPT-4o-mini và Claude 3.5 Haiku) ở lớp filter đầu rất hiệu quả về kinh tế so với việc nhét mọi thứ qua GPT-4o mà không suy tính.

## 3. Problem Solving
- **Vấn đề mô hình giám khảo bất đồng quan điểm:** Khi một Judge khắt khe cho 3 điểm, còn người khác châm trước cho 5 điểm.
- **Cách giải quyết:** Áp dụng ràng buộc khoảng cách tin cậy `diff > 1`. Nếu sự lệch phân quá lớn tôi không dùng trung bình cộng mà sử dụng hàm `min()` để lấy điểm thấp nhất phạt con Agent, đồng thời ghi nhận Agreement Rate là `0.0`. Điều này chặn triệt để hiện tượng gian lận điểm từ một Judge ảo bảo vệ Agent V2 một cách mù quáng.
17: 
18: ## 4. Bước 13: Phân tích nguyên nhân (Root Cause Analysis)
19: 
20: Dựa trên kết quả Regression Testing giữa V1 (Base) và V2 (Optimized), hệ thống ghi nhận sự cải thiện vượt bậc về mọi mặt (Judge Score từ 2.18 lên 4.07).
21: 
22: ### Tại sao V2 tốt hơn?
23: - **Sự cải tiến của Retrieval (Hit Rate +0.78):** V1 có Hit Rate cực thấp (0.06) vì hệ thống cũ chỉ lấy chunk ngẫu nhiên, khiến AI không có dữ liệu để trả lời. V2 sử dụng Hybrid RRF (Dense + Sparse) kết hợp Reranking giúp Hit Rate tăng vọt lên 0.84, cung cấp Context "sạch" và chính xác cho LLM.
24: - **Giảm thiểu Hallucination (-0.69):** Khi Context đầu vào chuẩn xác, hiện tượng AI "bịa" thông tin (Hallucination) giảm từ 0.90 xuống còn 0.21. Đây là yếu tố then chốt giúp Pass Rate tăng từ 4.29% lên 70.0%.
25: 
26: ### Tốt ở đâu?
27: - **Độ tin cậy của Judge (Agreement Rate 0.94):** Hai mô hình giám khảo (GPT và Claude) có độ đồng thuận rất cao (94%) khi đánh giá V2, chứng tỏ câu trả lời của V2 không chỉ đúng về thông tin mà còn tốt về giọng văn (Tone) và tính công bằng (Fairness).
28: - **Hiệu năng và Chi phí:** Mặc dù logic phức tạp hơn, nhưng nhờ Reranking lọc bớt noise, lượng token sử dụng cho mỗi request giảm nhẹ, giúp chi phí vận hành V2 thấp hơn V1 (~$0.037 so với $0.039 cho 70 cases).
29: 
30: ### Rủi ro còn lại
31: - **Hallucination Rate (0.21):** Vẫn còn 21% trường hợp bị gắn cờ ảo tưởng, chủ yếu rơi vào các câu hỏi Multi-hop (truy vấn đa tầng). Điều này cho thấy hệ thống cần cải thiện khả năng liên kết dữ liệu giữa các văn bản khác nhau.
32: - **Agreement Rate không đạt 100%:** Ở một số case biên (Edge cases), các Judge vẫn có sự lệch phân nhẹ (lệch ~1 điểm), đòi hỏi phải tinh chỉnh Prompt của Judge để đồng nhất tiêu chí chấm điểm hơn nữa.
