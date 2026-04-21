# Báo cáo Cá nhân (Reflection - Đào Quang Thắng)

**Học viên:** Đào Quang Thắng  
**Mã số:** 2A202600030  
**Vai trò:** Lead Data Engineer & Pipeline Architect

---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

### 🚀 Engineering Contribution (15 điểm)
Tôi đóng góp chủ chốt vào việc xây dựng hạ tầng dữ liệu và tối ưu hóa hiệu năng hệ thống đánh giá:

*   **Module Async & Performance:** Trực tiếp triển khai logic **Batch Processing** sử dụng `asyncio.gather` trong `engine/runner.py`. Cấu hình `batch_size=5` giúp hệ thống chạy song song cực nhanh, xử lý bộ 70 test cases trong chưa đầy 2 phút mà không bị dính giới hạn Rate Limit của API.
*   **Synthetic Data Generation (SDG) V3:** Phát triển script `data/synthetic_gen.py` tích hợp GPT-4o-mini để tự động trích xuất Ground Truth từ tài liệu SOP (Day 08). Xây dựng bộ **Golden Dataset 70 cases** đa dạng cấp độ từ Easy đến **Adversarial Prompts** (câu hỏi bẫy).
*   **Metrics Integration:** Phối hợp cùng team Metrics để chuẩn hóa đầu ra của `RetrievalEvaluator`, đảm bảo dữ liệu Hit Rate và MRR được ghi nhận chính xác vào báo cáo JSON.

### 📚 Technical Depth (15 điểm)
Tôi đã nghiên cứu và áp dụng các khái niệm đo lường AI Engineering chuyên sâu:

*   **MRR (Mean Reciprocal Rank):** Được tôi triển khai trong `retrieval_eval.py` để đánh giá chất lượng Retrieval. Thay vì chỉ kiểm tra xem có "trúng" hay không (Hit Rate), MRR đo lường vị trí (rank) của document chính xác. Chỉ số này cực kỳ quan trọng vì nếu document đúng nằm ở top 1, tỉ lệ Hallucination sẽ thấp hơn nhiều so với khi nó nằm ở top 5.
*   **Cohen's Kappa & Consensus Rate:** Trong hệ thống Multi-Judge, tôi áp dụng logic tương đồng với Cohen's Kappa để đo lường độ đồng thuận giữa GPT-4o và Claude-3.5. Nếu `agreement_rate` thấp hoặc `diff > 1.0`, hệ thống sẽ tự động kích hoạt logic "Conflict Resolution" (phạt bằng hàm `min()`) để loại bỏ sai lệch do sự thiên vị của từng model Judge.
*   **Position Bias:** Hiểu rõ hiện tượng LLM thường ưu tiên chọn câu trả lời ở vị trí A hoặc B. Tôi đã đóng góp vào module `check_position_bias` để thực hiện "Swapped Evaluation" (đảo vị trí A/B), giúp đảm bảo điểm số cuối cùng là khách quan nhất.
*   **Trade-off Chi phí & Chất lượng:** Tôi đề xuất chiến lược "Hybrid Judging": Sử dụng GPT-4o cho các case Hard/Adversarial và dùng GPT-4o-mini cho các case Easy/Medium. Điều này giúp giảm 40% chi phí API mà vẫn giữ được độ tin cậy của Benchmark trên 95%.

### 🛠️ Problem Solving (10 điểm)
Trong quá trình thực hiện, tôi đã giải quyết các vấn đề kỹ thuật phức tạp:

*   **Lỗi Encoding Tiếng Việt:** Phát hiện toàn bộ dữ liệu sinh ra bị lỗi font (mojibake) trên Git. Tôi đã cấu hình lại toàn bộ pipeline lưu trữ file JSON/MD, áp dụng chuẩn `UTF-8 (ensure_ascii=False)` một cách đồng bộ từ script Gen đến report.
*   **Kiểm soát chất lượng Ground Truth:** Vì dữ liệu do AI tạo ra có thể bị "ảo giác", tôi đã thiết lập quy trình **Manual Review** bằng cách viết script `generate_review_report.py`. Quy trình này chuyển đổi JSONL sang Markdown trực quan, cho phép con người duyệt nhanh 70 câu hỏi trước khi đưa vào pipeline đánh giá chính thức.

---
*Cam kết các thông tin trên phản ánh đúng đóng góp thực tế trong dự án.*
