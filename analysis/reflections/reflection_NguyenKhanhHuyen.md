# Báo cáo Cá nhân (Reflection)

**Họ và tên:** Nguyễn Khánh Huyền - 2A202600171  
**Vai trò:** Evaluation & Benchmarking  
**Phạm vi công việc chính:** `engine/retrieval_eval.py`, `engine/llm_judge.py`, `main.py`

## 1. Engineering Contribution
Phần em phụ trách tập trung vào 3 file chính của pipeline đánh giá. Trong `engine/retrieval_eval.py`, em hoàn thiện các metric retrieval như Hit Rate, Retrieval Accuracy, MRR và NDCG để hệ thống có thể đo được chất lượng truy xuất tài liệu trước khi đánh giá câu trả lời. Đây là phần quan trọng vì nếu retrieval sai thì answer quality thường cũng giảm theo.

Trong `engine/llm_judge.py`, em xây dựng phần chấm điểm tự động bằng LLM theo hướng multi-judge. Hệ thống dùng 2 judge độc lập để chấm theo các tiêu chí như accuracy, tone, fairness, consistency, đồng thời theo dõi thêm hallucination và bias. Em cũng thêm logic tính agreement rate và xử lý xung đột khi 2 judge chênh điểm quá lớn, để kết quả đánh giá ổn định hơn so với chỉ dùng một judge đơn lẻ.

Trong `main.py`, em phụ trách orchestration cho benchmark V1 vs V2: load dataset, chạy benchmark cho từng version, tổng hợp summary, so sánh regression và xuất các file report như `summary.json`, `benchmark_results.json`, `judge_audit.jsonl`. Em cũng thêm release gate để hệ thống có thể đưa ra quyết định pass hay block dựa trên ngưỡng chất lượng đã đặt.

## 2. Technical Depth
Qua phần việc của mình, em hiểu rõ hơn các khái niệm đánh giá cốt lõi. MRR là chỉ số đo vị trí của tài liệu đúng đầu tiên trong danh sách retrieve, nên rất hữu ích để biết hệ thống có đưa đúng tài liệu lên sớm hay không. Nếu tài liệu đúng đứng đầu thì MRR cao, còn nếu đúng nhưng nằm sâu phía dưới thì MRR thấp dù Hit Rate vẫn có thể bằng 1.

Về multi-judge, hệ thống hiện đang dùng agreement rate để đo mức độ đồng thuận giữa hai judge. So với agreement rate, Cohen’s Kappa là một thước đo chặt hơn vì có tính đến xác suất đồng thuận do ngẫu nhiên. Em hiểu rằng nếu muốn mở rộng hệ thống theo hướng nghiên cứu hoặc production nghiêm túc hơn, Cohen’s Kappa sẽ phù hợp hơn cho việc đánh giá độ tin cậy của judge.

Em cũng hiểu Position Bias là hiện tượng judge bị ảnh hưởng bởi thứ tự trình bày phương án hoặc câu trả lời. Vì vậy trong `engine/llm_judge.py` em có phần kiểm tra position bias để hỗ trợ phát hiện trường hợp model chấm lệch chỉ vì đổi thứ tự input. Ngoài ra, khi làm multi-judge em cũng thấy rõ trade-off giữa chi phí và chất lượng: dùng 2 model giúp tăng độ tin cậy, phát hiện hallucination tốt hơn, nhưng đồng thời tăng token usage và cost cho mỗi benchmark run.

## 3. Problem Solving
Khó khăn lớn nhất em gặp là làm sao để pipeline đánh giá vừa chạy được end-to-end, vừa phản ánh đúng chất lượng của từng version agent. Trong quá trình làm, có lúc benchmark chạy được nhưng report chưa đúng format để checker đọc, hoặc summary chưa chứa đủ key cần thiết. Em xử lý bằng cách rà lại luồng dữ liệu từ lúc chạy benchmark đến lúc ghi report để đảm bảo file đầu ra đồng nhất và có thể dùng lại cho cả kiểm tra kỹ thuật lẫn chấm lab.

Một vấn đề khác là kết quả judge giữa hai model có thể chênh lệch khá mạnh ở các case gần đúng hoặc có thêm chi tiết ngoài expected answer. Để giải quyết, em không lấy trực tiếp một judge làm kết quả cuối, mà thêm conflict resolution theo hướng bảo thủ: nếu lệch nhỏ thì lấy trung bình, nếu lệch lớn thì ưu tiên điểm thấp hơn. Cách này giúp pipeline tránh bị quá lạc quan với những câu trả lời nghe hợp lý nhưng chưa thật sự bám ground truth.

Cuối cùng, em cũng phải xử lý bài toán liên kết giữa retrieval quality và answer quality. Trong thực tế có case retrieval đúng nhưng answer vẫn sai do model thêm suy diễn, và cũng có case answer nghe hợp lý nhưng retrieval hoàn toàn không chính xác. Vì vậy em tách riêng phần retrieval metrics và LLM judge, rồi ghép chúng lại ở `main.py` thành benchmark summary hoàn chỉnh. Nhờ vậy nhóm có thể nhìn rõ lỗi nằm ở truy xuất, ở sinh câu trả lời, hay ở chính logic đánh giá.
