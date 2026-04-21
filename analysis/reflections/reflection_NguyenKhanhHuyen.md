# Báo cáo Cá nhân (Reflection)

**Họ và tên:** Nguyễn Khánh Huyền  
**Vai trò:** Evaluation & Benchmarking  
**Nhóm:** Team Evaluation Factory  

## 1. Engineering Contribution
Trong nhóm, mình phụ trách chính phần **đánh giá hệ thống**. Cụ thể, mình viết code để chạy benchmark cho hai phiên bản agent là **V1** và **V2**, sau đó tổng hợp kết quả và so sánh giữa hai phiên bản. Phần mình làm bao gồm chạy toàn bộ test cases, thu thập các chỉ số đánh giá, xuất file báo cáo và hỗ trợ nhóm nhìn rõ agent đang tốt lên hay còn yếu ở đâu.

Ngoài việc chạy benchmark, mình cũng tham gia xây dựng cơ chế **multi-judge** để chấm chất lượng câu trả lời của agent. Phần này giúp hệ thống không chỉ dựa vào một kết quả đánh giá duy nhất mà có thể xem mức độ đồng thuận giữa nhiều judge, từ đó phát hiện rõ hơn các câu trả lời đúng, gần đúng hoặc có dấu hiệu hallucination.

Bên cạnh đó, mình còn phụ trách tổng hợp kết quả benchmark thành các file báo cáo như `summary.json`, `benchmark_results.json` và phần `failure_analysis.md` để phục vụ cho việc kiểm tra và chấm lab. Nhờ đó, pipeline đánh giá của nhóm có thể chạy hoàn chỉnh từ bước benchmark đến bước phân tích kết quả.

## 2. Technical Depth
Qua phần việc của mình, mình hiểu rõ hơn cách đánh giá một hệ thống RAG hoặc QA agent không chỉ bằng một chỉ số duy nhất, mà cần nhìn trên nhiều khía cạnh khác nhau. Mình làm việc trực tiếp với các chỉ số như:

- **Pass Rate** để theo dõi tỉ lệ test case agent vượt qua
- **Judge Score** để đánh giá chất lượng câu trả lời cuối cùng
- **Hit Rate, Retrieval Accuracy, MRR** để đo chất lượng retrieval
- **Agreement Rate** để xem các judge có đồng thuận với nhau không
- **Hallucination Rate** để phát hiện các câu trả lời có thêm thông tin không được hỗ trợ bởi tài liệu

Trong quá trình làm, mình cũng hiểu rõ hơn sự khác nhau giữa lỗi đến từ **retrieval** và lỗi đến từ **answer generation**. Có những case agent trả lời sai vì không retrieve được đúng context, nhưng cũng có những case retrieval đúng mà agent vẫn trả lời dài dòng, thêm suy diễn hoặc không bám sát expected answer. Điều này giúp mình nhìn bài toán đánh giá agent một cách toàn diện hơn.

Ngoài ra, khi làm phần multi-judge, mình cũng hiểu thêm cách sử dụng nhiều judge để tăng độ tin cậy trong đánh giá. Những trường hợp judge không đồng thuận thường là các câu gần đúng hoặc câu có thêm chi tiết ngoài tài liệu, và đây chính là những case cần manual review hoặc cần tối ưu prompt sinh đáp án.

## 3. Problem Solving
Một vấn đề lớn mình gặp là khi benchmark sinh ra rất nhiều kết quả, nếu chỉ nhìn các số tổng hợp thì rất khó biết agent thực sự sai ở đâu. Ban đầu có thể thấy pass rate thấp hoặc hallucination rate cao, nhưng chưa thể kết luận nguyên nhân là do retrieval yếu, do prompt sinh đáp án chưa tốt hay do agent trả lời quá dài.

Để xử lý việc đó, mình đi theo hướng đọc trực tiếp từng case trong `benchmark_results.json`, đối chiếu giữa:
- câu hỏi
- câu trả lời của agent
- đáp án mong đợi
- retrieval metrics
- judge result

Từ đó mình phân loại lỗi thành các nhóm dễ hiểu hơn như:
- không retrieve được đúng context
- agent fallback sai kiểu “tài liệu không có thông tin”
- agent trả lời đúng ý chính nhưng thêm chi tiết ngoài tài liệu
- agent trả lời quá dài thay vì trả lời đúng fact cần hỏi

Sau khi phân nhóm lỗi, mình viết phần **failure analysis** để mô tả rõ các lỗi tiêu biểu, chọn các case xấu nhất để phân tích theo hướng nguyên nhân gốc, rồi đề xuất action plan cụ thể cho nhóm. Nhờ phần này, kết quả benchmark không chỉ dừng ở việc “đạt hay không đạt” mà còn trở thành căn cứ để cải thiện hệ thống ở vòng sau.

## 4. Kết quả và bài học rút ra
Từ phần mình thực hiện, nhóm có thể so sánh trực tiếp giữa V1 và V2 thay vì chỉ nhìn cảm tính. Qua benchmark, có thể thấy V2 cải thiện rõ rệt hơn V1 ở nhiều mặt như pass rate, retrieval quality và hallucination rate. Điều đó cho thấy việc xây dựng một pipeline đánh giá rõ ràng là rất cần thiết, vì nó giúp nhóm biết được thay đổi nào thực sự có ích.

Cá nhân mình học được rằng đánh giá agent không chỉ là chạy test rồi lấy điểm trung bình, mà quan trọng hơn là phải đọc được bản chất của từng lỗi. Một hệ thống có thể trả lời nghe rất hợp lý nhưng vẫn sai vì không bám tài liệu; ngược lại có những câu gần đúng nhưng bị phạt vì thiếu chi tiết bắt buộc. Vì vậy, phần evaluation cần vừa có chỉ số định lượng, vừa có phân tích định tính.

Qua bài lab này, mình hiểu rõ hơn vai trò của evaluation trong một pipeline AI: không chỉ để chấm điểm mô hình, mà còn để định hướng cải tiến hệ thống một cách có cơ sở.