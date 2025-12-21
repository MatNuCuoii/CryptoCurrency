📊 Social Sentiment Analysis – Implementation Guide (FREE API)
1. Mục tiêu của phần Social Sentiment

Phần Social Sentiment được thêm vào dashboard nhằm trả lời các câu hỏi phục vụ quyết định đầu tư, không chỉ để minh hoạ:

Tâm lý thị trường (sentiment) có ảnh hưởng đến lợi nhuận (return) hay không?

Sentiment đi trước (lead) hay chỉ phản ứng theo biến động giá?

Có thể dùng sentiment như tín hiệu hỗ trợ quản trị rủi ro hay không?

👉 Sentiment được xem như một chuỗi thời gian (time-series feature) và được phân tích song song với dữ liệu giá.

2. Nguồn dữ liệu Social Sentiment (CHỈ FREE)
2.1. Crypto Fear & Greed Index (nguồn chính)

Nhà cung cấp: Alternative.me

API (FREE – không cần API key)

https://api.alternative.me/fng/?limit=0


Đặc điểm

Chỉ số từ 0–100

Dữ liệu theo ngày (daily)

Phản ánh tâm lý chung toàn thị trường crypto

Phù hợp cho:

phân tích market regime

event study

correlation với return / volatility

Ý nghĩa chỉ số

Khoảng giá trị	Trạng thái
0 – 25	Extreme Fear
26 – 49	Fear
50 – 59	Neutral
60 – 74	Greed
75 – 100	Extreme Greed

⚠️ Chỉ số này không dùng độc lập để mua/bán, mà dùng như risk & sentiment indicator.

3. Phạm vi dữ liệu áp dụng cho project

Khoảng thời gian dữ liệu giá:
24-03-2023 → 17-12-2025

Mục tiêu sentiment:

Lấy toàn bộ lịch sử Fear & Greed

Lọc đúng date range trùng với dữ liệu coin

Chuẩn hoá về daily time-series

4. Quy trình triển khai Social Sentiment (End-to-End)
Bước 1: Gọi API lấy toàn bộ lịch sử sentiment

Gọi API với limit=0 để lấy full historical data

API trả về danh sách gồm:

value (0–100)

value_classification

timestamp (UNIX seconds)

Bước 2: Chuẩn hoá dữ liệu sentiment

Thực hiện các bước xử lý:

Convert timestamp → date (YYYY-MM-DD, UTC)

Ép kiểu value → numeric

Giữ lại các cột:

date

fng_value

fng_label

Bước 3: Lọc theo đúng date range của project

Chỉ giữ các dòng thoả mãn:

2023-03-24 ≤ date ≤ 2025-12-17


👉 Kết quả là sentiment dataset khớp 100% với dữ liệu giá.

Bước 4: Lưu sentiment thành dataset trung gian

Lưu ra file (CSV hoặc Parquet)

File này được dùng lại cho dashboard & phân tích

Tránh gọi API mỗi lần chạy Streamlit

Ví dụ:

fear_greed_daily_2023_2025.csv

Bước 5: Join sentiment với dữ liệu giá coin

Dữ liệu giá của coin phải ở daily frequency

Join theo cột date

Mỗi dòng dữ liệu coin sẽ có thêm:

fng_value

fng_label

⚠️ Lưu ý:

Thống nhất timezone

Không join theo timestamp intraday

5. Feature Engineering cho Sentiment Analysis
5.1. Biến giá (bắt buộc)

Không dùng price level, mà dùng:

return_1d (log-return hoặc % return)

volatility_7d (rolling std của return)

5.2. Biến sentiment (bắt buộc)

Tạo các biến trễ (lag):

fng_lag_0

fng_lag_1

fng_lag_3

fng_lag_7

fng_lag_14

👉 Mục tiêu: kiểm tra sentiment đi trước bao nhiêu ngày.

6. Các phân tích cốt lõi cần thực hiện
6.1. Lag Correlation Analysis

Tính correlation giữa:

fng(t − k)  ↔  return(t)


với:

k = 0, 1, 3, 7, 14


Kết quả mong muốn

Xác định lag nào sentiment có ảnh hưởng mạnh nhất

Phân biệt:

sentiment dẫn dắt

sentiment phản ứng theo giá

6.2. Event Study (Trọng tâm chính)

Định nghĩa event

Extreme Fear: fng_value ≤ 25

Extreme Greed: fng_value ≥ 75

Phân tích sau event

Return tại:

+1 ngày

+3 ngày

+7 ngày

+14 ngày

Thống kê

Median return

Hit-rate (% số ngày return dương)

Max drawdown sau event (nếu có)

👉 Đây là phần trả lời trực tiếp câu hỏi: sentiment có ảnh hưởng không?

6.3. Strategy Backtest (Decision Support)

Xây dựng chiến lược rule-based đơn giản:

Ví dụ:

Buy / Increase exposure: Extreme Fear

Reduce / Risk-off: Extreme Greed

So sánh với:

Buy & Hold

Chỉ số đánh giá:

Equity curve

Max drawdown

Win-rate

⚠️ Mục đích: hỗ trợ quyết định, không phải khuyến nghị đầu tư.

7. Hiển thị trong Dashboard (Social Sentiment Tab)

Tab Social Sentiment nên có:

Line chart Fear & Greed theo thời gian

Overlay sentiment vs return

Correlation theo lag (bar/heatmap)

Event study table (Extreme Fear/Greed)

(Tuỳ chọn) Backtest equity curve

8. Nguyên tắc phân tích cần tuân thủ

Không dùng sentiment để dự đoán giá tuyệt đối

Luôn phân tích trên return

Không cherry-pick timeframe

Chia market regime nếu cần (bull/bear)

Sentiment = risk signal, không phải nguyên nhân tuyệt đối

9. Lộ trình mở rộng (Sau khi xong FREE)

Reddit sentiment (FREE nhưng phức tạp hơn)

News sentiment

Coin-level sentiment (LunarCrush / Santiment – trả phí)

ML model: sentiment là feature đầu vào

10. Kết luận

Với:

Fear & Greed Index (FREE)

Dữ liệu giá có sẵn của bạn

Bạn đã đủ dữ liệu và cơ sở phân tích để:

Đánh giá ảnh hưởng sentiment

Tạo insight có giá trị cho nhà đầu tư

Nâng cấp dashboard từ “visualization” → “decision support”