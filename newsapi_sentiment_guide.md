# News Sentiment Analysis with NewsAPI (FREE) – Implementation Guide

## 1. Mục tiêu khi thêm News Sentiment

News Sentiment được thêm vào dashboard nhằm:

- Nắm bắt **narrative & bối cảnh thị trường** (ETF, regulation, hack, macro…)
- Giải thích **nguyên nhân biến động giá**
- Cảnh báo **rủi ro FOMO / panic** khi sentiment news quá lệch

👉 News sentiment **KHÔNG dùng để timing buy/sell**, mà dùng cho **context & risk analysis**.

---

## 2. Nguồn dữ liệu News (FREE & ổn định)

### Nhà cung cấp chính
**:contentReference[oaicite:0]{index=0}**

### Lý do chọn NewsAPI
- Có **FREE tier**
- API ổn định, rõ ràng
- Hợp pháp cho project học tập / dashboard public
- Text tin tức “sạch” → NLP sentiment dễ

---

## 3. Giới hạn của NewsAPI (FREE tier)

- ~**100 requests / ngày**
- Có thể query theo:
  - keyword
  - date range
- Phù hợp để lấy:
  - **tin gần đây (7–30 ngày)**

👉 **KHÔNG cần** lấy lịch sử nhiều năm.

---

## 4. Phạm vi dữ liệu News Sentiment (khuyến nghị)

### Time range
- **7 – 30 ngày gần nhất**

Lý do:
- News sentiment có **độ trễ**
- Narrative cũ nhanh chóng mất tác dụng
- Dữ liệu dài chỉ thêm nhiễu

---

### Keywords nên dùng
Ví dụ:
- `bitcoin`
- `crypto`
- `cryptocurrency`
- `ethereum`
- `blockchain`

👉 Có thể gom nhiều keyword trong 1 query.

---

## 5. Quy trình triển khai News Sentiment (End-to-End)

### Bước 1: Đăng ký NewsAPI
- Tạo tài khoản NewsAPI
- Lấy **API key FREE**

---

### Bước 2: Gọi API lấy tin tức
Mỗi request lấy:
- `title`
- `description`
- `publishedAt`
- `source.name`

Không cần:
- full article body
- author info

---

### Bước 3: Chuẩn hoá dữ liệu tin tức
- Gộp text:
full_text = title + " " + description

- Convert `publishedAt` → `date` (YYYY-MM-DD)

---

### Bước 4: Chấm sentiment cho từng bài báo

#### Công cụ FREE khuyến nghị
- **VADER (NLTK)**
- Phù hợp headline & mô tả ngắn
- Nhẹ, dễ triển khai

#### Output mỗi bài:
- `sentiment_score` ∈ [-1, 1]
- phân loại:
- positive
- neutral
- negative

---

### Bước 5: Aggregate News Sentiment theo ngày
Không dùng sentiment từng bài, mà gom theo ngày:

Các chỉ số nên có:
- `news_sentiment_mean`
- `news_sentiment_median`
- `news_positive_ratio`
- `news_count`

Ví dụ:
date | news_sentiment_mean | news_count


---

### Bước 6: Lưu thành dataset trung gian
- Lưu ra CSV / Parquet
- Dashboard chỉ đọc file này, **không gọi API trực tiếp**

Ví dụ:
news_sentiment_daily.csv

---

## 6. Join News Sentiment với dữ liệu giá

Join theo `date` với:
- OHLCV (coin)
- Fear & Greed Index
- Reddit recent sentiment (nếu có)

Kết quả:
> Mỗi ngày có **price + market sentiment + social sentiment + news sentiment**

---

## 7. Các phân tích nên làm với News Sentiment

### 7.1. Context Analysis
- News sentiment tăng mạnh → narrative tích cực
- News sentiment giảm mạnh → rủi ro vĩ mô / sự kiện xấu

---

### 7.2. Lag Analysis
So sánh:
news_sentiment(t − k) ↔ return(t)

với:
- k = 0, 1, 3, 7 ngày

👉 News thường **lead ngắn hạn**, không dài.

---

### 7.3. Divergence Analysis
Ví dụ:
- Price ↑ nhưng news sentiment ↓  
→ uptrend thiếu nền tảng narrative

- Price ↓ nhưng news sentiment ↑  
→ khả năng quá bán do fear ngắn hạn

---

## 8. Hiển thị trong Dashboard

Tab News Sentiment nên có:

1. Line chart: News sentiment theo ngày
2. Bar chart: Số lượng tin mỗi ngày
3. Overlay: News sentiment vs return
4. Highlight:
   - ngày sentiment cực đoan
   - sự kiện lớn (ETF, hack, regulation)

---

## 9. Insight tự động (gợi ý)

Dashboard có thể sinh insight dạng:
- “News sentiment 7 ngày qua nghiêng mạnh về tích cực, chủ yếu từ tin ETF.”
- “News sentiment giảm mạnh trong khi Fear & Greed vẫn cao → rủi ro điều chỉnh.”
- “Số lượng tin tăng đột biến kèm sentiment âm → panic-driven move.”

---

## 10. Những điều KHÔNG nên làm

❌ Không dùng news sentiment để dự đoán giá trực tiếp  
❌ Không lấy lịch sử news quá dài  
❌ Không coi news sentiment là social emotion

---

## 11. Best Practice kết hợp Sentiment (FREE)

- **Fear & Greed** → market psychology
- **Reddit (recent)** → cảm xúc retail
- **NewsAPI** → narrative & vĩ mô

👉 Khi 3 nguồn **đồng thuận** → signal mạnh  
👉 Khi **mâu thuẫn** → cảnh báo rủi ro

---

## 12. Kết luận

- NewsAPI là nguồn **FREE, ổn định, hợp pháp**
- News sentiment rất phù hợp cho:
  - context
  - risk analysis
  - giải thích biến động
- Không cần Glassnode nếu mục tiêu là dashboard sentiment

👉 Đây là hướng đi **thực tế & bền vững** cho project crypto analysis.

