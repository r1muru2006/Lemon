# Đồ án mini: Xây detector DDoS từ NetFlow/PCAP với CICDDoS (gần ý tưởng paper, quy mô nhỏ)

## 1) Bài toán và mục tiêu

### Bài toán

Xây một hệ thống phát hiện DDoS gần thời gian thực (near real-time) dựa trên luồng (flow-based), sử dụng dữ liệu:

- PCAP thô (để mô phỏng pipeline thực tế).
- NetFlow-like records (trích từ PCAP).
- Dataset CICDDoS (ưu tiên CIC-DDoS2019) để huấn luyện/đánh giá.

### Mục tiêu kỹ thuật

- Phát hiện nhị phân: `Benign` vs `DDoS`.
- Phân loại theo họ tấn công (tùy chọn): SYN/UDP/HTTP/DNS/NTP...
- Đánh giá theo tiêu chí vận hành:
  - F1, Recall, Precision, PR-AUC.
  - TPR tại FPR thấp (ví dụ FPR <= 1%).
  - Detection delay theo cửa sổ thời gian.
  - Tốc độ xử lý flows/s ở máy cá nhân.

### Mục tiêu học thuật (gần ý tưởng paper)

Mô phỏng ý tưởng phổ biến trong paper DDoS flow-based:

1. Đặc trưng theo flow + theo cửa sổ thời gian (không phụ thuộc payload).
2. Giảm phụ thuộc vào thông tin định tuyến (routing-oblivious): chỉ dùng metadata flow.
3. Tối ưu cho giám sát online: sliding window + cập nhật liên tục.

## 2) Phạm vi (quy mô nhỏ, khả thi)

### In-scope

- Dữ liệu CIC-DDoS2019 (CSV + một phần PCAP nếu có).
- Trích đặc trưng NetFlow-like từ PCAP bằng script Python.
- Huấn luyện 2 mô hình:
  - Baseline: Logistic Regression / RandomForest.
  - Main: XGBoost/LightGBM (khuyến nghị), hoặc MLP nhỏ.
- So sánh offline và mô phỏng online theo cửa sổ 1s/5s.

### Out-of-scope

- Hạ tầng phân tán lớn (Kafka/Flink/DPDK).
- Deep model cực lớn.
- Triển khai production đa node.

## 3) Kiến trúc detector đề xuất

```text
PCAP stream/file
   -> Flow Builder (5-tuple + timeout)
   -> Feature Extractor (flow stats + window stats)
   -> Detector Model (binary or multi-class)
   -> Alert + Score + Attack type
```

### 3.1 Định nghĩa flow

- Khóa flow: `(src_ip, dst_ip, src_port, dst_port, protocol)`.
- Active timeout: 30s, idle timeout: 10s (có thể tune).
- Mỗi flow xuất record gồm thống kê packet/byte/timing/flags.

### 3.2 Feature set (gần paper flow-based)

Nhóm A - per-flow cơ bản:

- `flow_duration_ms`
- `fwd_pkt_count`, `bwd_pkt_count`
- `fwd_bytes`, `bwd_bytes`
- `pkt_rate`, `byte_rate`
- `avg_pkt_size`, `min_pkt_size`, `max_pkt_size`
- `syn_count`, `ack_count`, `rst_count`, `psh_count`, `urg_count`
- `tcp_flag_ratio_syn_ack`, `bwd_fwd_pkt_ratio`

Nhóm B - timing:

- `iat_mean`, `iat_std`, `iat_min`, `iat_max`
- `burstiness = iat_std / (iat_mean + eps)`

Nhóm C - cửa sổ thời gian (1s/5s), routing-oblivious:

- `unique_src_per_dst`
- `unique_dst_per_src`
- `flows_per_src`, `flows_per_dst`
- `syn_rate_per_dst`
- `entropy_src_ip_on_dst_window`
- `entropy_src_port_on_dst_window`

Nhóm D - optional (mô phỏng paper measurement constraints):

- Sketch-like counters (Count-Min Sketch giả lập) cho top talker/burst score.

### 3.3 Chiến lược phân tầng (khuyến nghị)

- Stage 1 (fast filter): mô hình nhẹ phát hiện bất thường/attack.
- Stage 2 (refine): mô hình mạnh hơn để giảm false positive và phân loại loại tấn công.

## 4) Dữ liệu CICDDoS và cách chia tập

### 4.1 Dữ liệu

- Dùng CIC-DDoS2019 (hoặc CICIDS2017/2018 nếu thiếu tài nguyên).
- Ưu tiên dùng split theo ngày/session để giảm data leakage.

### 4.2 Quy tắc split (rất quan trọng)

Không random toàn cục từng flow vì dễ leak.

- Train: ngày A/B.
- Validation: ngày C.
- Test: ngày D (chứa kịch bản chưa thấy trong train nếu có).

### 4.3 Cân bằng lớp

DDoS thường lệch lớp mạnh:

- Dùng class weights hoặc focal loss (nếu deep).
- Hoặc undersample benign có kiểm soát.
- Luôn report confusion matrix theo từng attack family.

## 5) Pipeline triển khai chi tiết

## 5.1 Bước 1 - Chuẩn hóa dữ liệu

- Đọc CSV CICFlowMeter sẵn có (nếu dataset cung cấp).
- Làm sạch:
  - bỏ cột ID bị leak thời gian thực,
  - xử lý NaN/Inf,
  - chuẩn hóa tên cột,
  - map label về `benign` và `attack_family`.

Output:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

## 5.2 Bước 2 - Trích NetFlow-like từ PCAP

Dùng Python (scapy/pyshark/dpkt) để parse packet và build flow:

- Update counter theo 5-tuple.
- Tính IAT theo timestamp packet.
- Flush flow khi timeout.

Output:

- `data/flows/*.parquet`

## 5.3 Bước 3 - Tạo feature cửa sổ thời gian

- Gom flow theo cửa sổ (1s hoặc 5s).
- Tạo aggregate/entropy/count-distinct.
- Join lại với per-flow feature.

Output:

- `data/features/*.parquet`

## 5.4 Bước 4 - Huấn luyện

- Baseline model:
  - Logistic Regression (scaled features) hoặc RandomForest.
- Main model:
  - XGBoost/LightGBM với early stopping.
- Calibrate threshold theo mục tiêu FPR thấp.

Output:

- `models/baseline.pkl`
- `models/main_xgb.json`
- `reports/metrics_offline.json`

## 5.5 Bước 5 - Mô phỏng online detector

- Đọc flow theo thứ tự thời gian.
- Tạo sliding window state trong RAM.
- Inference từng batch nhỏ (100-1000 flow).
- Xuất cảnh báo khi score > threshold.

Output:

- `reports/online_eval.json`
- `reports/alerts.csv`

## 6) Thiết kế thư mục đồ án

```text
Project/
  ddos_detector/
    data/
      raw/
      processed/
      flows/
      features/
    src/
      config.py
      data_prep.py
      pcap_to_flow.py
      feature_window.py
      train.py
      evaluate.py
      online_detect.py
      utils/
        io.py
        metrics.py
        net.py
    models/
    reports/
    notebooks/
      eda.ipynb
      error_analysis.ipynb
    requirements.txt
    README.md
```

## 7) Ý tưởng mô hình gần paper (đề xuất chính)

### Option A (khuyến nghị cho đồ án)

`Flow + Window Features -> XGBoost`

- Rất mạnh với tabular data.
- Huấn luyện nhanh, giải thích được bằng SHAP.
- Dễ đạt kết quả tốt trên CICDDoS.

### Option B (nâng cao)

`Flow sequence per destination -> 1D-CNN/LSTM nhỏ`

- Bắt mẫu temporal tốt hơn.
- Chi phí tuning cao hơn.
- Chỉ làm nếu còn thời gian.

### Option C (hybrid gần thực tế)

- Stage 1: XGBoost nhanh.
- Stage 2: mini sequence model cho alert nghi ngờ.

## 8) Tiêu chí đánh giá và báo cáo

### 8.1 Offline

- PR-AUC (ưu tiên khi lệch lớp).
- ROC-AUC.
- F1 macro/micro.
- Recall của từng family.

### 8.2 Online

- `Detection delay`: thời gian từ lúc attack bắt đầu đến lúc alert đầu tiên.
- `Alerts per minute` và false alerts.
- Throughput flows/s.

### 8.3 Bảng kết quả tối thiểu cần có

- Bảng 1: baseline vs main model.
- Bảng 2: ablation:
  - chỉ per-flow,
  - per-flow + window,
  - per-flow + window + sketch-like.
- Bảng 3: theo từng loại tấn công.

## 9) Lộ trình thực hiện (4 tuần)

### Tuần 1

- Setup môi trường, tải/kiểm tra dataset.
- Xây data_prep + split đúng theo thời gian/ngày.
- EDA cơ bản.

### Tuần 2

- Viết pcap_to_flow + feature_window.
- Kiểm thử trên sample PCAP nhỏ.
- Đảm bảo feature ổn định và không leak.

### Tuần 3

- Huấn luyện baseline + main.
- Tuning threshold theo FPR mục tiêu.
- Chạy ablation.

### Tuần 4

- Online simulation + benchmark tốc độ.
- Error analysis (false positive/false negative).
- Viết báo cáo, slide, demo.

## 10) Rủi ro và cách giảm rủi ro

- Data leakage do split sai:
  - Bắt buộc split theo ngày/session.
- Lệch lớp quá lớn:
  - Class weight + PR-AUC + threshold tuning.
- Overfit vào CICDDoS:
  - Test trên ngày/attack khác, ablation rõ ràng.
- Thiếu tài nguyên máy:
  - Sample dữ liệu, ưu tiên XGBoost thay deep lớn.

## 11) Kế hoạch viết báo cáo/slide

### Mục lục báo cáo đề xuất

1. Giới thiệu và động lực.
2. Liên quan công trình (paper tương tự).
3. Thiết kế hệ thống detector.
4. Dataset và tiền xử lý.
5. Mô hình và cấu hình huấn luyện.
6. Kết quả thực nghiệm và phân tích lỗi.
7. Hạn chế và hướng mở rộng.

### Điểm nhấn cần thể hiện

- Vì sao chọn flow-based thay payload-based.
- Vì sao thêm feature cửa sổ giúp tăng recall với low-and-slow hoặc distributed attack.
- Trade-off độ chính xác vs tốc độ.

## 12) Checklist build thực tế (copy chạy dần)

1. Tạo môi trường và cài thư viện:

- pandas, numpy, scikit-learn, xgboost/lightgbm, pyarrow, scapy, tqdm, matplotlib, seaborn, shap.

2. Chạy data prep:

- đọc raw csv -> clean -> split -> lưu parquet.

3. Chạy pcap_to_flow:

- parse pcap -> flow parquet.

4. Chạy feature_window:

- aggregate theo 1s/5s -> merge per-flow features.

5. Huấn luyện baseline và main:

- lưu model + threshold.

6. Đánh giá offline + online:

- lưu metrics json + confusion matrix + alerts.

7. Viết report tự động:

- export bảng/plot vào thư mục reports.

## 13) Mức hoàn thiện mong muốn

### Mức đạt

- Có pipeline end-to-end chạy được.
- Có so sánh baseline/main.
- Có online simulation và detection delay.

### Mức tốt

- Có ablation + SHAP interpretability.
- Có mô phỏng sketch-like counter.

### Mức xuất sắc

- Có test chéo dataset hoặc cross-day unseen attack.
- Có demo live từ PCAP stream và dashboard cảnh báo.

## 14) Gợi ý mở rộng sau đồ án

- Distillation để giảm latency.
- Adaptive threshold theo thời gian trong ngày.
- Kết hợp detector với rule-based mitigation (rate limit / ACL auto suggest).

---

Nếu bạn muốn, bước kế tiếp mình có thể build luôn bộ khung code trong `Project/ddos_detector` (các file `pcap_to_flow.py`, `feature_window.py`, `train.py`, `online_detect.py`) để bạn chạy trực tiếp ngay trên CICDDoS.
