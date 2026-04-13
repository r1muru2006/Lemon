# Mini DDoS Detector (Flow-based, CICDDoS)

Project này hiện thực pipeline detector DDoS quy mô nhỏ theo hướng flow-based + window features:

- Input: CSV CICFlowMeter và/hoặc PCAP.
- Flow builder: NetFlow-like records từ PCAP.
- Feature: per-flow + aggregate theo cửa sổ thời gian.
- Model: baseline + main model.
- Eval: offline và online simulation.

## 1. Cấu trúc

```text
ddos_detector/
  data/
    raw/          # CSV/PCAP đầu vào
    processed/    # split train/val/test từ CSV
    flows/        # flow parquet từ PCAP
    features/     # feature parquet sau window aggregation
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
  requirements.txt
```

## 2. Cài đặt

```bash
cd ddos_detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu CSV CIC

Copy CSV CIC vào `data/raw/` rồi chạy:

```bash
python src/data_prep.py --raw-dir data/raw --out-dir data/processed
```

Output:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

## 4. Trích flow từ PCAP (tùy chọn)

```bash
python src/pcap_to_flow.py --pcap data/raw/sample.pcap --output data/flows/sample_flows.parquet
```

## 5. Tạo feature cửa sổ

Với dữ liệu split từ CSV:

```bash
python src/feature_window.py --input data/processed/train.parquet --output data/features/train.parquet
python src/feature_window.py --input data/processed/val.parquet --output data/features/val.parquet
python src/feature_window.py --input data/processed/test.parquet --output data/features/test.parquet
```

Với flow từ PCAP thì chỉ cần đổi đường dẫn input.

## 6. Huấn luyện

```bash
python src/train.py \
  --train data/features/train.parquet \
  --val data/features/val.parquet \
  --model-dir models \
  --report-dir reports
```

Output:

- `models/baseline.pkl`
- `models/main.pkl`
- `models/metadata.json`
- `reports/metrics_offline.json`

## 7. Đánh giá offline

```bash
python src/evaluate.py \
  --test data/features/test.parquet \
  --model-dir models \
  --report-dir reports
```

Output:

- `reports/metrics_test.json`
- `reports/confusion_matrix.csv`

## 8. Mô phỏng online detection

```bash
python src/online_detect.py \
  --input data/features/test.parquet \
  --model-dir models \
  --report-dir reports \
  --batch-size 500
```

Output:

- `reports/online_eval.json`
- `reports/alerts.csv`

## 9. Tich hop Lemon (GitHub)

Neu ban chay Lemon controller va thu duoc log dang:

```text
<dst_ip>,<counter>,<estimate>
```

hoac co prefix logger (vi du `INFO:root:<dst_ip>,<counter>,<estimate>`),
co the chuyen sang format detector bang:

```bash
python src/lemon_bridge.py \
  --lemon-log /path/to/lemon.log \
  --output data/features/lemon_online.parquet \
  --step-seconds 1
```

Sau do chay online detector truc tiep:

```bash
python src/online_detect.py \
  --input data/features/lemon_online.parquet \
  --model-dir models \
  --report-dir reports/lemon_online
```

Luu y: adapter se map cac thong so Lemon (counter, estimate) sang mot tap feature toi thieu,
nhung van giu tuong thich voi model metadata hien tai.

## 10. Luu y quan trong

- Ưu tiên split theo thời gian để giảm data leakage.
- Nhãn benign được map bởi từ khóa `benign`/`normal`; phần còn lại xem là attack.
- Nếu không có `xgboost`, script tự fallback sang `RandomForest`.
- Thư viện parse PCAP dùng `scapy`; với file lớn, nên chạy trên sample trước.
