# DDoS Detector: Hybrid Lemon Sketch + XGBoost

Đây là dự án Phát hiện Tấn công Mạng (DDoS Volumetric) lai ghép (Hybrid) giữa hạ tầng phân mảnh mạng (Data Plane) và thuật toán Học Máy (Machine Learning).
Dự án được truyền cảm hứng và mô phỏng lại thuật toán **Lemon Sketch** (thuộc bài báo hội nghị USENIX Security 2025) để giải quyết tình trạng đếm trùng gói tin (over-counting) do các luồng mạng chạy đa tuyến.

Đồng thời, thay vì sử dụng luật cảnh báo tĩnh truyền thống, lớp Phân tích hệ thống (Layer 3) được thay thế bởi **XGBoost** để tăng độ chính xác trong việc phân loại các họ tấn công dồn dập (SYN Flood, UDP Flood, DNS/NTP Amplification).

## Cấu trúc thư mục (Directory Structure)

```text
ddos_detector/
├── requirements.txt
└── src/
    ├── data_prep/
    │   ├── __init__.py
    │   └── csv_to_packets.py    # (Giả lập đẩy packets vào Lemon Network từ file CSV tĩnh)
    ├── ml_detection/
    │   └── train_main.py        # (File gốc mô phỏng chu trình end-to-end huấn luyện XGBoost)
    └── sketch/
        ├── __init__.py
        ├── controller.py        # (Lớp Mạng Tầng 2: Hợp nhất Sketches giải quyết Over-Counting)
        └── lemon_node.py        # (Lớp Mạng Tầng 1: Thuật toán Băm phân tầng Lemon Sketch cho từng gói)
```

## Yêu cầu Hệ thống (Prerequisites)

- **OS:** Linux (Ubuntu/WSL khuyến nghị)
- **Python:** 3.10 trở lên.
- Nên dùng Virtual Environment (Môi trường ảo) để cài đặt.

## Hướng dẫn Cài đặt & Chạy Thực Nghiệm

**Bước 1: Di chuyển vào thư mục dự án**
Mở terminal và trỏ đường dẫn tới `ddos_detector`:
```bash
cd /mnt/d/Lemon/ddos_detector
```

**Bước 2: Tạo môi trường ảo (Tùy chọn nhưng Rất Khuyến Dùng)**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện phụ thuộc**
```bash
pip install -r requirements.txt
```
*(Các thư viện chính bao gồm: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `pyarrow`)*

**Bước 4: Chạy Mô Phỏng Luồng Base Code**
Chạy trực tiếp file `train_main.py` để kiểm thử toàn bộ tiến trình hệ thống:
```bash
python src/ml_detection/train_main.py
```

Khi chạy thành công, hệ thống sẽ in ra màn hình trạng thái Console từ lúc **phân mảnh gói tin**, **hợp nhất (Aggregation) Lemon Sketches ở Controller**, đến **Huấn luyện mô hình XGBoost** và đánh giá chuẩn xác trên file dữ liệu giả lập.

> **Ghi chú mở rộng cho Đồ án thật:** 
> Khi bạn tích hợp Data thật từ **CIC-DDoS2019**, vui lòng đưa đường dẫn file CSV vào code đọc `pandas.read_csv(...)` tại script `train_main.py` (Block số 3: 'Simulate Data Injection'). Data khổng lồ sẽ tự động được chạy qua `CSVPacketSimulator` và "vắt sữa" lại bằng Engine Lemon Sketch siêu nhẹ mà không làm tốn RAM hệ điều hành.
