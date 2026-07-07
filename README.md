# THPT Score Analysis

Project Data Science bằng Python để phân tích điểm thi THPTQG từ file dữ liệu điểm thô (`scores_*.csv`).

## Mục tiêu

- Làm sạch dữ liệu điểm thi.
- Khám phá dữ liệu bằng EDA.
- Trực quan hóa phân bố điểm.
- Thực hiện phân tích thống kê mô tả.
- Sinh báo cáo kết quả tự động.

## Cấu trúc thư mục

```text
THPT-Score-Analysis/
├── data/
│   ├── raw/
│   │   └── scores_*.csv
│   └── processed/
│       └── cleaned_scores.csv
├── output/
│   ├── figures/
│   ├── reports/
│   └── tables/
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── visualization.py
│   ├── analysis.py
│   ├── utils.py
│   └── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Cài đặt

```bash
cd THPT-Score-Analysis
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Cách chạy

Chỉ cần chạy:

```bash
python src/main.py
```

Chương trình sẽ tự động:

1. Đọc file CSV đầu vào trong `data/raw/`.
2. Làm sạch dữ liệu.
3. Lưu dữ liệu sạch vào `data/processed/cleaned_scores.csv`.
4. Sinh bảng thống kê trong `output/tables/`.
5. Sinh biểu đồ PNG trong `output/figures/`.
6. Sinh báo cáo text trong `output/reports/report.txt`.

## Thư viện sử dụng

- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

## Mô tả module

- `src/data_cleaning.py`: chuẩn hóa cột, làm sạch dữ liệu, lưu file kết quả.
- `src/eda.py`: tạo các bảng EDA và thống kê mô tả.
- `src/visualization.py`: sinh toàn bộ biểu đồ phân tích.
- `src/analysis.py`: phân tích thống kê chuyên sâu và tạo báo cáo.
- `src/utils.py`: hàm dùng chung, logging, xử lý đường dẫn.
- `src/main.py`: điểm vào của toàn bộ pipeline.

## Kết quả đầu ra

- `data/processed/cleaned_scores.csv`
- `output/tables/*.csv`
- `output/figures/*.png`
- `output/reports/report.txt`
- `output/reports/report.html`

## Xem báo cáo HTML

Sau khi chạy xong, mở file `output/reports/report.html` bằng trình duyệt để xem báo cáo dạng web.

## Hình minh họa

Sau khi chạy xong, các biểu đồ được lưu trong `output/figures/` và có thể chèn trực tiếp vào báo cáo hoặc slide.
