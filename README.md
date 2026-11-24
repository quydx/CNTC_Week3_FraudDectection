# Week 3 – Fraud Detection (Bài B – Pre-class)

Phân tích dữ liệu giao dịch để phát hiện gian lận: EDA, kiểm tra leakage, baseline Logistic Regression và so sánh với mô hình phi tuyến (Random Forest). Đề bài: `Bai tap (2).pdf`.

## Cấu trúc

| Tệp | Mô tả |
|---|---|
| `b_preclass.py` | Script chính: EDA → kiểm tra leakage → đánh giá mô hình bằng cross-validation |
| `b_preclass_results.csv` | Bảng kết quả các mô hình (sinh ra khi chạy script) |
| `Bai tap (2).pdf` | Đề bài |
| `pyproject.toml`, `uv.lock`, `.python-version` | Khai báo và khóa dependency cho uv |
| `requirements.txt` | Dependency cho pip (xuất từ `uv.lock`) |

Dữ liệu đầu vào `B_transactions_Pre_class.csv` không nằm trong repo (đã thêm vào `.gitignore`); đặt tệp này cạnh script trước khi chạy.

## Cài đặt

Dùng [uv](https://docs.astral.sh/uv/) (Python 3.13, phiên bản thư viện được khóa trong `uv.lock`):

```powershell
uv sync
```

Hoặc dùng pip với `requirements.txt` (xuất từ `uv.lock`):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sau khi đổi dependency trong `pyproject.toml`, cập nhật lại: `uv lock; uv export --no-hashes --no-header --no-emit-project -o requirements.txt`.

## Chạy

```powershell
uv run python b_preclass.py B_transactions_Pre_class.csv
```

Script in kết quả EDA và đánh giá ra màn hình, đồng thời ghi `b_preclass_results.csv`. Tệp dữ liệu gốc không bị sửa; mọi tiền xử lý nằm trong `Pipeline` và chỉ fit trên tập train của từng fold.

## Nội dung phân tích

1. **EDA** – số dòng, tỷ lệ gian lận, giá trị thiếu, trùng `tx_id`, `src == dst`, thống kê theo lớp, ngoại lệ `amount_vnd` theo IQR, tỷ lệ gian lận theo `night_tx` / `new_payee`.
2. **Kiểm tra leakage** – PR-AUC / ROC-AUC của từng feature đơn lẻ, kiểm tra khả năng tách lớp hoàn hảo (max hợp lệ vs. min gian lận), phân tích tài khoản nguồn/đích.
3. **Mô hình** – features: `amount_vnd` (log1p), `night_tx`, `new_payee`, `tx_count_1h`, `risk_score`; không dùng `tx_id`, `src_account`, `dst_account`.
   - Dummy (prior)
   - LR – đủ feature
   - LR – bỏ `risk_score`
   - Random Forest (`max_depth=3`)

## Thiết lập đánh giá

- `RepeatedStratifiedKFold`: 3 fold × 30 lần, seed 42 (mỗi fold validation có 2 giao dịch gian lận).
- Chỉ số: PR-AUC (chính), ROC-AUC, Recall@top 10%, Accuracy@0.5, chi phí tại ngưỡng 0.5.
- Giả định chi phí: bỏ sót 1 gian lận (FN) = 20, cảnh báo sai (FP) = 1.

## Kết quả

| Mô hình | PR-AUC | ROC-AUC | Recall@top10% | Accuracy@0.5 | Cost/fold | FP | FN |
|---|---|---|---|---|---|---|---|
| Dummy (prior) | 0.100 | 0.500 | 0.006 | 0.900 | 40.00 | 0 | 180 |
| LR – đủ feature | 1.000 | 1.000 | 1.000 | 0.999 | 0.02 | 2 | 0 |
| LR – bỏ `risk_score` | 1.000 | 1.000 | 1.000 | 0.998 | 0.04 | 4 | 0 |
| RF – đủ feature | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 0 | 0 |

PR-AUC = 1.0 ở mọi mô hình (kể cả khi bỏ `risk_score`) cho thấy dữ liệu Pre-class gần như tách lớp hoàn hảo với cỡ mẫu nhỏ — cần thận trọng khi diễn giải, và kết quả cần được kiểm chứng trên dữ liệu mới.
