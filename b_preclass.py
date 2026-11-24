"""Bài B – Pre-class: EDA, kiểm tra leakage, baseline LR, đề xuất mô hình phi tuyến.

Chạy:  python b_preclass.py B_transactions_Pre_class.csv
Chỉ dùng tệp Pre-class. Không sửa tệp gốc; mọi tiền xử lý nằm trong Pipeline.
"""
import sys, json, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             confusion_matrix, accuracy_score)

SEED = 42
N_SPLITS, N_REPEATS = 3, 30          # 3-fold: mỗi fold validation có 2 giao dịch gian lận
COST_FN, COST_FP = 20, 1             # giả định: bỏ sót 1 gian lận = 20 lần chi phí 1 cảnh báo sai
TOP_FRAC = 0.10                      # ngân sách điều tra: top 10% giao dịch

path = sys.argv[1] if len(sys.argv) > 1 else "B_transactions_Pre_class.csv"
df = pd.read_csv(path)
y = df["fraud_label"].values

# ---------------- 1. EDA ----------------
print("Số dòng:", len(df), "| gian lận:", y.sum(), f"({y.mean():.1%})")
print("Thiếu:", int(df.isna().sum().sum()), "| trùng tx_id:", int(df.tx_id.duplicated().sum()),
      "| src==dst:", int((df.src_account == df.dst_account).sum()))
num = ["amount_vnd", "night_tx", "new_payee", "tx_count_1h", "risk_score"]
eda = df.groupby("fraud_label")[num].agg(["min", "median", "max", "mean"]).T
print(eda.round(1))

# ngoại lệ theo IQR
q1, q3 = df.amount_vnd.quantile([.25, .75]); iqr = q3 - q1
out = df[(df.amount_vnd > q3 + 1.5 * iqr)]
print("Ngoại lệ amount (IQR):", len(out), "dòng; trong đó gian lận:", int(out.fraud_label.sum()))

# tỷ lệ gian lận theo cờ nhị phân
for c in ["night_tx", "new_payee"]:
    print(c, df.groupby(c).fraud_label.agg(["count", "sum", "mean"]).round(3).to_dict("index"))

# ---------------- 2. Kiểm tra leakage: tách lớp đơn biến ----------------
print("\nPR-AUC khi dùng từng feature đơn lẻ làm điểm xếp hạng:")
for c in num:
    print(f"  {c:12s} PR-AUC={average_precision_score(y, df[c]):.3f}  ROC-AUC={roc_auc_score(y, df[c]):.3f}")
for c in ["amount_vnd", "tx_count_1h", "risk_score"]:
    print(f"  {c}: max(hợp lệ)={df.loc[y==0,c].max()}  min(gian lận)={df.loc[y==1,c].min()}")

# tài khoản
acc = pd.concat([df.src_account, df.dst_account]).value_counts()
print("\nSố tài khoản khác nhau:", acc.size, "| xuất hiện 1 lần:", int((acc == 1).sum()))
fr = df[y == 1]
print("Tài khoản nguồn của gian lận:", sorted(fr.src_account), "đích:", sorted(fr.dst_account))

# ---------------- 3. Mô hình ----------------
log_amt = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
def lr_pipe(cols):
    pre = ColumnTransformer([
        ("amt", Pipeline([("log", log_amt), ("sc", StandardScaler())]), ["amount_vnd"]),
        ("num", StandardScaler(), [c for c in cols if c != "amount_vnd"]),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))])

FEATS = num                                   # KHÔNG dùng tx_id, src_account, dst_account
FEATS_NO_RS = [c for c in num if c != "risk_score"]
models = {
    "Dummy (prior)": (DummyClassifier(strategy="prior"), FEATS),
    "LR – đủ feature": (lr_pipe(FEATS), FEATS),
    "LR – bỏ risk_score": (lr_pipe(FEATS_NO_RS), FEATS_NO_RS),
    "RF – đủ feature": (RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=2,
                        class_weight="balanced_subsample", random_state=SEED), FEATS),
}

cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
rows = []
for name, (m, cols) in models.items():
    ap, roc, rk, acc_, cost, cms = [], [], [], [], [], np.zeros((2, 2), int)
    for tr, va in cv.split(df, y):
        Xtr, Xva = df.iloc[tr][cols], df.iloc[va][cols]
        m.fit(Xtr, y[tr])                     # mọi thứ học từ dữ liệu chỉ fit trên train
        p = m.predict_proba(Xva)[:, 1]
        yv = y[va]                            # validation giữ nguyên prevalence 10%
        ap.append(average_precision_score(yv, p)); roc.append(roc_auc_score(yv, p))
        k = max(1, int(np.ceil(TOP_FRAC * len(va))))
        top = np.argsort(-p, kind="stable")[:k]
        rk.append(yv[top].sum() / yv.sum())   # recall @ top 10%
        pred = (p >= 0.5).astype(int)
        acc_.append(accuracy_score(yv, pred))
        cm = confusion_matrix(yv, pred, labels=[0, 1]); cms += cm
        cost.append(COST_FN * cm[1, 0] + COST_FP * cm[0, 1])
    rows.append({"model": name, "PR-AUC": np.mean(ap), "PR-AUC sd": np.std(ap),
                 "ROC-AUC": np.mean(roc), "Recall@top10%": np.mean(rk),
                 "Accuracy@0.5": np.mean(acc_), "Cost/fold@0.5": np.mean(cost),
                 "TN": cms[0, 0], "FP": cms[0, 1], "FN": cms[1, 0], "TP": cms[1, 1]})
res = pd.DataFrame(rows)
print("\nKết quả RepeatedStratifiedKFold (3 fold × 30 lần, seed 42):")
print(res.round(3).to_string(index=False))
res.to_csv("b_preclass_results.csv", index=False)

# hệ số LR (fit toàn bộ Pre-class, chỉ để mô tả, không dùng để đánh giá)
lr = lr_pipe(FEATS).fit(df[FEATS], y)
coef = dict(zip(lr[:-1].get_feature_names_out(), lr[-1].coef_[0].round(2)))
print("\nHệ số LR (chuẩn hóa):", coef)

print("\nPhiên bản:", json.dumps({"python": platform.python_version(), "numpy": np.__version__,
      "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "seed": SEED}))
