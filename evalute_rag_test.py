import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, cohen_kappa_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

# === Load dữ liệu ===
df = pd.read_excel("grading_results_rag_test_metrics.xlsx")  # hoặc pd.read_csv("scores.csv")

# === Làm tròn điểm để tính accuracy ===
y_true_int = df["diem_thuc_te"].round().astype(int)
# y_pred_noRAG_int = df["diem_mo_hinh_norag"].round().astype(int)
y_pred_RAG_int = df["diem_mo_hinh_rag_test"].round().astype(int)


# === Tính MAE, MSE, RMSE, R2, QWK (Quadratic Weighted Kappa) ===
# mae_noRAG = mean_absolute_error(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
mae_RAG = mean_absolute_error(df["diem_thuc_te"], df["diem_mo_hinh_rag_test"])

# mse_noRAG = mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
mse_RAG = mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_rag_test"])

# rmse_noRAG = root_mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
rmse_RAG = root_mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_rag_test"])

y_true = df["diem_thuc_te"].round().astype(int)
# y_pred_noRAG = df["diem_mo_hinh_norag"].round().astype(int)
y_pred_RAG = df["diem_mo_hinh_rag_test"].round().astype(int)

# qwk_noRAG = cohen_kappa_score(y_true, y_pred_noRAG, weights="quadratic")
qwk_RAG = cohen_kappa_score(y_true, y_pred_RAG, weights="quadratic")

# === In kết quả ===
print("=== Kết quả so sánh ===")
print("=== Đánh giá độ tương đồng ===")
print(f"MAE (RAG_TEST): {mae_RAG:.3f}") # Nếu MAE/RMSE của RAG thấp hơn → RAG tốt hơn.
print(f"MSE (RAG_TEST): {mse_RAG:.3f}") # Nếu MSE của RAG thấp hơn → RAG tốt hơn.
print(f"RMSE (RAG_TEST): {rmse_RAG:.3f}") # Nếu MAE/RMSE của RAG thấp hơn → RAG tốt hơn.
print(f"QWK (RAG_TEST): {qwk_RAG:.3f}") # QWK: giá trị từ -1 đến 1, càng gần 1 càng tốt