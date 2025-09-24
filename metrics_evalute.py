import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, cohen_kappa_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

# === Load dữ liệu ===
df = pd.read_excel("evalute_grading_results_metrics.xlsx")  # hoặc pd.read_csv("scores.csv")

# === Làm tròn điểm để tính accuracy ===
y_true_int = df["diem_thuc_te"].round().astype(int)
y_pred_noRAG_int = df["diem_mo_hinh_norag"].round().astype(int)
y_pred_RAG_int = df["diem_mo_hinh_rag"].round().astype(int)

# Accuracy
acc_noRAG = accuracy_score(y_true_int, y_pred_noRAG_int)
acc_RAG = accuracy_score(y_true_int, y_pred_RAG_int)

# === Tính MAE, MSE, RMSE, R2, QWK (Quadratic Weighted Kappa), Pearson Correlation, Spearman correlation ===
mae_noRAG = mean_absolute_error(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
mae_RAG = mean_absolute_error(df["diem_thuc_te"], df["diem_mo_hinh_rag"])

mse_noRAG = mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
mse_RAG = mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_rag"])

rmse_noRAG = root_mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
rmse_RAG = root_mean_squared_error(df["diem_thuc_te"], df["diem_mo_hinh_rag"])

r2_noRAG = r2_score(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
r2_RAG = r2_score(df["diem_thuc_te"], df["diem_mo_hinh_rag"])

y_true = df["diem_thuc_te"].round().astype(int)
y_pred_noRAG = df["diem_mo_hinh_norag"].round().astype(int)
y_pred_RAG = df["diem_mo_hinh_rag"].round().astype(int)

qwk_noRAG = cohen_kappa_score(y_true, y_pred_noRAG, weights="quadratic")
qwk_RAG = cohen_kappa_score(y_true, y_pred_RAG, weights="quadratic")

pearson_noRAG, _ = pearsonr(df["diem_thuc_te"], df["diem_mo_hinh_norag"])
pearson_RAG, _ = pearsonr(df["diem_thuc_te"], df["diem_mo_hinh_rag"])

# === In kết quả ===
print("=== Kết quả so sánh ===")
print("=== Đánh giá độ chính xác & tương đồng ===")
print(f"MAE (No RAG): {mae_noRAG:.3f} | MAE (RAG): {mae_RAG:.3f}") # Nếu MAE/RMSE của RAG thấp hơn → RAG tốt hơn.
print(f"MSE (No RAG): {mse_noRAG:.3f} | MSE (RAG): {mse_RAG:.3f}") # Nếu MSE của RAG thấp hơn → RAG tốt hơn.
print(f"RMSE (No RAG): {rmse_noRAG:.3f} | RMSE (RAG): {rmse_RAG:.3f}") # Nếu MAE/RMSE của RAG thấp hơn → RAG tốt hơn.
print(f"QWK (No RAG): {qwk_noRAG:.3f} | QWK (RAG): {qwk_RAG:.3f}") # QWK: giá trị từ -1 đến 1, càng gần 1 càng tốt

# === Bar Chart: so sánh metrics ===
# Tạo DataFrame chỉ gồm Accuracy (%) và QWK
acc_qwk_df = pd.DataFrame({
    "Metric": ["Accuracy", "QWK"],
    "No RAG": [acc_noRAG, qwk_noRAG],
    "RAG": [acc_RAG, qwk_RAG]
})

# Vẽ Bar Chart
ax = acc_qwk_df.set_index("Metric").plot(
    kind="bar", 
    figsize=(7,6), 
    color=["skyblue", "lightgreen"]
)

plt.title("So sánh Accuracy và QWK giữa RAG và No RAG")
plt.ylabel("Giá trị")
plt.xticks(rotation=0)

# Hiển thị giá trị trên cột
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R²", "Pearson Corr"],
    "No RAG": [mae_noRAG, mse_noRAG, rmse_noRAG, r2_noRAG, pearson_noRAG],
    "RAG": [mae_RAG, mse_RAG, rmse_RAG, r2_RAG, pearson_RAG]
})
# === Vẽ Bar Chart với giá trị trên cột ===
ax = metrics.set_index("Metric").plot(kind="bar", figsize=(9,6))
plt.title("So sánh Metrics: RAG vs No RAG")
plt.ylabel("Giá trị")
plt.xticks(rotation=0)

# Thêm số lên cột
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9)

plt.show()

# === Scatter Plot: điểm thật vs dự đoán ===
plt.figure(figsize=(8,6))
plt.scatter(df["diem_thuc_te"], df["diem_mo_hinh_norag"], alpha=0.6, label="No RAG", color="red")
plt.scatter(df["diem_thuc_te"], df["diem_mo_hinh_rag"], alpha=0.6, label="RAG", color="blue")
plt.plot([df["diem_thuc_te"].min(), df["diem_thuc_te"].max()],
            [df["diem_thuc_te"].min(), df["diem_thuc_te"].max()],
            color="green", linestyle="--", label="Perfect prediction")
plt.xlabel("Điểm thật")
plt.ylabel("Điểm mô hình")
plt.title("So sánh dự đoán: RAG vs No RAG")
plt.legend()
plt.show()

# === Boxplot: phân phối sai số ===
errors = pd.DataFrame({
    "No RAG": df["diem_mo_hinh_norag"] - df["diem_thuc_te"],
    "RAG": df["diem_mo_hinh_rag"] - df["diem_thuc_te"]
})

errors.plot(kind="box", figsize=(8,6))
plt.axhline(0, color="green", linestyle="--")
plt.title("Phân phối sai số: RAG vs No RAG")
plt.ylabel("Sai số (dự đoán - thực tế)")
plt.show()

# === Density Plot: so sánh phân phối điểm ===
plt.figure(figsize=(9,6))

sns.kdeplot(y_true, label="Điểm Thật", fill=True, alpha=0.4, linewidth=2)
sns.kdeplot(y_pred_noRAG, label="No RAG", fill=True, alpha=0.4, linewidth=2)
sns.kdeplot(y_pred_RAG, label="RAG", fill=True, alpha=0.4, linewidth=2)

plt.title("Density Plot: So sánh phân phối điểm", fontsize=14)
plt.xlabel("Điểm")
plt.ylabel("Mật độ")
plt.legend()
plt.tight_layout()
plt.show()

# === Heatmap ===
plt.figure(figsize=(6,4))
sns.heatmap(metrics.set_index("Metric"), annot=True, fmt=".3f", cmap="coolwarm", cbar=True)
plt.title("Heatmap so sánh Metrics: RAG vs No RAG")
plt.show()