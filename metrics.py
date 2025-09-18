import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from scipy.stats import pearsonr

def evaluate_essay_scoring(y_true, y_pred):
    """
    Đánh giá hiệu suất của mô hình chấm điểm bài luận.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Làm tròn điểm dự đoán và điểm thực tế để tính QWK
    y_true_int = np.round(y_true).astype(int)
    y_pred_int = np.round(y_pred).astype(int)
    qwk = cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')
    
    pearson_corr, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    evaluation_metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "QWK": qwk,
        "Pearson Correlation": pearson_corr,
        "R-squared (R^2)": r2
    }
    return evaluation_metrics

def analyze_metrics(metrics):
    """
    Phân tích và đưa ra nhận xét chi tiết về các chỉ số đánh giá.
    """
    print("\n--- Phân tích chi tiết các chỉ số đánh giá ---")
    
    mae = metrics['MAE']
    print(f"MAE (Sai số tuyệt đối trung bình): {mae:.4f}")
    if mae < 0.5:
        print("- Đánh giá: Rất tốt.")
    elif mae <= 1.5:
        print("- Đánh giá: Chấp nhận được.")
    else:
        print("- Đánh giá: Cần cải thiện.")
    
    mse = metrics['MSE']
    print(f"\nMSE (Sai số bình phương trung bình): {mse:.4f}")
    if mse < 1.0:
        print("- Đánh giá: Rất tốt.")
    elif mse <= 3.0:
        print("- Đánh giá: Chấp nhận được.")
    else:
        print("- Đánh giá: Cần cải thiện.")

    rmse = metrics['RMSE']
    print(f"\nRMSE (Căn bậc hai của sai số bình phương trung bình): {rmse:.4f}")
    if rmse < 0.7:
        print("- Đánh giá: Rất tốt.")
    elif rmse <= 2.0:
        print("- Đánh giá: Chấp nhận được.")
    else:
        print("- Đánh giá: Cần cải thiện.")
        
    qwk = metrics['QWK']
    print(f"\nQWK (Hệ số Kappa bình phương): {qwk:.4f}")
    if qwk > 0.8:
        print("- Đánh giá: Xuất sắc, mức độ đồng thuận rất cao.")
    elif qwk > 0.6:
        print("- Đánh giá: Rất tốt, mức độ đồng thuận đáng kể.")
    elif qwk > 0.4:
        print("- Đánh giá: Chấp nhận được, mức độ đồng thuận vừa phải.")
    else:
        print("- Đánh giá: Kém, mức độ đồng thuận thấp.")
        
    pearson_corr = metrics['Pearson Correlation']
    print(f"\nPearson Correlation (Hệ số tương quan Pearson): {pearson_corr:.4f}")
    if pearson_corr > 0.9:
        print("- Đánh giá: Rất tốt.")
    elif pearson_corr > 0.7:
        print("- Đánh giá: Chấp nhận được.")
    else:
        print("- Đánh giá: Cần cải thiện.")
        
    r2 = metrics['R-squared (R^2)']
    print(f"\nR-squared (R^2): {r2:.4f}")
    if r2 > 0.8:
        print("- Đánh giá: Rất tốt.")
    elif r2 > 0.6:
        print("- Đánh giá: Chấp nhận được.")
    else:
        print("- Đánh giá: Cần cải thiện.")

def plot_evaluation_charts_separate(y_true, y_pred, metrics):
    """
    Vẽ các biểu đồ để trực quan hóa kết quả đánh giá mô hình, mỗi biểu đồ
    sẽ nằm trong một cửa sổ riêng.
    """
    # --- Biểu đồ 1: Biểu đồ phân tán (Scatter Plot) ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.title('Biểu đồ phân tán: Điểm thực tế vs Điểm dự đoán')
    plt.xlabel('Điểm thực tế')
    plt.ylabel('Điểm dự đoán')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Điểm lý tưởng (y=x)')
    plt.legend()
    plt.show()

    # --- Biểu đồ 2: Density Plot (Phân bố điểm số) ---
    plt.figure(figsize=(10, 7))
    sns.kdeplot(x=y_true, label='Điểm thực tế', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(x=y_pred, label='Điểm mô hình', color='red', fill=True, alpha=0.5)
    plt.title('Phân bố điểm số')
    plt.xlabel('Điểm số')
    plt.ylabel('Mật độ')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Biểu đồ 3: Biểu đồ phân phối sai số (Error Distribution) ---
    plt.figure(figsize=(10, 7))
    errors = y_pred - y_true
    sns.histplot(errors, bins=20, kde=True)
    plt.title('Phân bố sai số (Error Distribution)')
    plt.xlabel('Sai số (Điểm dự đoán - Điểm thực tế)')
    plt.ylabel('Tần suất')
    plt.axvline(x=0, color='red', linestyle='--', label='Sai số = 0')
    plt.legend()
    plt.show()
    
    # --- Biểu đồ 4: Biểu đồ cột cho các chỉ số đánh giá ---
    plt.figure(figsize=(12, 7))
    metric_names = ['MAE', 'RMSE', 'QWK', 'Pearson Correlation', 'R-squared (R^2)']
    metric_values = [metrics[name] for name in metric_names]
    
    ax = sns.barplot(x=metric_names, y=metric_values, palette='viridis')
    plt.title('Các chỉ số đánh giá hiệu suất mô hình')
    plt.ylabel('Giá trị')
    plt.xlabel('Chỉ số')
    plt.xticks(rotation=30)
    
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# --- Phần chính: Đọc dữ liệu từ file và gọi hàm đánh giá ---
file_path = 'grading_results_metrics.xlsx'

try:
    df = pd.read_excel(file_path)
    scores_predicted = df['diem_mo_hinh'].values
    scores_actual = df['diem_thuc_te'].values

    if len(scores_predicted) == 0 or len(scores_actual) == 0:
        print("Lỗi: File Excel không có dữ liệu hoặc tên cột không đúng.")
    else:
        metrics = evaluate_essay_scoring(scores_actual, scores_predicted)
        
        print("--- Kết quả đánh giá mô hình ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        analyze_metrics(metrics)
        plot_evaluation_charts_separate(scores_actual, scores_predicted, metrics)

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn: {file_path}")
except KeyError as e:
    print(f"Lỗi: Không tìm thấy tên cột {e} trong file CSV. Vui lòng kiểm tra lại tên cột 'diem_mo_hinh' và 'diem_thuc_te'.")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")