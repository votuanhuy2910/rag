import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Thay đổi tên file Excel của bạn tại đây
file_path = 'metrics_grading_results.xlsx'

try:
    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(file_path)
    df.columns = ['ten_mon_hoc', 'filename', 'content_essays', 'diem_mo_hinh', 'diem_thuc_te']

    # Lấy dữ liệu điểm số
    diem_mo_hinh = df['diem_mo_hinh']
    diem_thuc_te = df['diem_thuc_te']

    # --- 1. Tính toán các chỉ số định lượng ---
    mae = mean_absolute_error(diem_thuc_te, diem_mo_hinh)
    mse = mean_squared_error(diem_thuc_te, diem_mo_hinh)
    rmse = np.sqrt(mse)
    r_squared = r2_score(diem_thuc_te, diem_mo_hinh)

    print("--- Kết quả Đánh giá Định lượng ---")
    print(f"1. Sai số trung bình tuyệt đối (MAE): {mae:.4f}")
    print(f"2. Sai số trung bình phương gốc (RMSE): {rmse:.4f}")
    print(f"3. Hệ số tương quan xác định (R-squared): {r_squared:.4f}")
    print("-------------------------------------")

    # --- 2. Kết luận dựa trên phân tích ---
    print("\n--- Phân tích Kết quả ---")
    if r_squared > 0.8:
        print("Mô hình có khả năng dự đoán tốt, giải thích được phần lớn sự biến thiên của điểm thực tế.")
    else:
        print("Mô hình cần cải thiện. R-squared thấp cho thấy mô hình chưa nắm bắt được mối quan hệ giữa nội dung và điểm số.")
    
    if mae < 1.0:
        print(f"Sai số MAE là {mae:.2f}, cho thấy trung bình mỗi bài chỉ chấm lệch dưới 1 điểm. Đây là một kết quả tốt.")
    else:
        print(f"Sai số MAE là {mae:.2f}, có thể cần cải thiện để giảm sai lệch trung bình.")
    
    print("\nĐể hiểu rõ hơn, hãy kiểm tra biểu đồ phân tán để xem các điểm sai lệch lớn nhất nằm ở đâu.")
    
    # --- 3. Trực quan hóa bằng biểu đồ ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Biểu đồ 1: Scatter Plot
    axes[0].scatter(diem_thuc_te, diem_mo_hinh, alpha=0.7, color='b')
    min_score = min(diem_thuc_te.min(), diem_mo_hinh.min())
    max_score = max(diem_thuc_te.max(), diem_mo_hinh.max())
    axes[0].plot([min_score, max_score], [min_score, max_score], 'r--', label='Đường lý tưởng (y=x)')
    axes[0].set_title('Biểu đồ phân tán điểm số')
    axes[0].set_xlabel('Điểm thực tế')
    axes[0].set_ylabel('Điểm mô hình')
    axes[0].grid(True)
    axes[0].legend()

    # Biểu đồ 2: Density Plot
    sns.kdeplot(x=diem_thuc_te, ax=axes[1], label='Điểm thực tế', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(x=diem_mo_hinh, ax=axes[1], label='Điểm mô hình', color='red', fill=True, alpha=0.5)
    axes[1].set_title('Phân bố điểm số')
    axes[1].set_xlabel('Điểm số')
    axes[1].set_ylabel('Mật độ')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Đánh giá toàn diện Hiệu suất Mô hình', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{file_path}'. Vui lòng kiểm tra lại tên file và đường dẫn.")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")