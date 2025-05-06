import pandas as pd

# Đọc file CSV
df = pd.read_csv('housing_processed_clean.csv')  # Thay bằng tên file thực tế

# Kiểm tra tổng số giá trị NaN trên mỗi cột
nan_counts = df.isnull().sum()

# In ra các cột có NaN
print("Các cột có giá trị NaN:")
print(nan_counts[nan_counts > 0])
