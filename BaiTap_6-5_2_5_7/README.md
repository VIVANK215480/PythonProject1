# Phân Tích Dự Đoán Giá Nhà(bài 2)

## Tổng Quan
Dự án này phân tích dữ liệu giá nhà (lưu trong `housing_processed_clean.csv`) để dự đoán giá trị nhà trung bình (`MEDV`) bằng các mô hình học máy. Code ban đầu gặp vấn đề **overfitting nghiêm trọng** với các chỉ số như MSE ≈ 0 và R² ≈ 1.0. Phiên bản cập nhật này khắc phục overfitting bằng cách kiểm tra dữ liệu, chọn đặc trưng, thêm điều chuẩn (regularization), và sử dụng cross-validation.

Các bước phân tích bao gồm:
- Tiền xử lý dữ liệu (kiểm tra rò rỉ dữ liệu, trùng lặp, và tương quan cao).
- Chọn đặc trưng bằng `SelectKBest`.
- Huấn luyện mô hình với Ridge Regression, Gradient Boosting, Neural Network, và Stacking.
- Tối ưu siêu tham số cho Gradient Boosting bằng Optuna.
- Đánh giá mô hình bằng MSE, RMSE, R², MAPE, và MSE cross-validation.
- Vẽ biểu đồ residual và phân tích SHAP để giải thích.
- So sánh hiệu suất mô hình bằng biểu đồ.

## Yêu Cầu
- **Python**: 3.8 trở lên
- **Thư viện**:
  - pandas
  - numpy
  - scikit-learn
  - optuna
  - shap
  - matplotlib
  - seaborn

Cài đặt thư viện:
```bash
pip install pandas numpy scikit-learn optuna shap matplotlib seaborn
```

## Dữ Liệu
- **Tệp**: `housing_processed_clean.csv`
- **Các cột** (giả định):
  - `MEDV`: Biến mục tiêu (giá trị nhà trung bình).
  - `is_outlier`: Chỉ báo ngoại lai (tùy chọn, có thể gây rò rỉ dữ liệu).
  - Các đặc trưng khác: Các đặc trưng số liên quan đến nhà ở (ví dụ: số phòng, tỷ lệ tội phạm).
- **Lưu ý**: Đảm bảo dữ liệu không chứa đặc trưng được suy ra trực tiếp từ `MEDV` để tránh rò rỉ dữ liệu.

## Hướng Dẫn Sử Dụng
1. **Đặt tệp dữ liệu**:
   - Đảm bảo `housing_processed_clean.csv` nằm trong cùng thư mục với script.

2. **Chạy script**:
   ```bash
   python Housing_Analysis_Improved.py
   ```

3. **Kết quả đầu ra**:
   - **Console**:
     - Kích thước dữ liệu và danh sách cột.
     - Tương quan của các đặc trưng với `MEDV`.
     - Số hàng trùng lặp.
     - Các đặc trưng được chọn.
     - Kết quả tối ưu siêu tham số.
     - Chỉ số đánh giá mô hình (MSE, RMSE, R², MAPE, CV_MSE, CV_MSE_STD).
     - 10 đặc trưng quan trọng nhất.
   - **Tệp**:
     - `residual_plot.png`: Biểu đồ residual cho Gradient Boosting.
     - `shap_importance.png`: Biểu đồ tầm quan trọng đặc trưng SHAP.
     - `model_evaluation_results.csv`: Chỉ số hiệu suất mô hình.
     - `feature_importance.csv`: Tầm quan trọng đặc trưng.
     - `model_comparison.png`: Biểu đồ so sánh RMSE giữa các mô hình.

## Phương Pháp
1. **Kiểm Tra Dữ Liệu**:
   - Xóa các hàng trùng lặp.
   - Loại bỏ đặc trưng có tương quan > 0.95 với `MEDV` để tránh rò rỉ dữ liệu.
   - Kiểm tra tính toàn vẹn dữ liệu (kích thước, cột).

2. **Chọn Đặc Trưng**:
   - Sử dụng `SelectKBest` với `f_regression` để chọn 10 đặc trưng quan trọng nhất.

3. **Huấn Luyện Mô Hình**:
   - **Ridge Regression**: Mô hình tuyến tính với điều chuẩn L2 (`alpha=1.0`).
   - **Gradient Boosting**: Tối ưu bằng Optuna, giới hạn độ phức tạp (số cây, độ sâu, v.v.).
   - **Neural Network**: Kiến trúc đơn giản với điều chuẩn L2 và dừng vroeg.
   - **Stacking**: Kết hợp Ridge, Gradient Boosting, và Neural Network, với Ridge làm meta-learner.

4. **Tối Ưu Siêu Tham Số**:
   - Optuna tối ưu các tham số Gradient Boosting (`n_estimators`, `learning_rate`, `max_depth`, v.v.) để giảm thiểu MSE cross-validation.

5. **Đánh Giá**:
   - Chỉ số: MSE, RMSE, R², MAPE trên tập kiểm tra.
   - MSE cross-validation (`CV_MSE`) để đánh giá khả năng khái quát hóa.
   - Biểu đồ residual để trực quan hóa sai số dự đoán.
   - Phân tích SHAP để giải thích đóng góp của đặc trưng.

6. **Khắc Phục Overfitting**:
   - Loại bỏ đặc trưng tương quan cao để tránh rò rỉ dữ liệu.
   - Áp dụng điều chuẩn (Ridge, L2 trong Neural Network).
   - Giới hạn độ phức tạp mô hình (mạng nhỏ hơn, cây nông hơn).
   - Sử dụng chọn đặc trưng để giảm số chiều.
   - Đảm bảo chia dữ liệu huấn luyện-kiểm tra ngẫu nhiên với trộn dữ liệu.

## Kết Quả Đánh Giá
Dưới đây là kết quả đánh giá hiệu suất các mô hình trên tập kiểm tra và cross-validation:

### Ridge Regression
- **MSE**: 1.2272
- **RMSE**: 1.1078
- **R²**: 0.9770
- **MAPE**: 0.0397
- **CV_MSE**: 1.5220
- **CV_MSE_STD**: 0.3216

### Gradient Boosting
- **MSE**: 1.1178
- **RMSE**: 1.0573
- **R²**: 0.9790
- **MAPE**: 0.0352
- **CV_MSE**: 2.3403
- **CV_MSE_STD**: 0.9425

### Neural Network
- **MSE**: 0.8938
- **RMSE**: 0.9454
- **R²**: 0.9832
- **MAPE**: 0.0336
- **CV_MSE**: 1.8957
- **CV_MSE_STD**: 0.5357

### Stacking
- **MSE**: 0.8362
- **RMSE**: 0.9144
- **R²**: 0.9843
- **MAPE**: 0.0340
- **CV_MSE**: 1.4140
- **CV_MSE_STD**: 0.3859

### Nhận Xét Kết Quả
- **Hiệu suất tổng thể**:
  - Tất cả các mô hình đạt hiệu suất cao với R² > 0.97, cho thấy chúng giải thích tốt biến động của `MEDV`. Stacking hoạt động tốt nhất với MSE thấp nhất (0.8362) và R² cao nhất (0.9843), nhờ kết hợp ưu điểm của Ridge, Gradient Boosting, và Neural Network.
  - Neural Network có MSE thấp (0.8938) và MAPE thấp nhất (0.0336), cho thấy sai số tương đối nhỏ, nhưng CV_MSE (1.8957) cao hơn MSE, gợi ý một chút overfitting.
  - Gradient Boosting có MSE thấp (1.1178) nhưng CV_MSE cao (2.3403) và CV_MSE_STD lớn (0.9425), cho thấy mô hình có thể không ổn định trên các fold cross-validation, có khả năng bị overfitting nhẹ.
  - Ridge Regression có hiệu suất ổn định với CV_MSE (1.5220) gần MSE (1.2272) và CV_MSE_STD thấp (0.3216), cho thấy khả năng khái quát hóa tốt nhưng kém hơn Stacking về MSE.

- **Overfitting**:
  - So với kết quả ban đầu (MSE ≈ 0, R² ≈ 1.0), các mô hình hiện tại đã giảm đáng kể overfitting nhờ điều chuẩn, chọn đặc trưng, và cross-validation. Tuy nhiên, Gradient Boosting và Neural Network vẫn có dấu hiệu overfitting nhẹ do CV_MSE cao hơn MSE.
  - Stacking và Ridge Regression cho thấy khả năng khái quát hóa tốt hơn, với khoảng cách giữa MSE và CV_MSE nhỏ.

- **Đề xuất cải thiện**:
  - Tăng điều chuẩn cho Gradient Boosting (ví dụ: giảm `max_depth`, tăng `min_samples_split`) hoặc Neural Network (tăng `alpha`).
  - Giảm số đặc trưng trong `SelectKBest` (ví dụ: `k=5`) để đơn giản hóa mô hình.
  - Kiểm tra lại `housing_processed_clean.csv` để đảm bảo không có rò rỉ dữ liệu (ví dụ: đặc trưng `is_outlier` suy ra từ `MEDV`).
  - Thêm nhiễu nhỏ vào dữ liệu để tăng tính đa dạng:
    ```python
    X += np.random.normal(0, 0.01, X.shape)
    ```

## Kết Quả Mong Đợi
- **Chỉ số**: Kỳ vọng:
  - MSE: ~0.8-2.0.
  - RMSE: ~0.9-1.4.
  - R²: ~0.95-0.98.
  - MAPE: ~0.03-0.05.
  - `CV_MSE` gần với `MSE`, cho thấy mô hình khái quát hóa tốt.
- **Biểu đồ**:
  - Biểu đồ residual nên có phân bố ngẫu nhiên quanh 0.
  - Biểu đồ SHAP làm nổi bật các đặc trưng chính (ví dụ: số phòng, đặc trưng vị trí).
  - Biểu đồ so sánh mô hình cho thấy Stacking hoặc Gradient Boosting thường hoạt động tốt nhất.

## Khắc Phục Sự Cố
- **Overfitting vẫn xảy ra**:
  - Kiểm tra `housing_processed_clean.csv` xem có rò rỉ dữ liệu không (ví dụ: đặc trưng suy ra từ `MEDV`).
  - Tăng `alpha` trong Ridge (ví dụ: `alpha=10.0`).
  - Giảm `k` trong `SelectKBest` (ví dụ: `k=5`).
  - Thêm nhiễu vào dữ liệu:
    ```python
    X += np.random.normal(0, 0.01, X.shape)
    ```
- **Lỗi**:
  - Thiếu cột: Kiểm tra tên cột trong `housing_processed_clean.csv`.
  - Vấn đề thư viện: Cập nhật thư viện (`pip install --upgrade matplotlib seaborn`).
- **Vấn đề dữ liệu**:
  - Nếu dữ liệu quá nhỏ hoặc quá sạch, sử dụng tập dữ liệu chuẩn (ví dụ: California Housing từ `sklearn.datasets.fetch_california_housing`).

## Tệp
- **Đầu vào**:
  - `housing_processed_clean.csv`: Dữ liệu nhà ở đã xử lý.
- **Đầu ra**:
  - `residual_plot.png`: Biểu đồ residual.
  - `shap_importance.png`: Tầm quan trọng đặc trưng SHAP.
  - `model_evaluation_results.csv`: Chỉ số hiệu suất mô hình.
  - `feature_importance.csv`: Tầm quan trọng đặc trưng.
  - `model_comparison.png`: So sánh RMSE giữa các mô hình.
- **Script**:
  - `Housing_Analysis_Improved.py`: Script phân tích chính.



# Phân tích dữ liệu giáo dục và y tế

Dự án này bao gồm hai phần phân tích dữ liệu riêng biệt:

## bài 5: Phân tích dữ liệu giáo dục (Data_Number_5.csv)

### Tính năng chính:
1. Tính toán chỉ số hiệu suất học tập tổng hợp
2. Phân tích ảnh hưởng của hoạt động ngoại khóa
3. Đánh giá cân bằng học tập
4. Phân tích rủi ro học tập
5. Xây dựng mô hình dự đoán SVM

### Các chỉ số được tính toán:
1. **Hiệu suất học tập tổng hợp**: 
   - Kết hợp điểm các môn học (trọng số: Toán 30%, Văn 30%, Khoa học 40%)
   - Thêm bonus tối đa 20% dựa trên số giờ tự học

2. **Cân bằng học tập**: 
   - Độ lệch chuẩn của điểm các môn học
   - Giá trị càng thấp càng thể hiện sự cân bằng tốt

3. **Rủi ro học tập**: 
   - Kết hợp số buổi vắng mặt và số giờ tự học
   - Chuẩn hóa và lấy trung bình

## bài 7: Phân tích dữ liệu y tế (Data_Number7.csv)

### Tính năng chính:
1. Tính toán chỉ số nguy cơ biến chứng
2. Phân tích mối liên hệ giữa tuổi và biến chứng
3. Đánh giá xu hướng đường huyết
4. Phân tích mức độ nghiêm trọng
5. Xây dựng mô hình dự đoán biến chứng
6. Xử lý mất cân bằng dữ liệu bằng SMOTE

### Các chỉ số được tính toán:
1. **Nguy cơ biến chứng**: 
   - Kết hợp BMI (40%), đường huyết (40%), và số lần nhập viện (20%)
   - Chuẩn hóa về thang đo [0,1]

2. **Xu hướng đường huyết**: 
   - Phân loại thành 3 mức: tăng (>180), giảm (<70), ổn định (70-180)
   - Dựa trên mức đường huyết hiện tại

3. **Mức độ nghiêm trọng**: 
   - Kết hợp đường huyết (60%) và số lần nhập viện (40%)
   - Chuẩn hóa về thang đo [0,1]

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Đảm bảo các file dữ liệu nằm trong cùng thư mục với script:
   - `Data_Number_5.csv` cho phân tích giáo dục
   - `Data_Number7.csv` cho phân tích y tế

## Chạy phân tích

### Phân tích dữ liệu giáo dục:
```bash
python analyze_education.py
```

### Phân tích dữ liệu y tế:
```bash
python analyze_health.py
```

## Kết quả

### Phân tích giáo dục:
1. Các thống kê mô tả về dữ liệu
2. Kết quả kiểm định ANOVA
3. Kết quả mô hình SVM
4. File ảnh `education_analysis.png`

### Phân tích y tế:
1. Các thống kê mô tả về dữ liệu
2. Kết quả kiểm định Chi-squared
3. Kết quả mô hình Logistic Regression và Random Forest
4. File ảnh `health_analysis.png`

## Xử lý mất cân bằng dữ liệu (Phần y tế)

Dự án sử dụng kỹ thuật SMOTE (Synthetic Minority Over-sampling Technique) để:
- Tạo mẫu tổng hợp cho lớp thiểu số
- Cân bằng tỷ lệ giữa các lớp
- Cải thiện hiệu suất mô hình dự đoán

## Mô hình dự đoán

### Phân tích giáo dục:
- **SVM (Support Vector Machine)**:
  - Phân loại sinh viên thành 2 nhóm
  - Tự động tối ưu siêu tham số
  - Xử lý tốt dữ liệu phi tuyến

### Phân tích y tế:
1. **Logistic Regression**:
   - Mô hình cơ bản để dự đoán biến chứng
   - Dễ giải thích và triển khai

2. **Random Forest**:
   - Mô hình nâng cao với khả năng học phi tuyến
   - Tự động tối ưu siêu tham số
   - Xử lý tốt dữ liệu nhiễu 

### Kiểm định Chi-squared:
p-value = 0.1037 > 0.05
Kết luận: Biến chứng không có mối liên hệ đáng kể với nhóm tuổi
Mô hình Logistic Regression:
Độ chính xác cho lớp 0 (không biến chứng): 65%
Độ chính xác cho lớp 1 (có biến chứng): 32%
Hiệu suất khá thấp, đặc biệt là với việc dự đoán biến chứng
Mô hình Random Forest:
Tham số tối ưu: max_depth=20, min_samples_split=2, n_estimators=300
Độ chính xác cross-validation: 75.09%
Hiệu suất trên tập test:
Độ chính xác cho lớp 0: 68%
Độ chính xác cho lớp 1: 37%
Hiệu suất tổng thể: 59%
Thống kê mô tả:
Nguy cơ biến chứng trung bình: 0.57 (độ lệch chuẩn: 0.25)
Mức độ nghiêm trọng trung bình: 0.61 (độ lệch chuẩn: 0.30)