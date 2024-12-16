# DỰ ĐOÁN GIÁ NHÀ SỬ DỤNG HỌC MÁY TRONG PYTHON

#### Các thư viện được sử dụng
* pandas: Dùng để thao tác với dữ liệu dưới dạng bảng (DataFrame).
* matplotlib: Dùng để vẽ đồ thị.
* seaborn: Dùng để vẽ đồ thị với giao diện thân thiện hơn matplotlib.
* openpyxl: Hỗ trợ đọc file Excel (.xlsx).
* scikit-learn: Dùng để tiền xử lý dữ liệu, chia tập dữ liệu, và các thuật toán học máy như RandomForestRegressor, LinearRegression, v.v.
* catboost: Thư viện machine learning hiệu suất cao cho các bài toán hồi quy và phân loại.

***Dùng lệnh này để cài đặt các thư viện trên:***
```pip install pandas matplotlib seaborn openpyxl scikit-learn catboost```

**Dataset:** HousePricePredition.xlsx

***Mô tả về dataset:***
* Id: Mã định danh (chỉ dùng để đếm, không có ý nghĩa trong dự đoán).
* MSSubClass: Loại nhà (vd: tầng lớp của tòa nhà trong ngữ cảnh phân vùng).
* MSZoning: Phân vùng sử dụng đất (vd: RL - khu dân cư với mật độ thấp).
* LotArea: Diện tích lô đất (tính bằng feet vuông).
* LotConfig: Cách bố trí của lô đất (vd: Inside - lô đất nằm bên trong, Corner - lô đất ở góc).
* BldgType: Loại nhà (vd: 1Fam - nhà dành cho một gia đình, 2fmCon - nhà cho hai hộ gia đình).
* OverallCond: Đánh giá tổng quan về tình trạng ngôi nhà (từ 1 - rất tệ, đến 10 - xuất sắc).
* YearBuilt: Năm xây dựng ban đầu của ngôi nhà.
* YearRemodAdd: Năm được sửa chữa hoặc cải tạo (nếu không có, sẽ giống YearBuilt).
* Exterior1st: Vật liệu bên ngoài của ngôi nhà (vd: VinylSd - Vinyl Siding, MetalSd - Metal Siding).
* BsmtFinSF2: Diện tích tầng hầm loại 2 (phần diện tích được hoàn thiện).
* TotalBsmtSF: Tổng diện tích tầng hầm (kể cả các phần chưa hoàn thiện).
* SalePrice: Giá bán nhà (đây là giá trị cần dự đoán).