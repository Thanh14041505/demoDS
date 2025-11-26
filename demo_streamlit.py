import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from scipy import stats
import pandas as pd

# Using menu
st.title("Trung Tâm Tin Học")
st.image("xe_may_cu.jpg", caption="Xe máy cũ")
menu = ["Home", "Capstone Project", "Sử dụng các điều khiển", "Gợi ý điều khiển project 1", "Gợi ý điều khiển project 2"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""### Có 2 chủ đề trong khóa học:    
    - Topic 1: Dự đoán giá xe máy cũ, phát hiện xe máy bất thường
    - Topic 2: Hệ thống gợi ý xe máy dựa trên nội dung, phân cụm xe máy
             """)

elif choice == 'Sử dụng các điều khiển':
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)
    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")
    # 3. Checkbox
    st.subheader("3. Checkbox")
    if st.checkbox("I agree"):
        st.write("Great!")
    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive"))
    st.write("You are", status)
    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox("What is your occupation?", ["Student", "Teacher", "Others"])
    st.write("You are a", occupation)
    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect("Where do you live?", ("Hanoi", "HCM", "Danang", "Hue"))
    st.write("You live in", location)
    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)    
    # 9. Date Input
    st.subheader("9. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)
    # 10. Time Input
    st.subheader("10. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)
    # 11. Display JSON
    st.subheader("11. Display JSON")
    json = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json)
    # 12. Display Raw Code
    st.subheader("12. Display Raw Code")
    code = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code)
    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json)
        st.write("You entered", code)
          
elif choice == 'Gợi ý điều khiển project 1':
    st.write("##### Project 1: Dự đoán giá xe cũ và phát hiện xe bất thường")
    st.write("##### Dữ liệu mẫu")

    # đọc dữ liệu từ file data_motobikes.xlsx
    # Trường hợp 1: Đọc dữ liệu từ file mẫu có sẵn
    st.write("### Đọc dữ liệu từ file mẫu có sẵn")
    DATA_PATH = "./data_motobikes.xlsx"
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    st.dataframe(df.head())

    # Trường hợp 2: Đọc dữ liệu từ file csv hoặc excel do người dùng tải lên
    st.write("### Đọc dữ liệu từ file csv hoặc excel do người dùng tải lên")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.write("Dữ liệu đã nhập:")
        st.dataframe(df.head())

    st.write("### 1. Dự đoán giá xe cũ")
    # Tạo điều khiển để người dùng nhập các thông tin về xe
    thuong_hieu = st.selectbox("Chọn hãng xe", df['Thương hiệu'].unique())
    dong_xe = st.selectbox("Chọn dòng xe", df['Dòng xe'].unique())
    tinh_trang = st.selectbox("Chọn tình trạng", df['Tình trạng'].unique())
    loai_xe = st.selectbox("Chọn loại xe", df['Loại xe'].unique())
    dung_tich_xe = st.selectbox("Dung tích xe (cc)", df['Dung tích xe'].unique())
    xuat_xu = st.selectbox("Chọn xuất xứ", df['Xuất xứ'].unique())
    nam_dang_ky = st.slider("Năm đăng ký", 1980, 2025, 2015)
    so_km_da_di = st.number_input("Số km đã đi", min_value=0, max_value=200000, value=50000, step=1000)
    du_doan_gia = st.button("Dự đoán giá")
    model = joblib.load('car_price_gbr_pipeline.pkl')  # Sử dụng joblib.load
    if du_doan_gia:
        # In ra các thông tin đã chọn
        st.write("Hãng xe:", thuong_hieu)
        st.write("Dòng xe:", dong_xe)
        st.write("Tình trạng:", tinh_trang)
        st.write("Loại xe:", loai_xe)
        st.write("Dung tích xe (cc):", dung_tich_xe)
        st.write("Xuất xứ:", xuat_xu)
        st.write("Năm đăng ký:", nam_dang_ky)
        st.write("Số km đã đi:", so_km_da_di)
        #dự đoán
        
        input_data = pd.DataFrame({
            'Thương hiệu': [thuong_hieu],
            'Dòng xe': [dong_xe],
            'Tình trạng': [tinh_trang],
            'Loại xe': [loai_xe],
            'Dung tích xe': [dung_tich_xe],
            'Xuất xứ': [xuat_xu],
            'Năm đăng ký': [nam_dang_ky],
            'Số Km đã đi': [so_km_da_di]
        })
        gia_du_doan = model.predict(input_data)[0]
        st.write("Giá dự đoán:", gia_du_doan)

    def detect_residual_anomaly(df, model, residual_threshold=10_000_000):
        """
        df: DataFrame phải có đủ input để predict + cột 'Giá'
        model: mô hình predict giá
        residual_threshold: ngưỡng VND để xác định bất thường

        Trả về DataFrame có thêm:
            - pred_price: giá dự đoán
            - residual: phần dư
            - anomaly_resid: 'Bất thường' hoặc 'Bình thường'
        """
        df_pred = df.copy()

        # ---- 1. Predict giá ----
        X = df_pred.drop(columns=["Giá"])
        df_pred["pred_price"] = model.predict(X)

        # ---- 2. Residual ----
        df_pred["residual"] = df_pred["Giá"] - df_pred["pred_price"]

        # ---- 3. Đánh dấu bất thường dựa trên threshold ----
        df_pred["anomaly_resid"] = df_pred["residual"].apply(
            lambda r: "Bất thường" if abs(r) > residual_threshold else "Bình thường"
        )

        # ---- 4. Không dùng z-score cho 1 xe ----
        df_pred["z_resid"] = None  # chỉ để giữ column, không dùng

        return df_pred
    # Làm tiếp cho phần phát hiện xe bất thường
    st.write("### 2. Phát hiện ô tô bất thường dựa trên Residual của mô hình")

    so_km_bat_thuong = st.number_input("Số km đã đi", min_value=0, max_value=300000, value=50000)
    gia_thuc_te = st.number_input("Giá thực tế (VND)", min_value=0, max_value=500000000, value=150000000)
    btn_check = st.button("Kiểm tra bất thường")

    if btn_check:
        # -----------------------
        # 1. Tạo 1 data point để predict
        # -----------------------
        input_row = {
            "Thương hiệu": thuong_hieu,
            "Dòng xe": dong_xe,
            "Tình trạng": tinh_trang,
            "Loại xe": loai_xe,
            "Dung tích xe": dung_tich_xe,
            "Xuất xứ": xuat_xu,
            "Năm đăng ký": nam_dang_ky,
            "Số Km đã đi": so_km_bat_thuong,
            "Giá": gia_thuc_te   # quan trọng
        }

        df_test = pd.DataFrame([input_row])

        # -----------------------
        # 2. Gọi hàm detect anomaly
        # -----------------------
        df_res = detect_residual_anomaly(df_test, model)

        # -----------------------
        # 3. Hiển thị
        # -----------------------
        residual = df_res["residual"].iloc[0]
        status = df_res["anomaly_resid"].iloc[0]

        st.write("### Kết quả kiểm tra:")
        st.write(f"Residual: **{residual:,.0f}**")

        if status == "Bất thường":
            st.error("Xe ô tô bất thường theo mô hình Residual với sai số là 10 triệu VND.")
        else:
            st.success("Xe ô tô bình thường theo mô hình Residual với sai số là 10 triệu VND.")

elif choice=='Gợi ý điều khiển project 2':
    st.write("##### Gợi ý điều khiển project 2: Recommender System")
    st.write("##### Dữ liệu mẫu")
    # Tạo dataframe có 3 cột là id, title, description
    # Đọc dữ liệu từ file mau_xe_may.xlsx
    df = pd.read_excel("mau_xe_may.xlsx")    
    st.dataframe(df)
    st.write("### 1. Tìm kiếm xe tương tự")
    # Tạo điều khiển để người dùng chọn công ty
    selected_bike = st.selectbox("Chọn xe", df['title'])
    st.write("Xe đã chọn:", selected_bike) 
    # Từ xe đã chọn này, người dùng có thể xem thông tin chi tiết của xe
    # hoặc thực hiện các xử lý khác
    # tạo điều khiển để người dùng tìm kiếm xe dựa trên thông tin người dùng nhập
    search = st.text_input("Nhập thông tin tìm kiếm")
    # Tìm kiếm xe dựa trên thông tin người dùng nhập vào search, chuyển thành chữ thường trước khi tìm kiếm
    # Trên thực tế sử dụng content-based filtering (cosine similarity/ gensim) để tìm kiếm xe tương tự
    result = df[df['title'].str.lower().str.contains(search.lower())]    
    # tạo button submit
    tim_kiem = st.button("Tìm kiếm")
    if tim_kiem:
        st.write("Danh sách xe tìm được:")
        st.dataframe(result)
       
# Done