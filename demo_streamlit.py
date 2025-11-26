# demo_streamlit.py
import streamlit as st
import pandas as pd
import joblib
from scipy import stats
import os

st.set_page_config(page_title="Dự đoán giá & Phát hiện bất thường - Xe máy cũ", layout="centered")

# ---------- Sidebar (3 tabs) ----------
st.sidebar.title("Menu")
menu = ["Overall", "Dự đoán giá", "Phát hiện bất thường"]
choice = st.sidebar.selectbox("Chọn trang", menu)

# ---------- Load data (mẫu) + allow upload ----------
DATA_PATH = "./data_motobikes.xlsx"
df = None

def load_default_data(path=DATA_PATH):
    if os.path.exists(path):
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            st.warning(f"Lỗi đọc file mẫu {path}: {e}")
            return None
    return None

df = load_default_data()

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload CSV hoặc Excel (thay dữ liệu mẫu)", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.sidebar.success("Đã load file upload.")
    except Exception as e:
        st.sidebar.error(f"Lỗi khi đọc file upload: {e}")
        df = None

# ---------- Load model once ----------
MODEL_PATH = "car_price_gbr_pipeline.pkl"
model = None
model_load_error = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model_load_error = e
else:
    model_load_error = FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

# ---------- Pages ----------
if choice == "Overall":
st.title("Trang tổng quan (Overall)")
    # hero image (placeholder) - không gắn link
    if os.path.exists("hero.jpg"):
        st.image("hero.jpg", caption="Dự án: Dự đoán giá & Phát hiện bất thường (hình minh họa)")
    else:
        st.image("xe_may_cu.jpg", caption="Hình minh họa (xe máy cũ)")

    st.markdown("""
    ### Business Objective
    Project này nhằm triển khai hai tính năng cho nền tảng Chợ Tốt (giả sử chưa có): 
    - **Dự đoán giá (Price Prediction)**: Xây dựng mô hình ML để ước tính giá bán hợp lý cho xe máy cũ dựa trên các đặc trưng như thương hiệu, năm đăng ký, số km đã đi, v.v. Giúp người bán định giá chính xác và người mua tham khảo.
    - **Phát hiện bất thường (Anomaly Detection)**: Xác định các xe có giá quá thấp (có thể lỗi nhập liệu, khuyến mãi đặc biệt, hoặc hàng giả) hoặc quá cao (thổi phồng giá hoặc xe hiếm sưu tầm). Tập trung vào xe máy cũ tại TP.HCM (dữ liệu trước 01/07/2025), sử dụng residual từ mô hình dự đoán để detect.
    
    Giá trị kinh doanh: Tăng độ tin cậy nền tảng, giảm gian lận, cải thiện trải nghiệm user.
    """)

    st.markdown("""
    ### Tóm tắt EDA (Exploratory Data Analysis)
    - **Kích thước data**: 7208 rows, 18 columns (id, Tiêu đề, Giá, Khoảng giá min/max, Địa chỉ, Mô tả, Thương hiệu, Dòng xe, Năm đăng ký, Số Km đã đi, Tình trạng, Loại xe, Dung tích xe, Xuất xứ, Chính sách bảo hành, Trọng lượng, Href).
    - **Missing values**: Cao nhất ở Khoảng giá min (202), max (197); thấp ở Giá (2). Tổng missing thấp (~3-5% ở vài cột) → Data khá sạch, fill bằng mode/mean.
    - **Phân bố target (Giá)**: Skewed phải (median ~16.5tr, mean ~49tr do outliers); min=0 (bất thường), max=136 tỷ (lỗi/outlier). Histogram filter <200tr cho thấy peak ~10-30tr (xe phổ thông).
    - **Insights**: Correlation mạnh: Giá giảm khi Số Km tăng hoặc Năm đăng ký cũ. Categorical: Honda/Yamaha chiếm đa số. Outliers ở Giá cao/thấp → Cần clean cho modeling.
    """)

    st.markdown("""
    ### So sánh các Models (Regression)
    Dưới đây là bảng tóm tắt hiệu suất các mô hình đã thử (trên test set):
    
    | Mô hình                  | RMSE (triệu VND) | MAE (triệu VND) | R²   | Ghi chú |
    |--------------------------|------------------|-----------------|------|---------|
    | Linear Regression        | 9.39            | 5.88           | 0.62 | Cơ bản, tuyến tính, chưa nắm được quan hệ phi tuyến giữa các biến. |
    | Random Forest            | 8.92            | 5.42           | 0.66 | Học tốt hơn nhờ bắt được quan hệ phi tuyến, giảm sai số đáng kể. |
    | Gradient Boosting Regressor | 8.86         | 5.22           | 0.66 | Hiệu quả cao hơn nhẹ, học sâu dần để sửa lỗi từng bước. |
    | XGBoost Regressor        | 8.81            | 5.29           | 0.66 | Ổn định, huấn luyện nhanh hơn, hiệu năng gần tương đương GBoost. |
    
    """)

    st.markdown("""
    ### Lý do chọn Gradient Boosting Regressor
    Bài toán dự đoán giá xe máy cũ là regression với data có quan hệ phi tuyến (e.g., Giá không giảm tuyến tính theo Số Km, mà phụ thuộc phức tạp vào Thương hiệu/Dòng xe). Gradient Boosting Regressor phù hợp vì bản chất là ensemble boosting: Xây dựng nhiều trees yếu, sửa lỗi dần dần từ tree trước, giúp capture patterns phức tạp mà không overfit nặng. Kết quả: RMSE/MAE thấp nhất trong các model thử, cân bằng giữa accuracy và tốc độ.
    """)

    if model_load_error:
        st.warning(f"Model chưa load: {model_load_error}")
    else:
        st.success("Model đã load sẵn (nếu cần dùng sẽ hoạt động trong các tab khác).")

elif choice == "Dự đoán giá":
    st.header("1. Dự đoán giá xe máy cũ")

    if df is None:
        st.warning("Chưa có dữ liệu mẫu. Hãy upload file CSV/XLSX có chứa các cột cần thiết.")
        st.stop()

    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.head())

    # Inputs
    try:
        thuong_hieu = st.selectbox("Chọn hãng xe", df['Thương hiệu'].dropna().unique())
        dong_xe = st.selectbox("Chọn dòng xe", df['Dòng xe'].dropna().unique())
        tinh_trang = st.selectbox("Chọn tình trạng", df['Tình trạng'].dropna().unique())
        loai_xe = st.selectbox("Chọn loại xe", df['Loại xe'].dropna().unique())
        dung_tich_xe = st.selectbox("Dung tích xe (cc)", df['Dung tích xe'].dropna().unique())
        xuat_xu = st.selectbox("Chọn xuất xứ", df['Xuất xứ'].dropna().unique())
    except Exception:
        st.error("Dữ liệu mẫu thiếu một số cột bắt buộc (Thương hiệu, Dòng xe, Tình trạng, Loại xe, Dung tích xe, Xuất xứ).")
        st.stop()

    nam_dang_ky = st.slider("Năm đăng ký", 1980, 2025, 2015)
    so_km_da_di = st.number_input("Số km đã đi", min_value=0, max_value=500000, value=50000, step=1000)

    # Load model check
    if model is None:
        st.error(f"Model chưa sẵn sàng: {model_load_error}")
        st.info("Bạn vẫn có thể nhập dữ liệu để kiểm tra UI, nhưng dự đoán sẽ không chạy.")
    du_doan_gia = st.button("Dự đoán giá")
    if du_doan_gia:
        st.write("Thông tin xe:")
        st.write(f"Hãng: {thuong_hieu} — Dòng: {dong_xe} — Tình trạng: {tinh_trang}")
        st.write(f"Loại: {loai_xe} — Dung tích: {dung_tich_xe} — Xuất xứ: {xuat_xu}")
        st.write(f"Năm đăng ký: {nam_dang_ky} — Số Km: {so_km_da_di:,}")

        if model is None:
            st.error("Không thể dự đoán vì model chưa load được.")
        else:
            input_data = pd.DataFrame([{
                'Thương hiệu': thuong_hieu,
                'Dòng xe': dong_xe,
                'Tình trạng': tinh_trang,
                'Loại xe': loai_xe,
                'Dung tích xe': dung_tich_xe,
                'Xuất xứ': xuat_xu,
                'Năm đăng ký': nam_dang_ky,
                'Số Km đã đi': so_km_da_di
            }])
            try:
                pred = model.predict(input_data)[0]
                st.success(f"Giá dự đoán: {pred:,.0f} VND")
            except Exception as e:
                st.error("Lỗi khi gọi model.predict(). Kiểm tra tên cột/format data sao cho khớp với lúc train.")
                st.exception(e)

elif choice == "Phát hiện bất thường":
    st.header("2. Phát hiện bất thường (Anomaly Detection)")

    if df is None:
        st.warning("Chưa có dữ liệu mẫu. Hãy upload file CSV/XLSX có chứa các cột cần thiết.")
        st.stop()

    st.write("Phương pháp: residual = Giá thực - Giá dự đoán. Nếu |residual| > threshold => Bất thường.")
    st.write("Bạn có thể điều chỉnh ngưỡng bằng slider (VND).")

    # Inputs for sample
    try:
        thuong_hieu_a = st.selectbox("Chọn hãng xe (anomaly)", df['Thương hiệu'].dropna().unique(), key="a1")
        dong_xe_a = st.selectbox("Chọn dòng xe (anomaly)", df['Dòng xe'].dropna().unique(), key="a2")
        tinh_trang_a = st.selectbox("Chọn tình trạng (anomaly)", df['Tình trạng'].dropna().unique(), key="a3")
        loai_xe_a = st.selectbox("Chọn loại xe (anomaly)", df['Loại xe'].dropna().unique(), key="a4")
        dung_tich_a = st.selectbox("Dung tích xe (anomaly)", df['Dung tích xe'].dropna().unique(), key="a5")
        xuat_xu_a = st.selectbox("Chọn xuất xứ (anomaly)", df['Xuất xứ'].dropna().unique(), key="a6")
    except Exception:
        st.error("Dữ liệu mẫu thiếu các cột cần thiết (Thương hiệu, Dòng xe, Tình trạng, Loại xe, Dung tích xe, Xuất xứ).")
        st.stop()

    nam_dk_a = st.slider("Năm đăng ký (anomaly)", 1980, 2025, 2015, key="a7")
    so_km_a = st.number_input("Số Km đã đi (anomaly)", min_value=0, max_value=500000, value=50000, step=1000, key="a8")
    gia_thuc_te = st.number_input("Giá thực tế (VND)", min_value=0, max_value=1_000_000_000, value=150_000_000, step=100_000)
    residual_threshold = st.slider("Ngưỡng residual (VND) để coi là bất thường", min_value=0, max_value=200_000_000, value=10_000_000, step=500_000)

    btn_check = st.button("Kiểm tra bất thường")
    if btn_check:
        if model is None:
            st.error(f"Model chưa sẵn sàng: {model_load_error}")
        else:
            input_row = {
                "Thương hiệu": thuong_hieu_a,
                "Dòng xe": dong_xe_a,
                "Tình trạng": tinh_trang_a,
                "Loại xe": loai_xe_a,
                "Dung tích xe": dung_tich_a,
                "Xuất xứ": xuat_xu_a,
                "Năm đăng ký": nam_dk_a,
                "Số Km đã đi": so_km_a,
                "Giá": gia_thuc_te
            }
            df_test = pd.DataFrame([input_row])

            # detect residual anomaly
            def detect_residual_anomaly_single(df_single, model, threshold):
                X = df_single.drop(columns=["Giá"])
                pred_price = model.predict(X)[0]
                residual = df_single["Giá"].iloc[0] - pred_price
                is_anom = abs(residual) > threshold
                return pred_price, residual, is_anom

            try:
                pred_price, residual, is_anom = detect_residual_anomaly_single(df_test, model, residual_threshold)
                st.write(f"Giá dự đoán (model): {pred_price:,.0f} VND")
                st.write(f"Residual (Giá thực - Giá dự đoán): {residual:,.0f} VND")
                if is_anom:
                    st.error(f"🚨 Bất thường: |residual| > {residual_threshold:,} VND")
                else:
                    st.success(f"✔ Bình thường (|residual| ≤ {residual_threshold:,} VND)")
            except Exception as e:
                st.error("Lỗi khi kiểm tra bất thường (kiểm tra tên cột/định dạng input so với pipeline).")
                st.exception(e)

# End of file

