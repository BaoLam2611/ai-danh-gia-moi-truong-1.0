import streamlit as st
import pandas as pd
import pickle

# Cấu hình tiêu đề trang web
st.set_page_config(page_title="AI Đánh Giá Môi Trường", page_icon="🌿")

# --- 1. TẢI MÔ HÌNH AI ĐÃ HUẤN LUYỆN ---
@st.cache_resource # Giúp web load nhanh hơn ở các lần sau
def load_model():
    with open('mo_hinh_ca_phe.pkl', 'rb') as f:
        return pickle.load(f)

models = load_model()

# --- 2. THIẾT KẾ GIAO DIỆN ---
st.title("Ứng Dụng AI Đánh Giá Tác Động Môi Trường")
st.subheader("Dự án: Sản xuất giấy từ lá cà phê")
st.write("Nhập số liệu vào bên dưới để AI dự đoán các chỉ số môi trường tích cực mang lại.")

# Khung nhập liệu cho người dùng
la_ca_phe_input = st.number_input(
    "Nhập khối lượng lá/bã cà phê tái chế (kg):", 
    min_value=1.0, 
    value=10.0, 
    step=1.0
)

# Nút bấm thực thi
if st.button("Bắt đầu Dự đoán bằng AI", type="primary"):
    
    # --- 3. XỬ LÝ DỮ LIỆU & DỰ ĐOÁN ---
    # Tạo bảng dữ liệu nhỏ chứa input để đưa vào mô hình
    input_df = pd.DataFrame({'La_Ca_Phe_kg': [la_ca_phe_input]})
    
    # Gọi AI ra dự đoán
    giay = int(models['Giay_Tao_Ra_To'].predict(input_df)[0])
    co2 = models['CO2_Giam_kg'].predict(input_df)[0]
    rac = models['Rac_Huu_Co_kg'].predict(input_df)[0]
    cay = models['Cay_Xanh_Tuong_Duong'].predict(input_df)[0]
    
    # --- 4. HIỂN THỊ KẾT QUẢ ---
    st.success("Phân tích hoàn tất! Dưới đây là kết quả đánh giá:")
    
    # Dùng các khung (metrics) để hiển thị cho đẹp và chuyên nghiệp
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="📄 Số tờ giấy thu được", value=f"{giay} tờ")
        st.metric(label="🌱 Rác hữu cơ giảm", value=f"{rac:.1f} kg")
        
    with col2:
        st.metric(label="☁️ Lượng CO2 giảm phát thải", value=f"{co2:.1f} kg")
        st.metric(label="🌳 Số cây xanh tương đương", value=f"{cay:.2f} cây")
        