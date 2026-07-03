import streamlit as st
import pandas as pd
import pickle

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="AI Đánh Giá Môi Trường", page_icon="🌿", layout="wide")

# --- TẢI MÔ HÌNH AI ĐÃ HUẤN LUYỆN ---
@st.cache_resource 
def load_model():
    with open('mo_hinh_ca_phe.pkl', 'rb') as f:
        return pickle.load(f)

models = load_model()

# --- 1. TẠO THANH BÊN (SIDEBAR) ĐƠN GIẢN ---
with st.sidebar:
    st.header("📌 Thông tin dự án")
    st.write("**Đề tài:** Sản xuất giấy từ lá cà phê")
    st.write("**Thực hiện:** Cô Tuyền (THPT Võ Văn Kiệt)")
    st.write("**Hỗ trợ AI:** Bảo Lâm")

# --- 2. THIẾT KẾ GIAO DIỆN CHÍNH ---
st.title("🌿 Ứng Dụng AI Đánh Giá Tác Động Môi Trường")
st.write("Vui lòng nhập khối lượng lá/bã cà phê để AI tự động tính toán các lợi ích môi trường mang lại.")
st.divider()

# Sử dụng ô nhập số thay vì thanh kéo
la_ca_phe_input = st.number_input(
    "👉 Nhập khối lượng lá/bã cà phê tái chế (kg):", 
    min_value=1.0, 
    value=10.0, 
    step=1.0
)

# Nút bấm thực thi
if st.button("🚀 Bắt đầu Dự đoán bằng AI", type="primary"):
    
    # --- 3. XỬ LÝ DỮ LIỆU & DỰ ĐOÁN ---
    input_df = pd.DataFrame({'La_Ca_Phe_kg': [la_ca_phe_input]})
    
    giay = int(models['Giay_Tao_Ra_To'].predict(input_df)[0])
    co2 = models['CO2_Giam_kg'].predict(input_df)[0]
    rac = models['Rac_Huu_Co_kg'].predict(input_df)[0]
    cay = models['Cay_Xanh_Tuong_Duong'].predict(input_df)[0]
    
    # --- 4. HIỂN THỊ KẾT QUẢ DẠNG SỐ ---
    st.success("✅ Phân tích hoàn tất! Dưới đây là kết quả đánh giá:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📄 Số tờ giấy thu được (Ước tính)", value=f"{giay} tờ")
        st.metric(label="🌱 Rác hữu cơ giảm", value=f"{rac:.1f} kg")
        
    with col2:
        st.metric(label="☁️ Lượng CO2 giảm phát thải", value=f"{co2:.1f} kg")
        st.metric(label="🌳 Số cây xanh tương đương", value=f"{cay:.2f} cây")
        
    st.divider()
    
    # --- 5. HIỂN THỊ BIỂU ĐỒ TRỰC QUAN ---
    st.write("### 📊 Biểu đồ hiệu quả môi trường")
    
    # Dữ liệu cho biểu đồ
    chart_data = pd.DataFrame(
        {
            "Khối lượng (kg)": [la_ca_phe_input, rac, co2],
            "Phân loại": ["Lá cà phê đầu vào", "Rác hữu cơ giảm", "CO2 giảm phát thải"]
        }
    )
    
    # Vẽ biểu đồ cột
    st.bar_chart(chart_data.set_index("Phân loại"))
    
    st.caption("Lưu ý: Dữ liệu hiện tại là bản Demo mô phỏng. Thông số sẽ được cập nhật lại khi có kết quả thực nghiệm chính thức.")
