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

# --- 1. TẠO THANH BÊN (SIDEBAR) CHUYÊN NGHIỆP ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2922/2922037.png", width=100) # Thêm 1 icon lá cây cho đẹp
    st.header("📌 Thông tin dự án")
    st.markdown("**Tên đề tài:** Ứng dụng AI trong đánh giá tác động môi trường của việc sản xuất giấy từ lá cà phê")
    st.markdown("**Nghiên cứu & Thực hiện:** Cô Tuyền")
    st.markdown("**Đơn vị:** Trường THPT Võ Văn Kiệt, Đắk Lắk")
    st.markdown("**Thiết kế & Lập trình AI:** Bảo Lâm")
    
    st.divider()
    st.info("💡 Hệ thống sử dụng mô hình Học máy (Machine Learning) để phân tích và dự đoán các chỉ số môi trường dựa trên dữ liệu đầu vào.")

# --- 2. THIẾT KẾ GIAO DIỆN CHÍNH ---
st.title("🌿 Ứng Dụng AI Đánh Giá Tác Động Môi Trường")
st.write("Vui lòng kéo thanh trượt bên dưới để chọn khối lượng lá/bã cà phê, AI sẽ tự động tính toán các lợi ích môi trường mang lại.")
st.divider()

# Thay thế ô nhập tay bằng Thanh trượt (Slider)
la_ca_phe_input = st.slider(
    "👉 Chọn khối lượng lá/bã cà phê tái chế (kg):", 
    min_value=1.0, 
    max_value=200.0, 
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
    st.write("So sánh tương quan giữa khối lượng vật liệu sử dụng và hiệu quả giảm phát thải.")
    
    # Dữ liệu cho biểu đồ
    chart_data = pd.DataFrame(
        {
            "Khối lượng (kg)": [la_ca_phe_input, rac, co2],
            "Phân loại": ["Lá cà phê đầu vào", "Rác hữu cơ giảm", "CO2 giảm phát thải"]
        }
    )
    
    # Vẽ biểu đồ cột
    st.bar_chart(chart_data.set_index("Phân loại"))
    
    st.caption("Lưu ý: Dữ liệu hiện tại là bản Demo mô phỏng. Thông số sẽ được cập nhật lại khi có kết quả thực nghiệm chính thức từ phòng thí nghiệm của trường THPT Võ Văn Kiệt.")
