import streamlit as st
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
import pandas as pd
import os
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import io
import zipfile
from datetime import datetime
import sqlite3
from datetime import date
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import re
import bcrypt
import gdown


def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "🔐 Đăng nhập"


DB_PATH = "tb_predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            PatientID TEXT PRIMARY KEY,
            Name TEXT,
            Birthday TEXT,
            Phone TEXT,
            Address TEXT,
            Prediction TEXT,
            ImageFile TEXT,
            CreatedAt TEXT
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            Username TEXT PRIMARY KEY,
            Password TEXT,
            Email TEXT UNIQUE,
            ResetCode TEXT,
            CreatedAt TEXT
        )
    """
    )
    conn.commit()
    conn.close()


# Load .env nếu chạy local
load_dotenv()

# Lấy email/password
try:
    # Streamlit Cloud
    EMAIL_ADDRESS = st.secrets["email"]["address"]
    EMAIL_PASSWORD = st.secrets["email"]["password"]
except Exception:
    # Local
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Kiểm tra
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    st.warning(
        "⚠ Email hoặc mật khẩu chưa được cấu hình. Vui lòng thêm .env hoặc st.secrets"
    )


init_user_db()


def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)


def is_valid_password(password):
    if len(password) < 8:
        return False, "Mật khẩu phải có ít nhất 8 ký tự"
    if not re.search(r"[A-Z]", password):
        return False, "Mật khẩu phải có ít nhất 1 chữ hoa"
    if not re.search(r"[a-z]", password):
        return False, "Mật khẩu phải có ít nhất 1 chữ thường"
    if not re.search(r"[0-9]", password):
        return False, "Mật khẩu phải có ít nhất 1 chữ số"
    return True, ""


def normalize_username(username: str) -> str:
    """
    Loại bỏ khoảng trắng đầu/cuối và gộp nhiều khoảng trắng thành 1
    """
    return re.sub(r"\s+", " ", username.strip())


def register_user(username, password, email):
    username = normalize_username(username)  # ✅ chuẩn hóa
    try:
        hashed_pw = hash_password(password)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (Username, Password, Email, CreatedAt)
            VALUES (?, ?, ?, ?)
        """,
            (username, hashed_pw, email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def is_valid_username(username: str):
    """
    Kiểm tra username hợp lệ:
    - Không để trống
    - Không cho phép khoảng trắng đầu/cuối
    - Độ dài 3-20 ký tự
    """
    username = re.sub(r"\s+", " ", username.strip())
    if len(username) < 3 or len(username) > 20:
        return False, "Tên đăng nhập phải từ 3 đến 20 ký tự"

    return True, ""


def login_user(username, password):
    username = normalize_username(username)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT Password FROM users WHERE Username = ?
    """,
        (username,),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return False

    hashed_pw = row[0]
    return check_password(password, hashed_pw)


def show_register_page():
    st.markdown(
        "<h1 style='text-align:center;'>📝 ĐĂNG KÝ</h1>", unsafe_allow_html=True
    )

    username = st.text_input("👤 Tên đăng nhập")
    email = st.text_input("📧 Email")
    password = st.text_input("🔑 Mật khẩu", type="password")
    confirm = st.text_input("🔁 Nhập lại mật khẩu", type="password")
    st.info(
        """
                🔐 **Yêu cầu mật khẩu:**
                - Ít nhất 8 ký tự  
                - Có chữ hoa  
                - Có chữ thường  
                - Có số
                """
    )

    if st.button("📝 Đăng ký", use_container_width=True):

        if not username.strip() or not email.strip() or not password:
            st.warning("⚠ Vui lòng nhập đầy đủ: Tên đăng nhập, Email và Mật khẩu")
            st.info(
                """
                🔐 **Yêu cầu mật khẩu:**
                - Ít nhất 8 ký tự  
                - Có chữ hoa  
                - Có chữ thường  
                - Có số
                """
            )
            return

        # ✅ Kiểm tra username
        valid_un, msg_un = is_valid_username(username)
        if not valid_un:
            st.error(f"❌ {msg_un}")
            return

        # ✅ Kiểm tra email
        if not is_valid_email(email):
            st.error("❌ Email không đúng định dạng (vd: example@gmail.com)")
            return

        # ✅ Kiểm tra mật khẩu
        valid_pw, msg = is_valid_password(password)
        if not valid_pw:
            st.error(f"❌ {msg}")
            return

        if password != confirm:
            st.error("❌ Mật khẩu không khớp")
            return

        success = register_user(username, password, email)

        if success:
            st.success("✅ Đăng ký & đăng nhập thành công!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("❌ Tên đăng nhập hoặc email đã tồn tại")


def show_login_page():
    st.markdown(
        "<h1 style='text-align:center;'>🔐 ĐĂNG NHẬP</h1>", unsafe_allow_html=True
    )

    username = st.text_input("👤 Tên đăng nhập")
    password = st.text_input("🔑 Mật khẩu", type="password")

    if st.button("🔓 Đăng nhập", use_container_width=True):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Đăng nhập thành công")
            st.rerun()
        else:
            st.error("❌ Sai tài khoản hoặc mật khẩu")


def send_reset_email(to_email, username, code):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = "🔐 Khôi phục mật khẩu"

    body = f"""
Xin chào,

Bạn đã yêu cầu khôi phục mật khẩu.

👤 Tên đăng nhập: {username}
🔢 Mã xác nhận: {code}

Vui lòng quay lại ứng dụng và nhập mã để đặt lại mật khẩu.

Nếu không phải bạn yêu cầu, hãy bỏ qua email này.
"""
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


def show_forgot_password_page():
    st.markdown(
        "<h1 style='text-align:center;'>🔁 QUÊN MẬT KHẨU</h1>", unsafe_allow_html=True
    )

    email = st.text_input("📧 Email đã đăng ký")

    if st.button("📨 Gửi mã xác nhận", use_container_width=True):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT Username FROM users WHERE Email = ?", (email,))
        row = cursor.fetchone()

        if not row:
            st.error("❌ Email không tồn tại")
            conn.close()
            return

        username = row[0]
        code = str(random.randint(100000, 999999))

        cursor.execute("UPDATE users SET ResetCode = ? WHERE Email = ?", (code, email))
        conn.commit()
        conn.close()

        try:
            send_reset_email(email, username, code)
        except Exception as e:
            st.error(f"❌ Không gửi được email: {e}")
            return

        st.success("✅ Đã gửi mã xác nhận qua email")

        st.session_state.page = "🔐 Đặt lại mật khẩu"
        st.rerun()


def show_reset_password_page():
    st.markdown(
        "<h1 style='text-align:center;'>🔐 ĐẶT LẠI MẬT KHẨU</h1>",
        unsafe_allow_html=True,
    )

    username = st.text_input("👤 Tên đăng nhập")
    code = st.text_input("🔢 Mã xác nhận")
    new_pw = st.text_input("🔑 Mật khẩu mới", type="password")
    confirm = st.text_input("🔁 Nhập lại mật khẩu", type="password")

    if st.button("💾 Đổi mật khẩu", use_container_width=True):
        if not username.strip() or not code.strip() or not new_pw:
            st.warning(
                "⚠ Vui lòng nhập đầy đủ: Tên đăng nhập, Mã xác nhận và Mật khẩu mới"
            )
            return

        # ✅ Kiểm tra username hợp lệ
        valid_un, msg_un = is_valid_username(username)
        if not valid_un:
            st.error(f"❌ {msg_un}")
            return

        if new_pw != confirm:
            st.error("❌ Mật khẩu không khớp")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 1 FROM users
            WHERE Username = ? AND ResetCode = ?
        """,
            (username, code),
        )

        if cursor.fetchone() is None:
            st.error("❌ Mã xác nhận hoặc tên đăng nhập không đúng")
            conn.close()
            return

        # 🔐 HASH PASSWORD
        hashed_pw = hash_password(new_pw)

        cursor.execute(
            """
            UPDATE users
            SET Password = ?, ResetCode = NULL
            WHERE Username = ?
        """,
            (hashed_pw, username),
        )

        conn.commit()
        conn.close()

        st.success("✅ Đổi mật khẩu thành công! Vui lòng đăng nhập")
        st.session_state.page = "🔐 Đăng nhập"
        st.rerun()


# ==========================
# THƯ MỤC LƯU ẢNH
# ==========================
IMAGE_SAVE_DIR = "saved_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ==========================
# CẤU HÌNH
# ==========================
MODEL_PATH = "final_modelv3.h5"
IMG_SIZE = (224, 224)
THRESHOLD = 0.8


# ==========================
# CUSTOM LAYER
# ==========================
class CustomTFOpLambda(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs


# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = "final_modelv3.h5"
MODEL_DRIVE_ID = "1-HLa6PSW_x3DaVEyImZIMF4E3DSk7blf"


@st.cache_resource(show_spinner=False)
def load_best_model():

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        try:
            st.info("📥 Đang tải model từ Google Drive…")
            gdown.download(
                url,
                MODEL_PATH,
                quiet=False,
                fuzzy=True,  # ⭐ Bật để hỗ trợ tải file lớn
            )
            st.success("✅ Tải model thành công")
        except Exception as e:
            st.error(f"❌ Không tải được model: {e}")
            return None

    # Load model
    try:
        with tf.keras.utils.custom_object_scope({"CustomTFOpLambda": CustomTFOpLambda}):
            model = load_model(MODEL_PATH, compile=False)

        return model
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {e}")
        return None


model = load_best_model()


# ==========================
# TIỀN XỬ LÝ ẢNH
# ==========================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# ==========================
# DỰ ĐOÁN
# ==========================
def predict_image(image: Image.Image):
    img = preprocess_image(image)
    prob = model.predict(img)[0][0]
    label = "Bệnh lao" if prob >= THRESHOLD else "Bình thường"
    return label, prob


# ==========================
# NHÃN THẬT TỪ FILE
# ==========================
def get_true_label(filename):
    filename = filename.lower()
    if "normal" in filename:
        return "Bình thường"
    elif "tuberculosis" in filename or "tb" in filename:
        return "Bệnh lao"
    return "Không xác định"


# ==========================
# LƯU CSV VÀ ẢNH
# ==========================
def save_patient_result(name, Birthday, Phone, address, pred, uploaded_file):
    patient_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_clean = "".join(c for c in name if c.isalnum())
    image_filename = f"{patient_id}_{name_clean}_{timestamp}_{uploaded_file.name}"
    image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
    image = Image.open(uploaded_file)
    image.save(image_path)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (PatientID, Name, Birthday, Phone, Address, Prediction, ImageFile, CreatedAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            patient_id,
            name,
            Birthday.strftime("%Y-%m-%d"),
            Phone,
            address,
            pred,
            image_filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )

    conn.commit()
    conn.close()
    return image_path, patient_id


# ==========================
# TRANG GIỚI THIỆU
# ==========================
def show_intro_page():
    st.markdown(
        """
    <div style="text-align:center; font-size:38px; font-weight:600; padding:px">
        🏥 HỆ THỐNG PHÂN LOẠI LAO PHỔI
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
Ứng dụng **Deep Learning** hỗ trợ phát hiện **bệnh lao phổi** từ ảnh X-quang, giúp bác sĩ chẩn đoán nhanh và chính xác hơn.

### 🧠 Công nghệ sử dụng
- TensorFlow / Keras
- MobileNetV3
- Streamlit

### ⚠ Lưu ý
Hệ thống chỉ mang tính **hỗ trợ**, không thay thế hoàn toàn chẩn đoán y khoa.
Kết quả nên được bác sĩ chuyên môn xác nhận trước khi đưa ra quyết định.
"""
    )


# ==========================
# TRANG DỰ ĐOÁN
# ==========================
default_date = date(2000, 1, 1)


def show_prediction_page():
    st.markdown(
        """
<h1 style="
    text-align:center;
    font-size:clamp(18px, 4vw, 34px);
    font-weight:700;
    margin-bottom:20px;
">
🫁 PHÂN LOẠI LAO PHỔI TỪ ẢNH X-QUANG
</h1>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div style="
    background:#f0f8ff;
    padding:8px;
    border-radius:8px;
    margin-bottom:15px;
">
    <h4 style="color:#4B0082; text-align:center;">
        🧑‍⚕️ Thông tin bệnh nhân
    </h4>
</div>
""",
        unsafe_allow_html=True,
    )

    name = st.text_input("👤 Tên bệnh nhân")
    Birthday = st.date_input("🎂 Ngày sinh", value=default_date)
    Phone = st.text_input("📞 Số điện thoại")
    address = st.text_input("🏠 Địa chỉ")
    uploaded_file = st.file_uploader("📤 Tải ảnh X-quang", type=["jpg", "jpeg", "png"])

    # Nút dự đoán
    st.markdown(
        """
<style>
div.stButton > button:first-child {
    height:50px;
    width:220px;
    background-color:#4B0082;
    color:white;
    font-size:18px;
    font-weight:bold;
    border-radius:12px;
    margin:auto;
    display:block;
}
</style>
""",
        unsafe_allow_html=True,
    )

    if st.button("🔍 Dự đoán"):

        if not name.strip():
            st.warning("⚠ Vui lòng nhập tên bệnh nhân")
            return
        if not address.strip():
            st.warning("⚠ Vui lòng nhập địa chỉ bệnh nhân")
            return

        if not Phone.strip():
            st.warning("⚠ Vui lòng nhập số điện thoại bệnh nhân")
            return
        # Regex kiểm tra số điện thoại
        if not re.match(r"^\+?\d{9,15}$", Phone.strip()):
            st.warning(
                "⚠ Số điện thoại không hợp lệ (ít nhất 9 số, tối đa 15 số, có thể có +)"
            )
            return

        if uploaded_file is None:
            st.warning("⚠ Vui lòng tải ảnh X-quang")
            return

        if model is None:
            st.error("❌ Model chưa load")
            return

        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh X-quang", use_container_width=True)

        # Dự đoán
        label, prob = predict_image(image)

        # Lưu kết quả + ảnh
        _, patient_id = save_patient_result(
            name, Birthday, Phone, address, label, uploaded_file
        )

        # Hiển thị kết quả
        st.markdown(
            f"""
<h2 style="color:#4B0082; text-align:center;">
🧪 Kết quả: {label}<br>
🎯 Xác suất mắc bệnh lao: {prob:.2%}
</h2>
""",
            unsafe_allow_html=True,
        )

        st.success(f"✅ Đã lưu kết quả với PatientID: {patient_id}")


# ==========================
# TRANG LỊCH SỬ & BIỂU ĐỒ
# ==========================
def show_history_page():
    st.markdown(
        """
    <h1 style="text-align:center; color:#4B0082;">
        📊 THỐNG KÊ & BIỂU ĐỒ
    </h1>
    """,
        unsafe_allow_html=True,
    )

    # ===== ĐỌC SQLITE =====
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()

    if df.empty:
        st.warning("⚠ Chưa có dữ liệu.")
        return

    df_display = df.rename(
        columns={
            "PatientID": "Mã bệnh nhân",
            "Name": "Tên",
            "Birthday": "Ngày sinh",
            "Phone": "Số điện thoại",
            "Address": "Địa chỉ",
            "Prediction": "Dự đoán",
            "ImageFile": "Tên file ảnh",
            "CreatedAt": "Thời gian",
        }
    )

    # =========================
    # 📋 BẢNG DỮ LIỆU
    # =========================
    st.markdown("### 📋 Danh sách bệnh nhân")
    st.dataframe(df_display, use_container_width=True, height=320)

    st.divider()

    # =========================
    # 🔧 QUẢN LÝ BỆNH NHÂN
    # =========================

    # --- Chọn PatientID (HÀNG TRÊN) ---
    selected_id = st.selectbox(
        "🗑️ Chọn mã bệnh nhân cần xóa", df_display["Mã bệnh nhân"]
    )

    selected_row = df_display[df_display["Mã bệnh nhân"] == selected_id].iloc[0]

    # =========================
    # ✏️ SỬA THÔNG TIN
    # =========================
    st.markdown("#### ✏️ Chỉnh sửa thông tin")

    col1, col2 = st.columns(2)

    with col1:
        edit_name = st.text_input("👤 Tên", selected_row["Tên"])

        edit_birthday = st.date_input(
            "🎂 Ngày sinh", pd.to_datetime(selected_row["Ngày sinh"])
        )

        edit_Phone = st.text_input("📞 Số điện thoại", selected_row["Số điện thoại"])

    with col2:
        edit_address = st.text_input("🏠 Địa chỉ", selected_row["Địa chỉ"])
        edit_pred = st.selectbox(
            "🧪 Dự đoán",
            ["Bình thường", "Bệnh lao"],
            index=0 if selected_row["Dự đoán"] == "Bình thường" else 1,
        )

    if st.button("💾 Lưu thay đổi", use_container_width=True):
        if not edit_name.strip():
            st.warning("⚠ Vui lòng nhập tên bệnh nhân")
            return

        if not edit_address.strip():
            st.warning("⚠ Vui lòng nhập địa chỉ bệnh nhân")
            return

        if not edit_Phone.strip():
            st.warning("⚠ Vui lòng nhập số điện thoại")
            return

        if not re.match(r"^\+?\d{9,15}$", edit_Phone.strip()):
            st.warning(
                "⚠ Số điện thoại không hợp lệ (ít nhất 9 số, tối đa 15 số, có thể có +)"
            )
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE predictions
            SET Name = ?, Birthday = ?, Phone =?, Address = ?, Prediction = ?
            WHERE PatientID = ?
        """,
            (
                edit_name,
                edit_birthday.strftime("%Y-%m-%d"),
                edit_Phone,
                edit_address,
                edit_pred,
                selected_id,
            ),
        )
        conn.commit()
        conn.close()
        st.success("✅ Đã cập nhật thông tin")
        st.rerun()

    st.divider()

    st.markdown("#### 🖼 Ảnh X-quang")

    img_path = os.path.join(IMAGE_SAVE_DIR, selected_row["Tên file ảnh"])

    if os.path.exists(img_path):
        st.image(
            img_path,
            caption=f"Ảnh X-quang – Mã bệnh nhân: {selected_id}",
            use_container_width=True,
        )
    else:
        st.error("❌ Không tìm thấy ảnh X-quang")

    # =========================
    # ❌ HÀNH ĐỘNG NGUY HIỂM
    # =========================
    st.markdown("#### ❌ Hành động nguy hiểm")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Xóa bệnh nhân", use_container_width=True):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM predictions WHERE PatientID = ?", (selected_id,)
            )
            conn.commit()
            conn.close()
            st.warning("🗑️ Đã xóa bệnh nhân")
            st.rerun()

    with col2:
        if st.button("🔥 Xóa toàn bộ dữ liệu", use_container_width=True):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.error("🚨 Đã xóa toàn bộ dữ liệu")
            st.rerun()

    st.divider()

    # =========================
    # 📈 BIỂU ĐỒ
    # =========================
    st.markdown("### 📊 Thống kê kết quả chẩn đoán (toàn bộ bệnh nhân)")

    counts = df_display["Dự đoán"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax1)
        ax1.set_title("Số lượng ca theo kết quả")
        ax1.set_xlabel("Kết quả")
        ax1.set_ylabel("Số bệnh nhân")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax2.axis("equal")
        ax2.set_title("Tỷ lệ phân loại")
        st.pyplot(fig2)

    st.divider()

    # =========================
    # 📥 XUẤT BÁO CÁO
    # =========================
    st.markdown("### 📥 Xuất báo cáo")

    csv_bytes = df_display.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📄 Tải CSV",
            csv_bytes,
            "tb_predictions.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("tb_predictions.csv", csv_bytes)
            for img in df_display["Tên file ảnh"]:
                img_path = os.path.join(IMAGE_SAVE_DIR, img)
                if os.path.exists(img_path):
                    zipf.write(img_path, arcname=f"images/{img}")

        zip_buffer.seek(0)

        st.download_button(
            "📦 Tải ZIP (CSV + Ảnh)",
            zip_buffer,
            "tb_predictions_full.zip",
            "application/zip",
            use_container_width=True,
        )


# ==========================
# TRANG THÔNG TIN MÔ HÌNH
# ==========================
def show_model_info_page():
    st.markdown(
        "<h1 style='text-align:center; color:#4B0082;'>🧠 THÔNG TIN MÔ HÌNH</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="
        background: linear-gradient(to right, #f0f8ff, #e6e6fa);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
        font-size: 16px;
        line-height: 1.6;
    ">
        <p>🔹 <strong>Kiến trúc:</strong> MobileNetV3</p>
        <p>🔹 <strong>Input:</strong> 224 × 224 RGB</p>
        <p>🔹 <strong>Output:</strong> Normal / Tuberculosis</p>
        <p>🔹 <strong>Threshold:</strong> 0.8</p>
        <p>🔹 <strong>Trạng thái model:</strong> 
        """
        + (
            "✅ <span style='color:green;'>Đã load</span>"
            if model
            else "❌ <span style='color:red;'>Chưa load</span>"
        )
        + """</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ==========================
# MAIN
# ==========================
def main():
    st.sidebar.title("🧭 MENU")

    # =============================
    # CHƯA ĐĂNG NHẬP
    # =============================
    if not st.session_state.logged_in:

        # Chỉ menu 3 page chính
        pages = [
            "🔐 Đăng nhập",
            "📝 Đăng ký",
            "🔁 Quên mật khẩu",
        ]

        # Lấy index an toàn
        try:
            default_index = pages.index(st.session_state.page)
        except ValueError:
            default_index = 0

        # Chọn page hiện tại từ sidebar
        selected_page = st.sidebar.radio("Tài khoản", pages, index=default_index)

        # Nếu người dùng vừa gửi email xong, hiển thị page đổi mật khẩu
        if st.session_state.page == "🔐 Đặt lại mật khẩu":
            show_reset_password_page()

        else:
            # Đồng bộ page được chọn từ sidebar
            st.session_state.page = selected_page

            if selected_page == "🔐 Đăng nhập":
                show_login_page()
            elif selected_page == "📝 Đăng ký":
                show_register_page()
            elif selected_page == "🔁 Quên mật khẩu":
                show_forgot_password_page()

    # =============================
    # ĐÃ ĐĂNG NHẬP
    # =============================
    else:
        st.sidebar.success(f"👋 Xin chào {st.session_state.username}")

        page = st.sidebar.radio(
            "Chức năng",
            [
                "🏠 Giới thiệu",
                "🫁 Phân loại lao",
                "📊 Thống kê & Biểu đồ",
                "🧠 Thông tin mô hình",
            ],
        )

        if page == "🏠 Giới thiệu":
            show_intro_page()
        elif page == "🫁 Phân loại lao":
            show_prediction_page()
        elif page == "📊 Thống kê & Biểu đồ":
            show_history_page()
        elif page == "🧠 Thông tin mô hình":
            show_model_info_page()

    if st.sidebar.button("🚪 Đăng xuất"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "🔐 Đăng nhập"
        st.rerun()


# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    main()
