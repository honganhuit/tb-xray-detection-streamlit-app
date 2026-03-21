# 🫁 TB X-ray Detection Streamlit App

## 🎯 Goal

Support doctors in detecting tuberculosis (TB) from chest X-ray images, improving diagnostic speed and accuracy.

---

## 👤 User Stories & Use Cases

### **User Story 1: Upload & Predict TB**

- **As a** doctor
- **I want to** upload a patient’s chest X-ray image
- **So that** I can receive a TB risk score and make informed clinical decisions

**Use Case (Simplified Flow):**

1. User logs in with username/password
2. User inputs patient information: Name, Birthday, Phone, Address
3. User uploads X-ray image (JPG, JPEG, PNG)
4. System validates image and required fields
5. System runs deep learning model (MobileNetV3)
6. System displays prediction: Normal / Tuberculosis + probability
7. User can export prediction report (CSV or ZIP)

**Exceptions / Alternate Flows:**

- Invalid image format → Show error message
- Missing required fields → Show warning and block prediction

### **User Story 2: Manage Patient History**

- **As a** doctor
- **I want to** view and manage patient history
- **So that** I can track past predictions and monitor patient outcomes

**Use Case (Simplified Flow):**

1. Select “Patient History” from dashboard
2. System displays list of patients with prediction results
3. User can view X-ray images, edit info, or delete records
4. System allows export (CSV or ZIP)
5. Dashboard shows statistics and charts

**Exceptions / Alternate Flows:**

- No patient records → Show “No data” message
- Invalid edits → Show validation warning

### **User Story 3: Password Management**

- **As a** doctor
- **I want to** reset/change my password securely
- **So that** I can maintain account security

**Use Case (Simplified Flow):**

1. Click “Forgot Password”
2. Receive verification code via registered email
3. Enter code and new password
4. System validates and updates password
5. Log in with new password

**Exceptions / Alternate Flows:**

- Invalid email → Show error
- Wrong verification code → Show error
- Weak password → Show validation rules

---

## 📊 Workflow Overview

1. **User Login:** Secure login with username/password
2. **Patient Info Input:** Name, Birthday, Phone, Address
3. **Upload X-ray:** Image validation (JPG, PNG, JPEG)
4. **Model Prediction:** Binary classification (Normal / TB)
5. **Result Display:** Show label and probability
6. **Save & Export:** Save to SQLite, download CSV or ZIP (CSV + images)
7. **Visualization Dashboard:** Display patient history and charts

**Validation Rules:**

- Required fields: Name, Birthday, Phone, Address, Image
- Image format check
- Phone/email format check
- Prediction threshold = 0.8

---

## 🛠 Key Technologies

- **Web app & UI:** Python, Streamlit
- **Deep learning:** TensorFlow/Keras, MobileNetV3
- **Data processing & visualization:** Pandas, Matplotlib, Seaborn
- **Database:** SQLite for patient records
- **Utilities:** gdown (download model), dotenv (environment variables), bcrypt (password hashing)

---

## ⚙️ How to Run

1. Clone repository:

```bash
git clone https://github.com/honganhuit/tb-xray-detection-streamlit-app.git
cd tb-xray-detection-streamlit-app
```
