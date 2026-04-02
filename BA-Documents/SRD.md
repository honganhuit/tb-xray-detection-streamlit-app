# System Requirements Document (SRD) – X-ray AI Pulmonary Classification

## 1. Introduction

### 1.1 Purpose

Define system requirements for X-ray AI Pulmonary Classification: accept X-ray images, predict TB, store patient data, and display statistical analysis.

### 1.2 Scope

- Web application using Streamlit
- Process X-ray images, predict TB, manage patient data
- Display statistics, charts, and model information

## 2. System Overview

- Frontend: Streamlit
- Backend: Python
- AI model: TensorFlow / Keras MobileNetV3
- Database: SQLite
- Email service: SMTP (Gmail)

## 3. Functional Requirements

| ID     | Function                | Description                                                            |
| ------ | ----------------------- | ---------------------------------------------------------------------- |
| FR-001 | Authentication          | Registration, login, password reset, change password, password hashing |
| FR-002 | Patient Data Management | Upload X-ray, save, edit, delete, view history                         |
| FR-003 | AI Prediction           | Predict TB, return label and probability, threshold 0.8                |
| FR-004 | Data Visualization      | Charts, statistics, number of cases by prediction                      |
| FR-005 | Model Info              | Display architecture, input/output, model load status                  |
| FR-006 | Export                  | Export CSV and ZIP (data + images)                                     |

## 4. Non-Functional Requirements

| Category        | Requirement                                    |
| --------------- | ---------------------------------------------- |
| Performance     | Model load < 10s, prediction < 5s              |
| Usability       | User-friendly and intuitive web interface      |
| Security        | Password hashing, OTP email for password reset |
| Maintainability | Modular Python code, easily extensible         |
| Portability     | Runs locally or on Streamlit Cloud             |

## 5. Data Requirements

### Input

- X-ray images (PNG/JPG)
- Patient info: Name, Birthday, Phone, Address

### Output

- Prediction: Normal / Tuberculosis
- Prediction probability
- Stored in SQLite for management and reporting

## 6. Constraints

- Pre-trained AI model only
- Offline processing (no real-time hospital system integration)
- Deploy on local machine or Streamlit Cloud

## 7. Assumptions

- Users have basic computer skills
- Images are valid and readable
- SMTP email works for OTP

## 8. Error Handling

- Invalid file → warning to user
- Missing data → request full input
- Model load failure → display error

## 9. Success Criteria

- Login successful
- AI prediction accurate and stored
- Tables, statistics, and charts display correctly
- CSV/ZIP export works for data + images
