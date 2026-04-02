# System Requirements Specification (SRS) – X-ray AI Pulmonary Classification

---

## 1. Introduction

### 1.1 Purpose

This document defines the system requirements for the X-ray AI Pulmonary Classification application.  
The system is designed to process chest X-ray images, predict pulmonary tuberculosis (TB) using a pre-trained AI model, manage patient data, and provide statistical visualization.

---

### 1.2 Scope

The system includes:

- A web-based application built with Streamlit
- Image processing and AI-based TB prediction
- Patient data management (CRUD operations)
- Data visualization (charts and statistics)
- Model information display
- Data export functionality (CSV and ZIP)

---

## 2. System Overview

| Component     | Technology                       |
| ------------- | -------------------------------- |
| Frontend      | Streamlit                        |
| Backend       | Python                           |
| AI Model      | TensorFlow / Keras (MobileNetV3) |
| Database      | SQLite                           |
| Email Service | SMTP (Gmail)                     |
| File Storage  | Local directory (saved_images/)  |

---

## 3. Functional Requirements

| ID     | Requirement Description                                                               |
| ------ | ------------------------------------------------------------------------------------- |
| FR-001 | The system shall allow users to register with username, email, and password           |
| FR-002 | The system shall validate user credentials using encrypted password hashing (bcrypt)  |
| FR-003 | The system shall allow users to log in and log out                                    |
| FR-004 | The system shall allow users to reset passwords via email verification code           |
| FR-005 | The system shall validate email format and password strength                          |
| FR-006 | The system shall allow users to upload X-ray images (JPG, JPEG, PNG)                  |
| FR-007 | The system shall preprocess uploaded images to 224x224 RGB format                     |
| FR-008 | The system shall perform AI prediction using a pre-trained model                      |
| FR-009 | The system shall return prediction results (label and probability)                    |
| FR-010 | The system shall classify results based on threshold (0.8)                            |
| FR-011 | The system shall collect and store patient information (Name, DOB, Phone, Address)    |
| FR-012 | The system shall generate a unique PatientID for each record                          |
| FR-013 | The system shall store prediction results and images in SQLite and local storage      |
| FR-014 | The system shall allow users to view patient records in a table                       |
| FR-015 | The system shall allow users to edit patient information                              |
| FR-016 | The system shall allow users to delete individual patient records                     |
| FR-017 | The system shall allow users to delete all patient records                            |
| FR-018 | The system shall display X-ray images associated with patient records                 |
| FR-019 | The system shall generate statistical charts (bar chart and pie chart)                |
| FR-020 | The system shall allow exporting patient data as CSV                                  |
| FR-021 | The system shall allow exporting patient data and images as ZIP                       |
| FR-022 | The system shall display AI model information (architecture, input size, load status) |
| FR-023 | The system shall download the model from Google Drive if not available locally        |

---

## 4. Non-Functional Requirements

| Category        | Requirement                                                         |
| --------------- | ------------------------------------------------------------------- |
| Performance     | Model loading time shall be under 10 seconds (after first download) |
| Performance     | Prediction response time shall be under 5 seconds                   |
| Usability       | The UI shall be simple, intuitive, and user-friendly                |
| Security        | Passwords shall be securely hashed using bcrypt                     |
| Security        | Email verification code shall be used for password reset            |
| Reliability     | The system shall handle invalid input without crashing              |
| Maintainability | Code shall be modular and structured for easy updates               |
| Portability     | The system shall run on local machine and Streamlit Cloud           |
| Scalability     | The system supports small-scale usage (SQLite limitation)           |

---

## 5. Data Requirements

### 5.1 Input Data

- X-ray images (JPG, JPEG, PNG)
- Patient information:
  - Name
  - Date of Birth
  - Phone Number
  - Address

---

### 5.2 Output Data

- Prediction result:
  - Label (Normal / Tuberculosis)
  - Probability score
- Stored data:
  - PatientID
  - Patient info
  - Prediction result
  - Image file path
  - Timestamp

---

### 5.3 Data Storage

- Database: SQLite (`tb_predictions.db`)
- Image storage: Local directory (`saved_images/`)

---

## 6. Constraints

- The system uses a pre-trained AI model only (no training functionality)
- The application is limited to a web-based interface (Streamlit)
- SQLite is used for lightweight data storage (not suitable for large-scale systems)
- Internet connection is required for initial model download

---

## 7. Assumptions

- Users have basic computer and internet usage skills
- Uploaded images are valid chest X-ray images
- Email service (SMTP) is properly configured
- The AI model file is accessible via Google Drive

---

## 8. Error Handling

| Scenario              | System Behavior              |
| --------------------- | ---------------------------- |
| Missing input fields  | Display warning message      |
| Invalid phone number  | Show validation error        |
| Invalid file format   | Reject file and notify user  |
| Model load failure    | Display error message        |
| Email sending failure | Display error notification   |
| Database error        | Prevent crash and show error |

---

## 9. Success Criteria

- Users can successfully register, log in, and reset passwords
- X-ray images are uploaded and processed correctly
- AI predictions are generated accurately and stored
- Patient records can be viewed, edited, and deleted
- Charts and statistics are displayed correctly
- CSV and ZIP export features function properly
- System operates without critical runtime errors

---
