# Business Requirements Document (BRD) – X-ray AI Pulmonary Classification

---

## 1. Project Overview

X-ray AI Pulmonary Classification is a web-based application designed to support medical users in detecting pulmonary tuberculosis (TB) from chest X-ray images.

The system utilizes a pre-trained deep learning model to generate predictions, manage patient records, and provide data visualization for diagnostic support.

---

## 2. Business Objectives

- Reduce the time required for TB diagnosis
- Support medical decision-making with AI-assisted predictions
- Enable efficient storage and management of patient information
- Provide visualization of diagnosis results through charts and reports

---

## 3. Scope

### In Scope

- User authentication:
  - Registration
  - Login
  - Password reset via email verification
- Upload X-ray images and perform AI-based prediction
- Store patient information and prediction results
- Edit, delete, and view patient records
- Display statistical charts (bar chart and pie chart)
- Export data as CSV and ZIP (including images)
- Display AI model information

### Out of Scope

- Integration with hospital information systems (HIS)
- AI model training or retraining by users
- Real-time diagnosis outside the web application
- Advanced role-based access control (RBAC)

---

## 4. Stakeholders

| Stakeholder      | Type                 | Responsibility                                    |
| ---------------- | -------------------- | ------------------------------------------------- |
| Medical User     | Primary Stakeholder  | Upload X-ray images and review prediction results |
| System Owner     | Project Stakeholder  | Define requirements and validate system outputs   |
| System Developer | Internal Stakeholder | Design, develop, and maintain the system          |

---

## 5. User Roles

| Role | Description                                                       |
| ---- | ----------------------------------------------------------------- |
| User | Upload X-ray images, receive predictions, and manage patient data |

> Note: The system does not implement advanced role separation. All users have the same access level.

---

## 6. High-Level Workflow

1. User registers or logs into the system
2. User uploads a chest X-ray image
3. The system preprocesses the image
4. The AI model predicts TB classification
5. The system displays prediction results (label and probability)
6. The system stores patient information and prediction results
7. User can edit or delete patient records
8. The system displays statistical charts
9. User can export data (CSV / ZIP)
10. User can view AI model information

---

## 7. High-Level Business Requirements

| ID     | Requirement Description                                                              |
| ------ | ------------------------------------------------------------------------------------ |
| BR-001 | The system shall allow users to register, log in, and reset passwords                |
| BR-002 | The system shall validate user credentials securely using encrypted passwords        |
| BR-003 | The system shall allow users to upload X-ray images (JPG, PNG)                       |
| BR-004 | The system shall preprocess images before prediction                                 |
| BR-005 | The system shall perform AI-based prediction using a pre-trained model               |
| BR-006 | The system shall display prediction results including classification and probability |
| BR-007 | The system shall store patient information and prediction results                    |
| BR-008 | The system shall allow users to edit patient information                             |
| BR-009 | The system shall allow users to delete individual or all patient records             |
| BR-010 | The system shall display stored data in tabular format                               |
| BR-011 | The system shall generate statistical charts (bar chart and pie chart)               |
| BR-012 | The system shall allow exporting data as CSV                                         |
| BR-013 | The system shall allow exporting data and images as ZIP                              |
| BR-014 | The system shall display AI model information (architecture, input size, status)     |

---

## 8. Assumptions

- Users have basic computer literacy
- The AI model is pre-trained and ready for inference
- Users upload valid chest X-ray images
- Email service is configured for password reset functionality

---

## 9. Constraints

- The system is developed using the Streamlit framework
- Predictions rely solely on a pre-trained AI model (no training in system)
- The system uses a local SQLite database
- Internet connection is required for initial model download (Google Drive)

---

## 10. Success Criteria

- Users can successfully register, log in, and reset passwords
- X-ray images can be uploaded and processed without errors
- AI predictions are generated correctly and displayed
- Patient data is stored, updated, and deleted accurately
- Statistical charts are displayed correctly
- CSV and ZIP export functions work as expected
- The system runs smoothly without critical errors

---
