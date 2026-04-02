# Business Requirements Document (BRD) – X-ray AI Pulmonary Classification

## 1. Project Overview

X-ray AI Pulmonary Classification is a web application that assists doctors and medical staff in classifying pulmonary tuberculosis (TB) from X-ray images. The system provides AI predictions, stores patient information, displays history, and visualizes data.

## 2. Business Objectives

- Reduce diagnosis time for pulmonary TB
- Support medical decision-making with AI predictions
- Manage patient information and diagnosis results
- Visualize data through tables and charts

## 3. Scope

### In Scope

- User registration, login, password reset, change password
- Upload X-ray images and predict using AI
- Save, edit, delete, and view patient history
- Display statistical charts of predictions
- Show AI model information

### Out of Scope

- Integration with hospital systems or external databases
- Training or fine-tuning AI model by end-users
- Real-time prediction outside the web app
- Advanced role-based access control

## 4. Stakeholders

| Role         | Type                     | Responsibility                                      |
| ------------ | ------------------------ | --------------------------------------------------- |
| Student / BA | Project Owner            | Define requirements, test the application           |
| Developer    | Solo Developer           | Implement system and integrate AI                   |
| User         | Doctor / Medical Student | Upload images, receive predictions, manage patients |

## 5. User Roles

| Role             | Description                                         |
| ---------------- | --------------------------------------------------- |
| User             | Upload X-ray, view predictions, manage patient data |
| Admin (implicit) | Maintain system, manage users                       |

## 6. High-Level Workflow

1. User registration/login
2. Upload X-ray image
3. AI predicts TB status
4. Save patient information and results
5. Edit or delete patient information
6. Display statistics and charts
7. View AI model information

## 7. High-Level Business Requirements

| ID     | Requirement Description                                   | Priority |
| ------ | --------------------------------------------------------- | -------- |
| BR-001 | The system allows registration, login, and password reset | High     |
| BR-002 | Upload X-ray images and predict using AI                  | High     |
| BR-003 | Save patient information and prediction results           | High     |
| BR-004 | Edit, delete, and view patient history                    | Medium   |
| BR-005 | Display statistics and charts                             | Medium   |
| BR-006 | Show AI model information                                 | Low      |

## 8. Assumptions

- Users have basic computer skills
- AI model is pre-trained
- X-ray images are in valid format

## 9. Constraints

- Web-based application (Streamlit)
- Predictions only use pre-trained AI model
- No hospital system integration

## 10. Success Criteria

- Login and image upload are successful
- AI prediction is accurate and results are stored
- Tables, statistics, and charts display correctly
- CSV/ZIP export containing data and images works correctly
