# X-ray AI Patient Analytics

This system classifies **tuberculosis** from chest X-ray images, assisting doctors in faster and more accurate diagnosis.

It uses **Deep Learning (MobileNetV3)** combined with **Streamlit** to provide a user-friendly web interface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Objectives](#business-objectives)
3. [Scope](#scope)
   - [In Scope](#in-scope)
   - [Out of Scope](#out-of-scope)
4. [Role – Developer / Business Analyst](#role--developer--business-analyst)
5. [Deliverables](#deliverables)
6. [Business Workflow](#business-workflow)
7. [Key Business Rules](#key-business-rules)
8. [Technical Implementation](#technical-implementation)
9. [Repository Structure](#repository-structure)
10. [How to Run](#how-to-run)

---

## Project Overview

X-ray AI Patient Analytics enables:

- Detection and classification of tuberculosis from X-ray images
- Patient information management
- Storage of prediction results and images
- Visualization and statistics of diagnosis results
- Export of reports in CSV/ZIP format

---

## Business Objectives

- Support fast and accurate medical diagnosis
- Store patient information and prediction history
- Provide statistical analysis and visual charts
- Export reports for research and management purposes

---

## Scope

### In Scope

- Predict tuberculosis from X-ray images
- Manage patient information
- Display statistics and visualizations
- Export CSV and ZIP files including images

### Out of Scope

- Replace professional medical diagnosis
- Direct integration with hospital systems
- Real-time X-ray data processing

---

## Role – Developer / Business Analyst

- Designed and implemented **Streamlit user interface**
- Integrated **MobileNetV3 Deep Learning model** for tuberculosis prediction
- Configured SQLite database to store patient information and prediction results
- Implemented user management: registration, login, password reset
- Developed features for data visualization, charts, and report export
- Built workflow from prediction to result storage

---

## Deliverables

- Streamlit app (`xquang.py`)
- SQLite database (`tb_predictions.db`)
- Trained Deep Learning model (`final_modelv3.h5`)
- Folder `saved_images/` for patient X-ray images
- Key features:
  - User registration, login, password reset
  - Upload X-ray images and predict tuberculosis
  - Save and edit patient information
  - Visualize statistics and charts
  - Export CSV and ZIP reports

---

## Business Workflow

1. Login / Register
2. Enter patient information
3. Upload X-ray image
4. Predict tuberculosis
5. Display result and probability
6. Save patient information and image
7. View statistics and charts
8. Export CSV/ZIP report

---

## Key Business Rules

- X-ray images must be in JPG, JPEG, or PNG format
- Required patient information: Name, Birthday, Phone, Address
- Phone number format: 9–15 digits, optional leading +
- Prediction probability ≥ 0.8 → "Tuberculosis", < 0.8 → "Normal"

---

## Technical Implementation

- **Front-end / UI:** Streamlit
- **Back-end:** Python 3, SQLite
- **Machine Learning:** TensorFlow / Keras, MobileNetV3
- **Image Preprocessing:** Resize to 224×224, MobileNetV3 preprocessing
- **Email:** Gmail SMTP for password reset
- **Security:** Bcrypt password hashing

---

## Repository Structure

/
├── xquang.py # Main Streamlit app
├── MOBILENETV3.ipynb # Jupyter Notebook for model training
├── requirements.txt # Dependencies
├── runtime.txt # Runtime for deployment
├── README.md # Project overview
├── BA-Documents/ # Optional business analysis documents (BRD, Use Cases, etc.)
└── Data/ # Optional sample data or images

---

## How to Run

```bash
git clone https://github.com/honganhuit/tb-xray-detection-streamlit-app.git
cd tb-xray-detection-streamlit-app
pip install -r requirements.txt
streamlit run xquang.py
```
