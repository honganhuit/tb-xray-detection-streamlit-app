# Test Scenarios – X-ray AI Patient Analytics System

## 1. User Authentication

- **TC-001:** Register with valid username, email, and password → Account created successfully
- **TC-002:** Register with existing username or email → System displays duplication error
- **TC-003:** Login with correct credentials → User is authenticated successfully
- **TC-004:** Login with incorrect password → System displays authentication error
- **TC-005:** Reset password via email verification → Password reset successfully

---

## 2. AI Prediction

- **TC-006:** Upload a valid X-ray image (JPG/PNG) → System returns prediction result
- **TC-007:** Upload an invalid file format → System displays file validation error
- **TC-008:** Prediction probability ≥ threshold (0.8) → Label = Tuberculosis
- **TC-009:** Prediction probability < threshold (0.8) → Label = Normal

---

## 3. Patient Data Management

- **TC-010:** Save patient information with valid input → Data stored in database
- **TC-011:** Edit existing patient information → Updated data saved successfully
- **TC-012:** Delete patient record → Record removed from database
- **TC-013:** Input invalid phone number → System displays validation error

---

## 4. Reporting & Export

- **TC-014:** View statistical charts → Charts display correct aggregated data
- **TC-015:** Export CSV file → File contains complete patient dataset
- **TC-016:** Export ZIP file → ZIP includes CSV and associated images
