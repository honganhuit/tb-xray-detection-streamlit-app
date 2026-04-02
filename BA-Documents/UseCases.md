# Use Case Specification – X-ray AI Patient Analytics

---

## Use Case 1: User Authentication (Register / Login / Reset Password)

**Actor:** User

**Description:**  
User registers a new account, logs into the system, or resets password via email verification.

### Preconditions

- User has access to the web application
- Email service is configured

### Main Flow

1. User enters username, email, and password (for registration)
2. System validates input (format, password strength)
3. System hashes the password using bcrypt
4. System stores user information in database
5. User enters username and password to log in
6. System validates credentials
7. System grants access to the application

### Alternate Flow

- A1: Invalid email format → System displays error
- A2: Weak password → System displays validation message
- A3: Incorrect login credentials → System denies access
- A4: Forgot password →
  1. User enters registered email
  2. System sends verification code
  3. User enters code and new password
  4. System updates password

### Postconditions

- User is authenticated and can access system features

---

## Use Case 2: Upload X-ray & Predict

**Actor:** User

**Description:**  
User uploads a chest X-ray image and receives AI-based TB prediction.

### Preconditions

- User is logged into the system
- AI model is successfully loaded

### Main Flow

1. User enters patient information (Name, DOB, Phone, Address)
2. User uploads X-ray image (JPG/PNG)
3. System validates input and file format
4. System preprocesses the image (resize 224x224, normalize)
5. System performs prediction using AI model
6. System displays:
   - Prediction label
   - Probability score
7. System stores patient data and result in database

### Alternate Flow

- B1: Missing patient information → System displays warning
- B2: Invalid phone number → System displays validation error
- B3: No file uploaded → System requests upload
- B4: Model not loaded → System displays error

### Postconditions

- Prediction result is displayed and stored successfully

---

## Use Case 3: Manage Patient Records (View / Edit / Delete)

**Actor:** User

**Description:**  
User manages stored patient records.

### Preconditions

- User is logged in
- Patient records exist in database

### Main Flow

1. System displays patient records in a table
2. User selects a patient record
3. User edits patient information or prediction result
4. System validates updated data
5. System updates database
6. User may delete a selected record
7. System removes record from database

### Alternate Flow

- C1: Invalid input during edit → System displays error
- C2: Delete all records → System clears database

### Postconditions

- Patient records are updated or deleted successfully

---

## Use Case 4: View Statistics & Export Data

**Actor:** User

**Description:**  
User views statistical summaries and exports data.

### Preconditions

- Patient data exists in the system

### Main Flow

1. System aggregates prediction data
2. System displays:
   - Bar chart (number of cases)
   - Pie chart (distribution)
3. User selects export option
4. System generates:
   - CSV file (data only)
   - ZIP file (data + images)
5. User downloads the file

### Alternate Flow

- D1: No data available → System displays warning
- D2: Missing image file → Skip file in ZIP

### Postconditions

- User views statistics and successfully exports data

---
