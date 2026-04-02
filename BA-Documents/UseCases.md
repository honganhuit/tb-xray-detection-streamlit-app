# Use Cases – X-ray AI Patient Analytics

## Use Case 1: Register / Login

**Actor:** User  
**Description:** User registers, logs in, or resets password

**Steps:**

1. User enters username, email, and password
2. System validates input
3. System stores hashed password
4. User logs in

## Use Case 2: Upload X-ray & Predict

**Actor:** User (Doctor)  
**Description:** Predict tuberculosis from X-ray image

**Steps:**

1. User enters patient information
2. User uploads X-ray image
3. System preprocesses the image
4. System predicts and displays result
5. Result is saved in the database

## Use Case 3: Edit/Delete Patient Info

**Actor:** User  
**Description:** Manage patient records

**Steps:**

1. User selects a patient record
2. Edit information or delete record
3. System updates the database

## Use Case 4: View Statistics

**Actor:** User  
**Description:** View summary of predictions

**Steps:**

1. System aggregates predictions
2. Displays charts (bar, pie)
3. Option to export CSV/ZIP
