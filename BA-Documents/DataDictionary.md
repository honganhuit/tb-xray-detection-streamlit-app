# Data Dictionary – X-ray AI Patient Analytics

---

## 1. Table: `users`

| Column    | Data Type | Required | Description                                    |
| --------- | --------- | -------- | ---------------------------------------------- |
| Username  | TEXT      | Yes      | Unique username used for login (Primary Key)   |
| Password  | TEXT      | Yes      | Hashed password using bcrypt                   |
| Email     | TEXT      | Yes      | Unique email address for user account          |
| ResetCode | TEXT      | No       | Temporary verification code for password reset |
| CreatedAt | TEXT      | Yes      | Timestamp when the account was created         |

### Constraints

- `Username` must be unique
- `Email` must be unique
- `Password` must be stored in hashed format (bcrypt)
- `ResetCode` is nullable and cleared after password reset
- `CreatedAt` format: `YYYY-MM-DD HH:MM:SS`

---

## 2. Table: `predictions`

| Column     | Data Type | Required | Description                                             |
| ---------- | --------- | -------- | ------------------------------------------------------- |
| PatientID  | TEXT      | Yes      | Unique identifier for each patient record (Primary Key) |
| Name       | TEXT      | Yes      | Full name of the patient                                |
| Birthday   | TEXT      | Yes      | Patient's date of birth (YYYY-MM-DD)                    |
| Phone      | TEXT      | Yes      | Contact phone number                                    |
| Address    | TEXT      | Yes      | Patient's address                                       |
| Prediction | TEXT      | Yes      | AI prediction result ("Bình thường" / "Bệnh lao")       |
| ImageFile  | TEXT      | Yes      | Stored filename of uploaded X-ray image                 |
| CreatedAt  | TEXT      | Yes      | Timestamp when prediction was saved                     |

---

### Constraints

- `PatientID` must be unique (UUID-based)
- `Name` cannot be empty
- `Birthday` must follow format `YYYY-MM-DD`
- `Phone` must match regex: `^\+?\d{9,15}$`
- `Prediction` values:
  - "Bình thường"
  - "Bệnh lao"
- `ImageFile` must exist in directory: `saved_images/`
- `CreatedAt` format: `YYYY-MM-DD HH:MM:SS`

---

## 3. Relationships

- No direct foreign key relationship between `users` and `predictions`
- Each prediction is independent and not linked to a specific user

---

## 4. Data Storage Details

| Component     | Storage Type                      |
| ------------- | --------------------------------- |
| Database      | SQLite (`tb_predictions.db`)      |
| Image Storage | Local directory (`saved_images/`) |

---

## 5. Notes

- The system does not implement user-based ownership of patient records
- All users have access to all prediction data
- Data is stored locally, suitable for small-scale applications only

---
