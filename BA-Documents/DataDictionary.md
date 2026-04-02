# Data Dictionary – X-ray AI Patient Analytics

## Table: `users`

| Column    | Type | Description                |
| --------- | ---- | -------------------------- |
| Username  | TEXT | Unique login username      |
| Password  | TEXT | Hashed password            |
| Email     | TEXT | Unique user email          |
| ResetCode | TEXT | Code for password reset    |
| CreatedAt | TEXT | Account creation timestamp |

## Table: `predictions`

| Column     | Type | Description                                   |
| ---------- | ---- | --------------------------------------------- |
| PatientID  | TEXT | Unique patient ID                             |
| Name       | TEXT | Full name of the patient                      |
| Birthday   | TEXT | Date of birth                                 |
| Phone      | TEXT | Contact phone number                          |
| Address    | TEXT | Patient address                               |
| Prediction | TEXT | Prediction result ("Normal" / "Tuberculosis") |
| ImageFile  | TEXT | Filename of uploaded X-ray                    |
| CreatedAt  | TEXT | Timestamp when prediction was saved           |
