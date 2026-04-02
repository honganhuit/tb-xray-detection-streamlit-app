# Test Scenarios – X-ray AI Patient Analytics

## User Registration / Login

- **TC-001:** Register with valid username/email/password → Success
- **TC-002:** Register with existing username/email → Error
- **TC-003:** Login with correct credentials → Success
- **TC-004:** Login with incorrect password → Error
- **TC-005:** Reset password via email → Success

## Prediction

- **TC-006:** Upload valid X-ray → Returns prediction
- **TC-007:** Upload invalid file format → Error
- **TC-008:** Check probability threshold → Correct labeling

## Patient Management

- **TC-009:** Save patient information → Database updated
- **TC-010:** Edit patient information → Changes saved
- **TC-011:** Delete patient → Record removed

## Reporting

- **TC-012:** View charts → Displays correct statistics
- **TC-013:** Export CSV → File contains all patient records
- **TC-014:** Export ZIP → Contains CSV and images
