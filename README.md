📌 Project: Credit Card Fraud Detection API

This project builds and deploys a machine learning model to detect fraudulent credit card transactions.

🔹 Problem
Credit card fraud is a highly imbalanced classification problem where fraudulent transactions are very rare.
Missing a fraud (False Negative) is costly, so the model focuses on high recall.

🔹 Dataset
Source: Credit Card Transactions dataset
Rows: ~284,000
Target column: Class
0 → Normal transaction
1 → Fraud transaction

🔹 Model
Logistic Regression
class_weight="balanced" to handle class imbalance
Features Time and Amount are scaled
Other features are passed without modification

🔹 Threshold Selection
Instead of default threshold 0.5, a threshold of 0.3 is used to reduce false negatives and increase fraud detection recall.

🔹 Deployment
The trained model is deployed using Flask as a REST API.

🔹 API Endpoints
GET /
Health check
POST /predict
Input: list of transaction features
Output: fraud probability and prediction
