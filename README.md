# 💳 Credit Card Fraud Detection

A Machine Learning-based project that detects fraudulent credit card transactions using techniques like data preprocessing, class imbalance handling, and model evaluation.

## 📌 Overview

This project demonstrates how to apply data science and machine learning to identify fraudulent credit card transactions. The dataset is highly imbalanced, with a very small percentage of fraud cases. Various strategies like data preprocessing, feature scaling, and oversampling (SMOTE) are applied to build and evaluate a classification model.

## 📁 Project Structure

credit-card-fraud-detection/ │ ├── data/ # Contains training and testing datasets │ ├── fraudTrain.csv # Training data (not uploaded to GitHub) │ └── fraudTest.csv # Testing data (excluded due to size) │ ├── models/ # Saved model files │ ├── web.py # Streamlit web app ├── train.py # Model training and evaluation script ├── preprocess.py # Data preprocessing logic ├── requirements.txt # Python dependencies └── README.md # Project documentation

markdown
Copy
Edit

> **Note:** `fraudTest.csv` is excluded from this repository due to GitHub’s file size limitations (>100 MB). To run the full project locally, please download the data from the original [source](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

## ⚙️ Features

- Logistic Regression classifier
- Data cleaning and preprocessing
- SMOTE oversampling to handle class imbalance
- Accuracy, precision, recall, F1-score, and ROC AUC evaluation
- Interactive Streamlit dashboard

## 🧠 Model & Techniques Used

- **Preprocessing:** Label encoding, feature scaling
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique)
- **Model:** Logistic Regression
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC AUC

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/sc1030/credit-card-fraud-detection.git
cd credit-card-fraud-detection
2. Create Virtual Environment & Install Dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Add Data
Download the dataset and place fraudTrain.csv and fraudTest.csv inside the data/ directory.

4. Train the Model
bash
Copy
Edit
python train.py
5. Run the Web App
bash
Copy
Edit
streamlit run web.py
🖥️ Screenshots
Add your Streamlit app screenshots here (optional)

✅ To-Do
 Add more ML models for comparison (Random Forest, XGBoost)

 Deploy Streamlit app using Streamlit Cloud

 Improve feature engineering

📚 Dataset Source
Kaggle: Credit Card Fraud Detection Dataset

👨‍💻 Author
Shyam C
BCA | Python & ML Developer | Software Engineer at CBA
🔗 LinkedIn

Feel free to suggest improvements or raise issues!

yaml
Copy
Edit

---

Want me to include badges (like Python version, Streamlit app, etc.) or a deployment link if you publish it on Streamlit Cloud?


