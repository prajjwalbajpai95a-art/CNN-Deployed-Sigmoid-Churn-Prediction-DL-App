# 📊 Customer Churn Prediction System (ANN)

An interactive Machine Learning & Deep Learning web application built with **Artificial Neural Network (ANN)** using **Sigmoid Activation Function** to predict whether a customer will churn (leave the service) or stay[cite: 1, 2].

---

## 🌟 Live Demo
🔗 **Live Demo:** https://cnn-deployed-sigmoid-churn-prediction-dl-app-jb2desjqmudi68duy.streamlit.app/


---

## 📌 Project Overview
Customer Churn Prediction helps businesses identify high-risk customers who are likely to cancel their accounts. 

Using an **Artificial Neural Network (ANN)** trained for **Binary Classification**, this model predicts the churn probability output between `0.0` and `1.0` via a **Sigmoid Activation Function**[cite: 1, 2]:
- **Probability < 0.5**: ✅ **Low Churn Risk** (Customer stays)
- **Probability >= 0.5**: ⚠️ **High Churn Risk** (Customer leaves)

---

## 🚀 Key Features
- **Deep Learning Model**: Multi-layer Sequential ANN with ReLU hidden layers and a Sigmoid output layer[cite: 2].
- **Feature Standardization**: Normalized inputs using `StandardScaler` for accurate predictions[cite: 2].
- **Interactive UI**: User-friendly web interface created with **Streamlit**[cite: 2].
- **Modern Asset Format**: Model saved in Keras native `.keras` format[cite: 2, 3].

---

## 🛠️ Tech Stack
- **Language**: Python
- **Deep Learning**: TensorFlow / Keras[cite: 2]
- **Data Processing**: Pandas, NumPy, Scikit-Learn[cite: 2]
- **Web Interface**: Streamlit[cite: 2]

---

## 📂 Project Structure
```text
Customer_Churn_Prediction/
│
├── ANN-Sigmoid_Churn-Model_DL.ipynb   # Jupyter Notebook with Model Training[cite: 2, 3]
├── app.py                             # Streamlit UI Script[cite: 3]
├── churn_ann_model.keras               # Trained ANN Model File[cite: 2, 3]
├── scaler.pkl                          # Standard Scaler Object File[cite: 2, 3]
├── requirements.txt                    # Project Dependencies[cite: 2, 3]
└── README.md                           # Documentation

⚙️ Local Setup & Installation
1. Repository Clone Karein
Bash
git clone https://github.com/prajjwalbajpai95a-art/CNN-Deployed-Sigmoid-Churn-Prediction-DL-App.git
cd Customer_Churn_Prediction
2. Virtual Environment Banayein & Activate Karein
Bash
# Environment create karein
python -m venv venv

# Windows (CMD/PowerShell)
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3. Dependencies Install Karein
Bash
pip install -r requirements.txt
🏃 How to Run the App
Make sure your virtual environment is active.

Launch the Streamlit application:

Bash
streamlit run app.py
Open http://localhost:8501 in your browser.

👨‍💻 Author
Prajjwal bajpai

🐙 GitHub: @AbhishekMishra

💼 LinkedIn: Abhishek Mishra
