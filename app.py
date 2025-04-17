
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Web page title
st.title("💳 AI-Powered Credit Card Fraud Detection")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()

# Show dataset sample
st.subheader("Sample Data")
st.write(data.sample(5))

# Data prep
X = data.drop(['Class'], axis=1)  # Features
y = data['Class']                # Target

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

st.subheader("📊 Model Evaluation")
st.write(pd.DataFrame(report).transpose())

# Interactive Prediction
st.subheader("🧪 Try Your Own Transaction")

# Create input sliders for transaction features
input_features = []
for feature in X.columns:
    value = st.number_input(f"{feature}", value=float(X[feature].mean()), step=1.0)
    input_features.append(value)

# Predict Button
if st.button("Predict Fraud or Not"):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    result = 'Fraud ⚠️' if prediction[0] == 1 else 'Legit ✅'
    st.subheader(f"Prediction: {result}")
