import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load model and encoders
# -------------------------
model = joblib.load('model.pkl')             # Your trained sklearn model
scaler = joblib.load('scaler.pkl')           # StandardScaler
onehot_encoder_geo = joblib.load('onehot_encoder_geo.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')

# Get original training columns for scaler
training_columns = scaler.feature_names_in_

# -------------------------
# Streamlit UI
# -------------------------
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=0, max_value=120, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
num_of_products = st.number_input('Number of Products',1,4)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
credit_score = st.number_input('Credit Score', min_value=0.0, value=0.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=0.0)

# -------------------------
# Prepare input data
# -------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [1],  # Default active
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[input_data['Geography'][0]]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Drop original Geography and add encoded columns
input_data = pd.concat([input_data.drop("Geography", axis=1), geo_encoded_df], axis=1)

# -------------------------
# Align columns with training
# -------------------------
for col in training_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[training_columns]  # Reorder to match training

# -------------------------
# Scale and predict
# -------------------------
input_data_scaled = scaler.transform(input_data)
prediction_proba = model.predict(input_data_scaled)[0]

if prediction_proba > 0.5:
    st.success("The Customer is likely to churn.")
else:
    st.success("The Customer is not likely to churn.")

st.write(f"Prediction Probability: {prediction_proba:.2f}")
