import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Define a function to make predictions
def predict(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    # Create a dataframe with all features
    df = pd.DataFrame([[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]],
                      columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

    # Encode Gender
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['HasCrCard'] = df['HasCrCard'].map({'No': 0, 'Yes': 1})
    df['IsActiveMember'] = df['IsActiveMember'].map({'No': 0, 'Yes': 1})
    # Apply ColumnTransformer and StandardScaler
    df = ct.transform(df)
    df = sc.transform(df)
    prediction = rf.predict_proba(df)[:, 1]
    return prediction

# Load the saved model and transformers
rf = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('scaler.pkl', 'rb'))
ct = pickle.load(open('encoder.pkl', 'rb'))

# Background image and styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #eed4aa;
    opacity: 1.0;
    background-image: linear-gradient(30deg, #986b6e 12%, transparent 12.5%, transparent 87%, #986b6e 87.5%, #986b6e), linear-gradient(150deg, #986b6e 12%, transparent 10.5%, transparent 87%, #986b6e 87.5%, #986b6e), linear-gradient(30deg, #986b6e 12%, transparent 12.5%, transparent 87%, #986b6e 87.5%, #986b6e), linear-gradient(150deg, #986b6e 12%, transparent 12.5%, transparent 87%, #986b6e 87.5%, #986b6e), linear-gradient(60deg, #986b6e77 25%, transparent 25.5%, transparent 75%, #986b6e77 75%, #986b6e77), linear-gradient(60deg, #986b6e77 25%, transparent 25.5%, transparent 75%, #986b6e77 75%, #986b6e77);
    background-size: 48px 84px;
    background-position: 0 0, 0 0, 24px 42px, 24px 42px, 0 0, 24px 42px;
    background-size: cover;
    height: 100%;
    position: absolute;
    width: 100%;
}
.css-1dp5vir, .css-1j77c0c, .css-1cpxqw2, .css-1djfy51 {
    font-size: 20px;
    font-weight: bold;
    color: #000000;
}
.css-1cpxqw2, .css-1djfy51 {
    margin-bottom: 10px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and styling
title_style = """
<style>
.title {
    font-size: 50px;  /* Increase font size */
    font-weight: bold;
    color: black;  
    text-decoration: underline;  /* Underline the text */
    text-align: center;
    margin-bottom: 20px;
    font-family: 'Pacifico', cursive;  /* Use a cursive font */
}
.message {
    font-size: 20px;
    font-weight: normal;
    color: #000000;
    text-align: center;
    margin-bottom: 20px;
    font-family: 'Roboto', sans-serif;
}
input[type="text"], input[type="number"], select {
    border-radius: 5px;
    padding: 10px;
    border: 1px solid #ccc;
    width: 100%;
}
.stButton button {
    background-color: #4B0082;
    color: white;
    border-radius: 5px;
    padding: 10px;
    border: none;
    cursor: pointer;
    font-size: 16px;
}
.stButton button:hover {
    background-color: #3e0071;
}
</style>
"""

st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<div class='title'>CHURN PREDICTION APP</div>", unsafe_allow_html=True)
st.markdown("<div class='message'>[Please fill the below options]</div>", unsafe_allow_html=True)

show_guide = st.checkbox("Show Guide")

if show_guide:
    st.markdown("""
    <div style="background-color:rgb(211,211,211); border: 2px solid #ccc; border-radius: 10px; padding: 20px; margin-top: 20px;">
    <h2>Guide to Churn Prediction App</h2>
    <p><strong>Geography:</strong> Select the country where the customer is located.</p>
    <p><strong>Gender:</strong> Choose the gender of the customer.</p>
    <p><strong>Age:</strong> Enter the age of the customer.</p>
    <p><strong>Credit Score:</strong> Enter the customer's credit score.</p>
    <p><strong>Tenure:</strong> Enter the number of years the customer has been with the bank.</p>
    <p><strong>Bank Balance:</strong> Enter the customer's bank balance.</p>
    <p><strong>Number of Products:</strong> Enter the number of products the customer has with the bank.</p>
    <p><strong>Credit Card Availability:</strong> Select whether the customer has a credit card (Yes/No).</p>
    <p><strong>Active Member:</strong> Select whether the customer is an active member (Yes/No).</p>
    <p><strong>Estimated Salary:</strong> Enter the customer's estimated salary.</p>
    <p>Click the <strong>Predict</strong> button to get the probability of the customer leaving the bank.</p>
    </div>
    """, unsafe_allow_html=True)
    
# Input fields
st.markdown("<h3 style='font-size:20px;'>GEOGRAPHY</h3>", unsafe_allow_html=True)
Geography = st.selectbox("", ['France', 'Spain', 'Germany'], key="Geography")

st.markdown("<h3 style='font-size:20px;'>GENDER</h3>", unsafe_allow_html=True)
Gender = st.selectbox("", ['Female', 'Male'], key="Gender")

st.markdown("<h3 style='font-size:20px;'>AGE</h3>", unsafe_allow_html=True)
Age = st.text_input("", key="Age", placeholder="Please fill")

st.markdown("<h3 style='font-size:20px;'>CREDIT SCORE</h3>", unsafe_allow_html=True)
CreditScore = st.number_input("", key="CreditScore", placeholder="Please fill")

st.markdown("<h3 style='font-size:20px;'>TENURE</h3>", unsafe_allow_html=True)
Tenure = st.text_input("", key="Tenure", placeholder="Please fill")

st.markdown("<h3 style='font-size:20px;'>BANK BALANCE</h3>", unsafe_allow_html=True)
Balance = st.text_input("", key="Balance", placeholder="Please fill")

st.markdown("<h3 style='font-size:20px;'>NUMBER OF PRODUCTS </h3>", unsafe_allow_html=True)
NumOfProducts = st.text_input("", key="NumOfProducts", placeholder="Please fill")

st.markdown("<h3 style='font-size:20px;'>CREDIT CARD AVAILABILITY (YES/NO) </h3>", unsafe_allow_html=True)
HasCrCard = st.selectbox("", ['No', 'Yes'], key="HasCrCard")

st.markdown("<h3 style='font-size:20px;'>ACTIVE MEMBER (YES/NO) </h3>", unsafe_allow_html=True)
IsActiveMember = st.selectbox("", ['No', 'Yes'], key="IsActiveMember")

st.markdown("<h3 style='font-size:20px;'>ESTIMATED SALARY </h3>", unsafe_allow_html=True)
EstimatedSalary = st.text_input("", key="EstimatedSalary", placeholder="Please fill")

# Create a button to make a prediction
st.markdown("<div class='message'>(For prediction click [Predict] below)</div>", unsafe_allow_html=True)
if st.button("Predict"):
    try:
        # Check if any field is empty and handle accordingly
        CreditScore = float(CreditScore) if CreditScore else 0.0
        Age = float(Age) if Age else 0.0
        Tenure = float(Tenure) if Tenure else 0.0
        Balance = float(Balance) if Balance else 0.0
        NumOfProducts = float(NumOfProducts) if NumOfProducts else 0.0
        EstimatedSalary = float(EstimatedSalary) if EstimatedSalary else 0.0

        prediction = predict(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        st.write("**🔳 Probability of Customer Leaving:** ", prediction[0])
        if prediction > 0.5:
            st.write("**🔳 Customer will likely leave**")
        else:
            st.write("**🔳 Customer will likely stay**")
    except ValueError:
        st.error("Please ensure all numerical inputs are correctly filled.")

# JavaScript code for selectboxes
clear_input_script = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    var inputs = document.getElementsByTagName("input");
    for (var i = 0; i < inputs.length; i++) {
        inputs[i].addEventListener("focus", function(event) {
            if (this.value == "Please fill") {
                this.value = "";
            }
        });
    }
});
</script>
"""
st.components.v1.html(clear_input_script)
