import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Load your pre-trained model and scaler
model = joblib.load("classification_model_housing.pkl")
scaler = joblib.load("h_scaler.pkl")
columns = joblib.load("h_X_train.pkl")

# Function for prediction
def predict_loan_default(input_data):
    input_data_scaled = scaler.transform(input_data)  # Scaling the input
    prediction = model.predict(input_data_scaled)  # Prediction
    return prediction[0]

# Function to convert image to base64 string
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Function to set background image
def set_background(image_path):
    base64_str = get_base64_image(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_background("background.jpg")

# App header and description
st.title('Loan Default Risk Prediction')
st.write('This app predicts whether a loan is at risk of default based on customer and loan details.')

# Input form for the user
with st.form(key='loan_form'):
    qspurposedes = st.selectbox('Loan Purpose', ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
    qsector = st.selectbox('Industry Sector', ['OTHER SERVICES', 'CONSTRUCTION & INFRASTRUCTURE', 'TRADERS', 'FINANCIAL',  'MANUFACTURING & LOGISTIC','CONSUMPTION','PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV','AGRICULTURE & FISHING', 'TECHNOLOGY & INNOVATION','TOURISM'])
    lnbase = st.selectbox('Customer Group', ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
    sex = st.selectbox('Gender', ['M', 'F'])
    lnpayfreq = st.selectbox('Payment Frequency', ['2', '5', '6','12'])
    credit_card_used = st.radio('Credit Card Used', ['No', 'Yes'])
    debit_card_used = st.radio('Debit Card Used', ['No', 'Yes'])
    lnperiod_category = st.selectbox('Loan Period Category', ['Short-term', 'Medium-term', 'Long-term'])
    
    # Numerical inputs
    lnamount = st.text_input('Loan Amount')
    lninstamt = st.text_input('Installment Amount')
    average_sagbal = st.text_input('Average Savings Account Balance')
    age = st.text_input('Age')
    lnintrate = st.text_input('Interest Rate')

    submit_button = st.form_submit_button(label='Predict Default Risk')

# Show user input summary before prediction
if submit_button:
    st.write("### Your Input Summary:")
    st.write(f"**Loan Purpose**: {qspurposedes}")
    st.write(f"**Sector**: {qsector}")
    st.write(f"**Base**: {lnbase}")
    st.write(f"**Gender**: {sex}")
    st.write(f"**Payment Frequency**: {lnpayfreq}")
    st.write(f"**Used Credit Card**: {credit_card_used}")
    st.write(f"**Used Debit Card**: {debit_card_used}")
    st.write(f"**Loan Period Category**: {lnperiod_category}")
    st.write(f"**Loan Amount**: {lnamount}")
    st.write(f"**Installment Amount**: {lninstamt}")
    st.write(f"**Average Savings Account Balance**: {average_sagbal}")
    st.write(f"**Age**: {age}")
    st.write(f"**Interest Rate**: {lnintrate}")

    # Create a DataFrame from user inputs
    user_input = pd.DataFrame({
        'LNAMOUNT': [float(lnamount)] if lnamount else [0],
        'LNINTRATE': [float(lnintrate)] if lnintrate else [0],
        'LNINSTAMT': [float(lninstamt)] if lninstamt else [0],
        'AGE': [int(age)] if age else [0],
        'AVERAGE_SAGBAL': [float(average_sagbal)] if average_sagbal else [0],
        'QSPURPOSEDES': [qspurposedes],
        'QS_SECTOR': [qsector],
        'LNBASELDESC': [lnbase],
        'SEX': [sex],
        'LNPAYFREQ': [lnpayfreq],
        'CREDIT_CARD_USED': [credit_card_used],
        'DEBIT_CARD_USED': [debit_card_used],
        'LNPERIOD_CATEGORY': [lnperiod_category]
    })

    # Apply one-hot encoding to categorical inputs
    user_input = pd.get_dummies(user_input, columns=['QSPURPOSEDES', 'QS_SECTOR', 'LNBASELDESC', 'SEX', 'LNPAYFREQ', 'CREDIT_CARD_USED', 'DEBIT_CARD_USED', 'LNPERIOD_CATEGORY'], drop_first=True)

    # Check and add missing columns from the training set (to ensure consistency)
    missing_cols = set(columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0  # Add missing columns with value 0

    # Reorder columns to match the original training data
    user_input = user_input[columns]

    # Make the prediction
    prediction = predict_loan_default(user_input)

    # Display the result
    if prediction == 1:
        st.write("Prediction: The loan is at risk of default.")
    else:
        st.write("Prediction: The loan is not at risk of default.")
