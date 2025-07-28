import streamlit as st
import pandas as pd
import joblib
import base64

#Background
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: black;
            background-image: url("data:image/png;base64,{encoded}");
            background-repeat: no-repeat;
            background-position: bottom left;
            background-size: 300px auto;
        }}

        .stSelectbox > div[data-baseweb="select"] > div {{
            max-height: 200px;
            overflow-y: auto;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("laptop_bg.png")


# Load model and processed feature columns
model = joblib.load("brf.pkl")
processed_columns = joblib.load("processed_columns.pkl")

# Load your dataset (used only for dropdown options)
df = pd.read_csv("laptop_price.csv", encoding="latin1")

# Extract original columns used for dropdowns
# These are columns before OneHotEncoding — infer them from processed_columns
categorical_base_columns = sorted(set(col.split('_')[0] for col in processed_columns if '_' in col))

# Generate dropdowns
st.title("💻 Laptop Price Predictor (SGD)")
st.markdown("### Select Laptop Specifications:")

user_input = {}

for col in categorical_base_columns:
    if col in df.columns:
        options = sorted(df[col].dropna().unique())
        user_input[col] = st.selectbox(f"{col}", options, key=col)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# One-hot encode like in training
input_encoded = pd.get_dummies(input_df)

# Align with training columns
for col in processed_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[processed_columns]  # Maintain column order

# Prediction
if st.button("🔍 Predict Laptop Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Price: **SGD ${prediction:,.2f}**")
