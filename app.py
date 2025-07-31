import streamlit as st
import pandas as pd
import joblib
import base64

# Background
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

# Load dataset 
df = pd.read_csv("laptop_price.csv", encoding="latin1")

# Extract original columns used for dropdowns
categorical_base_columns = sorted(set(col.split('_')[0] for col in processed_columns if '_' in col))

# Generate dropdowns
st.title("💻 Laptop Price Predictor (SGD)")
st.markdown("### Select Laptop Specifications:")

user_budget = st.number_input("Enter your budget ($):", min_value=100.0, step=50.0)

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
input_encoded = input_encoded[processed_columns]

# State for storing prediction
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.show_suggestion_btn = False

# Prediction
if st.button("🔍 Predict Laptop Price"):
    prediction = model.predict(input_encoded)[0]
    st.session_state.prediction = prediction
    st.success(f"💰 Estimated Price: **SGD ${prediction:,.2f}**")

    # Check budget and show suggest button if over budget
    if user_budget > 0 and prediction > user_budget:
        st.session_state.show_suggestion_btn = True
        st.warning(f"⚠️ This laptop exceeds your budget of SGD ${user_budget:,.2f}.")
    elif prediction <= user_budget:
        st.session_state.show_suggestion_btn = False
        st.success(f" Your selection is within your budget of SGD ${user_budget:,.2f}.")

# Suggest downgrade button (only if needed)
if st.session_state.show_suggestion_btn:
    if st.button("💡 Suggest Downgrades"):
        st.markdown("### 💡 Suggested Changes to Lower the Price:")
        suggestions = []

        # Processor downgrade
        high_end_cpus = ['i7', 'i9', 'Ryzen 7', 'Ryzen 9']
        low_end_cpus = ['i5', 'i3', 'Ryzen 3', 'Ryzen 5']
        if 'Processor' in user_input:
            if any(cpu in user_input['Processor'] for cpu in high_end_cpus):
                cheaper_cpus = [cpu for cpu in df['Processor'].unique() if any(l in cpu for l in low_end_cpus)]
                if cheaper_cpus:
                    suggestions.append(f"- **Processor**: Consider {cheaper_cpus[0]} instead of {user_input['Processor']}")

        # RAM downgrade
        if 'RAM' in user_input:
            if '16GB' in user_input['RAM'] or '32GB' in user_input['RAM']:
                cheaper_rams = [ram for ram in df['RAM'].unique() if '8GB' in ram]
                if cheaper_rams:
                    suggestions.append(f"- **RAM**: Downgrade to {cheaper_rams[0]} instead of {user_input['RAM']}")

        # Storage downgrade
        if 'Memory' in user_input:
            if '1TB' in user_input['Memory'] or 'SSD' in user_input['Memory']:
                cheaper_memories = [m for m in df['Memory'].unique() if '512GB' in m or 'HDD' in m]
                if cheaper_memories:
                    suggestions.append(f"- **Storage**: Consider {cheaper_memories[0]} instead of {user_input['Memory']}")

        # GPU downgrade
        if 'Gpu' in user_input:
            if not user_input['Gpu'].lower().startswith('intel'):
                integrated_gpus = [g for g in df['Gpu'].unique() if 'intel' in g.lower()]
                if integrated_gpus:
                    suggestions.append(f"- **Graphics**: Choose integrated GPU like {integrated_gpus[0]} instead of {user_input['Gpu']}")

        # Brand suggestion
        expensive_brands = ['Apple', 'Dell', 'MSI']
        if 'Company' in user_input:
            if user_input['Company'] in expensive_brands:
                cheaper_brands = [b for b in df['Company'].unique() if b not in expensive_brands]
                if cheaper_brands:
                    suggestions.append(f"- **Brand**: Try cost-effective brand like {cheaper_brands[0]} instead of {user_input['Company']}")

        if suggestions:
            for tip in suggestions:
                st.markdown(tip)
        else:
            st.markdown("✅ No obvious downgrades found. Try adjusting other specs manually.")
