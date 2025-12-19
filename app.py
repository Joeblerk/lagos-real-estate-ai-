
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Lagos Real Estate AI", layout="centered")

st.title("🏠 Lagos Real Estate Price Predictor")
st.write("Predict property prices in Lagos using AI")

# Load data
data = pd.read_csv("lagos.csv")

# Encode categorical data
le_location = LabelEncoder()
le_type = LabelEncoder()

data["Location"] = le_location.fit_transform(data["Location"])
data["Property_Type"] = le_type.fit_transform(data["Property_Type"])

# Train model
X = data[["Location", "Property_Type", "Bedrooms"]]
y = data["Price"]

model = LinearRegression()
model.fit(X, y)

# User inputs
location = st.selectbox("Select Location", le_location.classes_)
property_type = st.selectbox("Select Property Type", le_type.classes_)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)

if st.button("Predict Price"):
    loc_num = le_location.transform([location])[0]
    type_num = le_type.transform([property_type])[0]
    prediction = model.predict([[loc_num, type_num, bedrooms]])
    st.success(f"Estimated Price: ₦{int(prediction[0]):,}")
