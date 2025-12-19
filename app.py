import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.title("🏠 Lagos Real Estate Price & Rent Predictor")
st.write("Predict property prices and average monthly rent in Lagos using AI")

# Load dataset
data = pd.read_csv("lagos.csv")

# Encode categorical variables
le_location = LabelEncoder()
le_type = LabelEncoder()

data["Location"] = le_location.fit_transform(data["Location"])
data["Property_Type"] = le_type.fit_transform(data["Property_Type"])

# Features and target for Price
X_price = data[["Location", "Property_Type", "Bedrooms"]]
y_price = data["Price"]

model_price = LinearRegression()
model_price.fit(X_price, y_price)

# Features and target for Rent
X_rent = data[["Location", "Property_Type", "Bedrooms"]]
y_rent = data["Rent"]

model_rent = LinearRegression()
model_rent.fit(X_rent, y_rent)

# User inputs
location = st.selectbox("Select Location", le_location.classes_)
property_type = st.selectbox("Select Property Type", le_type.classes_)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)

if st.button("Predict"):
    loc_num = le_location.transform([location])[0]
    type_num = le_type.transform([property_type])[0]
    
    predicted_price = model_price.predict([[loc_num, type_num, bedrooms]])
    predicted_rent = model_rent.predict([[loc_num, type_num, bedrooms]])
    
    st.success(f"💰 Estimated Sale Price: ₦{int(predicted_price[0]):,}")
    st.info(f"🏢 Estimated Monthly Rent: ₦{int(predicted_rent[0]):,}")
