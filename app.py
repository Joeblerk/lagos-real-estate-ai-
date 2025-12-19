import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# App title
st.title("🏠 Lagos Real Estate Price Predictor")

# Load dataset
data = pd.read_csv("lagos.csv")

# Encode categorical variables
le_location = LabelEncoder()
le_type = LabelEncoder()

data["Location"] = le_location.fit_transform(data["Location"])
data["Property_Type"] = le_type.fit_transform(data["Property_Type"])

# Features and target
X = data[["Location", "Property_Type", "Bedrooms"]]
y = data["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User inputs
location = st.selectbox("Select Location", le_location.classes_)
property_type = st.selectbox("Select Property Type", le_type.classes_)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)

# Predict button
if st.button("Predict Price"):
    loc_num = le_location.transform([location])[0]
    type_num = le_type.transform([property_type])[0]

    prediction = model.predict([[loc_num, type_num, bedrooms]])
    st.success(f"Estimated Price: ₦{int(prediction[0]):,}")

