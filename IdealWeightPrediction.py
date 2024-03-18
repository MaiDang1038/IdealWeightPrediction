import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# I calculated the ideal_weight for adult women based on their height using the Devine formula
st.title("Ideal Weight Prediction for Adult Female")
height = np.array([1.4, 1.45, 1.5,1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85])
ideal_weight = np.array([43.1, 45.5, 48.0, 50.4, 52.9, 55.3, 57.7, 60.2, 62.6, 65.1])

inputvar = st.number_input('Enter your height (in meters)')
if st.button("Predict"):
    model = LinearRegression()
    height = height.reshape(-1,1)
    model.fit(height, ideal_weight)
    inputvar = np.array([[inputvar]])
    result = model.predict(inputvar)
    result = result[0]
    st.write("Your ideal weight (in kg):")
    st.success(round(result,2))

# streamlit run IdealWeightPrediction.py  
