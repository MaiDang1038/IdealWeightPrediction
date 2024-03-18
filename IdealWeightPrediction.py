import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

#Using a BMI of 21.7, the midpoint of the healthy BMI range between 18.5 and 24.9, 
# I calculated the ideal_weight for adult women based on their height
st.title("Ideal Weight Prediction for Adult Female")
height = np.array([1.4, 1.45, 1.5,1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85])
ideal_weight = np.array([42.5, 45.6, 48.8, 52.1, 55.6, 59.1, 62.7, 66.5, 70.3, 74.3])

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