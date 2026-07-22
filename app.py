import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Medical Insurance Cost Predictor")

# age
age = st.number_input('Age', min_value=18, max_value=100, value=30)

# sex
sex = st.selectbox('Sex', df['sex'].unique())

# bmi
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)

# children
children = st.selectbox('Number of Children', [0, 1, 2, 3, 4, 5])

# smoker
smoker = st.selectbox('Smoker', df['smoker'].unique())

# region
region = st.selectbox('Region', df['region'].unique())

if st.button('Predict Insurance Cost'):
    # Create a DataFrame for the query matching the training feature structure
    query = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                         columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # Predict the charges
    prediction = pipe.predict(query)[0]

    st.title("The predicted insurance charge is $" + str(round(prediction, 2)))