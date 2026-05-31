
import pandas as pd
import streamlit as st
import joblib

model = joblib.load("best_wine_model.pkl")
df = pd.read_csv("winequality-red.csv")

st.set_page_config(page_title="Wine Quality Prediction AI", layout="wide")
st.title("🍷 Wine Quality Prediction AI System")
st.write("Enter the wine characteristics and predict its quality score.")
fixed_acidity = st.slider("fixed acidity",
                          min_value = float(df["fixed acidity"].min()),
                          max_value = float(df["fixed acidity"].max()),
                          value = float(df["fixed acidity"].mean()),
                          step = 0.1)

acidity = st.slider("volatile acidity",
                     min_value = float(df["volatile acidity"].min()),
                     max_value = float(df["volatile acidity"].max()),
                     value = float(df["volatile acidity"].mean()),
                     step = 0.01)

citric_acid = st.slider("citric acid",
                         min_value = float(df["citric acid"].min()),
                         max_value = float(df["citric acid"].max()),
                         value = float(df["citric acid"].mean()),   
                         step = 0.01)

residual_sugar = st.slider("residual sugar",
                           min_value = float(df["residual sugar"].min()), 
                           max_value = float(df["residual sugar"].max()),
                           value = float(df["residual sugar"].mean()),  
                           step = 0.1)
chlorides = st.slider("chlorides",
                       min_value = float(df["chlorides"].min()),    
                       max_value = float(df["chlorides"].max()),
                       value = float(df["chlorides"].mean()),
                       step = 0.001)    
free_sulfur_dioxide = st.slider("free sulfur dioxide",
                                  min_value = float(df["free sulfur dioxide"].min()),    
                                  max_value = float(df["free sulfur dioxide"].max()),
                                  value = float(df["free sulfur dioxide"].mean()),  
                                  step = 1.0)
total_sulfur_dioxide = st.slider("total sulfur dioxide",
                                min_value = float(df["total sulfur dioxide"].min()),      
                                max_value = float(df["total sulfur dioxide"].max()),
                                value = float(df["total sulfur dioxide"].mean()),
                                step = 1.0)
density = st.slider("density",
                    min_value = float(df["density"].min()), 
                    max_value = float(df["density"].max()),
                    value = float(df["density"].mean()),
                    step = 0.0001)   
pH = st.slider("pH",
                min_value = float(df["pH"].min()),  
                max_value = float(df["pH"].max()),
                value = float(df["pH"].mean()),
                step = 0.01)
sulphates = st.slider("sulphates",
                        min_value = float(df["sulphates"].min()),  
                        max_value = float(df["sulphates"].max()),
                        value = float(df["sulphates"].mean()),
                        step = 0.01)
alcohol = st.slider("alcohol",
                    min_value = float(df["alcohol"].min()),  
                    max_value = float(df["alcohol"].max()),
                    value = float(df["alcohol"].mean()),
                    step = 0.1)         

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'fixed acidity': fixed_acidity,
        'volatile acidity': acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Wine Quality Score: {prediction:.2f}")

