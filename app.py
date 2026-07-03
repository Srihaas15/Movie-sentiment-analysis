import streamlit as st
from joblib import load

st.title("MOVIE SENTIMENT ANALYZER")

st.warning("Enter a movie review below and I'll predict whether it is Positive or Negative.")     

review= st.text_area("Enter the movie review: ")

model = load("models/sentiment_model.pkl")
vectorizer = load("models/TF-IDFvectorizor.pkl")

if st.button("Predict"):    
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        features = vectorizer.transform([review])
        prediction_of_model = model.predict(features)
        if prediction_of_model[0] == "positive":
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")