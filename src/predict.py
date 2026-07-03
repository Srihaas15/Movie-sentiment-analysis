from joblib import load

model=load("../models/sentiment_model.pkl")
vectorizor=load("../models/TF-IDFvectorizor.pkl")

review=input("Enter your review: ")

features=vectorizor.transform([review])

predictions_of_model=model.predict(features)

print(predictions_of_model)