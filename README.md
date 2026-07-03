# 🎬 Movie Sentiment Analysis

A Machine Learning web application that predicts whether a movie review is **Positive** or **Negative** using **TF-IDF** and **Logistic Regression**.

The application is built with **Scikit-learn** and deployed locally using **Streamlit**.

---

## 📌 Features

- Movie review sentiment prediction
- Text preprocessing using NLP techniques
- TF-IDF feature extraction
- Logistic Regression classifier
- Interactive web interface using Streamlit

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- Streamlit

---

## 📊 Machine Learning Pipeline

IMDb Movie Reviews
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Logistic Regression
↓
Sentiment Prediction

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| TF-IDF + Logistic Regression | **91%** |

Hyperparameter tuning was performed using **GridSearchCV**.

---

## 📁 Project Structure

```
Sentiment Analysis/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── IMDB Dataset.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── TF-IDFvectorizor.pkl
│
└── src/
    ├── train.py
    └── predict.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Sentiment-Analysis
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Application

(Add a screenshot of the Streamlit application here.)

---

## 🚀 Future Improvements

- Improve the Streamlit UI
- Deploy the application online
- Add support for more ML models
- Compare multiple classifiers
- Display prediction confidence

---

## 👨‍💻 Author

**Srihaas Gunda**

B.Tech, Electrical Engineering
EE24BTECH11026
Indian Institute of Technology Hyderabad