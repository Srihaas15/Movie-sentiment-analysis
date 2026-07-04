##Movie Sentiment analysis
<br>
A Machine Learning model that predicts whether a movie review is "Positive" or "Negative" using "TF-IDF"and "Logistic Regression".<br>
<br>
##Features<br>
<br>
->Text preprocessing using Natural Language Processing(NLP) techniques<br>
->TF-IDF(Text frequency-Inverse document frequency)<br>
->Logistic regression classifier(sigmoid)<br>
->Web interface via streamlit<br>
<br>
##Python packages used<br>
<br>
->Pandas<br>
->Scikit-learn<br>
->NLTK<br>
->Joblib<br>
->Streamlit<br>
<br>

## Machine learning pipeline<br>
<br>
Movie Review<br>
    |
Preproscessing(lower case words and removal of puntuations,HTML tags and conjuction,prepositons etc..)<br>
                              |
TF-IDF vectorization (Less importance to words higher frequency like "movie" and higher importance to those with lower frequency like "Awesome")<br>
                                                |
Logistic regression training( With 80% reviews of IMDB data set for max. 1000 iterations)<br>
      |
Sentiment analysis<br>
<br>
--> Hyperparameter tuning performed via "GridSearch with best performance found at "C"=10 C = 1/λ ,i.e regularization strength<br>
<br>
--> Model accuracy after Hyperparameter tuning = 91% (Via test data also from IMDB ,i.e the remaining 20%)
