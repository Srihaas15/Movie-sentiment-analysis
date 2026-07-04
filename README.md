##Movie Sentiment analysis

A Machine Learning model that predicts whether a movie review is "Positive" or "Negative" using "TF-IDF"and "Logistic Regression".

##Features

->Text preprocessing using Natural Language Processing(NLP) techniques
->TF-IDF(Text frequency-Inverse document frequency)
->Logistic regression classifier(sigmoid)
->Web interface via streamlit

##Python packages used

->Pandas
->Scikit-learn
->NLTK
->Joblib
->Streamlit

## Machine learning pipeline

Movie Review----->Preproscessing(lower case words and removal of puntuations,HTML tags and conjuction,prepositons etc..)---->TF-IDF vectorization (Less importance to words higher frequency like "movie" and higher importance to those with lower frequency like "Awesome")----->Logistic regression training( With *0% reviews of IMDB data set for max. 1000 iterations)----->Sentiment analysis

--> Hyperparameter tuning performed via "GridSearch with best performance found at "C"=10 C = 1/λ ,i.e regularization strength

--> Model accuracy after Hyperparameter tuning = 91% (Via test data also from IMDB ,i.e the remaining 20%)
