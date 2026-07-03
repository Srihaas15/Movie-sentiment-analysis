import pandas as pd
import string



df = pd.read_csv("../data/IMDB Dataset.csv") 
#print(df["sentiment"].iloc[:5])

import nltk
nltk.download("stopwords")

df["word_count"] = df["review"].apply(lambda review: len(review.split()))
df["clean review"]=df["review"].apply(lambda review: review.lower())
df["proper verbatim"]=df["clean review"].apply(lambda  review:  review.replace("<br />",""))
df["no punctuation"]=df["proper verbatim"].apply(lambda review: review.translate(str.maketrans("","",string.punctuation)))
df["tokens"]=df["no punctuation"].apply(lambda review: review.split())

#stop_words = {
#    "the",
#    "a",
#    "an",
#    "is",
#    "was",
#    "are",
#   "were",
#   "this",
#    "that",
#    "of",
#    "to",
#    "and",
#    "in",
#    "on",
#    "for",
#    "with",
#   "it",
#    "i"
#
#}
from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))

df["clean tokens"]=df["tokens"].apply(lambda review: [word for word in review if word not in stop_words])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#vectorizor = CountVectorizer()
vectorizor = TfidfVectorizer()
#vectorizor.fit(df["clean review"])
x = vectorizor.fit_transform(df["clean review"])
y= df["sentiment"]
#print(type(x))
#print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


"""
#correct=0
#total=len(y_test)
#for i in range (0,total):
#    if predictions_of_model[i]==y_test.iloc[i]:
#        correct+=1

#accuracy=correct/total


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions_of_model)

print(accuracy)
print(predictions_of_model[:10])
print(y_test.iloc[:10].tolist())

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

confusionmatrix = confusion_matrix(y_test,predictions_of_model)
precision = precision_score(y_test,predictions_of_model,pos_label="positive")
recall = recall_score(y_test,predictions_of_model,pos_label="positive")
f1 = f1_score(y_test,predictions_of_model,pos_label="positive")

print(confusionmatrix)
print(precision)
print(recall)
print(f1)
"""

"""
#Naive bayes model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()       
"""

#Logisitic regression model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
from sklearn.model_selection import GridSearchCV
parameters_grid={
    "C": [0.01,0.1,1,10,100]
}

grid = GridSearchCV(
    estimator = model,
    param_grid=parameters_grid,
    cv=5,
    scoring="accuracy"
)
grid.fit(x_train,y_train)
predictions_of_model=grid.predict(x_test)
print(grid.best_score_)
print(grid.best_params_)
"""
model.fit(x_train,y_train)
probabilities_in_model=model.predict_proba(x_test)
predictions_of_model=model.predict(x_test)
"""
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions_of_model))
#print(model.coef_)

from joblib import dump

dump(grid.best_estimator_,"../models/sentiment_model.pkl")
#grid.best_estimator_ basically it is the model which we are saving , as model here is last used for logistic regression without any grid parameters
dump(vectorizor,"../models/TF-IDFvectorizor.pkl")