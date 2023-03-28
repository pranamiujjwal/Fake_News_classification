import re
import pickle
import numpy as np
import pandas as pd
from time import time

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



def process_author(author):
  author=author.lower()
  author=re.sub("[^a-zA-Z]"," ",author)
  return author

potterStemObj=PorterStemmer()
def steamming(content):
  # print("before:", content)
  content=re.sub("’","",content)
  content=re.sub("[^a-zA-Z]"," ",content)
  content=content.lower().split()
  content=[potterStemObj.stem(word) for word in content if not word in stopwords.words('english')]
  content=' '.join(content)
  # print("after:", content)
  return content



# load data
t0=time()
news_df=pd.read_csv("train.csv")
news_df=news_df.fillna('')
print("data loaded...", round(time()-t0,3)," sec")
# print(news_df.head())

# tranning -------------------------------------------------
# t0=time()
# news_df["author"]=news_df["author"].apply(process_author)
# print("author processed...", round(time()-t0,3)," sec")
# # # print(news_df["author"])


# t0=time()
# news_df["title"]=news_df["title"].apply(steamming)
# print("title stemmed...", round(time()-t0,3)," sec")


# X=news_df["author"]+" "+news_df["title"]


# t0=time()
# vectorizerObj=TfidfVectorizer()
# tfidf=vectorizerObj.fit(X)
# X=vectorizerObj.transform(X)
# print("vectorized...", round(time()-t0,3)," sec")
# # print(X)


# Y=news_df.iloc[:,4]
# # print(Y.head())


# # split data
# X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# # model
# classifier=LogisticRegression()
# t0=time()
# classifier.fit(X_train,Y_train)
# print("Model trained...", round(time()-t0,3)," sec")


# Y_pred=classifier.predict(X_test)
# score=accuracy_score(Y_pred,Y_test)
# print("score: ",score)

# with open('fakenews_classifier.pkl','wb') as file:
#   pickle.dump(classifier, file)
# with open('fakenews_vectorizer.pkl','wb') as file:
#   pickle.dump(vectorizerObj, file)
# with open("tfidf.pickle", "wb") as file:
#   pickle.dump(tfidf, file)




# prediction -------------------------------------------------
with open('fakenews_classifier.pkl','rb') as file:
  classifier=pickle.load(file)
with open('fakenews_vectorizer.pkl','rb') as file:
  vectorizerObj=pickle.load(file)
with open("tfidf.pickle", "rb") as file:
  tfidf=pickle.load(file)

# news=news_df.iloc[0]
# print(f"author: {news['author']}, \ntitle: {news['title']}")
# author=process_author(news["author"])
# title=steamming(news["title"])
# X=[author+" "+title]
# try:
#   X=tfidf.transform(X)
# except:
#   print("tfidf error")
  # print(f"Original: {'unreliable' if news['label']==1 else 'reliable'}".ljust(20," "),f"| predicted: {'unreliable' if pred==1 else 'reliable'}".ljust(23," "), f"| status: {'True' if news['label']==pred else 'False'}")

for i in range(1000):
  print("-"*79)
  news=news_df.iloc[i]
  # print(f"author: {news['author']}, \ntitle: {news['title']}")
  author=process_author(news["author"])
  title=steamming(news["title"])
  X=[author+" "+title]
  try:
    X=tfidf.transform(X)
  except:
    print("tfidf error")
  pred=classifier.predict(X)[0]
  print(f"Original: {'unreliable' if news['label']==1 else 'reliable'}".ljust(20," "),f"| predicted: {'unreliable' if pred==1 else 'reliable'}".ljust(23," "), f"| status: {'True' if news['label']==pred else 'False'}")