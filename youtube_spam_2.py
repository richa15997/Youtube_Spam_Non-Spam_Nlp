# -*- coding: utf-8 -*-
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('youtube_spam.txt', delimiter = '\t', quoting = 3)#quoting =3 is used for removing double quotes in reviews

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords#stopwords are used to remove irrelevant words like it,this,at
from nltk.stem.porter import PorterStemmer
corpus=[]



for i in range(0,1947):
    content = re.sub('[^a-zA-Z1-9:)(*-]', ' ', dataset['CONTENT'][i])#used to remove all characters except ones in bracket and replace the removed character with space
    content = content.lower()#convert all chars to lowercase
    content=content.split()#used to separate words from text in array
    ps= PorterStemmer()
    content=[ps.stem(word) for word in content if not word in set(stopwords.words('english'))]#set is used for lots of text
    content=' '.join(content)#join back the review
    corpus.append(content)
   


from sklearn.feature_extraction.text import CountVectorizer    
from nltk.tokenize import TweetTokenizer
cv = CountVectorizer(ngram_range=(1,1), stop_words=None, tokenizer=TweetTokenizer().tokenize,max_features = 3500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
  
#naive bayes     
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#approx accuracy=(160+117)/390
