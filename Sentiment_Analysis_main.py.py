#Load Packages

import pandas as pd
import numpy as np
#from keras.models import load_model
#Load Data Viz pkgs
import matplotlib.pyplot as plt
import seaborn as sns
#Text cleaning
import neattext.functions as nfx
# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.downloader.download('vader_lexicon')
nv_model = MultinomialNB()

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv(r'C:\Users\Harika\OneDrive\Desktop\Python Project\Working_dataset.csv')
pickle.dump(nv_model, open("working_sentiment_model.pkl", "wb"))
model = pickle.load(open("working_sentiment_model.pkl", "rb"))
#model=load_model(r'C:\Users\Harika\OneDrive\Desktop\Python Project\internship\working_final_model.h5')

#Sentiment Analysis
def get_sentiment(text):
     
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(text)
     
    print("Overall sentiment dictionary is : ", sentiment_dict)
   
    print("Sentence Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        print("Positive")
 
    elif sentiment_dict['compound'] <= - 0.05 :
        print("Negative")
 
    else :
        print("Neutral")
df['Clean_Text']=df['Text'].apply(nfx.remove_stopwords)
df['Clean_Text']=df['Clean_Text'].apply(nfx.remove_punctuations)
df['Clean_Text']=df['Clean_Text'].apply(nfx.remove_userhandles)

from collections import Counter 
def extract_keywords(text,num=50):
    tokens = [tok for tok in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    print(dict(most_common_tokens)) 
emotion_list= df['Emotion'].unique().tolist()
#print(emotion_list)

#Load ML Pkgsfrom
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#Vectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#Metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
#Split our dataset
from sklearn.model_selection import train_test_split
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']
#Vectorizer
cv = CountVectorizer()
X=cv.fit_transform(Xfeatures)

X_train,X_test,y_train,y_test = train_test_split(X,ylabels,test_size=0.3,random_state=42)
nv_model = MultinomialNB()
nv_model.fit(X_train,y_train)
nv_model.score(X_test,y_test)
y_pred_for_nv = nv_model.predict(X_test)
sample_text = [' I hate spicy ']
vect = cv.transform(sample_text).toarray()
nv_model.predict(vect)
nv_model.predict_proba(vect)
np.max(nv_model.predict_proba(vect))
def predict_emotion(review,model):
    myvect=cv.transform(review).toarray()
    prediction=model.predict(myvect)
    pred_proba=model.predict_proba(myvect)
    pred_percentage_for_all=dict(zip(model.classes_,pred_proba[0]))
    print("Prediction:{} , Prediction Score: {}".format(prediction[0],np.max(pred_proba)))
    print(pred_percentage_for_all) 
print("Enter your review !")
sample_text=input()
get_sentiment(sample_text)
#user_list = sample_text.split()
t=sample_text
text=[t]
#print(text)
predict_emotion(text,nv_model)
import pickle


import speech_recognition as sr
#Needed Module

r = sr.Recognizer()
#Initializes r for Recognizer()

with sr.Microphone() as source:
    print("You can speak now")
    audio = r.listen(source)
    print("Time Over")

#Default Mic as source, it listens

try:
    print("TEXT: "+r.recognize_google(audio));
    sentence=r.recognize_google(audio)
    get_sentiment(sentence)
    lst=[sentence]
    predict_emotion(lst,nv_model)
    
    #Prints the Output

except:
    pass;
    #Does nothing, if error occurred(No error is showed on-scree)