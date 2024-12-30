import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords # This will require to remove the stopwords
from nltk.stem.porter import PorterStemmer # The will require to stem the words
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk import PorterStemmer
PS = PorterStemmer()
from nltk import WordNetLemmatizer
lm = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
# from nltk.tokenize import word_tokenize
# nltk.download('punkt')

# Load model
# model = pickle.load(open("Twitter_sentiment_model.pkl", "rb"))
with open('Twitter_sentiment_model.pkl', 'rb') as f:
    Twitter_sentiment_model = pickle.load(f)

with open('TFIDF_Twitter_sentiment_model.pkl', 'rb') as f:
    TFIDF_Twitter_sentiment_model = pickle.load(f)


## Streamlit app

st.title("Twitter Sentiment Analysis Model")

## User Input
# tweet = "RT @vooda1: CNN Declines to Air White House Press Conference Live YES! THANK YOU @CNN FOR NOT LEGITIMI…"

tweet=st.text_input("Enter the Tweet")
result = st.button("Make Prediction")
if result:
  # tweet = "RT @vooda1: CNN Declines to Air White House Press Conference Live YES! THANK YOU @CNN FOR NOT LEGITIMI…"
     lower_tweet = tweet.lower()
     removing_stopword_lower_tweet = " ".join([word for word in str(lower_tweet).split() if word not in STOPWORDS])
     rem_url_sw_lowe_tweet = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',removing_stopword_lower_tweet)
     rem_no_url_sw_lw_tweet = re.sub('[0-9]+', '', rem_url_sw_lowe_tweet)
     rem_hash_no_url_sw_lw_tweet = re.sub(r'[@#]\w+', '', rem_no_url_sw_lw_tweet)
     tokernize_tweet = tokenizer.tokenize(rem_hash_no_url_sw_lw_tweet)
     stemmed_tweet = [PS.stem(word) for word in tokernize_tweet]
     lemmetize_tweet = [lm.lemmatize(word) for word in stemmed_tweet]
     converted_tweet = TFIDF_Twitter_sentiment_model.transform(lemmetize_tweet)
     prediction= Twitter_sentiment_model.predict(converted_tweet)
     st.write(f'Tweeter Prediction :{prediction}')
     if (prediction[0]==0):
      st.write('The Tweet is Negative')
     else:
      st.write('The Tweet is Positive')
else:
  st.write("Enter the tweet")


