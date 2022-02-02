import streamlit as st
import plotly.express as px
import numpy as np
import pickle
import time
from predict import predict_flower

import json
#import torch
from collections import Counter

st.title("Hi mom, are you wondering if an online post/headline is real or fake?")
st.header("Let's predict by comparing it to thousands of other social media posts.")
st.subheader("If the model classifies the given text as 'worldnews', it is likely trustworthy, otherwise it'll be classified as 'conspiracy', please don't spread on Whatsapp.")

#df_iris = px.data.iris()

# hist_sl = px.histogram(df_iris, x="sepal_length")
# hist_sl

#show_df = st.checkbox("Do .you want to see the likelihood?")


with open("subreddit.pkl", "rb") as f:
    classifier = pickle.load(f)

headline = [st.text_input('Enter the post text or headline here')]
def fakeorreal(headline, gs):
    #gs.best_estimator_.named_steps['countvectorizer'].transform(headline)
    headline = gs.best_estimator_.named_steps['countvectorizer'].transform(headline).reshape(1, -1)
    prediction = gs.best_estimator_.named_steps['multinomialnb'].predict(headline)
    if st.button('Generate Prediction'):
        generated_pred = prediction = gs.best_estimator_.named_steps['multinomialnb'].predict(headline)
        probability = gs.best_estimator_.named_steps['multinomialnb'].predict_proba(headline).max()
        if generated_pred[0] == 'worldnews':
            st.write(f"There is a {round(probability, 2)*100} percent chance that this text/headline is real or trustworthy")
            st.balloons()
        else:
            st.write(f"There is a {round(probability, 2)*100} percent chance that this text/headline is Fake News, please don't spread this on Whatsapp without double checking")
            
            # st.write(f"There is a {round(probability, 2)*100} percent chance that this text or headline c: "+ generated_pred[0])




fakeorreal(headline, classifier)


