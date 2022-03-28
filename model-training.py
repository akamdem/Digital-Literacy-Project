from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression  # pip install scikit-learn
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import requests
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import set_config
set_config(display='diagram')
from sklearn.impute import SimpleImputer
pd.options.display.max_columns = 999
set_config(display='diagram')
# load data

url = 'https://api.pushshift.io/reddit/search/submission'

params = {
    'subreddit': 'conspiracy',
    'size': 100
    #'before': 'utc'
}

res = requests.get(url, params)

res.status_code

res.json()

data = res.json()

posted = data['data']

df = pd.DataFrame(posted)

#posts = []
#for post in res.json()['data']:
#    post_dict = {}
 #   post_dict['title'] = post['title']
  #  post_dict['text'] = post['selftext']
   # post_dict['auth'] = post['author']
    #post_dict['time'] = post['created_utc']
   # post_dict['subreddit'] = post['subreddit']
  #  posts.append(post_dict)



def pushshift_query(subreddit, kind='submission', num_loops=2):
    current_time = 1642778603

    posts = []
    for query in range(num_loops):
        url = f'https://api.pushshift.io/reddit/search/{kind}/?subreddit={subreddit}&before={current_time}&size=1000'
        res = requests.get(url)

        for post in res.json()['data']:
            post_dict = {}
            post_dict['title'] = post['title']
            post_dict['text'] = post['selftext']
            post_dict['auth'] = post['author']
            post_dict['time'] = post['created_utc']
            post_dict['subreddit'] = post['subreddit']
            posts.append(post_dict)
            current_time = pd.DataFrame(posts)['time'].min()
        print(f'{subreddit} data frame has {len(posts)} rows')
        time.sleep(3)
#with open
    return pd.DataFrame(posts)

#with open
    #return pd.DataFrame(posts)
fake = pushshift_query(subreddit='conspiracy')
news = pushshift_query(subreddit='worldnews')

#news['title'].replace('', np.nan, inplace=True)
#news.dropna(subset=['title'], inplace=True)
#fake['title'].replace('', np.nan, inplace=True)
#fake.dropna(subset=['title'], inplace=True)

newser = [fake, news]
newsy = pd.concat(newser)

newsy = newsy.sample(frac=1)
fake.drop(fake.tail(7).index, inplace=True)

vecto = CountVectorizer()
X = newsy['title']
y = newsy['subreddit']
#y = np.where(newsy['subreddit'] == 'conspiracy', 0, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y)
nbayes = MultinomialNB()
pipe = make_pipeline(vecto, nbayes)
params = {
    'multinomialnb__alpha': [0.1, 2],
    'countvectorizer__ngram_range': [(1, 2), (1, 3)],
    'countvectorizer__max_df': [0.5, 0.7, 1]
}
gs = GridSearchCV(pipe, param_grid=params, n_jobs=-1)
gs.fit(X_train, y_train)
gs.score(X_train, y_train)
gs.score(X_test, y_test)
#headline = input()
def fakeorreal(headline, gs):
    #gs.best_estimator_.named_steps['vecto'].transform(headline)
    #prediction = gs.best_estimator_.named_steps['nbayes'].predict(headline)
    print('loading the headline')
    headline = gs.best_estimator_.named_steps['countvectorizer'].transform(headline).reshape(1, -1)
    prediction = gs.best_estimator_.named_steps['multinomialnb'].predict(headline)
    return prediction
    

# save the model
with open("subreddit.pkl", "wb") as file:
    pickle.dump(gs, file)
