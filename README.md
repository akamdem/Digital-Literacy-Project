# misinfo
#### This app scrapes over 5000 posts from the r/worldnews subreddit and r/conspiracy subreddit, trains a Naïve Bayes model on these posts to help the model identify patterns behind traditional credible news post/headlines and those that are more likely to be stemming from conspiracy theories. The app then takes in new inputs from users and predicts a classification based on the NLP model's training and returns a classigication with a percentage likelyhood for either class. Please see the adjacent app.py doc for a technical review.

#### The model is still pretty small, only pulling in about 5k posts and it mostly analyses syntax, subjectivity, polarity, length, word choice, etc. Most fake/conspiracy posts are structured in a similar way so you can get a pretty good guess just by looking at the content’s structure. However, this model won’t do as good a job at predicting if a minute detail in the post is actually true. 

#### Scrapping was done mostly through reddit, since they’re the most amenable to scrapping

#### Web App available here: https://share.streamlit.io/akamdem/misinfo/main/app.py
