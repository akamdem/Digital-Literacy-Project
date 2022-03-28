# Informing Digital Media Literacy among ESL(English Second Language) Populations 

![text](https://github.com/akamdem/digital-literacy-project/blob/main/misinformation.png)

### Problem Statement:
Today, over a quarter of all Americans receive their daily news from social media, and unfortunately, distinguishing real/credible news, headlines and post from conspiracy theories or less than credible posts can be increasingly difficult even for native english speakers, let alone populations that are not are familiar with the contextual undertones and nuances behind digital media and english more broadly leading to situations where these populations tend to take almost anything they see on social media at face value as oppose to exercising the healthy level of skepticism needed to when parsing through digital media. This is a massive problem reaching incredible scale, and I first became aware of the intricacies of this issue in my time working at Facebook. After building some technical skills and observing this problem within my own immigrant family, I built this project to be a first line of defense for these audiences, to better inform their incstinct to trus any and all posts or headlines that presents as credible across social media.

### Executive Summary
In this project, I collected thousands of posts and headlines through Reddit's Push Shift API, specifically targeting the r/worldnews subreddit as a source of text from generally credible news sources and publications such as the New York times, Economist and others. I also pulled thousands of posts from r/conspiracy which is a notrious subreddit credited with being the source or at least a common passage way for most conspiracy theories that eventually populate the broader social media landscape. I performed some standard EDA on this pulled data, converting the data to usable data frames, removing uneeded features such post as post times and post authors, exploring the relative polarity and objectivity between the text from both subreddits> I then seperated text from credible new sources and text from fake news sources, tested out a count vectorizer, tfidf vectorizer and lemmatizer on each, trained tested and split my data, stratifying along the way, initiated a pipeline with a multinomial naive bayse model with set ngram parameters and fit this model onto my training data before setting a function that takes user inputs and initiates a classification based on the model it was trained on. The confusion matrix for the model can be found below. Overall is yielded a strong Precision score at 88% and a more modest F1 score at 61% 

![text](https://github.com/akamdem/digital-literacy-project/blob/main/ConfusionMatrix.png)

### Contents:
I: app.py 
- this serves as the front end for where I deploy the model, it harbors the visual layout and functions that take in and return classifications and probabilities to users

II: model-training.py 
- contains the base data and model used to train and apply to new user inputs, this serves as the backend to the streamlit app where I deploy the model

III: [Presentation](https://docs.google.com/presentation/d/1ies7K6b3VlwlCfaWNGyr70nYlU0IDkbJOGiLuh-3haM/edit?usp=sharing)

IV: [Web Application](https://share.streamlit.io/akamdem/misinfo/main/app.py)


### Data Sources:
- r/worldnews
- r/conspiracy


### Further Study
Given additional time, I would gather even more initial posts from reddit, i most cases, the more data you can train on, the better your model becomes at generalizing to new data. I may also continue to tune parameters such as ngrams to and continuously review my confusion matrix untill I can land at a strong F1 score (the harmoneous blend bectween recall and precision), I would also allow users to input links from posts directly as well, as ooposed to copying and pasting the specified text and linking to credible news sources after the model returns a classification and probability. 

The model is currently limited in that it mostly analyses syntax, subjectivity, polarity, length, word choice, etc. Most fake/conspiracy posts are structured in a similar way so you can get a pretty good guess just by looking at the content’s structure. However, this model won’t do as good a job at predicting if minute details in the post is actually factual.

##### [Web App available here:](https://share.streamlit.io/akamdem/misinfo/main/app.py)
