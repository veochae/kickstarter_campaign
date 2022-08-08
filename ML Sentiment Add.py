#1. Packages Utilized
import numpy as np # linear algebraimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from wordcloud import WordCloud,STOPWORDS
import vaderSentiment
import twython


data = pd.read_csv('/Volumes/T7 Touch/Senior (2021)/Machine Learning/Final Project/ML_Clean_Data_with_Sentiment_v2.csv')

def clean(text):
     text = re.sub('https?://\S+|www\.\S+', '', text)
     text = re.sub(r'\s+', ' ', text, flags=re.I)
     text = re.sub('\[.*?\]', '', text)
     text = re.sub('\n', '', text)
     text = re.sub('\w*\d\w*', '', text)
     text = re.sub('<.*?>+', '', text)
     return text

data['slug'] = data['slug'].apply(lambda x:clean(x))


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
scores=[]
for i in range(len(data['slug'])):
     score = analyser.polarity_scores(data['slug'][i])
     score=score['compound']
     scores.append(score)  
     
sentiment=[]
for i in scores:
     if i>=0.05:
         sentiment.append('Positive')
     elif i<=(-0.05):
         sentiment.append('Negative')
     else:
         sentiment.append('Neutral')

data['slug_sentiment']=pd.Series(np.array(sentiment))

data = data.drop('photo', 1)



data.to_csv("/Volumes/T7 Touch/Senior (2021)/Machine Learning/Final Project/ML_Clean_Data_Final.csv")

