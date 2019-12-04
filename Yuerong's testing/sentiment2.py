import pandas as pd
from pandas.core.frame import DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
posScores=[]
negScores=[]
neuScores=[]
compScores=[]

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    neg=score['neg']
    negScores.append(neg)
    pos=score['pos']
    posScores.append(pos)
    neu=score['neu']
    neuScores.append(neu)
    comp=score['compound']
    compScores.append(comp)

    #text_file.write(str(score))
data=pd.read_csv('../cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv')
sentSet=data['Abstract'].tolist()
tagSet=data['Year'].tolist()
for i in sentSet:
    sentiment_analyzer_scores(i)

print(len(posScores),len(compScores),len(neuScores),len(posScores),len(tagSet))

c={"pos" : posScores,
   "neg" : negScores,
   "neu":neuScores,
   "comp":compScores,
   "tag":tagSet}
output=DataFrame(c)
print(output.info())
output.to_csv("NewsSentimentScores.csv")

