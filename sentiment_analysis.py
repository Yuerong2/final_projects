import pandas as pd
from pandas.core.frame import DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def get_sentiment_scores(filepath:str,sentenceColumn:str,TagColumn:str,outputFileName:str):
    posScores = []
    negScores = []
    neuScores = []
    compScores = []
    data = pd.read_csv(filepath)
    sentSet = data[sentenceColumn].tolist()
    tagSet = data[TagColumn].tolist()
    for sentence in sentSet:
        score = analyser.polarity_scores(sentence)
        neg = score['neg']
        pos = score['pos']
        neu = score['neu']
        comp = score['compound']
        negScores.append(neg)
        posScores.append(pos)
        neuScores.append(neu)
        compScores.append(comp)
    #print(len(posScores),len(compScores),len(neuScores),len(posScores),len(tagSet))
    c = {"pos": posScores,
         "neg": negScores,
         "neu": neuScores,
         "comp": compScores,
         "tag": tagSet}
    output = DataFrame(c)
    output.to_csv(outputFileName)
    print("This dataset is done")


#data=pd.read_csv('../cleaned_data/billboard_lyrics_2001-2015.csv')
get_sentiment_scores('cleaned_data/billboard_lyrics_2001-2015.csv','Lyrics','Year','LyricsSentimentScores.csv')
get_sentiment_scores('cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv','Abstract','Year','NewsSentimentScores.csv')



