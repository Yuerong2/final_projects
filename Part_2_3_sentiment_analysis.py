import pandas as pd
from pandas.core.frame import DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def get_sentiment_scores(filepath:str,sentenceColumn:str,TagColumn:str,outputFileName:str):
    """ Read a *.csv file as pandas dataframe, get the strings and produce sentiment scores of them
    :param filepath: a string for where the *.csv file is
    :param sentenceColumn: a string for the name of the column where the texts are
    :param TagColumn: a string for the name of the column where the annotations to the texts are
    :param outputFileName: a string for the name of the column where the texts are
    :return: c: a dataframe ; also generated a csv file
    >>> get_sentiment_scores('cleaned_data/test.csv','Lyrics','Year','testLyricsSentimentScores.csv')
    This dataset is done
    >>> get_sentiment_scores('cleaned_data/test.csv', 'Lyrics', 'Year', 123)
    Traceback (most recent call last):
    ValueError: input of each parameter must be a string
    >>> get_sentiment_scores([1,2,3], 123, 'hi', 'output')
    Traceback (most recent call last):
    ValueError: input of each parameter must be a string
    """
    # check input
    if (isinstance(filepath, str) and isinstance(sentenceColumn, str) and isinstance(TagColumn, str) and isinstance(outputFileName, str)) is False:
        raise ValueError('input of each parameter must be a string')
    else:
        pass
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
    c = {"pos": posScores,
         "neg": negScores,
         "neu": neuScores,
         "comp": compScores,
         "tag": tagSet}
    output = DataFrame(c)
    output.to_csv(outputFileName)
    print("This dataset is done")

get_sentiment_scores('cleaned_data/billboard_lyrics_2001-2015.csv','Lyrics','Year','LyricsSentimentScores.csv')
get_sentiment_scores('cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv','Abstract','Year','NewsSentimentScores.csv')

