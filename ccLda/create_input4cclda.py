import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemmatizer = WordNetLemmatizer()
f = open('input.txt', 'w')

#from nltk.stem import WordNetLemmatizer
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemmatizer.lemmatize(word) for word in punc_free.split())
    return normalized

def rewrite4cclda(num,filepath,column_name):
    df = pd.read_csv(filepath)
    note1 = df[column_name].tolist()
    content1 = str(note1)
    doc4collection = num + " " + clean(content1)
    f.write(doc4collection)
    f.write('\n')

rewrite4cclda('0','NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv',"Abstract")
rewrite4cclda('1','billboard_lyrics_2001-2015.csv','Lyrics')

f.close()