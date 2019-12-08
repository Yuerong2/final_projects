import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def clean(doc:str):
    """ Preprocess/clean the doc: remove stopwords, punctuations and nomalize(lemmatize) the words.
       :param doc: a string
       :return: transformed string after cleaning
       >>> clean('a@b**c12//34 E|||D&&')
       'abc1234 ed'
       >>> b = clean('apple is good')
       >>> b == 'apple is good'
       False
       >>> b = clean('apple is good')
       >>> b == 'apple good'
       True
    """
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemmatizer.lemmatize(word) for word in punc_free.split())
    return normalized

def rewrite4cclda(num:str,filepath:str,column_name:str):
    """ Re-structure the strings
       :param num: a numerical tag add a group of string, for instance, all songs are tagged with '1'
       :return: generate a txt file as input for ccLDA
       >>> rewrite4cclda('0','../cleaned_data/test.csv',"Lyrics")
       done
       >>> rewrite4cclda('hi','hi',[1,2,3])
       Traceback (most recent call last):
       ValueError: input of each parameter must be a string
    """
    if (isinstance(num, str) and isinstance(filepath, str) and isinstance(column_name, str)) is False:
        raise ValueError('input of each parameter must be a string')
    else:
        pass
    f = open('ccLDA/input.txt', 'w')
    df = pd.read_csv(filepath)
    note1 = df[column_name].tolist()
    content1 = str(note1)
    doc4collection = num + " " + clean(content1)
    f.write(doc4collection)
    f.write('\n')
    print('done')
    f.close()

rewrite4cclda('0','cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv',"Abstract")
rewrite4cclda('1','cleaned_data/billboard_lyrics_2001-2015.csv','Lyrics')



