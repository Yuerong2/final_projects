from collections import defaultdict
from collections import Counter
from collections import OrderedDict
from nltk import WordNetLemmatizer
import operator
import pandas as pd
import numpy as np

lemmatizer = WordNetLemmatizer()

path2song = r'cleaned_data/billboard_lyrics_2001-2015.csv'
path2news = r'cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv'
song = pd.read_csv('cleaned_data/billboard_lyrics_2001-2015.csv')
news = pd.read_csv('cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv')
print(song.shape)
print(song.columns)
print(news.shape)
print(news.columns)


def read_data(path2file: str, yr_loc: int, ti_loc: int, txt_loc: int):
    data_per_yr = defaultdict(list)
    with open(path2file, 'r') as fin:
        lines = fin.readlines()[1:]
        for line in lines:
            line = line.replace('\n', '').split(',')
            yr = int(line[yr_loc])
            full_txt = (line[ti_loc] + ' ' + line[txt_loc]).split()
            full_txt_out = []
            for word in full_txt:
                if word.endswith('ing') or word.endswith('ed'):
                    word = lemmatizer.lemmatize(word, pos='v')
                else:
                    word = lemmatizer.lemmatize(word)
                if len(word) > 1:
                    full_txt_out.append(word)
            data_per_yr[yr].append(full_txt_out)

    return data_per_yr


def cal_tf_idf(data: dict):
    tf_idf_dict = defaultdict(list)
    c = 0
    for yr, docs in data.items():
        unique_words_per_doc = []
        for doc in docs:
            for w in set(doc):
                unique_words_per_doc.append(w)
        df_dict = Counter(unique_words_per_doc)

        n_doc = len(docs)

        terms_in_doc = []
        for doc in docs:
            c += 1
            term_freq = Counter(doc)
            doc_id = str(yr) + '_' + str(c)
            terms_in_doc.append(doc_id)
            for term, freq in term_freq.items():
                tf = freq/sum(term_freq.values())
                df = df_dict[term]
                tf_idf = tf * np.log(n_doc/(df+1))
                terms_in_doc.append([term, tf_idf])
        tf_idf_dict[yr].append(terms_in_doc)

    return tf_idf_dict


def get_top_words(tfidf_dict: dict, n_words=10):
    header = ['year', 'term', 'tf-idf']
    dfs = []
    for year, docs in tfidf_dict.items():
        df_list = []
        for doc in docs:
            for items in doc[1:]:
                df_list.append([year, items[0], float(items[1])])
        yr_df = pd.DataFrame(df_list, columns=header)
        yr_df = yr_df.sort_values(by=['tf-idf'], ascending=False)
        yr_df = yr_df.iloc[:n_words].reset_index(drop=True)
        dfs.append(yr_df)

    df_out = pd.concat(dfs)

    return df_out


song_data = read_data(path2song, yr_loc=3, ti_loc=1, txt_loc=4)
song_tfidf = cal_tf_idf(song_data)
song_top = get_top_words(song_tfidf, n_words=50)

news_data = read_data(path2news, yr_loc=1, ti_loc=2, txt_loc=3)
news_tfidf = cal_tf_idf(news_data)
news_top = get_top_words(news_tfidf, n_words=50)

all_top_df = pd.concat([song_top, news_top], axis=1)
for v in all_top_df.values.tolist():
    #print(v)
    print('{:4} {:<20} {:.2f} {:4} {:<20} {:.2f}'.format(v[0], v[1], v[2], v[3], v[4], v[5]))









