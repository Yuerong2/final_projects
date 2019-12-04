from collections import defaultdict
from collections import Counter
from collections import OrderedDict
import operator

import pandas as pd
import numpy as np

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
            data_per_yr[yr].append(full_txt)
            # full_txt_out = []
            # for w in full_txt:
            #     if len(w) > 1:
            #         full_txt_out.append(w)
            # data_per_yr[yr].append(full_txt_out)

    return data_per_yr


def cal_tf_idf(data: dict):
    tf_idf_dict = defaultdict(dict)
    c = 0
    for yr, docs in data.items():
        unique_words_per_doc = []
        for doc in docs:
            for w in set(doc):
                unique_words_per_doc.append(w)
        df_dict = Counter(unique_words_per_doc)

        n_doc = len(docs)

        for doc in docs:
            c += 1
            term_freq = Counter(doc)
            doc_id = str(yr) + '_' + str(c)
            tf_idf_dict[yr] = OrderedDict()
            for term, freq in term_freq.items():
                tf = freq/sum(term_freq.values())
                df = df_dict[term]
                tf_idf = tf * np.log(n_doc/(df+1))
                term_id = doc_id + '_' + term
                tf_idf_dict[yr][term_id] = tf_idf

        tf_idf_sort = sorted(tf_idf_dict[yr].items(), key=operator.itemgetter(1), reverse=True)
        tf_idf_dict[yr] = OrderedDict(tf_idf_sort)

    return tf_idf_dict


def get_top_words(tfidf_dict: dict, n_words=10):
    top_term = []
    for year, score in tfidf_dict.items():
        w_count = 0
        for term_id, tfidf in score.items():
            w_count += 1
            term = term_id.split('_')[-1]
            if w_count <= n_words:
                top_term.append([year, term, tfidf])

    return top_term


song_data = read_data(path2song, yr_loc=3, ti_loc=1, txt_loc=4)
song_tfidf = cal_tf_idf(song_data)
song_top = get_top_words(song_tfidf)
song_top_df = pd.DataFrame(song_top, columns=['year', 'song_term', 'song_term_tfidf'])

news_data = read_data(path2news, yr_loc=1, ti_loc=2, txt_loc=3)
news_tfidf = cal_tf_idf(news_data)
news_top = get_top_words(news_tfidf)
news_top_df = pd.DataFrame(news_top, columns=['year', 'news_term', 'news_term_tfidf'])

all_top_df = pd.concat([song_top_df, news_top_df], axis=1)
for v in all_top_df.values.tolist():
    print('{:4} {:<20} {:.2f} {:4} {:<20} {:.2f}'.format(v[0], v[1], v[2], v[3], v[4], v[5]))









