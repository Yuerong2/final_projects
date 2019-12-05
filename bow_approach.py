from collections import defaultdict
from collections import Counter
from nltk import WordNetLemmatizer
from time import time
from numpy.linalg import norm
import multiprocessing as mp
import pandas as pd
import numpy as np


start_time = time()

path2song = r'cleaned_data/billboard_lyrics_2001-2015.csv'
path2news = r'cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv'
song = pd.read_csv('cleaned_data/billboard_lyrics_2001-2015.csv')
news = pd.read_csv('cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv')
print(song.shape)
print(song.columns)
print(news.shape)
print(news.columns)


def read_data(path2file: str, yr_loc: int, ti_loc: int, txt_loc: int):
    lemmatizer = WordNetLemmatizer()

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
            unique_words_per_doc = list(set(doc))

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


def jaccard_sim(news_data_dict: dict, song_data_dict: dict):

    jaccard_dict = defaultdict(list)

    for news_yr, news_txt in news_data_dict.items():
        news_txt_flat = []
        for nt in news_txt:
            news_txt_flat += nt
        news_txt_flat = set(news_txt_flat)
        if news_yr <= 2011:
            song_txt_flat = set()
            for i in range(5):
                song_yr = news_yr+i
                for st in song_data_dict[song_yr]:
                    for stw in st:
                        song_txt_flat.add(stw)
                shared_words = news_txt_flat.intersection(song_txt_flat)
                jaccard = len(shared_words) / (len(song_txt_flat) + len(news_txt_flat) + len(shared_words))
                jaccard_dict[news_yr].append(jaccard)

    return jaccard_dict


def l2_norm(word_list: list):
    l2_dict = {}
    word_freq = Counter(word_list)
    word_freq_val = np.asarray(list(word_freq.values()))
    l2 = norm(word_freq_val)
    for k, v in word_freq.items():
        l2_dict[k] = v / l2

    return l2_dict


def cosine_sim(news_data_dict: dict, song_data_dict: dict):

    cosine_sim11 = defaultdict(list)

    for news_yr, news_txt in news_data_dict.items():
        news_txt_flat = []
        for nt in news_txt:
            news_txt_flat += nt
        news_tf_l2 = l2_norm(news_txt_flat)

        if news_yr <= 2011:
            song_txt_flat = []
            for i in range(5):
                song_yr = news_yr+i
                for st in song_data_dict[song_yr]:
                    song_txt_flat += st
                song_tf_l2 = l2_norm(song_txt_flat)

                cosine_prep = {}
                all_words = list(set(song_txt_flat + news_txt_flat))
                for aw in all_words:
                    cosine_prep[aw] = [0, 0]
                    if aw in song_tf_l2.keys():
                        cosine_prep[aw][0] = song_tf_l2[aw]
                    if aw in news_tf_l2.keys():
                        cosine_prep[aw][1] = news_tf_l2[aw]
                consine = {}
                for aw, l2_score in cosine_prep.items():
                    consine[aw] = l2_score[0] * l2_score[1]
                sim = sum(consine.values())
                cosine_sim11[news_yr].append(sim)

    return cosine_sim11


song_data = read_data(path2song, yr_loc=3, ti_loc=1, txt_loc=4)
song_tfidf = cal_tf_idf(song_data)
song_top = get_top_words(song_tfidf, n_words=5)

news_data = read_data(path2news, yr_loc=1, ti_loc=2, txt_loc=3)
news_tfidf = cal_tf_idf(news_data)
news_top = get_top_words(news_tfidf, n_words=5)

all_top_df = pd.concat([song_top, news_top], axis=1)
for v in all_top_df.values.tolist():
    # print the top N terms having the highest tf-idf in songs and news
    print('{:4} {:<20} {:.2f} {:4} {:<20} {:.2f}'.format(v[0], v[1], v[2], v[3], v[4], v[5]))

print('\n')
print('Jaccard similarity:')
jaccard_11yr = jaccard_sim(news_data, song_data)
for news_yr, jac_val in jaccard_11yr.items():
    # print jaccard similarity between news and songs, using a five-year sliding window
    print(news_yr, jac_val)

print('\n')
print('Cosine similarity:')
cosine_11yr = cosine_sim(news_data, song_data)
for news_yr, cos_val in cosine_11yr.items():
    # print cosine similarity between news and songs, using a five-year sliding window
    print(news_yr, cos_val)


print('\n')
print('run time:', time()-start_time)











