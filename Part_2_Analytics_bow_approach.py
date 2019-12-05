from collections import defaultdict
from collections import Counter
from nltk import WordNetLemmatizer
from time import time
from numpy.linalg import norm
import multiprocessing as mp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

start_time = time()

path2song = r'cleaned_data/billboard_lyrics_2001-2015.csv'
path2news = r'cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv'
# song = pd.read_csv('cleaned_data/billboard_lyrics_2001-2015.csv')
# news = pd.read_csv('cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv')
# print(song.shape)
# print(song.columns)
# print(news.shape)
# print(news.columns)


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
        unique_words_docs_sum = []
        for doc in docs:
            unique_words_in_one = list(set(doc))
            unique_words_docs_sum += unique_words_in_one

        df_dict = Counter(unique_words_docs_sum)

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
                jaccard = round(jaccard, 3)
                jaccard_dict[news_yr].append(jaccard)

    return jaccard_dict


def cosine_sim(news_data_dict: dict, song_data_dict: dict):

    cosine_sim11 = defaultdict(list)

    for news_yr, news_txt in news_data_dict.items():
        news_txt_flat = []
        for nt in news_txt:
            news_txt_flat += nt
        news_tf = Counter(news_txt_flat)

        if news_yr <= 2011:
            song_txt_flat = []
            for i in range(5):
                song_yr = news_yr+i
                for st in song_data_dict[song_yr]:
                    song_txt_flat += st
                song_tf = Counter(song_txt_flat)

                all_words = list(set(song_txt_flat + news_txt_flat))
                news_array = []
                song_array = []
                for aw in all_words:
                    if aw in news_tf.keys():
                        news_array.append(news_tf[aw])
                    else:
                        news_array.append(0)
                    if aw in song_tf.keys():
                        song_array.append(song_tf[aw])
                    else:
                        song_array.append(0)

                norm_news = norm(news_array)
                norm_song = norm(song_array)

                sim = np.dot(news_array, song_array) / (norm_news * norm_song)
                sim = round(sim, 3)
                cosine_sim11[news_yr].append(sim)

    return cosine_sim11

n_top_words = 100
song_data = read_data(path2song, yr_loc=3, ti_loc=1, txt_loc=4)
song_tfidf = cal_tf_idf(song_data)
song_top = get_top_words(song_tfidf, n_words=n_top_words)

news_data = read_data(path2news, yr_loc=1, ti_loc=2, txt_loc=3)
news_tfidf = cal_tf_idf(news_data)
news_top = get_top_words(news_tfidf, n_words=n_top_words)

all_top_df = pd.concat([news_top, song_top], axis=1)
all_top_df.columns = ['Year', 'N_term', 'N_tfidf', 'S_yr', 'S_term', 'S_tfidf']
all_top_df = all_top_df[['Year', 'N_term', 'N_tfidf', 'S_term', 'S_tfidf']]
all_top_df.set_index('Year').to_csv('TFIDF_top_terms.csv')

shared = [['Year', 'N_in_both', 'words_in_both']]
for year in range(15):
    year = 2001 + year
    df_1yr = all_top_df.loc[all_top_df.Year == year]
    top_w_news = set(df_1yr.N_term.tolist())
    top_w_song = set(df_1yr.S_term.tolist())
    in_both = list(top_w_news.intersection(top_w_song))
    n_shared = len(in_both)
    if n_shared > 0:
        shared.append([year, n_shared, '|'.join(in_both)])
    else:
        shared.append([year, 0, '-'])

print('Number of high TF-IDF words found in both corpus: (among', n_top_words, 'words with highest TD-IDF)')
for each in shared:
    print('{:4}  {:<9}  {:<}'.format(each[0], each[1], each[2]))


print('\n')
print('Jaccard similarity:')
print('{:7} {:8} {:8} {:8} {:8} {:8}'.format('News_YR', 'sim2YR+0', 'sim2YR+1', 'sim2YR+2', 'sim2YR+3', 'sim2YR+4'))
# calculate jaccard similarity between news and songs, using a five-year sliding window
jaccard_11yr = jaccard_sim(news_data, song_data)
for news_year, jac_vals in jaccard_11yr.items():
    print(
        '{:<7} {:<8} {:<8} {:<8} {:<8} {:<8}'.format(news_year,
                                                     jac_vals[0], jac_vals[1], jac_vals[2], jac_vals[3], jac_vals[4]))
    # Example:
    # If news_yr == 2001,
    # jac_vals are the Jaccard similarity between news published in 2001 and songs published between 2001-2005

# make fig of Jaccard scores
t = ['news_yr', 'news_yr + 1', 'news_yr + 2', 'news_yr + 3', 'news_yr + 4']
y_labels = list(jaccard_11yr.keys())
colormap = plt.cm.Greens
color = [colormap(i) for i in np.linspace(0, 1, 12)]
fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.set_xlabel('year progression', fontsize=15)
ax1.set_ylabel('jaccard similarity', color='k', fontsize=13)
color_count = 0
for ylab, jac_values in jaccard_11yr.items():
    color_count += 1
    line = jac_values
    ax1.plot(t, line, color=color[color_count], label=ylab)
ax1.tick_params(axis='y', labelcolor='k')
fig.tight_layout()
plt.legend(loc='upper right', prop={'size': 15})
plt.title('Jaccard Similarity; 5-year sliding window', fontsize=20)
plt.savefig('Graphs/jaccard_similarity.png', dpi=300)
plt.show()

print('\n')
print('Cosine similarity:')
print('{:7} {:8} {:8} {:8} {:8} {:8}'.format('News_YR', 'sim2YR+0', 'sim2YR+1', 'sim2YR+2', 'sim2YR+3', 'sim2YR+4'))
cosine_11yr = cosine_sim(news_data, song_data)
for news_year, cos_vals in cosine_11yr.items():
    print(
        '{:<7} {:<8} {:<8} {:<8} {:<8} {:<8}'.format(news_year,
                                                     cos_vals[0], cos_vals[1], cos_vals[2], cos_vals[3], cos_vals[4]))

    # Example:
    # If news_yr == 2001,
    # cos_vals are the cosine similarity between news published in 2001 and songs published between 2001-2005

# make fig of cosine scores
t = ['news_yr', 'news_yr + 1', 'news_yr + 2', 'news_yr + 3', 'news_yr + 4']
y_labels = list(cosine_11yr.keys())
colormap = plt.cm.Reds
color = [colormap(i) for i in np.linspace(0, 1, 12)]
fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.set_xlabel('year progression', fontsize=15)
ax1.set_ylabel('cosine similarity', color='k', fontsize=13)
color_count = 0
for ylab, cos_values in cosine_11yr.items():
    color_count += 1
    line = cos_values
    ax1.plot(t, line, color=color[color_count], label=ylab)
ax1.tick_params(axis='y', labelcolor='k')
fig.tight_layout()
plt.legend(loc='upper right', prop={'size': 15})
plt.title('Cosine Similarity; 5-year sliding window', fontsize=20)
plt.savefig('Graphs/cosine_similarity.png', dpi=300)
plt.show()

print('\n')
print('run time:', time()-start_time)











