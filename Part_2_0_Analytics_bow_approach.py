# This script is for calculating TF-IDF and text similarity (using cosine and Jaccard).

from collections import defaultdict
from collections import Counter
from nltk import WordNetLemmatizer
from time import time
from numpy.linalg import norm
import multiprocessing as mp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def read_data(path2file: str, yr_loc: int, ti_loc: int, txt_loc: int):
    """ Read files and process files by organizing data by each year.

    :param path2file: path to files, should be a string
    :param yr_loc: an integer pointing to the column where "year" is in the file
    :param ti_loc: an integer pointing to the column where "title" is in the file
    :param txt_loc: an integer pointing to the column where the main text is in the file
    :return: a dictionary (defaultdict), in which years are the key, and values are lists of lists.
             Each sublist contains the data of a news/song.

    >>> p1 = r'cleaned_data/billboard_lyrics_2001-2015.csv'
    >>> all_songs = read_data(p1, 3, 1, 4)
    >>> type(all_songs)
    <class 'collections.defaultdict'>
    >>> len(all_songs.keys())
    15
    >>> songs_keys = list(all_songs.keys()) # check whether all the keys are years by datatype being numbers
    >>> sum([isinstance(k, int) for k in songs_keys])
    15
    >>> yr_len = [len(str(k)) for k in songs_keys] # check whether all the keys are year (if so, length should all be 4)
    >>> yr_len_should_be = [4]*len(songs_keys)
    >>> yr_len == yr_len_should_be
    True
    """
    lemmatizer = WordNetLemmatizer()
    data_per_yr = defaultdict(list)
    with open(path2file, 'r') as fin:
        lines = fin.readlines()[1:]
        for each_line in lines:
            each_line = each_line.replace('\n', '').split(',')
            yr = int(each_line[yr_loc])
            full_txt = (each_line[ti_loc] + ' ' + each_line[txt_loc]).split()

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
    """ calculate TD-IDF of words in data, by year

    :param data: a dict, values are lists of lists
    :return: a dict, keys are years and values are lists of lists.
             In each sublist, the first item is the word, the second item is the TF-IDF score of the word.
             i.e. [[term1, TF-IDF_1], [term2, TF-IDF2]...]

    >>> a = defaultdict(list)
    >>> a[2019].append(['IS590PR', 'best', 'class', 'ever'])
    >>> a[2019].append(['Doing', 'project', 'winter', 'break', 'best'])
    >>> a[2019].append(['best', 'break', 'glass'])
    >>> a_tfidf = cal_tf_idf(a)
    >>> type(a)
    <class 'collections.defaultdict'>
    >>> val2019 = a_tfidf[2019]
    >>> type(val2019)
    <class 'list'>
    >>> isinstance(val2019[0][0], str)
    True
    >>> isinstance(val2019[0][1], float)
    True
    >>> b = [['IS590PR', 'best', 'class', 'ever'], ['Doing', 'project', 'winter', 'break', 'best']]
    >>> cal_tf_idf(b)
    Traceback (most recent call last):
    ValueError: input must be an dictionary
    """
    if isinstance(data, dict) is False:
        raise ValueError('input must be an dictionary')

    tf_idf_dict = defaultdict(list)
    for yr, docs in data.items():
        unique_words_docs_sum = []
        for doc in docs:
            unique_words_in_one = list(set(doc))
            unique_words_docs_sum += unique_words_in_one

        df_dict = Counter(unique_words_docs_sum)

        n_doc = len(docs)

        for doc in docs:
            term_freq = Counter(doc)
            for term, freq in term_freq.items():
                tf = freq/sum(term_freq.values())
                df = df_dict[term]
                tf_idf = tf * np.log(n_doc/(df+1))
                tf_idf_dict[yr].append([term, tf_idf])

    return tf_idf_dict


def get_top_words(tfidf_dict: dict, n_words=10):
    """ Get N words with highest TF-IDF score.

    :param tfidf_dict: an dictionary, keys are year, values are [[term1, TF-IDF_1], [term2, TF-IDF2]...]
    :param n_words: Number of words to retrieve. The default value is 10.
    :return: a dataframe with three columns: year, term, and tf-idf

    >>> tfidf_exmaple = defaultdict(list)
    >>> tfidf_exmaple[2019].append(['IS590PR', 0.1013662770270411])
    >>> tfidf_exmaple[2019].append(['best', -0.07192051811294523])
    >>> tfidf_exmaple[2019].append(['class', 0.1013662770270411])
    >>> tfidf_exmaple[2019].append(['ever', 0.1013662770270411])
    >>> df1 = get_top_words(tfidf_exmaple, n_words = 2)
    >>> df1.shape
    (2, 3)
    >>> df2 = get_top_words(tfidf_exmaple, n_words = 5)
    Traceback (most recent call last):
    ValueError: input of n_words is more than the words in data!
    >>> tfidf_exmaple[2018].append(['cats', 0.8])
    >>> tfidf_exmaple[2018].append(['are', 0.1])
    >>> tfidf_exmaple[2018].append(['cute', 0.9])
    >>> df3 = get_top_words(tfidf_exmaple, n_words = 2)
    >>> df3.iloc[:,0].drop_duplicates().tolist()
    [2019, 2018]
    >>> cols = list(df3.columns)
    >>> len(cols)
    3
    """
    header = ['year', 'term', 'tf-idf']
    dfs = []
    for each_year, tfidf_scores in tfidf_dict.items():
        df_list = []
        for term_score in tfidf_scores:
            df_list.append([each_year, term_score[0], float(term_score[1])])
        yr_df = pd.DataFrame(df_list, columns=header)
        yr_df = yr_df.sort_values(by=['tf-idf'], ascending=False)
        if n_words < len(tfidf_scores):
            yr_df = yr_df.iloc[:n_words].reset_index(drop=True)
            dfs.append(yr_df)
        else:
            raise ValueError('input of n_words is more than the words in data!')

    df_out = pd.concat(dfs)

    return df_out


def find_shared_words(news_df, song_df):
    """ Find the words with high TD-IDF scores and appear in both news and songs.
        Count how many words appear in both news and songs
        Also:
        1. Write all the high TD-IDF scores in news and songs into a file, 'TFIDF_top_terms.csv"
        2. Write the counting result into a file, 'TFIDF_found_in_both.csv'

    :param news_df: a dataframe, containing the high TF-IDF words in news
    :param song_df: a dataframe, containing the high TF-IDF words in songs
    :return: a dataframe, containing the counting results and the words appear in both corpus.

    >>> ndf_data = [['year', 'term', 'tf-idf'], [2001, 'cat', 0.1], [2001, 'dog', 0.2], [2001, 'pig', 0.3]]
    >>> ndf = pd.DataFrame(ndf_data[1:], columns=ndf_data[0])
    >>> sdf_data = [['year', 'term', 'tf-idf'], [2001, 'cat', 0.1], [2001, 'dog', 0.2], [2001, 'pig', 0.3]]
    >>> sdf_data.append([[2002, 'cow', 0.1], [2002, 'bird', 0.2], [2002, 'pig', 0.3]])
    >>> sdf_data.append([[2003, 'cat', 0.1], [2003, 'bird', 0.2], [2003, 'dog', 0.3]])
    >>> sdf_data.append([[2004, 'baby', 0.1], [2003, 'chile', 0.2], [2003, 'men', 0.3]])
    >>> sdf_data.append([[2005, 'cattle', 0.1], [2003, 'bird', 0.2], [2003, 'cat', 0.3]])
    >>> sdf = pd.DataFrame(sdf_data[1:], columns=sdf_data[0])
    >>> shared_w = find_shared_words(ndf, sdf)
    >>> shared_w.shape
    (5, 4)
    >>> shared_w.iloc[0,1]
    '2001/2001'
    >>> shared_w.iloc[4,1]
    '2001/2005'
    """
    all_top_df = pd.concat([news_df, song_df], axis=1)
    all_top_df.columns = ['N_yr', 'N_term', 'N_tfidf', 'S_yr', 'S_term', 'S_tfidf']
    all_top_out = all_top_df[['N_yr', 'N_term', 'N_tfidf', 'S_term', 'S_tfidf']].rename(columns={'N_yr': 'Year'})
    all_top_out.set_index('Year').to_csv('TFIDF_top_terms.csv')

    shared = [['Window_ID', 'NewsYR/SongYR', 'N_in_both', 'words_in_both']]
    window_count = 0
    yr_unique = list(news_df.iloc[:, 0].unique())
    for year in range(len(yr_unique)):
        news_year = 2001 + year
        window_count += 1
        df_1yr_news = all_top_df.loc[all_top_df.N_yr == news_year]
        top_w_news = set(df_1yr_news.N_term.tolist())
        for each_yr in range(15 - year):
            song_year = news_year + each_yr
            if song_year >= news_year and song_year - news_year < 5:
                df_1yr_song = all_top_df.loc[all_top_df.S_yr == song_year]
                top_w_song = set(df_1yr_song.S_term.tolist())
                in_both = list(top_w_news.intersection(top_w_song))
                in_both.sort()
                n_shared = len(in_both)
                yr_pair = str(news_year) + '/' + str(song_year)
                if n_shared > 0:
                    shared.append([str(window_count), yr_pair, n_shared, '|'.join(in_both)])
                else:
                    shared.append([str(window_count), yr_pair, 0, '-'])

    shared_df = pd.DataFrame(shared[1:], columns=shared[0])
    shared_df.set_index('Window_ID').to_csv('TFIDF_found_in_both.csv')

    return shared_df


def jaccard_sim(news_data_dict: dict, song_data_dict: dict):
    """ Calculate Jaccard similarity between one year of news and songs published within the same and the next 4 years.
        For example, if news were published in 2001, this function calculated the similarity between the pairs below:
        - news (published in 2001) , songs (published in 2001)
        - news (published in 2001) , songs (published in 2002)
        - news (published in 2001) , songs (published in 2003)
        - news (published in 2001) , songs (published in 2004)
        - news (published in 2001) , songs (published in 2005)

    :param news_data_dict: an dictionary containing news data
    :param song_data_dict: an dictionary containing song data
    :return: an dictionary (defalutdict),
             keys are the year of the news being published, while
             values are the Jaccard similarity between news and the songs in the time window.

    >>> news1 = defaultdict(list)
    >>> news1[2001] = [['programming', 'healthy', 'activity'], ['cats', 'are' 'mystic']]
    >>> songs1 = defaultdict(list)
    >>> songs1[2001] = [['programming', 'healthy', 'activity'], ['cats', 'are' 'mystic']]
    >>> songs1[2002] = [['programming', 'brain', 'activity'], ['cats', 'are' 'cute']]
    >>> songs1[2003] = [['programming', 'good', 'brain'], ['cats', 'are' 'dangerous']]
    >>> songs1[2004] = [['programming', 'hard', 'activity'], ['dogs', 'are' 'cute']]
    >>> songs1[2005] = [['programming', 'healthy', 'thing'], ['dogs', 'are' 'loyal']]
    >>> j_sim = jaccard_sim(news1, songs1)
    >>> type(j_sim)
    <class 'collections.defaultdict'>
    >>> list(j_sim.keys())
    [2001]
    >>> len(j_sim[2001])
    5
    """
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
                jaccard = len(shared_words) / (len(song_txt_flat) + len(news_txt_flat) - len(shared_words))
                jaccard = round(jaccard, 3)
                jaccard_dict[news_yr].append(jaccard)

    return jaccard_dict


def cosine_sim(news_data_dict: dict, song_data_dict: dict):
    """ Calculate cosine similarity between one year of news and songs published within the same and the next 4 years.
        For example, if news were published in 2001, this function calculated the similarity between the pairs below:
        - news (published in 2001) , songs (published in 2001)
        - news (published in 2001) , songs (published in 2002)
        - news (published in 2001) , songs (published in 2003)
        - news (published in 2001) , songs (published in 2004)
        - news (published in 2001) , songs (published in 2005)

    :param news_data_dict: an dictionary containing news data
    :param song_data_dict: an dictionary containing song data
    :return: an dictionary (defalutdict),
             keys are the year of the news being published, while
             values are the cosine similarity between news and the songs in the time window.

    >>> news1 = defaultdict(list)
    >>> news1[2001] = [['programming', 'healthy', 'activity'], ['cats', 'are' 'mystic']]
    >>> songs1 = defaultdict(list)
    >>> songs1[2001] = [['programming', 'healthy', 'activity'], ['cats', 'are' 'fun']]
    >>> songs1[2002] = [['programming', 'brain', 'activity'], ['cats', 'are' 'cute']]
    >>> songs1[2003] = [['programming', 'good', 'brain'], ['cats', 'are' 'dangerous']]
    >>> songs1[2004] = [['programming', 'hard', 'activity'], ['dogs', 'are' 'cute']]
    >>> songs1[2005] = [['programming', 'healthy', 'thing'], ['dogs', 'are' 'loyal']]
    >>> cos_sim = cosine_sim(news1, songs1)
    >>> type(cos_sim)
    <class 'collections.defaultdict'>
    >>> list(cos_sim.keys())
    [2001]
    >>> len(cos_sim[2001])
    5
    """
    cosine_dict = defaultdict(list)

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

                norm_news = norm(np.asarray(news_array))
                norm_song = norm(np.asarray(song_array))

                cosine = np.dot(news_array, song_array) / (norm_news * norm_song)
                cosine = round(cosine, 3)
                cosine_dict[news_yr].append(cosine)

    return cosine_dict


def draw_pic(data2draw, cmp='Greens'):
    """ This function is for making a graph to show the 5-year trend of similarity

    :param data2draw: an dictionary, keys are the years while values are the similarities
    :param cmp: colormap for drawing the graph, default as Greens
    :return: pyplot axes

    >>> draw = dict()
    >>> draw[2001] = [0.1, 0.2, 0.3, 0.4, 0.3]
    >>> draw[2002] = [0.01, 0.02, 0.03, 0.04, 0.03]
    >>> a_pic = draw_pic(draw)
    >>> type(a_pic)
    <class 'matplotlib.axes._subplots.AxesSubplot'>
    >>> draw2 = dict()
    >>> draw2[2001] = [0.1, 0.2, 0.3, 0.4]
    >>> draw2[2002] = [0.01, 0.02, 0.03, 0.04]
    >>> draw_pic(draw2)
    Traceback (most recent call last):
    ValueError: The number of similarity scores should be five.
    """
    t = ['news_yr', 'news_yr + 1', 'news_yr + 2', 'news_yr + 3', 'news_yr + 4']
    colormap = plt.cm.get_cmap(str(cmp))
    color = [colormap(i) for i in np.linspace(0, 1, 12)]
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_count = 0
    for ylab, sim_values in data2draw.items():
        if len(sim_values) == 5:
            color_count += 1
            line = sim_values
            ax1.plot(t, line, color=color[color_count], label=ylab)
        else:
            raise ValueError('The number of similarity scores should be five.')
    ax1.tick_params(axis='y', labelcolor='k')
    fig.tight_layout()
    plt.legend(loc='upper right', prop={'size': 15})

    return ax1


if __name__ == "__main__":
    start_time = time()

    path2song = r'cleaned_data/billboard_lyrics_2001-2015.csv'
    path2news = r'cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv'

    n_top_words = 100
    song_data = read_data(path2song, yr_loc=3, ti_loc=1, txt_loc=4)
    song_tfidf = cal_tf_idf(song_data)
    song_top = get_top_words(song_tfidf, n_words=n_top_words)

    news_data = read_data(path2news, yr_loc=1, ti_loc=2, txt_loc=3)
    news_tfidf = cal_tf_idf(news_data)
    news_top = get_top_words(news_tfidf, n_words=n_top_words)

    shared_top_words = find_shared_words(news_top, song_top)
    print('Number of high TF-IDF words found in both corpus: (among', n_top_words, 'words with highest TD-IDF)')
    shared_top_words_list = [shared_top_words.columns.values.tolist()] + shared_top_words.values.tolist()
    for each in shared_top_words_list:
        print('{:<9}  {:<13}  {:<9}  {:<}'.format(each[0], each[1], each[2], each[3]))
    # calculate jaccard similarity between news and songs, using a five-year sliding window
    print('Jaccard similarity:')
    print('{:8} {:8} {:8} {:8} {:8} {:8}'.format('News_YR', 'sim2YR+0', 'sim2YR+1', 'sim2YR+2', 'sim2YR+3', 'sim2YR+4'))
    jaccard_11yr = jaccard_sim(news_data, song_data)
    for news_year, jac_vals in jaccard_11yr.items():
        print('{:<8} {:<8} {:<8} {:<8} {:<8} {:<8}'.format(
            news_year, jac_vals[0], jac_vals[1], jac_vals[2], jac_vals[3], jac_vals[4]))
    # make fig of Jaccard scores
    jpic = draw_pic(jaccard_11yr, cmp='Greens')
    jpic.set_xlabel(str('year progression'), fontsize=15)
    jpic.set_ylabel(str('Jaccard similarity'), color='k', fontsize=13)
    plt.title('Jaccard Similarity; 5-year sliding window', fontsize=20)
    plt.savefig('Graphs/jaccard_similarity.png', dpi=300)
    plt.show()
    # calculate cosine similarity between news and songs, using a five-year sliding window
    print('Cosine similarity:')
    print('{:8} {:8} {:8} {:8} {:8} {:8}'.format('News_YR', 'sim2YR+0', 'sim2YR+1', 'sim2YR+2', 'sim2YR+3', 'sim2YR+4'))
    cosine_11yr = cosine_sim(news_data, song_data)
    for news_year, cos_vals in cosine_11yr.items():
        print('{:<8} {:<8} {:<8} {:<8} {:<8} {:<8}'.format(
            news_year,cos_vals[0], cos_vals[1], cos_vals[2], cos_vals[3], cos_vals[4]))
    # make fig of cosine scores
    cpic = draw_pic(cosine_11yr, cmp='Reds')
    cpic.set_xlabel(str('year progression'), fontsize=15)
    cpic.set_ylabel(str('cosine similarity'), color='k', fontsize=13)
    plt.title('Cosine Similarity; 5-year sliding window', fontsize=20)
    plt.savefig('Graphs/cosine_similarity.png', dpi=300)
    plt.show()

    print('run time:', time()-start_time)
