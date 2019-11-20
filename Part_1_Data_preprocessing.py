# This script is for process the raw data and generate the data for further analyses.

from nltk.corpus import stopwords
from lxml import etree as ET
import pandas as pd
import numpy as np
import re
import csv


def clean_text(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = ' '.join(text.split()).lower()

    return text


def remove_stopwords(a_string, list_of_stopwords):
    filtered_words = []
    for w in a_string.split():
        if w not in list_of_stopwords:
            filtered_words.append(w)
    filtered_string = ' '.join(filtered_words)

    return filtered_string


sw = list(set(stopwords.words('english')))
sw = clean_text(' '.join(sw)).split()
customized_sw = ['na', 'la', 'ooh', 'oh']
sw = sw + customized_sw
print(sw)

# Process song data, and store data into a dataframe (song_df).
song_data = []
with open('raw_data/billboard_lyrics_1964-2015.csv', 'r', encoding='cp1252') as song_in:
    song_file = csv.reader(song_in)
    next(song_file)
    for line in song_file:

        rank = line[0]

        song_title = line[1]
        song_title = clean_text(song_title, remove_digits=True)
        song_title = remove_stopwords(song_title, sw)

        artist = line[2]

        song_yr = line[3]

        lyric_dirty = line[4]
        if lyric_dirty != 'NA':
            lyric = clean_text(lyric_dirty, remove_digits=True)
            lyric = remove_stopwords(lyric, sw)
            if len(lyric) > 0:
                # only append the lines in which lyric data is not na
                song_data.append([rank, song_title, artist, song_yr, str(lyric)])

song_df = pd.DataFrame(song_data, columns=['Rank', 'Song', 'Artist', 'Year', 'Lyrics'])
song_df['Year'] = song_df['Year'].astype(int)
song_df = song_df.loc[song_df.Year >= 2001]
song_df['Year'] = song_df['Year'].astype(str)


# Process news data, and news data into a dataframe (news_df).
news_xml = ['raw_data/NewYorkTimes_CoverStory_2001-2008_2013_2015.xml', 'raw_data/NewYorkTimes_CoverStory_2009-2012.xml']
news_data = []
for news_file in news_xml:
    tree = ET.parse(news_file)
    news = tree.xpath(".//rec")

    for each_news in news:
        news_id = each_news.xpath(".//header")[0].attrib['uiTerm']

        news_title = each_news.xpath(".//tig/atl")[0].text
        news_title = clean_text(news_title, remove_digits=True)
        news_title = remove_stopwords(news_title, sw)

        abstract = 'NONE'
        if len(each_news.xpath(".//ab")) > 0:
            abstract_text = each_news.xpath(".//ab")[0].text
            abstract = clean_text(abstract_text, remove_digits=True)
            abstract = remove_stopwords(abstract, sw)

        pubyr = 'NONE'
        year_node = each_news.xpath(".//pubinfo/dt[@year]")
        if len(year_node) > 0:
            pubyr = str(year_node[0].attrib['year'])

        news_data.append([news_id, pubyr, news_title, abstract])

news_df = pd.DataFrame(news_data, columns=['News_id', 'Year', 'Title', 'Abstract'])
news_df = news_df.replace('NONE', np.nan)
# only append the lines in which Year, Title, and Abstract are not na
news_df = news_df.dropna(subset=['Year', 'Title', 'Abstract'])


# Sample the news based on number of songs (by year)
song_count_per_yr = song_df.groupby(['Year']).count()[['Lyrics']].reset_index()
print(song_count_per_yr)
#print(news_df.groupby(['Year']).count())

song_count_per_yr_list = song_count_per_yr.values.tolist()
print(song_count_per_yr_list)

sampled_news = pd.DataFrame(columns=news_df.columns)
for pair in song_count_per_yr_list:
    song_yr = pair[0]
    n_song = pair[1]
    one_year_news = news_df.loc[news_df['Year'] == song_yr]
    one_year_sample = one_year_news.sample(n=n_song, random_state=1)
    sampled_news = pd.concat([sampled_news, one_year_sample])

print(sampled_news.groupby(['Year']).count())

song_df.set_index('Rank').to_csv('/Users/iwishsomeday/Documents/PYCharm/PR590/final_projects/cleaned_data/billboard_lyrics_2001-2015.csv')
news_df.set_index('News_id').to_csv('/Users/iwishsomeday/Documents/PYCharm/PR590/final_projects/cleaned_data/NewYorkTimes_CoverStory_2001-2015.csv')
sampled_news.set_index('News_id').to_csv('/Users/iwishsomeday/Documents/PYCharm/PR590/final_projects/cleaned_data/NewYorkTimes_CoverStory_2001-2015_SAMPLED.csv')


