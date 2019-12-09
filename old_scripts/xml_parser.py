# Author: Tzu-Kun Hsiao
# This is for parsing the XML obtained from Newspaper Source.
# If 'title' or 'abstract' is empty, the value will be 'NONE'.
# Only words and numbers in the titles and abstracts are preserved. All the punctuations are removed.

from lxml import etree as ET
import pandas as pd
import re


def clean_text(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = ' '.join(text.split()).lower()

    return text


tree = ET.parse('NewYorkTimes_CoverStory_2001-2015.xml')
news = tree.xpath(".//rec")

df_list = []
for each_news in news:
    news_id = each_news.xpath(".//header")[0].attrib['uiTerm']

    title = each_news.xpath(".//tig/atl")[0].text
    title = clean_text(title)

    abstract = 'NONE'
    if len(each_news.xpath(".//ab")) > 0:
        abstract_text = each_news.xpath(".//ab")[0].text
        abstract = clean_text(abstract_text)

    pubyr = 'NONE'
    year_node = each_news.xpath(".//pubinfo/dt[@year]")
    if len(year_node) > 0:
        pubyr = str(year_node[0].attrib['year'])

    df_list.append([news_id, pubyr, title, abstract])

df_news = pd.DataFrame(df_list, columns=['news_id', 'year', 'title', 'abstract'])

p_out = 'NewYorkTimes_CoverStory_2001-2015.csv'
with open(p_out, 'a') as fout:
    fout.write('news_id' + ',' + 'year' + ',' + 'title' + ',' + 'abstract' + '\n')
df_news.set_index('news_id').to_csv(p_out, mode='a', header=False)







