import pandas as pd
df = pd.read_csv('billboard_lyrics_1964-2015.csv', encoding='cp1252', dtype = {'Year' : 'object'})

list_of_texts=[]

def year_matcher(year_id):
    df_oneyear = df.loc[df['Year'] == year_id]
    content_list = df_oneyear.Lyrics.unique().tolist()
    content_text = " ".join(str(x) for x in content_list)
    list_of_texts.append(content_text)

# for i in range(2000, 2016):
#     print("'"+str(i)+"',")

year_list=['2000',
'2001',
'2002',
'2003',
'2004',
'2005',
'2006',
'2007',
'2008',
'2009',
'2010',
'2011',
'2012',
'2013',
'2014',
'2015']


for year in year_list:
    year_id=year
    content_id='Lyrics'
    year_matcher(year_id)

print(len(list_of_texts))

# Create a zipped list of tuples from above lists
zippedList = list(zip(year_list, list_of_texts))
dfObj = pd.DataFrame(zippedList, columns = ['Year' , 'CombinedLyrics'])
dfObj.to_csv('lyricsByYear.csv',sep='\t')