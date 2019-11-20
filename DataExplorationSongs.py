
import pandas as pd # type: ignore
import re

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s|]' if not remove_digits else r'[^a-zA-z\s|]'
    text = re.sub(pattern, '', text)
    return text

songs_file = pd.read_csv('billboard_lyrics_1964-2015.csv', encoding='cp1252', dtype = {'Year' : 'object'})

print(sum(songs_file['Lyrics'].isnull()))

print("Test 1: ", songs_file.loc[500, 'Lyrics'])
print("Test 2: ", type(songs_file.loc[192, 'Lyrics']))

for i in songs_file['Lyrics']:
    if isinstance(i, float):
        pass
    else:
        clean_lyrics = remove_special_characters(i, True)
        songs_file.loc[i, 'Lyrics'] = clean_lyrics

print("Test 3: ", songs_file.loc[500, 'Lyrics'])


