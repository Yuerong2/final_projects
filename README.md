# Final_project

## Group members:
- Tzu-Kun Hsiao(TK-Hsiao)
- Samantha Lynn Walkow(swalkow2)
- Yuerong Hu(Yuerong2)

## Project Name:
- Project Types II: The semantic agreement between fictional corpus and real-word incidents: a case study on the song lyrics and news between 2000-2015.

## Hypotheses:
- 1. Popular songs (lyrics) and news coming out during the same period of time (etc, five years from 2000 to 2005) share certain features that were shaped by the culture and theme of the society during that period.
- 2. There might be a time delay between the similar groups of topics extracted from popular songs and those of the news. Since news always comes out quickly right after the things happened, while it takes a long time to write and publish a song.

## What we are going to do

- Comparing the topics of creative works and factual data during the same time

 1. Data wrangling and preprocessing

 1.1 Collect and align datasets

 1.2 Slice the datasets and build up pairs of subsets for comparison (for instance, lyrics and news datasets from 2000 to   
     2005)

 1.3 Build up alternative pairs of datasets with a time delay (for instance, news dataset from 2000 to 005 and lyrics dataset      from 2001 to 2006)

 2. Topic modeling

 2.1 Model the topics in the top 100 songs and news (2000-2015)

 2.2 Analyze the within-group and between-group cohesion/similarity of topics in songs and news

 Or Text analytics with bag of words model

 2.1. word (feature) selection

 2.2  text analysis based on word features

 3. Discussion and Data Visualization

## Steps
 1. Get Top 100 songs, and aligned News dataset
 2. Subset data by 5-year time window (2000-2005, 2006-2010, 2011-2015)
 3. Perform topic modeling/text analysis on the song lyrics and news of each subset of data

## Datasets going to use

- Songs: https://www.kaggle.com/rakannimer/billboard-lyrics [coverage: 1965-2015)]
- News:  in progress of data wrangling  
  -- ProQuest Historical Newspapers: The New York Times (Database)  
    - https://search.proquest.com/hnpnewyorktimes?accountid=14553 (or can be found from UIUC library)
    - time coverage: 1851 - 2015
    - 843,476 news (document type limited to article and dates as 2000-2015)
    - limitation: can only download 100 record each time
    - what we can get from the downloaded data:    
      * Title, abstract, pub date, and some other extra but might not be useful columns
