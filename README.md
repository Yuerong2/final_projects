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

 1. Data wrangling and preprocessing (script: Part_1_Data_preprocessing.py; output:cleaned_data(folder))

 1.1 Collect and align datasets

 1.2 Slice the datasets and build up paired samples for comparison 

 1.3 Build up alternative pairs of datasets with a time delay (for instance, news dataset from 2000 to 005 and lyrics dataset  from 2001 to 2006)

 2. Text analysis 

 2.1 Topic Modeling : Model the topics in the top 100 songs and news (2000-2015) 
 2.1.1 Latent Dirichlet Allocation topic modeling
 
 2.1.2 Cross-collection latent Dirichlet allocation (ccLDA) topic modeling
 - Scripts
   in folder: ccLDA
   Part_2_1_2_/ccLDA/create_input4cclda.py
 - Output:
   ccLDA_output_topwords.txt
   

 2.2 Word (feature) selection and  text analysis based on word features. For example, analyze the within-group and between-group cohesion/similarity of topics in songs and news

 Or text analytics with bag of words model

 2.3 Sentiment analysis
 
 - Script: Part_2_3_sentiment analysis
 
 - Output: LyricsSentimentScores.csv & NewsSentimentScores.csv
 
 3. Discussion and Data Visualization


## Datasets used

- Songs: https://www.kaggle.com/rakannimer/billboard-lyrics [coverage: 1965-2015)]
- News:  in progress of data wrangling   
  -- Newspaper Source (Database)
    - http://web.a.ebscohost.com.proxy2.library.illinois.edu/ehost/search/advanced?vid=0&sid=3c64557f-5146-433d-891a-724ed9e12b3d%40sdc-v-sessmgr01 (or can be accessed from UIUC library database catalog)
    - Coverage: Identifies articles in regional U.S. newspapers, international newspapers, newswires and newspaper columns, as well as TV and radio news transcripts. Provides cover-to-cover full text for over 20 national (U.S.) and international newspapers, including USA Today, The Christian Science Monitor, The Washington Post, The Times (London), The Toronto Star, etc. Also contains selected full text from more than 200 regional (U.S.) newspapers, including The Boston Globe, The Chicago Tribune, The Detroit Free Press, The Miami Herald, The New York Daily News, The San Jose Mercury News, etc. In addition, full text television & radio news transcripts are provided from CBS News, FOX News, NPR, etc. 
    - Esther's Notes:
      - This database is great! It allows user to download searched result as a XML file. 
      - I searched for cover stories published in New York Times, between 2001-2015. (I changed the time range a bit so that we can get three equal time window:2001-2005, 2006-2010, and 2011-2015)
      - Got 21,055 records.
      - We can extract title, abstract and pub date from the XML file!
  
  -- ProQuest Historical Newspapers: The New York Times (Database)  
    - https://search.proquest.com/hnpnewyorktimes?accountid=14553 (or can be accessed from UIUC library database catalog)
    - time coverage: 1851 - 2015
    - 843,476 news (document type limited to article and dates as 2000-2015)
    - limitation: can only download 100 record each time
    - what we can get from the downloaded data:    
      * Title, abstract, pub date, and some other extra but might not be useful columns
      
 ## Outline of outcome


Notes to us:
Yuerong: I have created 3 functions with doctests. I see that Esther has created 2 in Part_1_Data_preprocessing.py with doctests and 5 in Part_2_Analytics_bow_approach.py without doctests. I think we have met the "10" requirments.