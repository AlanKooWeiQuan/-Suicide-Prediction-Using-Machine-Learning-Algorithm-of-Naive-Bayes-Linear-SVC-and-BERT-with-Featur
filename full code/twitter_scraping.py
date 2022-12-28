#### Module : Twitter Scraping


#### libraries
import snscrape.modules.twitter as sntwitter
from langdetect import detect
import re
import sys
import pandas as pd

######## getTweetDocument
def getTweetDocument(keyword):
    tweets = []
    scraped = sntwitter.TwitterSearchScraper(keyword).get_items()
    languages=['en']
    for i,tweet in enumerate(scraped):
        try:
            lan=detect(tweet.content)
        except:
            lan='error'
        if i>60:
            break
        if lan in languages:
            tweets.append(tweet.content)
            
    for n in range (0,len(tweets)):  
        tweets[n] = re.sub('@[_a-zA-Z0-9]*','',tweets[n])
        tweets[n] = re.sub('#[_a-zA-Z0-9]*','',tweets[n])
        tweets[n] = re.sub('http[://.a-zA-Z0-9]*','',tweets[n])
        tweets[n] = re.sub('\s+',' ',tweets[n]) 
        tweets[n] = re.sub('\A[ ]','',tweets[n])  
        tweets[n] = tweets[n].encode("utf-8")      
    return tweets


tweetsSentences = getTweetDocument("commit suicide since:2000-01-01 until:2022-12-31")
tweetsSentences1 = getTweetDocument("fuck this life since:2000-01-01 until:2022-12-31")
tweetsSentences2 = getTweetDocument("depress since:2000-01-01 until:2022-12-31")

data = []
n = 0
for t in tweetsSentences:
    print(str(n)+'. ')
    sys.stdout.buffer.write(t)
    data.append(t.decode('utf-8'))
    n +=1
    print("\n")

for t in tweetsSentences1:
    print(str(n)+'. ')
    sys.stdout.buffer.write(t)
    data.append(t.decode('utf-8'))
    n +=1
    print("\n")

for t in tweetsSentences2:
    print(str(n)+'. ')
    sys.stdout.buffer.write(t)
    data.append(t.decode('utf-8'))
    n +=1
    print("\n")

  
df = pd.DataFrame(data, columns=['text'])
ori_text = data
df = df.assign(ori_text=ori_text)
# df.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\2_twitter\twitter_testing_data.csv', index = False)
