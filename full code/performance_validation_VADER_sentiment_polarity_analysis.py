######### Performance Validation (VADER : sentiment polarity analysis)

#### libraries
from nltk.corpus import movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from spacy.lang import tn


vader = SentimentIntensityAnalyzer()
# print(movie_reviews.raw(movie_reviews.fileids()[0:2]))

positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')


# print(movie_reviews.raw(fileids=negative_fileids[0]))
# print(len(movie_reviews.fileids()))
# print(len(negative_fileids))

tp = 0
tn = 0
fp = 0
fn = 0
neutral = 0

# n=0
# print(vader.polarity_scores(movie_reviews.raw(fileids=positive_fileids[n]))['compound'])
# print(vader.polarity_scores(movie_reviews.raw(fileids=positive_fileids[n])))
# if (vader.polarity_scores(movie_reviews.raw(fileids=positive_fileids[n]))['compound']) < 0 :
#     fn+=1
#
# print(fn)

# for n in range(len(positive_fileids)):
#     if (vader.polarity_scores(movie_reviews.raw(fileids=positive_fileids[n]))['compound']) > 0 :
#         tp+=1
#     elif (vader.polarity_scores(movie_reviews.raw(fileids=positive_fileids[n]))['compound']) < 0 :
#         fn+=1
#     else :
#         neutral+=1
#
# print("true positive : " , tp)
# print("false negative : " , fn)
# print("neutral : " , neutral)
# print("total : " , (tp+fn+neutral))
# print("\n")
#
# for n in range(len(negative_fileids)):
#     if (vader.polarity_scores(movie_reviews.raw(fileids=negative_fileids[n]))['compound']) < 0 :
#         tn+=1
#     elif (vader.polarity_scores(movie_reviews.raw(fileids=negative_fileids[n]))['compound']) > 0 :
#         fp+=1
#     else :
#         neutral+=1
#
# print("true negative : " , tn)
# print("false positive : " , fp)
# print("neutral : " , neutral)
# print("total : " , (tn+fp+neutral))


#########################################################################################################################################
data = pd.read_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MovieReviewTrainingDatabase.csv')
# print(data['sentiment'].unique(),'\n')
# print(data['sentiment'].value_counts(),'\n')
# print(data.iloc[0][1])



for n in range(len(data)):
    if (data.iloc[n][0]) == "Positive" :
        if (vader.polarity_scores(data.iloc[n][1])['compound']) > 0 :
            tp+=1
        elif (vader.polarity_scores(data.iloc[n][1])['compound']) < 0 :
            fn+=1
        else :
            neutral+=1

    if (data.iloc[n][0]) == "Negative" :
        if (vader.polarity_scores(data.iloc[n][1])['compound']) < 0 :
            tn+=1
        elif (vader.polarity_scores(data.iloc[n][1])['compound']) > 0 :
            fp+=1
        else :
            neutral+=1

print("true positive : " , tp)
print("false negative : " , fn)
print("true negative : " , tn)
print("false positive : " , fp)
print("neutral : " , neutral)
print("total : " , (tp+tn+fp+fn+neutral))