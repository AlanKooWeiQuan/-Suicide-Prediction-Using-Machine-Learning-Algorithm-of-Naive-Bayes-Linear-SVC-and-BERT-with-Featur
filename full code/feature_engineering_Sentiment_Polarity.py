#### Module : feature engineering (Sentiment Polarity)


## libraries
import data_loading
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def feature_engineering_Sentiment_Polarity (data):
    vader = SentimentIntensityAnalyzer()
    data["polarity_pos"] =""
    data["polarity_neg"] =""
    data["polarity_compound"] =""
    for n in range(len(data)):
        data.polarity_pos.iloc[n] = vader.polarity_scores(data.text.iloc[n])['pos']
        data.polarity_neg.iloc[n] = vader.polarity_scores(data.text.iloc[n])['neg']
        data.polarity_compound.iloc[n] = (vader.polarity_scores(data.text.iloc[n])['compound']+1)

## Build Model
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_after_preprocessing.csv')
# feature_engineering_Sentiment_Polarity(data)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_sentiment_polarity.csv', index = False)


## kaggle testing
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\1_kaggle\kaggle_testing_data_preprocess.csv')
# feature_engineering_Sentiment_Polarity(data)
# data = data.drop(data.columns[[0]],axis = 1)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\1_kaggle\kaggle_testing_data_sentiment_polarity.csv')


## twitter testing
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\2_twitter\twitter_testing_data_preprocess.csv')
# feature_engineering_Sentiment_Polarity(data)
# data = data.drop(data.columns[[0]],axis = 1)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\2_twitter\twitter_testing_data_sentiment_polarity.csv')


## MAX
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MAX_after_preprocessing.csv')
# feature_engineering_Sentiment_Polarity(data)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MAX_sentiment_polarity.csv')  
    