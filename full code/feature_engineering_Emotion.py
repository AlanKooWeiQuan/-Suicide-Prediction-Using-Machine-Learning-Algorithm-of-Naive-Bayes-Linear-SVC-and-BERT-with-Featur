#### Module : feature engineering (Emotion)


## libraries
import data_loading
from nrclex import NRCLex


def feature_engineering_Emotion(data):
    data["emotion"] =""
    data["emotion_code"] = ""
    for n in range(len(data)):
        emotion = NRCLex(data.text.iloc[n])
        data.emotion.iloc[n] = emotion.top_emotions[0][0]
        
        if emotion.top_emotions[0][0] == "fear" :
            data.emotion_code.iloc[n] = 1
        elif emotion.top_emotions[0][0] == "anger":
            data.emotion_code.iloc[n] = 2
        elif emotion.top_emotions[0][0] == "anticipation" :
            data.emotion_code.iloc[n] = 3
        elif emotion.top_emotions[0][0] == "trust":
            data.emotion_code.iloc[n] = 4
        elif emotion.top_emotions[0][0] == "surprise":
            data.emotion_code.iloc[n] = 5
        elif emotion.top_emotions[0][0] == "positive" :
            data.emotion_code.iloc[n] = 6
        elif emotion.top_emotions[0][0] == "negative":
            data.emotion_code.iloc[n] = 7
        elif emotion.top_emotions[0][0] == "sadness":
            data.emotion_code.iloc[n] = 8
        elif emotion.top_emotions[0][0] == "disgust" :
            data.emotion_code.iloc[n] = 9
        elif emotion.top_emotions[0][0] == "joy":
            data.emotion_code.iloc[n] = 10
    
## Build Model
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_sentiment_polarity.csv')
# feature_engineering_Emotion(data)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_emotion.csv', index = False)
    
    
## kaggle testing
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\1_kaggle\kaggle_testing_data_sentiment_polarity.csv')
# feature_engineering_Emotion(data)
# data = data.drop(data.columns[[0]],axis = 1)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\1_kaggle\kaggle_testing_data_emotion.csv')    
    
    
## twitter testing
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\2_twitter\twitter_testing_data_sentiment_polarity.csv')
# feature_engineering_Emotion(data)
# data = data.drop(data.columns[[0]],axis = 1)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\2_twitter\twitter_testing_data_emotion.csv')


## MAX
data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MAX_sentiment_polarity.csv')
feature_engineering_Emotion(data)
data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MAX_emotion.csv')
