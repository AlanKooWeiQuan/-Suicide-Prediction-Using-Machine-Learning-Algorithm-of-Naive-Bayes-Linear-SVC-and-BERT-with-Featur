######### Main program 2 for (model training)

#### libraries
import data_loading
import model_01_NB_with_BOW
import model_02_NB_with_TFIDF
import model_03_NB_with_Sentiment_Polarity
import model_04_NB_with_Emotion
import model_05_NB_with_BOW_n_Sentiment_Polarity
import model_06_NB_with_BOW_n_Emotion
import model_07_NB_with_BOW_n_Sentiment_Polarity_n_Emotion
import model_08_NB_with_TFIDF_n_Sentiment_Polarity
import model_09_NB_with_TFIDF_n_Emotion
import model_10_NB_with_TFIDF_n_Sentiment_Polarity_n_Emotion
import model_11_LSVC_with_BOW
import model_12_LSVC_with_TFIDF
import model_13_LSVC_with_Sentiment_Polarity
import model_14_LSVC_with_Emotion
import model_15_LSVC_with_BOW_n_Sentiment_Polarity
import model_16_LSVC_with_BOW_n_Emotion
import model_17_LSVC_with_BOW_n_Sentiment_Polarity_n_Emotion
import model_18_LSVC_with_TFIDF_n_Sentiment_Polarity
import model_19_LSVC_with_TFIDF_n_Emotion
import model_20_LSVC_with_TFIDF_n_Sentiment_Polarity_n_Emotion


#### load data
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_after_preprocessing.csv')
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_sentiment_polarity.csv')
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_emotion.csv')
data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MAX_emotion.csv')


#### model training and evaluation
# ## model 01
# print("\n****** MODEL 01 (NAIVE BAYES with BOW) ******")
# model_01_NB_with_BOW.model_01(data)
#
# ## model 02
# print("\n\n****** MODEL 02 (NAIVE BAYES with TFIDF) ******")
# model_02_NB_with_TFIDF.model_02(data)
#
# ## model 03
# print("\n\n****** MODEL 03 (NAIVE BAYES with sentiment polarity) ******")
# model_03_NB_with_Sentiment_Polarity.model_03(data)
#
# ## model 04
# print("\n\n****** MODEL 04 (NAIVE BAYES with emotion) ******")
# model_04_NB_with_Emotion.model_04(data)
#
# ## model 05
# print("\n\n****** MODEL 05 (NAIVE BAYES with BOW and sentiment polarity) ******")
# model_05_NB_with_BOW_n_Sentiment_Polarity.model_05(data)
#
# ## model 06
# print("\n\n****** MODEL 06 (NAIVE BAYES with BOW and emotion) ******")
# model_06_NB_with_BOW_n_Emotion.model_06(data)
#
# ## model 07
# print("\n\n****** MODEL 07 (NAIVE BAYES with BOW, sentiment polarity and emotion) ******")
# model_07_NB_with_BOW_n_Sentiment_Polarity_n_Emotion.model_07(data)
#
# ## model 08
# print("\n\n****** MODEL 08 (NAIVE BAYES with TFIDF and sentiment polarity) ******")
# model_08_NB_with_TFIDF_n_Sentiment_Polarity.model_08(data)
#
# ## model 09
# print("\n\n****** MODEL 09 (NAIVE BAYES with TFIDF and emotion) ******")
# model_09_NB_with_TFIDF_n_Emotion.model_09(data)
#
# ## model 10
# print("\n\n****** MODEL 10 (NAIVE BAYES with TFIDF, sentiment polarity and emotion) ******")
# model_10_NB_with_TFIDF_n_Sentiment_Polarity_n_Emotion.model_10(data)
#
# ## model 11
# print("\n\n****** MODEL 11 (Linear SVC with BOW) ******")
# model_11_LSVC_with_BOW.model_11(data)
#
# ## model 12
# print("\n\n****** MODEL 12 (Linear SVC with TFIDF) ******")
# model_12_LSVC_with_TFIDF.model_12(data)
#
# ## model 13
# print("\n\n****** MODEL 13 (Linear SVC with sentiment polarity) ******")
# model_13_LSVC_with_Sentiment_Polarity.model_13(data)
#
# ## model 14
# print("\n\n****** MODEL 14 (Linear SVC with emotion) ******")
# model_14_LSVC_with_Emotion.model_14(data)
#
# ## model 15
# print("\n\n****** MODEL 15 (Linear SVC with BOW and sentiment polarity) ******")
# model_15_LSVC_with_BOW_n_Sentiment_Polarity.model_15(data)

## model 16
print("\n\n****** MODEL 16 (Linear SVC with BOW and emotion) ******")
model_16_LSVC_with_BOW_n_Emotion.model_16(data)

## model 17
print("\n\n****** MODEL 17 (Linear SVC with BOW, sentiment polarity and emotion) ******")
model_17_LSVC_with_BOW_n_Sentiment_Polarity_n_Emotion.model_17(data)

## model 18
print("\n\n****** MODEL 18 (Linear SVC with TFIDF and sentiment polarity) ******")
model_18_LSVC_with_TFIDF_n_Sentiment_Polarity.model_18(data)

## model 19
print("\n\n****** MODEL 19 (Linear SVC with TFIDF and emotion) ******")
model_19_LSVC_with_TFIDF_n_Emotion.model_19(data)

## model 20
print("\n\n****** MODEL 20 (Linear SVC with TFIDF, sentiment polarity and emotion) ******")
model_20_LSVC_with_TFIDF_n_Sentiment_Polarity_n_Emotion.model_20(data)