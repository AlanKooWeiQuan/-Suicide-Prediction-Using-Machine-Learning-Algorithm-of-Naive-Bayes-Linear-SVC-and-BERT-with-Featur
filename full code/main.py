##### Main program for (data preprocessing)

## libraries
import data_loading
import data_preprocessing

## load data
data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2.csv')
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\1_kaggle\kaggle_testing_data.csv')
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\2_twitter\twitter_testing_data.csv')
# data = data_loading.load_data2(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MAX.csv')
# data = data.drop(data.iloc[:, 3:87],axis = 1)


##data preprocessing
data = data.drop(['no'], axis = 1) #### need remove this syntax and remove the no column in csv
data = data_preprocessing.row_removal_missing_data(data)
data = data_preprocessing.row_removal_of_unwated_class(data)
data = data_preprocessing.expand_contractions(data)
data = data_preprocessing.punctuation_removal(data)
data = data_preprocessing.stop_words_removal(data) 
data = data_preprocessing.lower_casing(data)
data = data_preprocessing.stemming(data) 
data = data_preprocessing.lemmatization(data)
data = data_preprocessing.digit_removal(data) 
data = data_preprocessing.removal_rephrase_unwanted_pattern_text(data)
data = data_preprocessing.removal_duplicate_data(data)


import pandas as pd
pd.set_option("display.max_colwidth", None)
# print(data[0:5],'\n')
# for n in range(5):
#     print(data[n:n+1],'\n')
pd.set_option("display.max_colwidth", 60)
print (data.describe(include=['object']),'\n')
print(data['class'].unique(),'\n')
print(data['class'].value_counts(),'\n')


# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_after_preprocessing.csv', index = False)
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\1_kaggle\kaggle_testing_data_preprocess.csv')
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\testing\2_twitter\twitter_testing_data_preprocess.csv')
# data.to_csv(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\MAX_after_preprocessing.csv', index = False)
