import data_loading

## load data
data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2.csv')



# import pandas as pd
# pd.set_option("display.max_colwidth", None)
# # print(data[0:5],'\n')
# for n in range(5):
#     print(data[n:n+1],'\n')
# pd.set_option("display.max_colwidth", 60)
# print (data.describe(include=['object']),'\n')
# print(data['class'].unique(),'\n')
# print(data['class'].value_counts(),'\n')


#### check whihc data is empth and print number row
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\test.csv')
# for n in range(len(data)):
#     if data['sentiment polarity level'].iloc[n] == "PROBLEM" :
#         print(n)


numbers = [175, 347, 3, 1593, 1, 5011, 1588, 81, 62,290]
numbers.sort()

print(numbers)






