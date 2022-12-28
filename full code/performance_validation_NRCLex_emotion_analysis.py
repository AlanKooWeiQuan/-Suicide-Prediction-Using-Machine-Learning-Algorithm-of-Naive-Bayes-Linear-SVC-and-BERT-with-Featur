"""
1 fear....
2 anger........
3 anticipation
4 trust
5 surprise.............
6 positive
7 negative
8 sadness......
9 disgust
10 joy.......
"""
from nrclex import NRCLex

# text = "im feeling quite sad and sorry for myself but ill snap out of it soon"
# emotion = NRCLex(text)
# print(emotion.top_emotions)
# print(len(emotion.top_emotions))

# print(emotion.top_emotions[0][0])
# print(emotion.top_emotions[1][0])
# print(emotion.top_emotions[2][0])
# print("\n\n\n")


############################################
import pandas as pd
data = pd.read_csv(r'C:\Users\AlanKoo99\Desktop\val.csv')
#
# print (data.describe(include=['object']),'\n')
# print(data['emotion'].unique(),'\n')
# print(data['emotion'].value_counts(),'\n')
# print(data.iloc[2][0])

t = 0
f =0
neutral = 0

for n in range(len(data)):
    emotion = NRCLex(data.iloc[n][0])
    for n2 in range(len(emotion.top_emotions)):
        if n2 == len(emotion.top_emotions):
            break
        elif emotion.top_emotions[n2][0] == data.iloc[n][1]:
            print(n)
            print (emotion.top_emotions[n2][0] , "==" , data.iloc[n][1])
            t+=1
            

print("true : " , t)
print("false : " , f)
print("total : " , (t+f))







