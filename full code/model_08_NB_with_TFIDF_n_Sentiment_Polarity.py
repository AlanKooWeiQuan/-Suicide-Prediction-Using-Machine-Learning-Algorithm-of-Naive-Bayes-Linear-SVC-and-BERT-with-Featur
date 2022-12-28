#### Module : model 08 (naive bayes with TFIDF and sentiment polarity)


## libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np


def model_08(data):
    
    #### feature engineering
    df = pd.DataFrame(data=data)
    X = df[['text', 
            'polarity_pos', 
            'polarity_neg', 
            'polarity_compound']]
    vectorizer = TfidfVectorizer(max_features = 20000, ngram_range =(1,3), analyzer='char')
    column_transformer = ColumnTransformer(
        [('tfidf', vectorizer, 'text')],
        remainder='passthrough')
    X = column_transformer.fit_transform(X)
    y = df['class']
    
    #### train test data split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
    
    #### classifier and report
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Model's Performance Metrics Report : ")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Model's Confusion Matrix Report : ")
    print(conf_matrix)
    
    #### report visualization
    sns.set(font_scale=1.4)
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2, in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(conf_matrix, annot=labels, fmt="", cmap='Blues')
    ax.set_title('Suicide Prediction Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['non-suicide','suicide'])
    ax.yaxis.set_ticklabels(['non-suicide','suicide'])
    plot.show()
    