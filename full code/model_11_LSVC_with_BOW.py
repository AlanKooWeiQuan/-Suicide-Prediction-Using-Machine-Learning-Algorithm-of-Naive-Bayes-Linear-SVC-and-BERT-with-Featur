#### Module : model 11 (linear SVC with bag of words)


## libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np


def model_11(data):
    
    #### feature engineering
    count = CountVectorizer(max_features = 20000, ngram_range =(1,3), analyzer='char')
    X = count.fit_transform(data['text'])
    y = data['class']
    
    #### train test data split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
    
    #### classifier and report
    clf = LinearSVC(dual=True, max_iter=1000)
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
    