#### Module : model 21 (BERT)


## libraries
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plot
# from numba import jit, cuda
# @jit (target_backend='cuda')
# @jit(nopython=True)


def model_21(data):
    data['suicide']= data['class'].apply(lambda x: 1 if x=='suicide' else 0)
    data.sample(5)


    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['suicide'], stratify=data['suicide'])

    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    def get_sentence_embeding(sentences):
        preprocessed_text = bert_preprocess(sentences)
        return bert_encoder(preprocessed_text)['pooled_output']


    # Bert layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # Neural network layers
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

    # Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs = [l])
    # print(model.summary())


    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=METRICS)

    #### train model    
    model.fit(X_train, y_train, epochs=10)
    model.evaluate(X_test, y_test)
    y_predicted = model.predict(X_test)
    y_predicted = y_predicted.flatten()
    y_predicted = np.where(y_predicted > 0.5, 1, 0)
    cm = confusion_matrix(y_test, y_predicted)
    print(cm)


    # print time now
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


    #### confusion table visualization
    ax = sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')
    ax.set_title('Suicide Prediction Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['non-suicide','suicide'])
    ax.yaxis.set_ticklabels(['non-suicide','suicide'])
    plot.show()    

### classification model report
    print(classification_report(y_test, y_predicted))




    print("OK DONE")



import data_loading
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_emotion.csv')
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_1_after_preprocessing.csv')
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\bert_test_1.csv')
data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\experiment_test_2_emotion.csv')
# data = data_loading.load_data(r'C:\Users\AlanKoo99\Desktop\FYP coding\data\bert_model_data_20text.csv')



from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


model_21(data)



