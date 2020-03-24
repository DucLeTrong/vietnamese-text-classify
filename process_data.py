import os
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy

def remove_stopwords(data, stopwords):

    for i in range(len(data)):
        text = data[i]
        try:
            split_words =  [x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'').lower() for x in text.split()]
        except TypeError:
            split_words =  []
        data[i] = ' '.join([word for word in split_words if word not in stopwords])
        
    return data


def preprocess_data(data_path, model_path):

    X_data = pickle.load(open(data_path + "x_train.pkl",'rb'))
    y_data = pickle.load(open(data_path + "y_train.pkl",'rb'))
    X_test = pickle.load(open(data_path + "x_test.pkl",'rb'))
    y_test = pickle.load(open(data_path + "y_test.pkl",'rb'))

    with open(data_path + 'vietnamese-stopwords-dash.txt', 'r') as f:
        stopwords = set([w.strip() for w in f.readlines()])

    X_data = remove_stopwords(X_data, stopwords)
    X_test = remove_stopwords(X_test, stopwords)


    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=10000)
    tfidf_vect.fit(X_data)
    pickle.dump(tfidf_vect, open(model_path+"vectorizer.pickle", "wb"))
    tfidf_X_data =  tfidf_vect.transform(X_data)
    tfidf_X_test =  tfidf_vect.transform(X_test)

    svd = TruncatedSVD(n_components=500, random_state=1998)
    svd.fit(tfidf_X_data)
    pickle.dump(svd, open(model_path+"selector.pickle", "wb"))

    tfidf_X_data_svd = svd.transform(tfidf_X_data)
    tfidf_X_test_svd = svd.transform(tfidf_X_test)

    encoder = preprocessing.LabelEncoder()
    y_data_one_hot = encoder.fit_transform(y_data)
    y_test_one_hot = encoder.fit_transform(y_test)
    numpy.save(model_path+'classes.npy', encoder.classes_)

    return tfidf_X_data_svd, y_data_one_hot, tfidf_X_test_svd, y_test_one_hot