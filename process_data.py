import os
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def remove_stopwords(data, stopwords):

    for i in range(len(data)):
        text = data[i]
        try:
            split_words =  [x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'').lower() for x in text.split()]
        except TypeError:
            split_words =  []
        data[i] = ' '.join([word for word in split_words if word not in stopwords])
        
    return data


def preprocess_data(root_path = '/content/drive/My Drive/NLP/10Topics/Ver1.1'):
    X_test = pickle.load(open(root_path + "/data/X_data.pkl",'rb'))
    y_test = pickle.load(open(root_path + "/data/y_data.pkl",'rb'))
    X_data = pickle.load(open(root_path + "/data/X_test.pkl",'rb'))
    y_data = pickle.load(open(root_path + "/data/y_test.pkl",'rb'))

    with open(root_path + '/data/vietnamese-stopwords-dash.txt', 'r') as f:
        stopwords = set([w.strip() for w in f.readlines()])

    X_data = remove_stopwords(X_data)
    X_test = remove_stopwords(X_test)


    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=10000)
    tfidf_vect.fit(X_data)
    pickle.dump(tfidf_vect, open(root_path+"/model/vectorizer.pickle", "wb"))
    tfidf_X_data =  tfidf_vect.transform(X_data)
    tfidf_X_test =  tfidf_vect.transform(X_test)

    svd = TruncatedSVD(n_components=500, random_state=1998)
    svd.fit(tfidf_X_data)
    pickle.dump(svd, open(root_path+"/model/selector.pickle", "wb"))

    tfidf_X_data_svd = svd.transform(tfidf_X_data)
    tfidf_X_test_svd = svd.transform(tfidf_X_test)

    return tfidf_X_data_svd, tfidf_X_test_svd