from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import argparse
import os 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from keras.models import model_from_json
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', type=str,
    #                     default='../classification/data/VNTC/corpus/test/')
    parser.add_argument('--prime', type=str,
                        default="đá bóng với đá cầu nhảy dây bắn bi trốn tìm")
    args = parser.parse_args()
    lines = args.prime
    lines = ' '.join(lines)
    lines = gensim.utils.simple_preprocess(lines)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    try:
        split_words =  [x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'').lower() for x in lines.split()]
    except TypeError:
        split_words =  []
    lines = ' '.join([word for word in split_words if word not in stopwords])
    x = [lines]
        

    with open('data/vietnamese-stopwords-dash.txt', 'r') as f:
        stopwords = set([w.strip() for w in f.readlines()])


    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load('model/classes.npy')

    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
    tfidf_vect = pickle.load(open("model/vectorizer.pickle", "rb"))
    tfidf_x = tfidf_vect.transform(x)
    svd = TruncatedSVD(n_components=500, random_state=1998)

    svd = pickle.load(open("model/selector.pickle", "rb"))
    tfidf_x_svd = svd.transform(tfidf_x)

    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights int|o new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded model from disk")
    print(encoder.inverse_transform([np.argmax(loaded_model.predict(np.array(tfidf_x_svd))[0])])[0])