from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import pickle
import numpy as np
import gensim # thư viện NLP
import os 

def get_data(folder_path):
    data = []
    labels = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in (file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
        
                lines = ViTokenizer.tokenize(lines)
                data.append(lines)
                labels.append(path)

    return data, labels

if __name__ == "__main__":
    folder_path = "data/"
    X_data, y_data = get_data(folder_path+'Train_Full')
    pickle.dump(X_data, open(folder_path+'x_train.pkl', 'wb'))
    pickle.dump(y_data, open(folder_path+'y_train.pkl', 'wb'))

    X_test, y_test = get_data(folder_path+'Test_Full')
    pickle.dump(X_test, open(folder_path+'x_test.pkl', 'wb'))
    pickle.dump(y_test, open(folder_path+'y_test.pkl', 'wb'))