
from keras.models import Sequential
from keras import models
from keras.models import *
from keras.layers import Dense, Dropout, LSTM, BatchNormalization,RNN
from keras.layers import *
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import metrics


def create_classifier():
    input_layer = Input(shape=(500,))
    layer = Reshape((1, 500))(input_layer)
    layer = LSTM(256, activation='relu',dropout=0.4,recurrent_dropout=0.3)(layer)
#     layer = LSTM(128, activation='relu',dropout=0.4,recurrent_dropout=0.3)(layer)
#     keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
    layer = Dense(512, activation='relu')(layer)
#     layer = Dropout(0.3)(layer)
#     layer = Reshape((16, 32))(layer)
#     layer = LSTM(256, activation='relu',dropout=0.4,recurrent_dropout=0.3)(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(256, activation='relu')(layer)
#     layer = Dropout(0.3)(layer)
    layer = Dense(256, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
#     layer = Dropout(0.3)(layer)
    layer = Dense(128, activation='relu')(layer)
    output_layer = Dense(10, activation='softmax')(layer)
    model = models.Model(input_layer, output_layer)
#     models.Model.summary()
    
    model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# def create_classifier():
#     input_layer = Input(shape=(500,))
    
#     layer = Reshape((1, 500))(input_layer)
# #     layer = LSTM(256, activation='relu',dropout=0.4,recurrent_dropout=0.3)(layer)
#     layer = Bidirectional(GRU(128, activation='relu'))(layer)

# #     layer = LSTM(128, activation='relu',dropout=0.4,recurrent_dropout=0.3)(layer)
# #     keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
#     layer = Dense(512, activation='relu')(layer)
# #     layer = Dropout(0.3)(layer)
# #     layer = Reshape((16, 32))(layer)
# #     layer = LSTM(256, activation='relu',dropout=0.4,recurrent_dropout=0.3)(layer)
#     layer = Dense(512, activation='relu')(layer)
#     layer = Dense(256, activation='relu')(layer)
#     layer = Dropout(0.3)(layer)
#     layer = Dense(256, activation='relu')(layer)
#     layer = Dense(128, activation='relu')(layer)
#     layer = Dropout(0.3)(layer)
#     layer = Dense(128, activation='relu')(layer)
    
#     output_layer = Dense(10, activation='softmax')(layer)
    
#     classifier = models.Model(input_layer, output_layer)
# #     models.Model.summary()
    
#     classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     return classifier
