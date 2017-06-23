import os

from keras.models import Sequential
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import pickle
from keras.callbacks import ModelCheckpoint
from data_prepare_8classes import *
from auxilary.function import one_hot
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.metrics import confusion_matrix

import keras.backend as K
K.set_learning_phase(0)

os.environ['CUDA_VISIBLE_DEVICES']='0'

# Building the LSTM
input_dim = X_train[0].shape
output_dim=y_train.max()+1
hidden_num=200
epoch_num=300
dropout_ratio=0.5
learning_rate=0.0008


modelName = 'blstm' + '_' + str(hidden_num) + '_' + str(learning_rate) + '_' + str(epoch_num) + '_' + str(X_train.shape[0]) + '_' + str(dropout_ratio)
print('Mode name is {}'.format(modelName))
print('Num. of samples is {:}'.format(X_train.shape[0]))

model=Sequential()
model.add(Bidirectional(LSTM(output_dim=hidden_num,return_sequences=False,dropout_W=dropout_ratio),input_shape=input_dim))
model.add(Dense(output_dim=output_dim))
model.add(Activation('softmax'))

optimizer= Adam(lr=learning_rate)
loss='categorical_crossentropy'
metrics=['accuracy']

# path to save weights
model_weights_name = "model_" + modelName + ".hdf5"
model_weights_path = './models/' + model_weights_name
checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc', verbose=1,  save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.summary()

history=model.fit(X_train, one_hot(y_train), nb_epoch=epoch_num, batch_size=80, validation_split=0.2, verbose=1, shuffle=True, callbacks=callbacks_list)
print(history.history.keys())
