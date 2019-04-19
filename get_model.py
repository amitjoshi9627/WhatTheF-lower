from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import keras
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.convolutional import *
import os

def save_model(model):
	if not os.path.exists('Data/Model/'):
		os.makedirs('Data/Model/')
	saving_path = os.path.join('Data/Model/','Model_save.h5')
	model.save(saving_path)
	return

def get_model(num_classes = 5):
	model = Sequential()

	model.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (48, 48,3), name = 'Conv2d_1'))
	model.add(Conv2D(32,(3,3),activation = 'relu', name = 'Conv2d_2'))
	model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2D_1'))
	model.add(Dropout(0.25, name = 'Dropout_1'))

	model.add(Conv2D(64,(3,3),activation = 'relu', name = 'Conv2d_3'))
	model.add(Conv2D(64,(3,3),activation = 'relu', name = 'Conv2d_4'))
	model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2D_2'))
	model.add(Dropout(0.25, name = 'Dropout_2'))

	model.add(Conv2D(128,(3,3),activation = 'relu', name = 'Conv2d_5'))
	model.add(Conv2D(128,(3,3),activation = 'relu', name = 'Conv2d_6'))
	model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2D_3'))
	model.add(Flatten(name = 'Flatten'))
	model.add(Dense(1024,activation='relu', name = 'Dense_1'))
	model.add(Dropout(0.5, name = 'Dropout_3'))
	model.add(Dense(512,activation='relu',name = 'Dense_2'))
	model.add(Dense(64,activation='relu',name = 'Dense_3'))
	model.add(Dense(num_classes,activation='softmax', name = 'Dense_4'))

	model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

	print(model.summary())

	return model

if __name__ == '__main__':
	save_model(get_model())