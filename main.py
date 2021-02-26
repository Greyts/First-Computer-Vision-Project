import pandas as pd
from tensorflow.keras.models import Sequential
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
import cv2
import matplotlib as plt
from image_data_generator import DataGenerator

import sklearn.model_selection

data_attr = pd.read_csv(open('D:\Projekt Praktyczny CV\Archive\list_attr_celeba.csv', 'r'))
data_bbox = pd.read_csv(open('D:\Projekt Praktyczny CV\Archive\list_bbox_celeba.csv', 'r'))
data_land = pd.read_csv(open('D:\Projekt Praktyczny CV\Archive\list_landmarks_align_celeba.csv', 'r'))

id = np.array(data_attr['image_id'])
male = np.array(data_attr['Male'])

for index, item in enumerate(id):
    id[index] = (f'D:\Projekt Praktyczny CV\Archive\img_align_celeba\img_align_celeba\{item}')

train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(id, male, test_size = 0.2, random_state=42)

#Model Structure
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(218, 178, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='softmax'))

opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

generatorTrain = DataGenerator(images=train_x, labels=train_y, batch_size=64,shuffle=True, augment=False)
generatorTest = DataGenerator(images=test_x, labels=test_y, batch_size=64,shuffle=True, augment=False)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

history = model.fit_generator(generatorTrain, epochs=50, verbose=True,
                              validation_data=generatorTest,
                              callbacks = [es])