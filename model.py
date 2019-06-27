import os
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Lambda, Dense, Conv2D, Flatten, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 70, 160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

#class CTools:
def bgr2rgb(self, image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def flip(image):
    return cv2.flip(image, 1)

def crop(image):
    return image[70:160, :]

def resize(image, shape=(160, 70)):
    return cv2.resize(image, shape)

def pretty(image):
    return resize(crop(image))

class CHyperParams:
    def __init__(self):
        self.dataDirectory = 'data/myData/'
        self.dataFile = 'driving_log.csv'
        self.learningRate = 0.0009
        self.samplesPerEpoch = 20000
        self.batchSize = 200
        self.testSize = 0.15
        self.epochs = 10
        self.loss = losses.mean_squared_error
    
    def getModel(self, inputShape=(160,70)):
        model = Sequential()
        model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(inputShape[1], inputShape[0], 3)))
        model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1)) # output steering value

        model.compile(loss = self.loss, optimizer = Adam(lr = self.learningRate))

        return model

def loadData(dataDir, dataFile, testSize):
    print("data file: ", os.path.join(dataDir, dataFile))
    data_df = pd.read_csv(os.path.join(dataDir, dataFile))
    
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=testSize, random_state=0)
    return X_train, X_valid, y_train, y_valid

def getBatch(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    #tools = CTools()
    #print("Batch size: ", batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            
            filePath = os.path.join(data_dir, center.strip())
            if os.path.isfile(filePath):
                image = mpimg.imread(filePath)

                # add the image and steering angle to the batch
                #images[i] = tools.pretty(image)
                images[i] = pretty(image)
                steers[i] = steering_angle
                i += 1
                #print ("File path:", filePath, "| i =", i, end="\r")
                if i == batch_size - 1:
                    break
                    
        yield images, steers
        
def train_model(hP, model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor = 'val_loss',
                                 verbose = 0,
                                 save_best_only = 'true',
                                 mode = 'auto')

    #model.fit_generator(training_generator, 
    #                    validation_data = validation_data_generator,
    #                    samples_per_epoch = samples_per_epoch, 
    #                    nb_epoch = 3, 
    #                    nb_val_samples = 3000)

    model.fit_generator(getBatch(hP.dataDirectory, X_train, y_train, hP.batchSize, True),
                        samples_per_epoch = hP.samplesPerEpoch,
                        epochs = hP.epochs,
                        max_q_size = 1,
                        validation_data = getBatch(hP.dataDirectory, X_valid, y_valid, hP.batchSize, False),
                        nb_val_samples = len(X_valid),
                        #callbacks = [checkpoint],
                        verbose = 1)
    
if __name__ == '__main__':
    hP = CHyperParams()
    model = hP.getModel()

    X_train, X_valid, y_train, y_valid = loadData(hP.dataDirectory, hP.dataFile, hP.testSize)
    train_model(hP, model, X_train, X_valid, y_train, y_valid)
    
# end of model.py    