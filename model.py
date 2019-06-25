import csv
import cv2
import argparse
import numpy as np
import pandas as pd

import keras
#from keras import losses
from keras.models import Sequential
from keras.layers import Lambda, Dense, Conv2D, Flatten, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class CTools:
    def bgr2rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def flip(image):
        return cv2.flip(image, 1)

    def crop(image):
        return image[60:130, :]

    def resize(image, shape=(160, 70)):
        return cv2.resize(image, shape)

    def pretty(image):
        return resize(crop(image))

class CHyperParams:
    def __init__(self):
        self.dataDirectory = 'data',
        self.learningRate = 0.0009,
        self.samplesPerEpoch = 20000,
        self.batchSize = 128,
        self.epochs = 10,
        self.loss = 'mse', #losses.mean_squared_error,
    
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

def loadData(dataDir = 'data', testSize = 0.15):
    data_df = pd.read_csv(os.path.join(dataDir, '.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    return train_test_split(X, y, test_size=testSize, random_state=0)

def getBatch(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    tools = CTools()
    i = 0
    while i < batch_size:
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            #if is_training and np.random.rand() < 0.6:
            #    image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            #else:
            image = mpimg.imread(os.path.join(data_dir, center.strip()))

            # add the image and steering angle to the batch
            images[i] = tools.pretty(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
        
def train_model(hP, model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor = 'val_loss',
                                 verbose = 0,
                                 save_best_only = 'true',
                                 mode = 'auto')

    model.fit_generator(training_generator, 
                        validation_data = validation_data_generator,
                        samples_per_epoch = samples_per_epoch, 
                        nb_epoch = 3, 
                        nb_val_samples = 3000)

    model.fit_generator(getBatch(hP.dataDirectory, X_train, y_train, hP.batchSize, True),
                        samples_per_epoch = hP.samplesPerEpoch,
                        epochs = hP.epochs,
                        max_q_size = 1,
                        validation_data = getBatch(hP.dataDirectory, X_valid, y_valid, hP.batchSize, False),
                        nb_val_samples = len(X_valid),
                        callbacks=[checkpoint],
                        verbose = 1)
    
if __name__ == '__main__':
    hP = CHyperParams()
    model = hP.getModel()

    X_train, X_valid, y_train, y_valid = load_data()
    train_model(hP, model, X_train, X_valid, y_train, y_valid)
    
# end of model.py    