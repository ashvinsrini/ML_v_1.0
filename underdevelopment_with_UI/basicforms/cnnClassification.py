import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
class cnn:
    def __init__(self, batch_size, num_classes, epochs, opt, data_augment):
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.epochs  = int(epochs)
        self.opt = opt
        self.data_augment = data_augment

    def designModel(self, inpShape, num_classes, opt, lr = 0.0001):
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same',
                             input_shape = inpShape))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes, activation = "softmax"))


            # initiate RMSprop optimizer
            #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

            # Let's train the model using RMSprop
            model.compile(loss='categorical_crossentropy',
                          optimizer= opt,
                          metrics=['accuracy'])
            return model

    def modelTraining(self, model, x_train, x_test, y_train, y_test, dataaugment = False):
        print('data augmentation --------->{}'.format(dataaugment))
        if dataaugment == 'False':
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_data=(x_test, y_test),
                      shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=self.batch_size),
                                epochs=self.epochs,
                                validation_data=(x_test, y_test))
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        return scores


    def demo(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                # Convert class vectors to binary class matrices.

        #print('num pf classes ------> is {}'.format(self.num_classes))
        #print('opt ------> is {}'.format(self.opt))

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        shape = x_train.shape[1:]
        mod = self.designModel(shape, self.num_classes, self.opt)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        scores = self.modelTraining(mod, x_train, x_test, y_train, y_test, self.data_augment)

    def trainFromCSV(self, filepath, target, imght, imgwdt, rgb):
        df = pd.read_csv(filepath)
        feats = df.drop(columns = [target])
        lab = df[target]
        x_train, x_test, y_train, y_test = train_test_split(feats, lab, test_size=0.33, random_state=42)
        x_train.reset_index(inplace = True, drop = True)
        x_test.reset_index(inplace = True, drop = True)
        trainImgs = np.array([np.array(x_train.loc[i,:]).reshape(imght,imgwdt,rgb) for i in range(len(x_train))])
        testImgs = np.array([np.array(x_test.loc[i,:]).reshape(imght,imgwdt,rgb) for i in range(len(x_test))])
        y_train = keras.utils.to_categorical(y_train.values, self.num_classes)
        y_test = keras.utils.to_categorical(y_test.values, self.num_classes)
        shape = trainImgs.shape[1:]
        mod = self.designModel(shape, self.num_classes, self.opt)

        trainImgs = trainImgs.astype('float32')
        testImgs = testImgs.astype('float32')
        trainImgs /= 255
        testImgs /= 255
        scores = self.modelTraining(mod, trainImgs, testImgs, y_train, y_test, self.data_augment)
