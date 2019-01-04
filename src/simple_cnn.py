from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop, adadelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy

def create_cnn(input_shape, loss):

    '''Create a simple CNN model

    args:
    input_shape (tuple): tuple with image dimensions (height, width, channel number)
    loss (string): loss function for network (categorical_crossentropy or binary_crossentropy)
    '''

    kernel_size = (3, 3)
    pool_size = (2,2)
    strides = 1

    if loss == 'categorical_crossentropy':
        last_activation = 'softmax'
        output_num = 2
    else:
        last_activation = 'sigmoid'
        output_num = 1

    model = Sequential()

    model.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape,
                            kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))


    model.add(Convolution2D(64, kernel_size[0], kernel_size[1],
                            kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))


    model.add(Convolution2D(64, kernel_size[0], kernel_size[1],
                            kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

    # transition to an mlp
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(64, kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_num))
    model.add(Activation(last_activation))

    return model


if __name__=='__main__':

    hold_path = '/Users/christopherlawton/final_test_train_hold/hold'
    train_path = '/Users/christopherlawton/final_test_train_hold/train'
    test_path = '/Users/christopherlawton/final_test_train_hold/test'

    scale = 65535
    batch_size = 60
    nb_epoch = 1
    target_size = (32, 32)
    input_shape = (32, 32, 1)
    class_mode = 'categorical'
    loss_function = 'categorical_crossentropy'

    model_name = 'testing_model'

    model = create_cnn(input_shape, loss=loss_function)


    call_backs = [ModelCheckpoint(filepath='/Users/christopherlawton/galvanize/module_2/capstone_2/save_model/{}'.format(model_name),
                                monitor='val_loss',
                                save_best_only=True),
                                EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    train_datagen = ImageDataGenerator(
                    rescale=1./scale,
                    rotation_range=0.4,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True)

    validatiobn_datagen = ImageDataGenerator(
                    rescale=1./scale)

    train_generator = train_datagen.flow_from_directory(
                        train_path,
                        color_mode='grayscale',
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode=class_mode,
                        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
                        test_path,
                        color_mode='grayscale',
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode=class_mode,
                        shuffle=True)

    model.compile(loss=loss_function,
                  optimizer='adadelta',
                  metrics=['accuracy'])

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=520 // batch_size,
            epochs=nb_epoch,
            validation_data=validation_generator,
            callbacks=call_backs,
            validation_steps=148 // batch_size)
