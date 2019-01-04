import keras
from keras.applications import InceptionV3, Xception, VGG16
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop, adadelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from simple_cnn import create_cnn
import os
import glob
from instantiate_transfer_model import create_transfer_model


class ClassificationCNN():

    '''
    Create either a simple cnn model using simple_cnn.py
    '''

    def __init__(self, project_name, target_size, channels=1, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255):
        '''
        Instantiate cnn model

        args:
        project_name (string): name of project to save best model -> see callbacks in fit method to alter path if needed.
                               Otherwise create a 'save_model' directory.
        target_size (tuple): width and height dimenstions of target images. This will be put into the datagenerators in the build_generators method
        channels (int): specify channel number for images (1 for grayscale, 3 for rgb).
        augmentation_stregth (float): magnitude of image augmentation under train Image data generator.
        preprocessing (function): function to input into image data generators if additional processing is needed.
        batch_size (int): size of batches to pull images from directory using Keras flow_from_directory
        scale (int): specify image scale (max pixel value) for example: 255 for uint8 images and 65535 for uint16.
        '''

        self.project_name = project_name
        self.target_size = target_size
        if channels == 3:
            self.input_size = self.target_size + (3,)
            self.color_mode = 'rgb'
        else:
            self.input_size = self.target_size + (1,)
            self.color_mode = 'grayscale'
        self.train_datagen = None
        self.train_generator= None
        self.validation_datagen = None
        self.validation_generator = None
        self.holdout_generator = None
        self.augmentation_strength = augmentation_strength
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.scale = scale #scale is included for to use either uint8 or uint16 images
        self.class_names =  None

        self.loss_function = None
        self.class_mode = None
        self.last_activation = None
        self.history = None

    def get_data(self, train_folder, validation_folder, holdout_folder):

        '''
        Get number of images in train, validation, and hold out folder. This is used in the fit generator under the fit method

        args:
        train_folder (string): path to train folder
        validation_folder (string): path to validation folder
        holdout_folder (string): path to holdout folder
        '''

        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.holdout_folder = holdout_folder

        self.num_train = sum(len(files) for _, _, files in os.walk(self.train_folder)) #: number of training samples

        self.num_val = sum(len(files) for _, _, files in os.walk(self.validation_folder)) #: number of validation samples

        self.num_holdout = sum(len(files) for _, _, files in os.walk(self.holdout_folder)) #: number of holdout samples

        self.num_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(self.train_folder)) #: number of categories


    def build_generators(self):
        '''Create generators to read images from directory

        args:
        self -> this is called under the fit method
        '''

        self.train_datagen = ImageDataGenerator(
                        preprocessing_function = self.preprocessing,
                        rescale=1./self.scale,
                        rotation_range=self.augmentation_strength,
                        width_shift_range=self.augmentation_strength,
                        height_shift_range=self.augmentation_strength,
                        shear_range=self.augmentation_strength,
                        zoom_range=self.augmentation_strength,
                        horizontal_flip=True)

        self.validation_datagen = ImageDataGenerator(
                        preprocessing_function = self.preprocessing,
                        rescale=1./self.scale)

        self.train_generator = self.train_datagen.flow_from_directory(
                            self.train_folder,
                            color_mode=self.color_mode,
                            target_size=self.target_size,
                            batch_size=self.batch_size,
                            class_mode=self.class_mode,
                            shuffle=True)

        self.validation_generator = self.validation_datagen.flow_from_directory(
                            self.validation_folder,
                            color_mode=self.color_mode,
                            target_size=self.target_size,
                            batch_size=self.batch_size,
                            class_mode=self.class_mode,
                            shuffle=True)

    def fit(self, input_model, train_folder, validation_folder, holdout_folder, epochs, loss, optimizer='adadelta'):

        '''
        This fit method calls on simple_cnn.py. See script for more details.
        This method with fit, train, and evaluate the model.

        args:
        input_model (object): model instantiated using create cnn from simple_cnn.py under main block
        train_folder (string): path to train folder
        validation_folder (string): path to validation folder
        holdout_folder (string): path to holdout folder
        epochs (int): number of epochs to train model
        loss (string): specified loss function. These models are classification, so you may only enter
                       'categorical_crossentopy' or 'binary_crossentropy'.
        optimizer (string): keras optimizer to use for the model
        '''

        self.loss_function = loss
        if self.loss_function == 'categorical_crossentropy':
            self.class_mode = 'categorical'
            self.last_activation = 'softmax'
        elif self.loss_function == 'binary_crossentropy':
            self.class_mode = 'binary'
            self.last_activation = 'sigmoid'
        else:
            print('WARNING: Please specify loss function as categorical or binary crossentropy')

        self.get_data(train_folder, validation_folder, holdout_folder)
        self.build_generators()
        model = input_model(self.input_size, loss)

        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['accuracy'])

        #initialize tensorboard for monitoring
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.project_name, histogram_freq=0, batch_size=self.batch_size, write_graph=True, embeddings_freq=0)

        #initialize model checkpoint to save the best model
        save_name = 'save_model/'+self.project_name+'.hdf5'
        call_backs = [ModelCheckpoint(filepath=save_name,
                                    monitor='val_loss',
                                    save_best_only=True),
                                    EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

        self.history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.num_train// self.batch_size,
                epochs=epochs,
                validation_data=self.validation_generator,
                validation_steps=self.num_val // self.batch_size,
                callbacks=call_backs)

        best_model = load_model(save_name)
        print('evaluating simple model')
        accuracy = self.evaluate_model(best_model, self.holdout_folder)

        return save_name

    def evaluate_model(self, model, holdout_folder):
        """
        evaluates model on holdout data -> called from fit method
        Args:
            model (keras classifier model): model to evaluate
            holdout_folder (str): path of holdout data
        Returns:
            list(float): metrics returned by the model, typically [loss, accuracy]
            """

        self.holdout_generator = self.validation_datagen.flow_from_directory(
                            self.holdout_folder,
                            color_mode=self.color_mode,
                            target_size=self.target_size,
                            batch_size=self.batch_size,
                            class_mode=self.class_mode,
                            shuffle=False)


        metrics = model.evaluate_generator(self.holdout_generator,
                                           steps=self.num_holdout/self.batch_size,
                                           use_multiprocessing=True,
                                           verbose=1)
        print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
        return metrics

    def plot_model(self, save_name=None):

        '''
        This method will plot accuracy and loss on individual graphs. This will allow saving of separate graphs

        args:
        save_name (string): path to save gaph. If not specified, graph will only show.
        '''

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) +1)

        plt.plot(epochs, acc, 'g-', label='Training acc')
        plt.plot(epochs, val_acc, 'b-', label='Validation acc')
        plt.title('Training and validation accuracy', fontsize=18)
        plt.xlabel('epochs', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'g-', label='Training loss')
        plt.plot(epochs, val_loss, 'b-', label='Validation loss')
        plt.title('Training and validation loss', fontsize=18)
        plt.xlabel('epochs', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.legend()
        if save_name:
            plt.savefig(save_name)
        else:
            plt.show()

    def plot_model_2(self, save_name=None):

        '''
        This method will plot accuracy and loss on the same figure with 2 subplots.

        args:
        save_name (string): path to save gaph. If not specified, graph will only show.
        '''

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) +1)

        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9,5))
        ax0.plot(epochs, acc, 'g-', label='Training acc')
        ax0.plot(epochs, val_acc, 'b-', label='Validation acc')
        ax0.set_title('Training and validation accuracy', fontsize=16)
        ax0.set_xlabel('epochs', fontsize=14)
        ax0.set_ylabel('loss', fontsize=14)
        ax0.legend()

        ax1.plot(epochs, loss, 'g-', label='Training loss')
        ax1.plot(epochs, val_loss, 'b-', label='Validation loss')
        ax1.set_title('Training and validation loss', fontsize=16)
        ax1.set_xlabel('epochs', fontsize=14)
        ax1.set_ylabel('loss', fontsize=14)
        ax1.legend()
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)
        else:
            plt.show()

class TransferCNN(ClassificationCNN):

    '''
    Create either a simple cnn model using instaniate_transfer_model.py. See script for more details.
    This inherits from ClassificationCNN. Instatiate using same __init__ method above.
    '''

    def fit(self, model_name, train_folder, validation_folder, holdout_folder, input_model, n_categories, loss, optimizers, epochs, freeze_indices, warmup_epochs=5):

        '''
        This fit method calls on simple_cnn.py. See script for more details.
        This method with fit, train, and evaluate the model.

        args:
        model_name (string): name of model to save. Saves in local 'save_model' directory by default.
        train_folder (string): path to train folder
        validation_folder (string): path to validation folder
        holdout_folder (string): path to holdout folder
        input_model (object): model instantiated using create_tranfer_model from instantiate_tranfer_model.py under main block
        loss (string): specified loss function. These models are classification, so you may only enter
                       'categorical_crossentopy' or 'binary_crossentropy'.
        optimizers (list): Keras optimizers with learning rates for warm up epochs and first level fine tuning epochs.
        epochs (int): number of epochs to train model
        freeze_indices (list): first index for warming up head second + indices for fine tuning layers. This is used in the change_trainable_layers method
        warmup_epochs (int): number of epochs for head
        '''

        self.n_categories = n_categories

        self.loss_function = loss
        if self.loss_function == 'categorical_crossentropy':
            self.class_mode = 'categorical'
            self.last_activation = 'softmax'
        elif self.loss_function == 'binary_crossentropy':
            self.class_mode = 'binary'
            self.last_activation = 'sigmoid'
        else:
            print('WARNING: Please specify loss function as categorical or binary crossentropy')

        self.get_data(train_folder, validation_folder, holdout_folder)
        self.build_generators()

        model = input_model(self.input_size, self.n_categories, self.last_activation, model=model_name)
        self.change_trainable_layers(model, freeze_indices[0])

        model.compile(optimizer=optimizers[0],
                        loss=self.loss_function, metrics=['accuracy'])

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.project_name, histogram_freq=0, batch_size=self.batch_size, write_graph=True, embeddings_freq=0)

        save_name = 'save_model/'+self.project_name+'.hdf5'
        call_backs = [ModelCheckpoint(filepath=save_name,
                                    monitor='val_loss',
                                    save_best_only=True),
                                    EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

        self.history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.num_train// self.batch_size,
                epochs=warmup_epochs,
                validation_data=self.validation_generator,
                validation_steps=self.num_val // self.batch_size,
                callbacks=call_backs)

        self.change_trainable_layers(model, freeze_indices[1])

        model.compile(optimizer=optimizers[1],
                      loss=self.loss_function, metrics=['accuracy'])

        self.history = model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.num_train// self.batch_size,
                epochs=epochs,
                validation_data=self.validation_generator,
                validation_steps=self.num_val // self.batch_size,
                callbacks=call_backs)

        if len(freeze_indices) > 2:
            for i in range(len(freeze_indices) - 2):
                i = i + 2
                self.change_trainable_layers(model, freeze_indices[i])

                model.compile(optimizer=optimizers[1],
                              loss=self.loss_function, metrics=['accuracy'])

                self.history = model.fit_generator(
                        self.train_generator,
                        steps_per_epoch=self.num_train// self.batch_size,
                        epochs=epochs,
                        validation_data=self.validation_generator,
                        validation_steps=self.num_val // self.batch_size,
                        callbacks=call_backs)


        best_model = load_model(save_name)
        print('evaluating transfer model')
        accuracy = self.evaluate_model(best_model, self.holdout_folder)

        return save_name


    def change_trainable_layers(self, model, trainable_index):

        '''
        Make all layers that are to keep current weights untrainable and all layers to be trained during specified epochs trainable.
        This is called in the fit mehtod above

        args:
        model (object): model object to be trained
        trainable_index (list): fit method will iteratively supply the index to keep all layers before the index layer untrainable and after trainable
        '''

        for layer in model.layers[:trainable_index]:
            layer.trainable = False
        for layer in model.layers[trainable_index:]:
            layer.trainable = True


if __name__ == '__main__':

    train_folder = '/Users/christopherlawton/galvanize/module_3/mammogram_data/cropped_train_test_split/3_channel/CC/train'
    validation_folder = '/Users/christopherlawton/galvanize/module_3/mammogram_data/cropped_train_test_split/3_channel/CC/test'
    holdout_folder = '/Users/christopherlawton/galvanize/module_3/mammogram_data/cropped_train_test_split/3_channel/CC/hold'

    #simple cnn
    input_shape = (250, 250, 3)
    target_size = (250,250)
    scale = 255
    epochs = 20

    # simple_model = create_cnn
    # simple_cnn = ClassificationCNN('class_test_one', target_size, channels=1, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255)
    # simple_cnn.fit(simple_model, train_folder, validation_folder, holdout_folder, epochs, 'categorical_crossentropy', optimizer='adadelta')

    inception
    model_name = InceptionV3
    warmup_epochs = 5
    epochs = epochs - warmup_epochs
    optimizers = [Adam(lr=0.0006), Adam(lr=0.0001)] # keep learning rates low to keep from wrecking weights
    train_head_idx = [311, 299, 294]
    transfer_model = create_transfer_model
    transfer_cnn = TransferCNN('transfer_test_one', target_size, channels=3, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255)
    savename = transfer_cnn.fit(train_folder, validation_folder, holdout_folder, transfer_model, 2, \
                        'categorical_crossentropy', optimizers, epochs, train_head_idx, warmup_epochs=warmup_epochs)

    #xception
    # model_name = Xception
    # warmup_epochs = 1
    # epochs = epochs - warmup_epochs
    # optimizers = [RMSprop(lr=0.0006), RMSprop(lr=0.0001)] # keep learning rates low to keep from wrecking weights
    # train_head_idx = [132, 126]
    # transfer_model = create_transfer_model
    # transfer_cnn = TransferCNN('transfer_test_one', target_size, channels=3, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255)
    # savename = transfer_cnn.fit(model_name, train_folder, validation_folder, holdout_folder, transfer_model, 2, \
    #                     'categorical_crossentropy', optimizers, epochs, train_head_idx, warmup_epochs=warmup_epochs)

    #VGG16
    # model_name = VGG16
    # warmup_epochs = 5
    # epochs = epochs - warmup_epochs
    # optimizers = [RMSprop(lr=0.0006), RMSprop(lr=0.0001)] # keep learning rates low to keep from wrecking weights
    # train_head_idx = [19, 17, 15]
    # transfer_model = create_transfer_model
    # transfer_cnn = TransferCNN('transfer_test_one', target_size, channels=3, augmentation_strength=0.1, preprocessing=None, batch_size=50, scale=255)
    # savename = transfer_cnn.fit(model_name, train_folder, validation_folder, holdout_folder, transfer_model, 2, \
    #                     'categorical_crossentropy', optimizers, epochs, train_head_idx, warmup_epochs=warmup_epochs)
