import pandas as pd
import numpy as np
from keras.models import Model
from keras.applications import VGG16, InceptionV3, Xception
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import os

def get_layer_info(model):
    '''
    Print out layer name and output. This will allow you to see the layer to you need to stop at the the dimensions to specify

    args:
    model (keras model object): keras model object from applications
    '''

    for layer in model.layers:
        print('layer name: {}'.format(layer.name))
        print('layer output: {}\n'.format(layer.output))

def make_base_model(model, layer_name, optimizer, loss):

    '''
    This function will only be used to trained and loaded models, not for out of the box keras application models.
    Remove head of model, instatiate new model without head, and recompile model so it can be used to predict.

    args:
    model (object): pre-trained model to remove head
    layer_name (string): model layer to make output
    optimizer (string): optimizer to recompile model with
    loss (string): loss function to recompile model with
    '''

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    for layer in intermediate_layer_model.layers:
        layer.trainable = False

    intermediate_layer_model.compile(optimizer=optimizer, loss=loss)

    return intermediate_layer_model


def get_data(train_folder, validation_folder, holdout_folder):

    '''
    Get number of images from train, validation, and holdout folders. Used in the extract_features function.

    args:
    train_folder (string): path to train folder
    validation_folder (string): path to validation folder
    holdout_folder (String): path to holdout folder
    '''

    num_train = sum(len(files) for _, _, files in os.walk(train_folder)) #: number of training samples

    num_val = sum(len(files) for _, _, files in os.walk(validation_folder)) #: number of validation samples

    num_holdout = sum(len(files) for _, _, files in os.walk(holdout_folder)) #: number of holdout samples

    return num_train, num_val, num_holdout


def extract_features(model, directory, sample_count, dims):

    '''
    Predict on model with removed head. Extract feautre maps of specified layer.

    note: Keep batch_size == 1 unless entering a multiple of the total directory image count.
          If your batch size is not a multiple of the total image count for the specified
          directory then Keras will exclude the remaining images.

    args:
    model (object): out of the box model from keras or newly instantiated model from loaded pre-trained model
    directory (string): directory from where to pull the images to predict on
    sample_count (int): count of images from target directory. This is received from the get_data function
    dims: dimensions of feature map (width, height, number of feature maps). This is received from the get_layer_info function
    '''

    features = np.zeros(shape=(sample_count, dims[0], dims[1], dims[2]))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
                directory,
                target_size=(250,250),
                batch_size=batch_size,
                class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        print(features_batch.shape)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

if __name__=='__main__':

    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 1

    dims = (6, 6, 2048)

    #3 channel padded images
    pad_train = '../mammogram_data/cropped_train_test_split/3_channel/combined/train'
    pad_test = '../mammogram_data/cropped_train_test_split/3_channel/combined/test'
    pad_hold = '../mammogram_data/cropped_train_test_split/3_channel/combined/hold'

    #3 channel original images
    orig_train = '../mammogram_data/cropped_train_test_split/original_3_channel/combined/train'
    orig_test = '../mammogram_data/cropped_train_test_split/original_3_channel/combined/test'
    orig_hold = '../mammogram_data/cropped_train_test_split/original_3_channel/combined/hold'

    #Create model for original images and save feature maps into flattened numpy arrays
    inceptionV3_orig_model = load_model('./inception_orig_FT3_Adam_aug1_lrsmall.hdf5')

    inception_orig_compiled = make_base_model(inceptionV3_orig_model, 'mixed10', 'rmsprop', 'categorical_crossentropy')

    num_orig_train, num_orig_test, num_orig_hold = get_data(orig_train, orig_test, orig_hold)


    orig_train_features, orig_train_labels = extract_features(inception_orig_compiled, orig_train, num_orig_train, dims)
    orig_test_features, orig_test_labels = extract_features(inception_orig_compiled, orig_test, num_orig_test, dims)
    orig_hold_features, orig_hold_labels = extract_features(inception_orig_compiled, orig_hold, num_orig_hold, dims)

    orig_train_features = np.reshape(orig_train_features, (num_orig_train, 6*6*2048))
    orig_test_features = np.reshape(orig_test_features, (num_orig_test, 6*6*2048))
    orig_hold_features = np.reshape(orig_hold_features, (num_orig_hold, 6*6*2048))

    np.save('./numpy_arrays/orig_feature_ext_inception_train_X.npy', orig_train_features)
    np.save('./numpy_arrays/orig_feature_ext_inception_train_y.npy', orig_train_labels)

    np.save('./numpy_arrays/orig_feature_ext_inception_test_X.npy', orig_test_features)
    np.save('./numpy_arrays/orig_feature_ext_inception_test_y.npy', orig_test_labels)

    np.save('./numpy_arrays/orig_feature_ext_inception_hold_X.npy', orig_hold_features)
    np.save('./numpy_arrays/orig_feature_ext_inception_hold_y.npy', orig_hold_labels)




    #Create model for original images and save feature maps into flattened numpy arrays
    inceptionV3_padded_model = load_model('./inception_pad_FT4_Adam_aug1_lrsmall.hdf5')

    inception_pad_compiled = make_base_model(inceptionV3_padded_model, 'mixed10', 'rmsprop', 'categorical_crossentropy')

    num_pad_train, num_pad_test, num_pad_hold = get_data(pad_train, pad_test, pad_hold)


    pad_train_features, pad_train_labels = extract_features(inception_pad_compiled, pad_train, num_pad_train, dims)
    pad_test_features, pad_test_labels = extract_features(inception_pad_compiled, pad_test, num_pad_test, dims)
    pad_hold_features, pad_hold_labels = extract_features(inception_pad_compiled, pad_hold, num_pad_hold, dims)

    pad_train_features = np.reshape(pad_train_features, (num_pad_train, 6*6*2048))
    pad_test_features = np.reshape(pad_test_features, (num_pad_test, 6*6*2048))
    pad_hold_features = np.reshape(pad_hold_features, (num_pad_hold, 6*6*2048))

    np.save('./numpy_arrays/pad_feature_ext_inception_train_X.npy', pad_train_features)
    np.save('./numpy_arrays/pad_feature_ext_inception_train_y.npy', pad_train_labels)

    np.save('./numpy_arrays/pad_feature_ext_inception_test_X.npy', pad_test_features)
    np.save('./numpy_arrays/pad_feature_ext_inception_test_y.npy', pad_test_labels)

    np.save('./numpy_arrays/pad_feature_ext_inception_hold_X.npy', pad_hold_features)
    np.save('./numpy_arrays/pad_feature_ext_inception_hold_y.npy', pad_hold_labels)




    #feature extraction with regular inceptionv3 for original images
    incpetionv3_base = InceptionV3(weights='imagenet',
                                    include_top=False,
                                    input_shape=(250,250,3))

    num_orig_train_v3, num_orig_test_v3, num_orig_hold_v3 = get_data(orig_train, orig_test, orig_hold)

    orig_train_v3_features, orig_train_v3_labels = extract_features(incpetionv3_base, orig_train, num_orig_train_v3, dims)
    orig_test_v3_features, orig_test_v3_labels = extract_features(incpetionv3_base, orig_test, num_orig_test_v3, dims)
    orig_hold_v3_features, orig_hold_v3_labels = extract_features(incpetionv3_base, orig_hold, num_orig_hold_v3, dims)

    orig_train_v3_features = np.reshape(orig_train_v3_features, (num_orig_train_v3, 6*6*2048))
    orig_test_v3_features = np.reshape(orig_test_v3_features, (num_orig_test_v3, 6*6*2048))
    orig_hold_v3_features = np.reshape(orig_hold_v3_features, (num_orig_hold_v3, 6*6*2048))

    np.save('./numpy_arrays/orig_fev3_train_X.npy', orig_train_v3_features)
    np.save('./numpy_arrays/orig_fev3_train_y.npy', orig_train_v3_labels)

    np.save('./numpy_arrays/orig_fev3_test_X.npy', orig_test_v3_features)
    np.save('./numpy_arrays/orig_fev3_test_y.npy', orig_test_v3_labels)

    np.save('./numpy_arrays/pad_fev3_hold_X.npy', orig_hold_v3_features)
    np.save('./numpy_arrays/pad_fev3_hold_y.npy', orig_hold_v3_labels)




    #feature extraction with regular inceptionv3 for padded images
    incpetionv3_base = InceptionV3(weights='imagenet',
                                    include_top=False,
                                    input_shape=(250,250,3))

    get_layer_info(incpetionv3_base)

    num_pad_train_v3, num_pad_test_v3, num_pad_hold_v3 = get_data(orig_train, orig_test, orig_hold)

    pad_train_v3_features, pad_train_v3_labels = extract_features(incpetionv3_base, pad_train, num_pad_train_v3, dims)
    pad_test_v3_features, pad_test_v3_labels = extract_features(incpetionv3_base, pad_test, num_pad_test_v3, dims)
    pad_hold_v3_features, pad_hold_v3_labels = extract_features(incpetionv3_base, pad_hold, num_pad_hold_v3, dims)

    pad_train_v3_features = np.reshape(pad_train_v3_features, (num_pad_train_v3, 6*6*2048))
    pad_test_v3_features = np.reshape(pad_test_v3_features, (num_pad_test_v3, 6*6*2048))
    pad_hold_v3_features = np.reshape(pad_hold_v3_features, (num_pad_hold_v3, 6*6*2048))

    np.save('./numpy_arrays/pad_fev3_train_X.npy', pad_train_v3_features)
    np.save('./numpy_arrays/pad_fev3_train_y.npy', pad_train_v3_labels)

    np.save('./numpy_arrays/pad_fev3_test_X.npy', pad_test_v3_features)
    np.save('./numpy_arrays/pad_fev3_test_y.npy', pad_test_v3_labels)

    np.save('./numpy_arrays/pad_fev3_hold_X.npy', pad_hold_v3_features)
    np.save('./numpy_arrays/pad_fev3_hold_y.npy', pad_hold_v3_labels)
