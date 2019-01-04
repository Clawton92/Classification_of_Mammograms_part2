from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def get_hold_out_batch_size(target_folder):
    '''
    Get the number of images in target directory for steps
    in evaluate generator and predict generator

    args:
    Target folder (string): path to images to use for prediction
    '''

    num_targets = sum(len(files) for _, _, files in os.walk(target_folder))

    return num_targets


def evaluate_model(model, holdout_generator, batch_size):

    '''evaluate model on holdout set to get loss and accuracy

    args:

    model (object): loaded trained model for evaluation
    holdout_generator (object): images data generator object to supply images
    batch_size (int): specified batch size to flow from directory
    '''

    metrics = model.evaluate_generator(holdout_generator,
                                        steps=num_hold//batch_size,
                                        use_multiprocessing=True,
                                        verbose=1)

    print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
    return metrics[0], metrics[1]

def predictions_to_csv(holdout_generator, model, batch_size, name_csv):

    '''Save predictions to csv. Includes file path, predictions, and predicted probabilities.

    Note: the holdout generator is reset in the event that the evaluation is executed
    before the predictions

    args:

    holdout_generator (object): images data generator object to supply images
    model (object): loaded trained model for evaluation
    batch_szie (int): specified batch size to predict on
    name_csv (string): path to save csv (include name of csv with extension)
    '''
    # pdb.set_trace()
    holdout_generator.reset()
    predictions = model.predict_generator(holdout_generator, steps = num_hold // batch_size)
    pred_vals = predictions
    if class_mode == 'binary':
        vec = np.vectorize(lambda x: 1 if x>0. else 0)
        predicted_class_indices = vec(predictions)
    else:
        predicted_class_indices = np.argmax(predictions, axis=1)
    labels = (holdout_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices.ravel()]
    filenames=holdout_generator.filenames[:len(predictions)]
    actual_vals = ['BENIGN' if file[0] == 'B' else 'MALIGNANT' for file in filenames]
    if class_mode == 'binary':
        results=pd.DataFrame({"Filename":filenames,
                              "Predictions":predictions,
                              "Actual": actual_vals,
                              "probability_class":np.round(pred_vals.ravel()[::2], decimals=2)})
    else:
        results=pd.DataFrame({"Filename":filenames,
                              "Predictions":predictions,
                              "Actual": actual_vals,
                              "prob_benign":np.round(pred_vals.ravel()[::2], decimals=2),
                              "prob_malignant":np.round(pred_vals.ravel()[1::2], decimals=2)})

    results.to_csv("/Users/christopherlawton/galvanize/module_3/cap_3_dir/predictions_csvs/{}.csv".format(name_csv),index=False)

if __name__=='__main__':

    holdout_folder = '../../mammogram_data/cropped_train_test_split/original_3_channel/combined/hold'
    train_folder = '../../mammogram_data/cropped_train_test_split/original_3_channel/combined/train'
    test_folder = '../../mammogram_data/cropped_train_test_split/original_3_channel/combined/test'

    num_hold = get_hold_out_batch_size(holdout_folder)
    num_train = get_hold_out_batch_size(train_folder)
    num_test = get_hold_out_batch_size(test_folder)
    batch_size = 20
    scale = 255
    target_size = (250,250)
    color_mode = 'rgb'
    class_mode = 'categorical'

    model_name = 'inception_orig_FT3_Adam_aug1_lrsmall.hdf5'
    model = load_model('../{}'.format(model_name))

    test_datagen = ImageDataGenerator(
                    rescale=1./scale)

    holdout_generator = test_datagen.flow_from_directory(
                        holdout_folder,
                        color_mode=color_mode,
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode=class_mode,
                        shuffle=True)

    model_loss, model_accuracy = evaluate_model(model, holdout_generator, batch_size)

    # predictions_to_csv(holdout_generator, model, batch_size, 'inception_orig_FT3_Adam_aug1_lrsmall')
