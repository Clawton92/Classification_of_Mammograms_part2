import pandas as pd
import numpy as np
from keras.models import Model
from keras.applications import VGG16, InceptionV3, Xception
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import pickle
from skimage.transform import resize
import imageio
import matplotlib.pyplot as plt

class MassPredictor():

    def __init__(self):
        self.network = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_shape=(250,250,3))
        self.classifier = pickle.load(open('../save_model/rf_final_model.pkl', 'rb'))

        self.predicted_class = None
        self.prob_benign = None
        self.prob_malignant = None

    def predict(self, image):
        if len(image.shape) == 2:
            self.image = np.stack((image,)*3, axis=-1)
        else:
            self.image = image
        resized_image = resize(self.image, (250,250), preserve_range=True)
        self.reshaped_image = np.reshape(resized_image, (1, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2]))
        feature_map = self.network.predict(self.reshaped_image)
        flat_map = np.reshape(feature_map, (1, 6*6*2048))
        predicted_class = self.classifier.predict(flat_map)
        if predicted_class == 1:
            self.predicted_class = 'Malignant'
        else:
            self.predicted_class = 'Benign'
        predicted_proba = self.classifier.predict_proba(flat_map)
        self.prob_benign = np.round(predicted_proba[0][0], 2)
        self.prob_malignant = np.round(predicted_proba[0][1], 2)

    def plot_dist(self, filepath):
        arr = self.image.ravel()
        new_arr = arr[arr > 0]
        plt.hist(new_arr, bins=255, color='g')
        plt.axvline(x=np.mean(self.image.ravel()), color='c', label='mean')
        plt.xlabel("Pixel values", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.title('Submitted Mass Pixel Distribution', fontsize=18)
        plt.legend(loc='upper center')
        plt.savefig(filepath)
        plt.close()





if __name__=='__main__':

    test_image = imageio.imread('/Users/christopherlawton/galvanize/module_3/mammogram_data/cropped_train_test_split/original/combined/hold/BENIGN/Mass-Training_P_01826_RIGHT_MLO_crop.png')

    test = MassPredictor()
    test.predict(test_image)
    print(test.predicted_class)
    print(test.prob_benign)
    print(test.prob_malignant)
