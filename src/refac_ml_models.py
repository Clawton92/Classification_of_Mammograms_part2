import pandas as pd
import numpy as np
from scipy import misc
import os
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import cluster, decomposition, ensemble, manifold, random_projection, preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from skimage.transform import resize


class PCAModel():

    '''
    Create a PCA model
    Make scree plot
    Make 2d component plot
    '''

    def __init__(self, X_array):
        '''
        Instantiate class with X vals only

        args
        X_array (numpy array): X matrix to apply PCA
        '''
        self.X = X_array
        self.y = None
        self.pca_model = None
        self.X_pca = None

    def make_pca_model(self, n_components=2):
        '''
        Create sklearn pca object

        args:
        n_components (int): integer to specify the number of principal components\
                            to include in the model
                            NOTE: You do not have to specify 2 components to plot your model,\
                            this is accounted for in the plotting method
        '''
        scaler = preprocessing.StandardScaler() #always scale values for PCA
        X_scaled = scaler.fit_transform(self.X)
        self.pca_model = decomposition.PCA(n_components=n_components)
        self.X_pca = self.pca_model.fit_transform(X_scaled)
        return self.X_pca, self.pca_model


    def scree_plot(self, ax, n_components_to_plot=10, title=None):
         """Make a scree plot showing the variance explained (i.e. variance
         of the projections) for the principal components in a fit sklearn
         PCA object.

         args:

         ax (matplot lib object): matplot lib axes object
         n_components_to_plot (int): number of principal components to plot to show\
                                     variance explained for each component
         title (optional) (string): title of plot
         """
         num_components = self.pca_model.n_components_
         ind = np.arange(num_components)
         vals = self.pca_model.explained_variance_ratio_
         ax.plot(ind, vals, color='blue')
         ax.scatter(ind, vals, color='blue', s=50)

         for i in range(num_components):
             ax.annotate(r"{:2.2f}%".format(vals[i]),
                    (ind[i]+0.2, vals[i]+0.005),
                    va="bottom",
                    ha="center",
                    fontsize=12)

         ax.set_xticklabels(ind, fontsize=12)
         ax.set_ylim(0, max(vals) + 0.05)
         ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
         ax.set_xlabel("Principal Component", fontsize=12)
         ax.set_ylabel("Variance Explained (%)", fontsize=12)
         if title:
             ax.set_title(title, fontsize=16)

    def plot_2d(self, y, ax, title=None):
         """Plot an embedding of your dataset onto a 2d plane.

         args:
         y (numpy array): numpy array of tragets corresponding to X_array
         ax (matplotlib axes object): axes to use for plotting 2d PCA plot
         title (optional) (string): title of plot
         """

         X_vals = self.X_pca[:, :2] #grabbing first two principal components to plot in 2d space
         self.y = y
         arr = self.y
         arr[arr == 0] = 2

         x_min, x_max = np.min(X_vals, 0), np.max(X_vals, 0)
         X_final = (X_vals - x_min) / (x_max - x_min)
         ax.axis('off')
         ax.patch.set_visible(False)
         for i in range(X_final.shape[0]):
             plt.text(X_final[i, 0], X_final[i, 1],
                      str(arr[i]),
                      color=plt.cm.Set1(arr[i] / 10),
                      fontdict={'weight': 'bold', 'size': 12})

         ax.set_xticks([]),
         ax.set_yticks([])
         ax.set_ylim([-0.1,1.1])
         ax.set_xlim([-0.1,1.1])
         ax.set_legend()

         if title:
             ax.set_title(title, fontsize=16)

        def visualize(self, path=None):
            '''
            Show or save scree and/or 2d plot depending on what method(s) have been called

            args:
            path (string): path including figure name to save figure to
            '''
            if path:
                plt.savefig(path)
            else:
                plt.show()

class ImgMLClassifier():
    '''
    Instatiate a sklearn classifier object

    methods:

    '''

    def __init__(self, X_arr=None, y_arr=None, arr_path=None):
        '''
        Instantiate object with X and y numpy arrays

        args:

        X_arr (numpy array): full x matrix to use for training and classification
        y_arr (numpy array): full y array to use for model evaluation and training
        arr_path (tuple of length == 2): This tuple must only contain two elements which are both strings \
                                         the first element is the path to the X matrix and the second \
                                         element is the path to the y matrix

        '''
        if arr_path:
            self.X = np.load(arr_path[0])
            self.y = np.load(arr_path[1])
        else:
            self.X = X_arr
            self.y = y_arr

        self.classifier_model = None
        self.grid_search = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def integrate_pca(self, PCAModel, integration_type=None):
        if integration_type == 'pca_only':
            self.X == PCAModel.X_pca
        else:
            self.X = np.hstack((self.X, PCAModel.X_pca))

    def split_data(self):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def fit(self, classifier, **kwargs):
        '''
        fit data to specified sklearn model

        args:
        classifier (object): sklearn model object
        **kwargs (keyword arguments): key word arguments for specific sklearn model
        '''

        self.classifier_model = classifier(**kwargs)
        self.classifier_model.fit(self.X_train, self.y_train)

    def pred_score(self):
        '''
        Get accuracy of prediction on test data
        '''
        accuracy = self.classifier_model.score(self.X_test, self.y_test)
        print('Accuracy: {}'.format(accuracy))
        return accuracy


    def grid_search(self, classifier, params, set_classifier=False):
        ''' gets a rough idea where the best parameters lie

        args:

        classifier (sklearn object): selected classifier sklearn object to search over
        params (dictionary): dictionary of hyperparameters. keys = parameter name, values = paramater values.
        set_classifier (boolean): if True this method will set the best estimator of the grid search as the classifier that can then be evaluated
        '''

        self.grid_search = GridSearchCV(classifier, params)
        print("Starting grid search")
        self.grid_search.fit(self.X_train, self.y_train)
        grid_params = self.grid_search.best_params_
        grid_score = self.grid_search.best_score_
        print("Coarse search best parameters:")
        for param, val in grid_params.items():
            print("{0:<20s} | {1}".format(param, val))
        print("Coarse search best score: {0:0.3f}".format(grid_score))
        if set_classifier:
            self.classifier = self.grid_search.best_estimator_


    def plot_roc_curve(self, img_type='original'):
        '''
        Plot ROC courve for trained classifier model

        args:
        img_type (string): specify if the image type is padded or original.
        '''

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

        probs = self.classifier.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probs[:,1])
        auc_score = round(roc_auc_score(self.y_test, probs[:,1]), 4)
        ax.plot(fpr, tpr, label= f'{self.classifier.__class__.__name__} = {auc_score} AUC')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance')
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.set_title("ROC plot of Mammogram mass classification with {} images".format(img_type), fontsize=18)
        ax.legend()
        plt.show()
        accuracy = self.classifier.score(self.X_test, self.y_test)
        print('Accuracy: {}'.format(accuracy))

    def save_ml_model(self, path):
        '''
        Save trained classifier model

        args:
        path (string): path including name of model to save
        '''
        with open('{}'.format(path), 'wb') as f:
            Pickle.dump(self.classifier, f)

def images_to_matrices(self, benign_dir, malignant_dir, resize_img=None):
    '''this is a rough function based off the total images for the benign class'''
    dirs = [benign_dir, malignant_dir]
    array_ls = []
    label_ls = []
    for source_dir in dirs:
        for file in glob.iglob(os.path.join(source_dir, "*.png")):
            dirname = file.split(os.path.sep)
            if dirname[-2][0] == 'B':
                label_ls.append(0)
            else:
                label_ls.append(1)
            img = misc.imread(file)
            if resize_img:
                resized_img = resize(img, (resize_img, resize_img), preserve_range=True)
                flat_img = resized_img.flatten()
            else:
                flat_img = img.flatten()
            array_ls.append(flat_img)

        X = array_ls[0]
        for array in array_ls[1:]:
            X = np.vstack((X, array))
        y = np.array(label_ls)
    return X, y


if __name__=='__main__':


    #randomf forest and pca for ORIGINAL image feature extractions
    extracted_orig_train_X = np.load('numpy_arrays/orig_feature_ext_inception_train_X.npy')
    extracted_orig_test_X = np.load('numpy_arrays/orig_feature_ext_inception_test_X.npy')
    extracted_orig_hold_X = np.load('numpy_arrays/orig_feature_ext_inception_hold_X.npy')

    extracted_orig_train_y = np.load('numpy_arrays/orig_feature_ext_inception_train_y.npy')
    extracted_orig_test_y = np.load('numpy_arrays/orig_feature_ext_inception_test_y.npy')
    extracted_orig_hold_y = np.load('numpy_arrays/orig_feature_ext_inception_hold_y.npy')

    extracted_orig_tt_X = np.vstack((extracted_orig_train_X, extracted_orig_test_X))
    extracted_orig_tt_y = np.hstack((extracted_orig_train_y, extracted_orig_test_y))


    # classifier_model = RandomForestClassifier(n_estimators=1000)
    # classifier_model.fit(extracted_orig_train_X, extracted_orig_train_y)
    # accuracy = classifier_model.score(extracted_orig_hold_X, extracted_orig_hold_y)
    # print('Accuracy: {}'.format(accuracy

    classifier_model = RandomForestClassifier(n_estimators=5000)
    classifier_model.fit(extracted_orig_tt_X, extracted_orig_tt_y)
    accuracy = classifier_model.score(extracted_orig_hold_X, extracted_orig_hold_y)
    print('Accuracy: {}'.format(accuracy))
