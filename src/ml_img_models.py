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


class ImgMLClassifier():

    '''
    Plan:
        1) read in both original images and padded images
            - while reading in original images, resize them to be the same as the padded
        2) PCA on both with 2 components
        3) ML models on original X_mat, y_mat, pca_X_mat, concatenated X_mat and pca_X_mat
        4) Grid search randome forest and boosting models
        '''
    def __init__(self, X_arr=None, y_arr=None, load=None, arr_path=None, img_path=None, resize_img=None):
        '''
        note: this class only allows loading from a saved numpy array or saving a directory
              of images to a numpy array which will then be loaded

        args:
            load (boolean): True -> load previously saved arrays
                             False -> iterate through img dirs, save generated arrays and load those save arrays

            arr_path (tuple or list of array paths): index_0 == path to X array, index_1 == path to y array
                                            note: in the case of load=False arr_path will serve as path to save and load

            img_path (tuple or list of img paths): index_0 == path to BENIGN images, imdex_1 == path to MALIGNANT images

            resize (int): if specified, resizes images to sqaure images with dimensions (int, int)
        '''
        self.X = X_arr
        self.y = y_arr
        if load == True:
            self.X = np.load(arr_path[0])
            self.y = np.load(arr_path[1])
        elif load == False:
            X, y = self.images_to_matrices(img_path[0], img_path[1], resize_img) #(benign_path, malignant_path, if resize is needed)
            np.save(arr_path[0], X)
            np.save(arr_path[1], y)
            self.X = np.load(arr_path[0])
            self.y = np.load(arr_path[1])
        else:
            print('Please instantiate with input arrays!')

        self.pca_model = None
        self.X_pca = None
        self.classifier_model = None
        self.coarse_search = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None



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

    def make_pca_model(self, n_components=2):
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        self.pca_model = decomposition.PCA(n_components=n_components)
        self.X_pca = self.pca_model.fit_transform(X_scaled)
        return self.X_pca, self.pca_model

    def make_and_fit_classifier(self, classifier, x_vals=None, **kwargs):
        '''
        Make specified sklearn model
        args:
        classifier (object): sklearn model object
        X_train (2d numpy array): X_train matrix from train test split
        y_train (1d numpy array): y_train matrix from train test split
        **kwargs (keyword arguments): key word arguments for specific sklearn model
        '''
        if x_vals == 'X_full':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        elif x_vals == 'X_pca':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_pca, self.y)
        elif x_vals == 'stacked':
            stacked = np.hstack((self.X, self.X_pca))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(stacked, self.y)
        else:
            pass

        self.classifier_model = classifier(**kwargs)
        self.classifier_model.fit(self.X_train, self.y_train)
        return self.classifier_model

    def pred_score(self):
        accuracy = self.classifier_model.score(self.X_test, self.y_test)
        print('Accuracy: {}'.format(accuracy))
        return accuracy


    def scree_plot(self, ax, n_components_to_plot=10, title=None):
         """Make a scree plot showing the variance explained (i.e. variance
         of the projections) for the principal components in a fit sklearn
         PCA object.

         Parameters
         ----------
         ax: matplotlib.axis object
           The axis to make the scree plot on.

         pca: sklearn.decomposition.PCA object.
           A fit PCA object.

         n_components_to_plot: int
           The number of principal components to display in the scree plot.

         title: str
           A title for the scree plot.
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
         if title is not None:
             ax.set_title(title, fontsize=16)

    def plot_2d(self, ax, title=None):
         """Plot an embedding of the mnist dataset onto a plane.

         Parameters
         ----------
         ax: matplotlib.axis object
           The axis to make the scree plot on.

         X: numpy.array, shape (n, 2)
           A two dimensional array containing the coordinates of the embedding.

         y: numpy.array
           The labels of the datapoints.  Should be digits.

         title: str
           A title for the plot.
         """
         if self.X_pca.shape[1] == 2:
             X_vals = self.X_pca
         else:
             X_vals, pca_model = self.make_pca_model(n_components=2)

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

         if title is not None:
             ax.set_title(title, fontsize=16)

        def visual(self, title=None):
            if title:
                plt.savefig(title)
            else:
                plt.show

    def grid_search(self, classifier, params):
        ''' gets a rough idea where the best parameters lie '''
        # note that the number of estimators is set at 100 while the learning rate varies

        self.coarse_search = GridSearchCV(classifier, params)
        print("\n4) Part 2 grid search" )
        print("Starting grid search - coarse (will take several minutes)")
        self.coarse_search.fit(self.X, self.y)
        coarse_params = self.coarse_search.best_params_
        coarse_score = self.coarse_search.best_score_
        print("Coarse search best parameters:")
        for param, val in coarse_params.items():
            print("{0:<20s} | {1}".format(param, val))
        print("Coarse search best score: {0:0.3f}".format(coarse_score))

    # def plot_roc(fitted_model, X, y, ax):
	# probs = fitted_model.predict_proba(X)
	# fpr, tpr, thresholds = roc_curve(y, probs[:,1])
	# auc_score = round(roc_auc_score(y,probs[:,1]), 4)
	# ax.plot(fpr, tpr, label= f'{fitted_model.__class__.__name__} = {auc_score} AUC')

    def plot_roc_curve(self):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

        # plot_roc(model, X_test, y_test, ax)

        probs = self.classifier_model.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probs[:,1])
        auc_score = round(roc_auc_score(self.y_test,probs[:,1]), 4)
        ax.plot(fpr, tpr, label= f'{self.classifier_model.__class__.__name__} = {auc_score} AUC')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance')
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.set_title("ROC plot of Mammogram mass classification", fontsize=18)
        ax.legend()
        plt.show()
        accuracy = self.classifier_model.score(self.X_test, self.y_test)
        print('Accuracy: {}'.format(accuracy))

def plot_roc_curve(model, X_test, y_test, img_type):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    # plot_roc(model, X_test, y_test, ax)

    probs = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    auc_score = round(roc_auc_score(y_test, probs[:,1]), 4)
    ax.plot(fpr, tpr, label= f'{model.__class__.__name__} = {auc_score} AUC')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance')
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_title("ROC plot of Mammogram mass classification with {} images".format(img_type), fontsize=18)
    ax.legend()
    plt.show()
    accuracy = model.score(X_test, y_test)
    print('Accuracy: {}'.format(accuracy))

def save_ml_model(model, path):
    with open('{}'.format(path), 'wb') as f:
        Pickle.dump(rf, f)

if __name__=='__main__':

    '''random forest params
        n_estimators, max_depth,
    '''
    # original_combined_benign = '/Users/christopherlawton/crop_img_pool/original/mass/combined/BENIGN'
    # original_combined_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/combined/MALIGNANT'
    # original_combined_250_X = 'numpy_arrays/original_combined_250_X.npy'
    # original_combined_250_y = 'numpy_arrays/original_combined_250_y.npy'
    #
    # padded_combined_benign = '/Users/christopherlawton/crop_img_pool/grayscale/mass/combined/BENIGN'
    # padded_combined_malignant = '/Users/christopherlawton/crop_img_pool/grayscale/mass/combined/MALIGNANT'
    # padded_combined_250_X = 'numpy_arrays/padded_combined_250_X.npy'
    # padded_combined_250_y = 'numpy_arrays/padded_combined_250_y.npy'
    #
    # original_combined = ImgMLClassifier(load=False, arr_path=(original_combined_250_X, original_combined_250_y),img_path=(original_combined_benign, original_combined_malignant), resize_img=250)
    # padded_combined = ImgMLClassifier(load=False, arr_path=(padded_combined_250_X, padded_combined_250_y), img_path=(padded_combined_benign, padded_combined_malignant), resize_img=250)

    # np.random.seed(2349723)
    # original_combined = ImgMLClassifier(load=True, arr_path=(original_combined_400_X, original_combined_400_y))
    # fig, ax = plt.subplots()
    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9,5))
    # original_combined.make_pca_model(n_components=2)
    # original_combined.scree_plot(ax0, n_components_to_plot=2)
    # original_combined.plot_2d(ax)
    # full_forest_model = original_combined.make_and_fit_classifier(RandomForestClassifier, x_vals='X_full', n_estimators=100)
    # original_combined.pred_score()
    # pca_forest_model = original_combined.make_and_fit_classifier(RandomForestClassifier, x_vals='X_pca', n_estimators=100)
    # original_combined.pred_score()
    # plt.show()


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


    # fig, ax = plt.subplots()
    # extracted_orig_X_full = np.vstack((extracted_orig_train_X, extracted_orig_test_X, extracted_orig_hold_X))
    # extracted_orig_y_full = np.hstack((extracted_orig_train_y, extracted_orig_test_y, extracted_orig_hold_y))
    #
    # extracted_orig_model = ImgMLClassifier(X_arr=extracted_orig_X_full, y_arr=extracted_orig_y_full)
    # extracted_forest_params = {'n_estimators': [1000]}
    # extracted_model.grid_search(RandomForestClassifier(), extracted_forest_params)
    # extracted_model.make_pca_model(n_components=2)
    # extracted_model.scree_plot(ax, n_components_to_plot=2)
    # extracted_model.plot_2d(ax)
    # plt.show()
    # extracted_orig_forest_model = extracted_orig_model.make_and_fit_classifier(RandomForestClassifier, x_vals='X_full', n_estimators=1000, n_jobs=2)
    # extracted_orig_model.pred_score()


    #randomf forest and pca for PADDED image feature extractions
    # extracted_pad_train_X = np.load('numpy_arrays/pad_feature_ext_inception_train_X.npy')
    # extracted_pad_test_X = np.load('numpy_arrays/pad_feature_ext_inception_test_X.npy')
    # extracted_pad_hold_X = np.load('numpy_arrays/pad_feature_ext_inception_hold_X.npy')
    #
    # extracted_pad_train_y = np.load('numpy_arrays/pad_feature_ext_inception_train_y.npy')
    # extracted_pad_test_y = np.load('numpy_arrays/pad_feature_ext_inception_test_y.npy')
    # extracted_pad_hold_y = np.load('numpy_arrays/pad_feature_ext_inception_hold_y.npy')
    #
    # fig, ax = plt.subplots()
    # extracted_pad_X_full = np.vstack((extracted_pad_train_X, extracted_pad_test_X, extracted_pad_hold_X))
    # extracted_pad_y_full = np.hstack((extracted_pad_train_y, extracted_pad_test_y, extracted_pad_hold_y))

    # fig, ax = plt.subplots()
    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9,5))
    # extracted_pad_model = ImgMLClassifier(X_arr=extracted_pad_X_full, y_arr=extracted_pad_y_full)
    # extracted_forest_params = {'n_estimators': [1000]}
    # extracted_model.grid_search(RandomForestClassifier(), extracted_forest_params)
    # extracted_pad_model.make_pca_model(n_components=0.95)
    # extracted_pad_model.scree_plot(ax0, n_components_to_plot=10)
    # extracted_pad_model.plot_2d(ax1)
    # plt.show()

    # extracted_pad_forest_model = extracted_pad_model.make_and_fit_classifier(RandomForestClassifier, x_vals='stacked', n_estimators=1000)                                                                            # n_jobs=4, max_depth=20000, min_samples_split=3)
    # extracted_pad_model.pred_score()

    '''feature extraction with ORIGINAL images on non trained inception'''
    # orig_train_v3_X = np.load('./numpy_arrays/orig_fev3_train_X.npy')
    # orig_train_v3_y = np.load('./numpy_arrays/orig_fev3_train_y.npy')
    #
    # orig_test_v3_X = np.load('./numpy_arrays/orig_fev3_test_X.npy')
    # orig_test_v3_y = np.load('./numpy_arrays/orig_fev3_test_y.npy')
    #
    # orig_hold_v3_X = np.load('./numpy_arrays/pad_fev3_hold_X.npy')
    # orig_hold_v3_y = np.load('./numpy_arrays/pad_fev3_hold_y.npy')
    #
    # orig_v3_stack_x = np.vstack((orig_train_v3_X, orig_test_v3_X))
    # orig_v3_stack_y = np.hstack((orig_train_v3_y, orig_test_v3_y))
    #
    # orig_classifier_model = RandomForestClassifier(n_estimators=1000)
    # orig_classifier_model.fit(orig_v3_stack_x, orig_v3_stack_y)
    # plot_roc_curve(orig_classifier_model, orig_hold_v3_X, orig_hold_v3_y, 'original')
    # accuracy = classifier_model.score(orig_hold_v3_X, orig_hold_v3_y)
    # print('Accuracy: {}'.format(accuracy))

    '''feature extraction with padded images on non trained inception'''
    # pad_train_v3_X = np.load('./numpy_arrays/pad_fev3_train_X.npy')
    # pad_train_v3_y = np.load('./numpy_arrays/pad_fev3_train_y.npy')
    #
    # pad_test_v3_X = np.load('./numpy_arrays/pad_fev3_test_X.npy')
    # pad_test_v3_y = np.load('./numpy_arrays/pad_fev3_test_y.npy')
    #
    # pad_hold_v3_X = np.load('./numpy_arrays/pad_fev3_hold_X.npy')
    # pad_hold_v3_y = np.load('./numpy_arrays/pad_fev3_hold_y.npy')
    #
    # pad_v3_stack_x = np.vstack((pad_train_v3_X, pad_test_v3_X))
    # pad_v3_stack_y = np.hstack((pad_train_v3_y, pad_test_v3_y))
    #
    # pad_classifier_model = RandomForestClassifier(n_estimators=1000)
    # pad_classifier_model.fit(pad_v3_stack_x, pad_v3_stack_y)
    # plot_roc_curve(pad_classifier_model, pad_hold_v3_X, pad_hold_v3_y, 'padded')
    # accuracy = classifier_model.score(pad_hold_v3_X, pad_hold_v3_y)
    # print('Accuracy: {}'.format(accuracy))
