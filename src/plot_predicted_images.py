import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import stats
import pdb
from matplotlib import rc
import cv2
import cPickle

def plot_grid(imgs, titles, supertitle, save_name=None):
    '''
    Plot image grid with titles above each image and a supertitle above the entire figure space

    args:
    imgs (list): list of image arrays
    titles (list): list of titles for each image
    supertitle (string): Title to describe the entire image grid
    save_name (optional) (string): path, including name of figure, to save plot. \
                                   Figure will be saved else it will be shown.
    '''

    fig, axes = plt.subplots(1,3, figsize=(15,10))
    images_and_labels = list(zip(imgs, titles))

    axes = axes.ravel()
    for i, (image, label) in enumerate(images_and_labels):
        axes[i].imshow(image, cmap=plt.cm.gray)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].set_title(label, fontsize=20)
    plt.tight_layout()
    plt.suptitle(supertitle, fontsize=30)
    if save_name:
        plt.savefig('/Users/christopherlawton/galvanize/module_3/cap_3_dir/visuals/{}.png'.format(save_name), dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_heatmap(imgs, titles, supertitle, save_name=None):
    '''
    Plot heatmap (density of pixel values) grid of images. This function is very simiar to the plot_grid function

    args:
    imgs (list): list of image arrays
    titles (list): list of titles for each imageb
    supertitle (stirng): Title for full figure
    save_name (optional) (string): path, including name of figure, to save plot. \
                                   Figure will be saved else it will be shown.
    '''
    gray_imgs = [cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE) for img in imgs]
    heatmap_imgs = [cv2.applyColorMap(gray_img, cv2.COLORMAP_JET) for gray_img in gray_imgs]
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    images_and_labels = list(zip(heatmap_imgs, titles))

    axes = axes.ravel()
    for i, (image, label) in enumerate(images_and_labels):
        axes[i].imshow(image)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].set_title(label, fontsize=20)
    plt.suptitle(supertitle, fontsize=30)
    if save_name:
        plt.savefig('/Users/christopherlawton/galvanize/module_3/cap_3_dir/visuals/{}.png'.format(save_name), dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

def create_heatmap_imgs(root_img_path, paths):
    '''
    Create heatmap images from a root image path combined with the file name of images (paths)

    args:
    root_img_path (string): path not including file names to images in paths
    paths (list): list of image file names
    '''
    cv_imgs = [cv2.resize(cv2.imread('{}/{}'.format(root_img_path, path), 1), (250,250)) for path in paths]
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in cv_imgs]
    heatmap_imgs = [cv2.applyColorMap(gray_img, cv2.COLORMAP_HSV) for gray_img in gray_imgs]
    return heatmap_imgs

def df_to_strings(x):
    '''convert vlaues in a dataframe to strings using the apply method'''
    return str(x)

def add_title_benign(x):
    '''Add following string before value in a pandas DataFrame'''
    return 'Probability Benign: ' + x

def add_title_malignant(x):
    '''Add following string before value in a pandas DataFrame'''
    return 'Probability Malignant: ' + x

def make_grid_title(df):
    '''Apply above three functions to full dataframe and add two new columns, 'malignant_title' and 'benign_title'
    args:
    df (Pandas DataFrame)
    '''
    new = df.applymap(df_to_strings)
    new['malignant_title'] = new['prob_malignant'].apply(add_title_malignant)
    new['benign_title'] = new['prob_benign'].apply(add_title_benign)
    return new

def get_top_correct(df, root_img_path, num_imgs):
    '''
    Get the most confident correct n predictions (i.e. the probabilities of one \
    class is high and the other class is low and predicted class is correct).

    df (pandas DataFrame): DataFrame that includes prob_benign, prob_malignant, \
                           predicted class, acutall class, and the file names of the images
    root_img_path (string): path to directory where image files are located
    num_imgs (int): number of images to include in most confident predictions
    '''
    df['absolute'] = np.absolute((df['prob_benign'] - df['prob_malignant']))
    df_correct = df[df['Predictions'] == df['Actual']]
    idx = np.argsort(df_correct['absolute'].values)[::-1][:num_imgs]
    df_small = df_correct.iloc[idx][['prob_malignant', 'prob_benign', \
                                    'Filename', 'Predictions', 'Actual']]
    df_titles = make_grid_title(df_small)
    paths = df_small['Filename'].values
    prob_mal = df_titles['malignant_title'].values
    prob_ben = df_titles['benign_title'].values
    zip_titles = list(zip(prob_mal, prob_ben))
    final_titles = [elem[0] + '\n' + elem[1] for elem in zip_titles]
    imgs = [resize(misc.imread('{}/{}'.format(root_img_path, path)), (250,250)) for path in paths]
    heatmap_imgs = create_heatmap_imgs(root_img_path, paths)
    return final_titles, imgs, heatmap_imgs

def get_top_incorrect(df, root_img_path, num_imgs):
    '''
    Get the most confident incorrect n predictions (i.e. the probabilities of one \
    class is high and the other class is low and predicted class is incorrect).

    df (pandas DataFrame): DataFrame that includes prob_benign, prob_malignant, \
                           predicted class, acutall class, and the file names of the images
    root_img_path (string): path to directory where image files are located
    num_imgs (int): number of images to include in most confident predictions
    '''

    df['absolute'] = np.absolute((df['prob_benign'] - df['prob_malignant']))
    df_correct = df[df['Predictions'] != df['Actual']]
    idx = np.argsort(df_correct['absolute'].values)[::-1][:num_imgs]
    df_small = df_correct.iloc[idx][['prob_malignant', 'prob_benign', \
                                    'Filename', 'Predictions', 'Actual']]
    df_titles = make_grid_title(df_small)
    paths = df_small['Filename'].values
    prob_mal = df_titles['malignant_title'].values
    prob_ben = df_titles['benign_title'].values
    zip_titles = list(zip(prob_mal, prob_ben))
    final_titles = [elem[0] + '\n' + elem[1] for elem in zip_titles]
    imgs = [resize(misc.imread('{}/{}'.format(root_img_path, path)), (250,250)) for path in paths]
    heatmap_imgs = create_heatmap_imgs(root_img_path, paths)
    return final_titles, imgs, heatmap_imgs

def get_confused_incorrect(df, root_img_path, num_imgs):
    '''
    Get the least confident incorrect n predictions (i.e. the probabilities of both classes are close and the predicted class is incorrect).

    df (pandas DataFrame): DataFrame that includes prob_benign, prob_malignant, predicted class, acutall class, and the file names of the images
    root_img_path (string): path to directory where image files are located
    num_imgs (int): number of images to include in most confident predictions
    '''

    df['absolute'] = np.absolute((df['prob_benign'] - df['prob_malignant']))
    df_correct = df[df['Predictions'] != df['Actual']]
    idx = np.argsort(df_correct['absolute'].values)[:num_imgs]
    df_small = df_correct.iloc[idx][['prob_malignant', 'prob_benign', \
                                    'Filename', 'Predictions', 'Actual']]
    df_titles = make_grid_title(df_small)
    paths = df_small['Filename'].values
    prob_mal = df_titles['malignant_title'].values
    prob_ben = df_titles['benign_title'].values
    zip_titles = list(zip(prob_mal, prob_ben))
    final_titles = [elem[0] + '\n' + elem[1] for elem in zip_titles]
    imgs = [resize(misc.imread('{}/{}'.format(root_img_path, path)), (250,250)) for path in paths]
    heatmap_imgs = create_heatmap_imgs(root_img_path, paths)
    return final_titles, imgs, heatmap_imgs

def get_confused_correct(df, root_img_path, num_imgs):
    '''
    Get the least confident correct n predictions (i.e. the probabilities of both classes are close and the predicted class is correct).

    df (pandas DataFrame): DataFrame that includes prob_benign, prob_malignant, predicted class, acutall class, and the file names of the images
    root_img_path (string): path to directory where image files are located
    num_imgs (int): number of images to include in most confident predictions
    '''

    df['absolute'] = np.absolute((df['prob_benign'] - df['prob_malignant']))
    df_correct = df[df['Predictions'] == df['Actual']]
    idx = np.argsort(df_correct['absolute'].values)[:num_imgs]
    df_small = df_correct.iloc[idx][['prob_malignant', 'prob_benign', \
                                    'Filename', 'Predictions', 'Actual']]
    df_titles = make_grid_title(df_small)
    paths = df_small['Filename'].values
    prob_mal = df_titles['malignant_title'].values
    prob_ben = df_titles['benign_title'].values
    zip_titles = list(zip(prob_mal, prob_ben))
    final_titles = [elem[0] + '\n' + elem[1] for elem in zip_titles]
    imgs = [resize(misc.imread('{}/{}'.format(root_img_path, path)), (250,250)) for path in paths]
    heatmap_imgs = create_heatmap_imgs(root_img_path, paths)
    return final_titles, imgs, heatmap_imgs

def get_top_predictions_for_classes(df, root_img_path, target_class, top_n, correct=True, confused=False, preserve_range=False):
    '''
    Get the top predictions for the predicted image class for images given the specified combinations:
        predicted class is correct -> correct == True
        predicted class is incorrect -> correct == False
        predicted probabilities for each calss are far apart -> confused == False
        predicted probabilities for each class are close together -> confused == True

    args:
    df (pandas DataFrame): DataFrame that inclused prob_benign, prob_malignant, predicted class, actual class, and file names for each image.
    root_img_path (string): path to folder containing target images in DataFrame
    target_class (string): class that you want to get predictions for
    top_n (int): number of images you want predictions for
    correct (boolean): if True give images where predicted class == actual class, \
                       else give images where predicted class != actual class
    confused (boolean): If False give the top images where the probabilities are far apart \
                        (i.e. the model is most confident that an image is a given class), \
                        else give the top images where the probabilities are close together \
                        (i.e. the model is leat confident that and image belongs to a given class)

    '''
    df['absolute'] = np.absolute((df['prob_benign'] - df['prob_malignant']))
    if correct:
        df_filtered = df[df['Predictions'] == df['Actual']]
    else:
        df_filtered = df[df['Predictions'] != df['Actual']]
    df_correct_class = df_filtered[df_filtered['Predictions'] == target_class].reset_index()
    if confused:
        idx = np.argsort(df_correct_class['absolute'].values)[:top_n]
    else:
        idx = np.argsort(df_correct_class['absolute'].values)[::-1][:top_n]
    df_small = df_correct_class.iloc[idx][['prob_malignant', 'prob_benign', \
                                    'Filename', 'Predictions', 'Actual']]
    df_titles = make_grid_title(df_small)
    paths = df_small['Filename'].values
    prob_mal = df_titles['malignant_title'].values
    prob_ben = df_titles['benign_title'].values
    zip_titles = list(zip(prob_mal, prob_ben))
    final_titles = [elem[0] + '\n' + elem[1] for elem in zip_titles]
    if preserve_range:
        imgs = [resize(misc.imread('{}/{}'.format(root_img_path, path)), (250,250), preserve_range=True) for path in paths]
    else:
        imgs = [resize(misc.imread('{}/{}'.format(root_img_path, path)), (250,250)) for path in paths]
    heatmap_imgs = create_heatmap_imgs(root_img_path, paths)
    return final_titles, imgs, heatmap_imgs


def get_distributions(imgs):
    '''
    Return an aggregation of image pixel values into a flattened array for plotting

    args:
    imgs (list): list of image arrays
    '''
    flat_img = [image.flatten() for image in imgs]
    total_arr = flat_img[0]
    for image in flat_img[1:]:
        total_arr = np.hstack((total_arr, image))
    return total_arr

def plot_two_distributions(first_set, second_set, first_label, second_label, title, x_title, y_title, save_name=None):
    '''
    Plot two distributions on the same graph (aggregated malignant distribution and aggregated benign distribution)

    args:
    first_set (numpy array): aggregated or individual pixel array (distribution 1)
    second_set (numpy array): aggregated or individual pixel array (distribution 2)
    first_label (string): label for the first_set array
    second_label (string): label for the second_set array
    title (string): label for the full visualization
    x_title (string): label for the x axis
    y_title (string): label for the y axis
    save_name (string): path including file name to save visualization. If specified, \
                        the visualization will be saved, else it will be shown
    '''
    plt.hist(first_set, bins=255, alpha=0.4, color='g', label=first_label)
    plt.axvline(x=np.mean(first_set), color='c', label='{} mean'.format(first_label))
    plt.hist(second_set, bins=255, alpha=0.4, color='b', label=second_label)
    plt.axvline(x=np.mean(second_set), color='m', label='{} mean'.format(second_label))
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend(loc='upper center')
    plt.text(20, 15000, 'Benign μ: {}, σ: {}\nMalignant μ: {}, σ: {}'.format(np.round(np.mean(first_set), 2), np.round(np.std(first_set),2),\
                                                                        np.round(np.mean(second_set), 2), np.round(np.std(second_set),2)))
    plt.tight_layout()
    if save_name:
        plt.savefig('/Users/christopherlawton/galvanize/module_3/cap_3_dir/visuals/{}.png'.format(save_name), dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

if __name__=='__main__':

    root_csv_path = './predictions_csvs'
    hold_csv_name = 'inception_orig_FT3_Adam_aug1_lrsmall.csv'
    hold_img_path = '/Users/christopherlawton/galvanize/module_3/mammogram_data/cropped_train_test_split/original_3_channel/combined/hold'
    hold_pred_df = pd.read_csv('{}/{}'.format(root_csv_path, hold_csv_name))

    root_csv_path = './predictions_csvs'
    train_csv_name = 'inception_orig_FT3_Adam_aug1_lrsmall_TRAIN.csv'
    train_img_path = '/Users/christopherlawton/galvanize/module_3/mammogram_data/cropped_train_test_split/original_3_channel/combined/train'
    train_pred_df = pd.read_csv('{}/{}'.format(root_csv_path, train_csv_name))


    titles_correct, imgs_correct, heatmap_correct = get_top_correct(hold_pred_df, hold_img_path, 3)
    titles_incorrect, imgs_incorrect, heatap_incorrect = get_top_incorrect(hold_pred_df, hold_img_path, 3)
    titles_confused_correct, imgs_confused_correct, heatmap_confused_correct = get_confused_correct(hold_pred_df, hold_img_path, 3)
    titles_confused_incorrect, imgs_confused_incorrect, heatmap_confused_incorrect = get_confused_incorrect(hold_pred_df,hold_img_path, 3)

    plot_grid(imgs_correct, titles_correct, "Most confident correct\n\n.")
    grid_2(imgs_correct, titles_correct, "Most confident correct\n", save_name='confident_correct_gird')
    grid_2(heatmap_correct, titles_correct, "Most confident correct\n")
    plot_grid(imgs_incorrect, titles_incorrect, "Most confident incorrect")
    grid_2(imgs_incorrect, titles_incorrect, "Most confident incorrect\n", save_name='confident_incorrect_gird')
    plot_grid(imgs_confused_correct, titles_confused_correct, "Most confused correct")
    grid_2(imgs_confused_correct, titles_confused_correct, "Most unconfident correct\n", save_name='unconfident_correct_grid')
    plot_grid(imgs_confused_incorrect, titles_confused_incorrect, "Most confused incorrect")
    grid_2(imgs_confused_incorrect, titles_confused_incorrect, "Most unconfident incorrect\n", save_name='unconfident_incorrect_grid')

    # NOT CONFUSED
    hold_titles_malig, hold_imgs_malig, heat_malig = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 20, correct=True)
    hold_titles_benign, hold_imgs_benign, heat_benign = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 20, correct=True)
    hold_top_malig_array = get_distributions(hold_imgs_malig)
    hold_top_benign_array = get_distributions(hold_imgs_benign)

    plot_two_distributions(hold_top_benign_array, hold_top_malig_array, 'Benign', 'Malignant', 'Distribution of pixel values for top 20\n high confidence, correct classifications', 'Pixel values', 'Count', save_name='confident_correct_distributions')

    hold_titles_malig_incorrect, hold_imgs_malig_incorrect, malig_con_heat = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 20, correct=False)
    grid_2(malig_con_heat[:6], hold_titles_malig_incorrect[:6], "Most confident correct\n", save_name='test_malig_incorrect')

    hold_titles_benign_incorrect, hold_imgs_benign_incorrect = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 20, correct=False)
    hold_top_malig_array_incorrect = get_distributions(hold_imgs_malig_incorrect)
    hold_top_benign_array_incorrect = get_distributions(hold_imgs_benign_incorrect)

    plot_two_distributions(hold_top_benign_array_incorrect, hold_top_malig_array_incorrect, 'Benign', 'Malignant', 'Distribution of pixel values for top 20\n high confidence, incorrect classifications', 'Pixel values', 'Count', save_name='confident_incorrect_distributions')


    # CONFUSED
    con_titles_malig, con_imgs_malig = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 20, correct=True, confused=True)
    con_titles_benign, con_imgs_benign = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 20, correct=True, confused=True)
    con_malig_array = get_distributions(con_imgs_malig)
    con_benign_array = get_distributions(con_imgs_benign)

    plot_two_distributions(con_benign_array, con_malig_array, 'Benign', 'Malignant', 'Distribution of pixel values for top 20\n low confidence, correct classifications', 'Pixel values', 'Count', save_name='not_confident_correct_distributions')

    con_titles_malig_incorrect, con_imgs_malig_incorrect = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 20, correct=False, confused=True)
    con_titles_benign_incorrect, con_imgs_benign_incorrect = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 20, correct=False, confused=True)
    con_malig_array_incorrect = get_distributions(con_imgs_malig_incorrect)
    con_benign_array_incorrect = get_distributions(con_imgs_benign_incorrect)

    plot_two_distributions(con_benign_array_incorrect, con_malig_array_incorrect, 'Benign', 'Malignant', 'Distribution of pixel values for top 20\n low confidence, incorrect classifications', 'Pixel values', 'Count', save_name='not_confident_incorrect_distributions')

    #getting only top 3 images
    #Confident CORRECT
    CC_malignant_titles, CC_malignant_imgs, CC_malignant_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 3, correct=True)
    CC_benign_titles, CC_benign_imgs, CC_benign_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 3, correct=True)
    grid_2(CC_malignant_imgs, CC_malignant_titles, "Most Confident Correct Malignant\n", save_name='3img_confident_correct_malig')
    grid_2(CC_malignant_heatmap_imgs, CC_malignant_titles, "Most Confident Correct Malignant\n")
    grid_2(CC_benign_imgs, CC_benign_titles, "Most Confident Correct Benign\n", save_name='3img_confident_correct_benign')
    grid_2(CC_benign_heatmap_imgs, CC_benign_titles, "Most Confident Correct Benign\n")


    CI_malignant_titles, CI_malignant_imgs, CI_malignant_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 3, correct=False)
    CI_benign_final_titles, CI_benign_imgs, CI_benign_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 3, correct=False)
    grid_2(CI_malignant_imgs, CI_malignant_titles, "Most Confident Incorrect Malignant\n", save_name='3img_confident_incorrect_malig')
    grid_2(CI_malignant_heatmap_imgs, CI_malignant_titles, "Most Confident Incorrect Malignant\n")
    grid_2(CI_benign_imgs, CI_benign_final_titles, "Most Confident Incorrect Benign\n", save_name='3img_confident_incorrect_benign')
    grid_2(CI_benign_heatmap_imgs, CI_benign_final_titles, "Most Confident Incorrect Benign\n")


    # #Not confident CORRECT
    NCC_malignant_final_titles, NCC_malignant_imgs, NCC_malignant_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 3, correct=True, confused=True)
    NCC_benign_final_titles, NCC_benign_imgs, NCC_benign_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 3, correct=True, confused=True)
    grid_2(NCC_malignant_imgs, NCC_malignant_final_titles, "Most Unconfident Correct Malignant\n", save_name='3img_unconfident_correct_malig')
    grid_2(NCC_benign_imgs, NCC_benign_final_titles, "Most Unconfident Correct Benign\n", save_name='3img_unconfident_correct_benign')

    NCI_malignant_final_titles, NCI_malignant_imgs, NCI_malignant_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'MALIGNANT', 3, correct=True, confused=True)
    NCI_benign_final_titles, NCI_benign_imgs, NCI_benign_heatmap_imgs = get_top_predictions_for_classes(hold_pred_df, hold_img_path, 'BENIGN', 3, correct=True, confused=True)
    grid_2(NCI_malignant_imgs, NCI_malignant_final_titles, "Most Unconfident Incorrect Malignant\n", save_name='3img_unconfident_incorrect_malig')
    grid_2(NCI_benign_imgs, NCI_benign_final_titles, "Most Unconfident Incorrect Benign\n", save_name='3img_unconfident_incorrect_benign')
