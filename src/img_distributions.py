import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt

def plot_two_distributions(first_set, second_set, first_label, second_label, title, x_title, y_title, save_name=None):

    '''
    Plot full image pixel distributions overlay all malignant images and all benign image.

    args:
    first_set (numpy array): flattened array of all pixel values (here I used Benign)
    second_set (numpy array): flattened array of all pixel values (here I used malignant)
    first_label (string): class of the first_set (here I used benign)
    second_label (string): class of the second_Set (here I used malignant)
    title (string): title of graph
    x_title (string): title of x axis
    y_title (string): title of y axis
    save_name (string): path to save figure if save_name is specified. Else it will just show the plot.
    '''

    plt.hist(first_set, bins=255, alpha=0.4, color='g', label=first_label)
    plt.axvline(x=np.mean(first_set), color='c', label='{} mean'.format(first_label))
    plt.hist(second_set, bins=255, alpha=0.4, color='b', label=second_label)
    plt.axvline(x=np.mean(second_set), color='m', label='{} mean'.format(second_label))
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.text(20, 300000, 'Benign μ: {}, σ: {}\nMalignant μ: {}, σ: {}'.format(np.round(np.mean(first_set), 2), np.round(np.std(first_set),2),\
                                                                        np.round(np.mean(second_set), 2), np.round(np.std(second_set),2)))
    if save_name:
        plt.savefig('/Users/christopherlawton/galvanize/module_3/cap_3_dir/visuals/{}.png'.format(save_name), dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()


if __name__=='__main__':

    orig_X = np.load('./numpy_arrays/original_combined_250_X.npy')
    orig_y = np.load('./numpy_arrays/original_combined_250_y.npy')
    idx_benign = np.argwhere(orig_y == 0)
    idx_malignant = np.argwhere(orig_y == 1)
    orig_benign = orig_X[:len(idx_benign)]
    final_benign = orig_X[:len(idx_malignant)]
    orig_malignant = orig_X[len(idx_benign):]

    pad_X = np.load('./numpy_arrays/padded_combined_250_X.npy')
    pad_y = np.load('./numpy_arrays/padded_combined_250_y.npy')
    idx_pad_benign = np.argwhere(orig_y == 0)
    idx_pad_malignant = np.argwhere(orig_y == 1)
    pad_benign = orig_X[:len(idx_pad_benign)]
    final_pad_benign = orig_X[:len(idx_pad_malignant)]
    pad_malignant = orig_X[len(idx_pad_benign):]

    plot_two_distributions(final_benign.flatten(), orig_malignant.flatten(), 'Benign', 'Malignant', 'Distribution of pixel values for all original images', 'Pixel values', 'Count', save_name='full_original_image_dist')
    # plot_two_distributions(final_pad_benign.flatten(), pad_malignant.flatten(), 'Benign', 'Malignant', 'Distribution of pixel values for all padded images', 'Pixel values', 'Count')
