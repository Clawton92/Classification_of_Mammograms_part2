import glob
import shutil
import os
import numpy as np
from sklearn.model_selection import train_test_split

def create_train_test_holdout_rand(source_dir, train_dir, test_dir, hold_dir):

    '''
    Create train, test, and hold out directories given a pooled image directory.
        Note: this gives a split train: 0.7, test: 0.2, hold: 0.1

    args:
    source_dir (string): source directory for images
    train_dir (string): destination directory for train set
    test_dir (string): destination directory for test set
    hold_dir (string): destination directory for hold set
    '''

    count = 0
    list_dir = [file for file in glob.iglob(os.path.join(source_dir, "*.png"))]
    train, test_and_holdout = train_test_split(list_dir, train_size=0.7)
    test, hold = train_test_split(test_and_holdout, train_size=(2/3))
    for path in train:
        shutil.copy(path, train_dir)
    for path in test:
        shutil.copy(path, test_dir)
    for path in hold:
        shutil.copy(path, hold_dir)
    return train, test, hold
    count +=1
    print(count)

if __name__=='__main__':

    source dirs
    source paths original grayscale
    orig_CC_benign = '/Users/christopherlawton/crop_img_pool/original/mass/CC/BENIGN'
    orig_CC_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/CC/MALIGNANT'
    orig_MLO_benign = '/Users/christopherlawton/crop_img_pool/original/mass/MLO/BENIGN'
    orig_MLO_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/MLO/MALIGNANT'
    orig_combined_benign = '/Users/christopherlawton/crop_img_pool/original/mass/combined/BENIGN'
    orig_combined_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/combined/MALIGNANT'

    '''original destination dirs'''
    #train CC
    orig_CC_train_benign = '/Users/christopherlawton/cropped_train_test_split/original/CC/train/BENIGN'
    orig_CC_train_malignant = '/Users/christopherlawton/cropped_train_test_split/original/CC/train/MALIGNANT'
    #test CC
    orig_CC_test_benign = '/Users/christopherlawton/cropped_train_test_split/original/CC/test/BENIGN'
    orig_CC_test_malignant = '/Users/christopherlawton/cropped_train_test_split/original/CC/test/MALIGNANT'
    #hold CC
    orig_CC_hold_benign = '/Users/christopherlawton/cropped_train_test_split/original/CC/hold/BENIGN'
    orig_CC_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/original/CC/hold/MALIGNANT'
    #destination CC
    create_train_test_holdout_rand(orig_CC_benign, orig_CC_train_benign, orig_CC_test_benign, orig_CC_hold_benign)
    create_train_test_holdout_rand(orig_CC_malignant, orig_CC_train_malignant, orig_CC_test_malignant, orig_CC_hold_malignant)

    #train MLO
    orig_MLO_train_benign = '/Users/christopherlawton/cropped_train_test_split/original/MLO/train/BENIGN'
    orig_MLO_train_malignant = '/Users/christopherlawton/cropped_train_test_split/original/MLO/train/MALIGNANT'
    #test MLO
    orig_MLO_test_benign = '/Users/christopherlawton/cropped_train_test_split/original/MLO/test/BENIGN'
    orig_MLO_test_malignant = '/Users/christopherlawton/cropped_train_test_split/original/MLO/test/MALIGNANT'
    #hold MLO
    orig_MLO_hold_benign = '/Users/christopherlawton/cropped_train_test_split/original/MLO/hold/BENIGN'
    orig_MLO_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/original/MLO/hold/MALIGNANT'
    #destination MLO
    create_train_test_holdout_rand(orig_MLO_benign, orig_MLO_train_benign, orig_MLO_test_benign, orig_MLO_hold_benign)
    create_train_test_holdout_rand(orig_MLO_malignant, orig_MLO_train_malignant, orig_MLO_test_malignant, orig_MLO_hold_malignant)

    #train combined
    orig_combined_train_benign = '/Users/christopherlawton/cropped_train_test_split/original/combined/train/BENIGN'
    orig_combined_train_malignant = '/Users/christopherlawton/cropped_train_test_split/original/combined/train/MALIGNANT'
    #test CC
    orig_combined_test_benign = '/Users/christopherlawton/cropped_train_test_split/original/combined/test/BENIGN'
    orig_combined_test_malignant = '/Users/christopherlawton/cropped_train_test_split/original/combined/test/MALIGNANT'
    #hold CC
    orig_combined_hold_benign = '/Users/christopherlawton/cropped_train_test_split/original/combined/hold/BENIGN'
    orig_combined_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/original/combined/hold/MALIGNANT'
    #destination CC
    create_train_test_holdout_rand(orig_combined_benign, orig_combined_train_benign, orig_combined_test_benign, orig_combined_hold_benign)
    create_train_test_holdout_rand(orig_combined_malignant, orig_combined_train_malignant, orig_combined_test_malignant, orig_combined_hold_malignant)


    '''source paths padded one channel'''
    gray_CC_benign = '/Users/christopherlawton/crop_img_pool/grayscale/mass/CC/BENIGN'
    gray_CC_malignant = '/Users/christopherlawton/crop_img_pool/grayscale/mass/CC/MALIGNANT'
    gray_MLO_benign = '/Users/christopherlawton/crop_img_pool/grayscale/mass/MLO/BENIGN'
    gray_MLO_malignant = '/Users/christopherlawton/crop_img_pool/grayscale/mass/MLO/MALIGNANT'
    gray_combined_benign = '/Users/christopherlawton/crop_img_pool/grayscale/mass/combined/BENIGN'
    gray_combined_malignant = '/Users/christopherlawton/crop_img_pool/grayscale/mass/combined/MALIGNANT'

    #train CC
    gray_CC_train_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/CC/train/BENIGN'
    gray_CC_train_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/CC/train/MALIGNANT'
    #test CCCC
    gray_CC_test_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/CC/test/BENIGN'
    gray_CC_test_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/CC/test/MALIGNANT'
    #hold CC
    gray_CC_hold_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/CC/hold/BENIGN'
    gray_CC_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/CC/hold/MALIGNANT'
    #destination CC
    create_train_test_holdout_rand(gray_CC_benign, gray_CC_train_benign, gray_CC_test_benign, gray_CC_hold_benign)
    create_train_test_holdout_rand(gray_CC_malignant, gray_CC_train_malignant, gray_CC_test_malignant, gray_CC_hold_malignant)

    #train MLO
    gray_MLO_train_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/MLO/train/BENIGN'
    gray_MLO_train_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/MLO/train/MALIGNANT'
    #test MLO
    gray_MLO_test_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/MLO/test/BENIGN'
    gray_MLO_test_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/MLO/test/MALIGNANT'
    #hold MLO
    gray_MLO_hold_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/MLO/hold/BENIGN'
    gray_MLO_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/MLO/hold/MALIGNANT'
    #destination MLO
    create_train_test_holdout_rand(gray_MLO_benign, gray_MLO_train_benign, gray_MLO_test_benign, gray_MLO_hold_benign)
    create_train_test_holdout_rand(gray_MLO_malignant, gray_MLO_train_malignant, gray_MLO_test_malignant, gray_MLO_hold_malignant)

    #train combined
    gray_combined_train_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/combined/train/BENIGN'
    gray_combined_train_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/combined/train/MALIGNANT'
    #test combined
    gray_combined_test_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/combined/test/BENIGN'
    gray_combined_test_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/combined/test/MALIGNANT'
    #hold combined
    gray_combined_hold_benign = '/Users/christopherlawton/cropped_train_test_split/grayscale/combined/hold/BENIGN'
    gray_combined_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/grayscale/combined/hold/MALIGNANT'
    #destination combined
    create_train_test_holdout_rand(gray_combined_benign, gray_combined_train_benign, gray_combined_test_benign, gray_combined_hold_benign)
    create_train_test_holdout_rand(gray_combined_malignant, gray_combined_train_malignant, gray_combined_test_malignant, gray_combined_hold_malignant)


    '''source paths padded 3 channel'''
    source paths padded 3 channel (required for transfer learning InceptionV3)
    three_CC_benign = '/Users/christopherlawton/crop_img_pool/3_channel/mass/CC/BENIGN'
    three_CC_malignant = '/Users/christopherlawton/crop_img_pool/3_channel/mass/CC/MALIGNANT'
    three_MLO_benign = '/Users/christopherlawton/crop_img_pool/3_channel/mass/MLO/BENIGN'
    three_MLO_malignant = '/Users/christopherlawton/crop_img_pool/3_channel/mass/MLO/MALIGNANT'
    three_combined_benign = '/Users/christopherlawton/crop_img_pool/3_channel/mass/combined/BENIGN'
    three_combined_malignant = '/Users/christopherlawton/crop_img_pool/3_channel/mass/combined/MALIGNANT'

    #train CC
    three_CC_train_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/CC/train/BENIGN'
    three_CC_train_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/CC/train/MALIGNANT'
    #test CC
    three_CC_test_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/CC/test/BENIGN'
    three_CC_test_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/CC/test/MALIGNANT'
    #hold CC
    three_CC_hold_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/CC/hold/BENIGN'
    three_CC_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/CC/hold/MALIGNANT'
    #destination CC
    create_train_test_holdout_rand(three_CC_benign, three_CC_train_benign, three_CC_test_benign, three_CC_hold_benign)
    create_train_test_holdout_rand(three_CC_malignant, three_CC_train_malignant, three_CC_test_malignant, three_CC_hold_malignant)

    #train MLO
    three_MLO_train_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/MLO/train/BENIGN'
    three_MLO_train_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/MLO/train/MALIGNANT'
    #test MLO
    three_MLO_test_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/MLO/test/BENIGN'
    three_MLO_test_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/MLO/test/MALIGNANT'
    #hold MLO
    three_MLO_hold_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/MLO/hold/BENIGN'
    three_MLO_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/MLO/hold/MALIGNANT'
    #destination MLO
    create_train_test_holdout_rand(three_MLO_benign, three_MLO_train_benign, three_MLO_test_benign, three_MLO_hold_benign)
    create_train_test_holdout_rand(three_MLO_malignant, three_MLO_train_malignant, three_MLO_test_malignant, three_MLO_hold_malignant)

    #train combined
    three_combined_train_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/combined/train/BENIGN'
    three_combined_train_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/combined/train/MALIGNANT'
    #test CC
    three_combined_test_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/combined/test/BENIGN'
    three_combined_test_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/combined/test/MALIGNANT'
    #hold CC
    three_combined_hold_benign = '/Users/christopherlawton/cropped_train_test_split/3_channel/combined/hold/BENIGN'
    three_combined_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/3_channel/combined/hold/MALIGNANT'
    #destination CC
    create_train_test_holdout_rand(three_combined_benign, three_combined_train_benign, three_combined_test_benign, three_combined_hold_benign)
    create_train_test_holdout_rand(three_combined_malignant, three_combined_train_malignant, three_combined_test_malignant, three_combined_hold_malignant)


    '''cource paths stacked original'''
    #cource paths stacked original (required for transfer learning InceptionV3)
    stack_CC_benign = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/CC/BENIGN'
    stack_CC_malignant = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/CC/MALIGNANT'
    stack_MLO_benign = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/MLO/BENIGN'
    stack_MLO_malignant = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/MLO/MALIGNANT'
    stack_combined_benign = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/combined/BENIGN'
    stack_combined_malignant = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/combined/MALIGNANT'

    #train CC
    stack_CC_train_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/CC/train/BENIGN'
    stack_CC_train_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/CC/train/MALIGNANT'
    #test CC
    stack_CC_test_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/CC/test/BENIGN'
    stack_CC_test_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/CC/test/MALIGNANT'
    #hold CC
    stack_CC_hold_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/CC/hold/BENIGN'
    stack_CC_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/CC/hold/MALIGNANT'
    #destination CC
    create_train_test_holdout_rand(stack_CC_benign, stack_CC_train_benign, stack_CC_test_benign, stack_CC_hold_benign)
    create_train_test_holdout_rand(stack_CC_malignant, stack_CC_train_malignant, stack_CC_test_malignant, stack_CC_hold_malignant)

    #train MLO
    stack_MLO_train_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/MLO/train/BENIGN'
    stack_MLO_train_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/MLO/train/MALIGNANT'
    #test MLO
    stack_MLO_test_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/MLO/test/BENIGN'
    stack_MLO_test_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/MLO/test/MALIGNANT'
    #hold MLO
    stack_MLO_hold_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/MLO/hold/BENIGN'
    stack_MLO_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/MLO/hold/MALIGNANT'
    #destination MLO
    create_train_test_holdout_rand(stack_MLO_benign, stack_MLO_train_benign, stack_MLO_test_benign, stack_MLO_hold_benign)
    create_train_test_holdout_rand(stack_MLO_malignant, stack_MLO_train_malignant, stack_MLO_test_malignant, stack_MLO_hold_malignant)

    #train combined
    stack_combined_train_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/combined/train/BENIGN'
    stack_combined_train_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/combined/train/MALIGNANT'
    #test CC
    stack_combined_test_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/combined/test/BENIGN'
    stack_combined_test_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/combined/test/MALIGNANT'
    #hold CC
    stack_combined_hold_benign = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/combined/hold/BENIGN'
    stack_combined_hold_malignant = '/Users/christopherlawton/cropped_train_test_split/original_3_channel/combined/hold/MALIGNANT'
    #destination CC
    create_train_test_holdout_rand(stack_combined_benign, stack_combined_train_benign, stack_combined_test_benign, stack_combined_hold_benign)
    create_train_test_holdout_rand(stack_combined_malignant, stack_combined_train_malignant, stack_combined_test_malignant, stack_combined_hold_malignant)
