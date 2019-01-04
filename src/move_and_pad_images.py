import glob
import shutil
import os
import numpy as np
import cv2

def zero_pad_img(path, desired_size=250, channel_num=3):

    '''
    Resize and zero pad image to keep aspect ratio constant

    args:
        path (string): source path of image to zero pad
        desired_size (int): desired diemnsion of padded image the sigle integer serves\
                            as the height and width value (square image)
        channel_num (int): Specify number of channels for your output image. Specifying \
                           < 3 will give a slice of the 3 channel image.
    '''

    desired_size = desired_size
    im_pth = path
    im = cv2.imread(im_pth)
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    if channel_num == 1:
        return new_im[:,:,0]
    else:
        return new_im

def stack_grayscale_to_three_channel(file):
        stacked_img = cv2.imread(file) #reading in a grayscale image in cv2 without specifying grayscale will automatically stack the image 3 times (3 channels)
        return stacked_img

def copy_and_move_imgs(source_dir, dstdir, stack=False, pad=False, desired_size=250, channel_num=3):
    count = 0
    for file in glob.iglob(os.path.join(source_dir, "*.png")):
        if pad == True:
            dirname = file.split(os.path.sep)
            final_path = dstdir + '/' + dirname[-1]
            padded_image = zero_pad_img(file, desired_size, channel_num)
            cv2.imwrite('{}'.format(final_path), padded_image)
        elif stack == True:
            dirname = file.split(os.path.sep)
            final_path = dstdir + '/' + dirname[-1]
            stacked_img = stack_grayscale_to_three_channel(file)
            cv2.imwrite('{}'.format(final_path), stacked_img)
        else:
            shutil.copy(file, dstdir)
        count += 1

if __name__ == '__main__':

    # source paths mass_train
    crop_train_CC_benign = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_train/CC/BENIGN'
    crop_train_CC_malignant = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_train/CC/MALIGNANT'
    crop_train_MLO_benign = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_train/MLO/BENIGN'
    crop_train_MLO_malignant = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_train/MLO/MALIGNANT'

    #source paths mass_test
    crop_test_CC_benign = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_test/CC/BENIGN'
    crop_test_CC_malignant = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_test/CC/MALIGNANT'
    crop_test_MLO_benign = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_test/MLO/BENIGN'
    crop_test_MLO_malignant = '/Users/christopherlawton/all_structured_scaled_images/cropped_images/mass_test/MLO/MALIGNANT'

    #destination paths original grayscale
    orig_CC_benign = '/Users/christopherlawton/crop_img_pool/original/mass/CC/BENIGN'
    orig_CC_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/CC/MALIGNANT'
    orig_MLO_benign = '/Users/christopherlawton/crop_img_pool/original/mass/MLO/BENIGN'
    orig_MLO_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/MLO/MALIGNANT'
    orig_combined_benign = '/Users/christopherlawton/crop_img_pool/original/mass/combined/BENIGN'
    orig_combined_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/combined/MALIGNANT'

    #pool all cropped CC benign
    copy_and_move_imgs(crop_train_CC_benign, orig_CC_benign, pad=False)
    copy_and_move_imgs(crop_test_CC_benign, orig_CC_benign, pad=False)
    #pool all cropped CC malignant
    copy_and_move_imgs(crop_train_CC_malignant, orig_CC_malignant, pad=False)
    copy_and_move_imgs(crop_test_CC_malignant, orig_CC_malignant, pad=False)
    #pool all cropped MLO benign
    copy_and_move_imgs(crop_train_MLO_benign, orig_MLO_benign, pad=False)
    copy_and_move_imgs(crop_test_MLO_benign, orig_MLO_benign, pad=False)
    #pool all cropped MLO malignant
    copy_and_move_imgs(crop_train_MLO_malignant, orig_MLO_malignant, pad=False)
    copy_and_move_imgs(crop_test_MLO_malignant, orig_MLO_malignant, pad=False)
    #pool all benign (both CC and MLO views)
    copy_and_move_imgs(crop_train_CC_benign, orig_combined_benign, pad=False)
    copy_and_move_imgs(crop_test_CC_benign, orig_combined_benign, pad=False)
    copy_and_move_imgs(crop_train_MLO_benign, orig_combined_benign, pad=False)
    copy_and_move_imgs(crop_test_MLO_benign, orig_combined_benign, pad=False)
    #pool all malignant (both CC and MLO views)
    copy_and_move_imgs(crop_train_MLO_malignant, orig_combined_malignant, pad=False)
    copy_and_move_imgs(crop_test_MLO_malignant, orig_combined_malignant, pad=False)
    copy_and_move_imgs(crop_train_CC_malignant, orig_combined_malignant, pad=False)
    copy_and_move_imgs(crop_test_CC_malignant, orig_combined_malignant, pad=False)


    #destination paths padded one channel (basic CNN model)
    gray_CC_benign = '/Users/christopherlawton/crop_img_pool/grayscale/mass/CC/BENIGN'
    gray_CC_malignant = '/Users/christopherlawton/crop_img_pool/grayscale/mass/CC/MALIGNANT'
    gray_MLO_benign = '/Users/christopherlawton/crop_img_pool/grayscale/mass/MLO/BENIGN'
    gray_MLO_malignant = '/Users/christopherlawton/crop_img_pool/grayscale/mass/MLO/MALIGNANT'
    gray_combined_benign = '/Users/christopherlawton/crop_img_pool/grayscale/mass/combined/BENIGN'
    gray_combined_malignant = '/Users/christopherlawton/crop_img_pool/grayscale/mass/combined/MALIGNANT'

    #pool all cropped CC benign
    copy_and_move_imgs(crop_train_CC_benign, gray_CC_benign, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_CC_benign, gray_CC_benign, pad=True, desired_size=400, channel_num=1)
    #pool all cropped CC malignant
    copy_and_move_imgs(crop_train_CC_malignant, gray_CC_malignant, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_CC_malignant, gray_CC_malignant, pad=True, desired_size=400, channel_num=1)
    #pool all cropped MLO benign
    copy_and_move_imgs(crop_train_MLO_benign, gray_MLO_benign, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_MLO_benign, gray_MLO_benign, pad=True, desired_size=400, channel_num=1)
    #pool all cropped MLO malignant
    copy_and_move_imgs(crop_train_MLO_malignant, gray_MLO_malignant, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_MLO_malignant, gray_MLO_malignant, pad=True, desired_size=400, channel_num=1)
    #pool all benign (both CC and MLO views)
    copy_and_move_imgs(crop_train_CC_benign, gray_combined_benign, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_CC_benign, gray_combined_benign, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_train_MLO_benign, gray_combined_benign, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_MLO_benign, gray_combined_benign, pad=True, desired_size=400, channel_num=1)
    #pool all malignant (both CC and MLO views)
    copy_and_move_imgs(crop_train_MLO_malignant, gray_combined_malignant, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_MLO_malignant, gray_combined_malignant, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_train_CC_malignant, gray_combined_malignant, pad=True, desired_size=400, channel_num=1)
    copy_and_move_imgs(crop_test_CC_malignant, gray_combined_malignant, pad=True, desired_size=400, channel_num=1)

    #destination paths padded 3 channel (required for transfer learning InceptionV3)
    three_CC_benign = '/Users/christopherlawton/crop_img_pool/3_channel/mass/CC/BENIGN'
    three_CC_malignant = '/Users/christopherlawton/crop_img_pool/3_channel/mass/CC/MALIGNANT'
    three_MLO_benign = '/Users/christopherlawton/crop_img_pool/3_channel/mass/MLO/BENIGN'
    three_MLO_malignant = '/Users/christopherlawton/crop_img_pool/3_channel/mass/MLO/MALIGNANT'
    three_combined_benign = '/Users/christopherlawton/crop_img_pool/3_channel/mass/combined/BENIGN'
    three_combined_malignant = '/Users/christopherlawton/crop_img_pool/3_channel/mass/combined/MALIGNANT'

    #pool all cropped CC benign
    copy_and_move_imgs(crop_train_CC_benign, three_CC_benign, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_CC_benign, three_CC_benign, pad=True, desired_size=400, channel_num=3)
    #pool all cropped CC malignant
    copy_and_move_imgs(crop_train_CC_malignant, three_CC_malignant, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_CC_malignant, three_CC_malignant, pad=True, desired_size=400, channel_num=3)
    #pool all cropped MLO benign
    copy_and_move_imgs(crop_train_MLO_benign, three_MLO_benign, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_MLO_benign, three_MLO_benign, pad=True, desired_size=400, channel_num=3)
    #pool all cropped MLO malignant
    copy_and_move_imgs(crop_train_MLO_malignant, three_MLO_malignant, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_MLO_malignant, three_MLO_malignant, pad=True, desired_size=400, channel_num=3)
    #pool all benign (both CC and MLO views)
    copy_and_move_imgs(crop_train_CC_benign, three_combined_benign, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_CC_benign, three_combined_benign, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_train_MLO_benign, three_combined_benign, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_MLO_benign, three_combined_benign, pad=True, desired_size=400, channel_num=3)
    #pool all malignant (both CC and MLO views)
    copy_and_move_imgs(crop_train_MLO_malignant, three_combined_malignant, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_MLO_malignant, three_combined_malignant, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_train_CC_malignant, three_combined_malignant, pad=True, desired_size=400, channel_num=3)
    copy_and_move_imgs(crop_test_CC_malignant, three_combined_malignant, pad=True, desired_size=400, channel_num=3)


    orig_CC_benign = '/Users/christopherlawton/crop_img_pool/original/mass/CC/BENIGN'
    orig_CC_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/CC/MALIGNANT'
    orig_MLO_benign = '/Users/christopherlawton/crop_img_pool/original/mass/MLO/BENIGN'
    orig_MLO_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/MLO/MALIGNANT'
    orig_combined_benign = '/Users/christopherlawton/crop_img_pool/original/mass/combined/BENIGN'
    orig_combined_malignant = '/Users/christopherlawton/crop_img_pool/original/mass/combined/MALIGNANT'

    #making 3 channel stack images for transfer learning (grayscale image stacked 3 times)
    #I am using the destination dirs for the single channel grayscale images as the source dirs
    #destination channel for original stacked 3 channel images
    stack_CC_benign = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/CC/BENIGN'
    stack_CC_malignant = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/CC/MALIGNANT'
    stack_MLO_benign = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/MLO/BENIGN'
    stack_MLO_malignant = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/MLO/MALIGNANT'
    stack_combined_benign = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/combined/BENIGN'
    stack_combined_malignant = '/Users/christopherlawton/crop_img_pool/original_3_channel/mass/combined/MALIGNANT'

    #stack all original CC images
    copy_and_move_imgs(orig_CC_benign, stack_CC_benign, stack=True)
    copy_and_move_imgs(orig_CC_malignant, stack_CC_malignant, stack=True)
    #stack all original MLO images
    copy_and_move_imgs(orig_MLO_benign, stack_MLO_benign, stack=True)
    copy_and_move_imgs(orig_MLO_malignant, stack_MLO_malignant, stack=True)
    #stack all original benign (both CC and MLO views)
    copy_and_move_imgs(orig_combined_benign, stack_combined_benign, stack=True)
    #stack all original malignant (both CC and MLO views)
    copy_and_move_imgs(orig_combined_malignant, stack_combined_malignant, stack=True)
