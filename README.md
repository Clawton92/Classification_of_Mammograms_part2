# Introduction
With the advent of an increasing elderly population and breast cancer awareness campaigns, the demand for routine
mammograms and early screening has increased dramatically. However, a 2009 RadiologyToday article discusses the the looming shortage of trained mammographers and a workforce struggling to keep up. Additionally, according to UCHealth, breast cancer is accurately diagnosed through mammography at a rate of 78%. These challenges raise new questions for what can be done to aid the healthcare industry to provide the quality care people need.

# Objectives
Typically, we see that masses with an irregular shape have a higher likelihood of being malignant. This can be supported by the plot below, this the the mass shape classification given by a mammographer prior to the release of this data.

![](https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/path_common_mass%20copy.png)

This plot suggests that inspecting mass shape could be a good method of discerning between malignant and benign masses.

My goal is to apply both machine and deep learning, specifically convolutional neural networks, methods on cropped mammogram images to create a model that can guide physicians and radiologists to high priority cases. These images contain masses that will be classified as benign or malignant. The priority images can then be reviewed by hospital staff.

# The Data
The Data comes from the Cancer Imaging Archive’s CBIS-DDSM. These are mammograms compiled from 2620 studies with 1600 cropped images. Each image contains a mass that has been classified as benign or malignant by a trained medical professional. The cropped images come from two different image views: Craniocaudal (side) and Mediolateral Oblique (below at a 45° angle). Each image is in the standard DICOM format, however, because these come from different studies and are cropped, each image has a unique shape.

![](https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/data_variability.png)

These images are just a few examples of variability in the data. The shape, resolution, noise, etc. of images clearly present some challenges in the attempt to classify based off of mass shape. The convolutional neural net model input's will require square images, so the images will have to be reshaped prior to modeling. Since resizing a rectangular image into a square image will distort the image, I will need to preserve the mass shapes. To accomplish this I used openCV to preserve the aspect ratio of and zero pad the images to meet the square requirements of the networks.

![](https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/pic_aspect_ratio.png)

The image on the right is an original, rectangular image. The Center image is the non-zero padded, resized image that distorts the shape of the original image. The right image is the zero padded image that preserves the aspect ratio. From here I decided to model with both the original images and the padded images to explore if shape does aid classification.

#Initial approach
As stated above, my approach will be to use a convolutional neural network (CNN) in order to hopefully pick up on the shapes of masses.

![](https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/conv_nn_picture.png)
[source](https://www.mdpi.com/2078-2489/7/4/61)

Here is a basic diagram depicting the structure and process of a simple CNN. The input layer takes an image, this is typically followed by several convolutional layers, layers in which a filter scans over the image to pick up on features (curves, edges, lines, etc.) in the image, this is done by performing matrix multiplication with weights in the filter and the pixel values in the image. The outputs can then be pooled in serval different ways. The following outputs are called feature maps, essentially they are "smaller images" that carry forward specific features deeper into the network. This process can be repeated iteratively on these feature maps to pick up on more complex features the deeper the network goes. The output of the network will then give a probability of an image belonging to each target class (if using the softmax activation function).  

Because CNNs are great at learning shapes and other features in images, I decided to start with a simple CNN built from scratch. The architecture of the simple CNN is based on a 2016 paper. (levy, D., Jain, A. 'Breast Mass Classification from Mammograms using Deep Convolutional Neural Networks.' 2016) [source](https://arxiv.org/pdf/1612.00542.pdf)

Results after training the simple cnn after numerous iterations of hyperparameter tuning.

| |Validation Accuracy|Validation Loss|
|--|--|--|
|Original Images|0.53|0.67|
|Padded Images|0.55|0.63|

This simple cnn was trained for 30 epochs yet did not achieve great results. The accuracy and loss remained consistent through each epoch.

#Transfer Learning
Since the simple CNN did not yield ideal results, I decided to move onto another method, Transfer learning. Transfer learning is utilizing a pre-trained network on a large amalgam of images. Because these networks are pre-trained, they have likely already learned certain features in images that can be similar to shapes in mammogram masses (curves, lines, ridges, etc.). I can then utilize two methods of transfer learning, feature extraction or fine tuning. Feature extraction allows me extract the last layer of feature maps and flatten them into one dimensional arrays. I can then send these arrays into another classification method, such as a random forest classifier. Fine turning is a method in which the final layer of the pre-trained network is removed and the output layer for my classes (bianry) is stitched to the end of the network for classification. I decided to use the network InceptionV3. This is version 3 of GoogLeNet, which was used in the above mentioned paper.  

####Feature extraction
Using feature extraction, I compiled all images and labels into new train and testing arrays and then trained a Random forest classifier on these arrays. The results follow:

|  |Original Images|Padded Images|
|--|--|--|
|Accuracy|0.70|0.68|

![](https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/feature_extraction_original_images_ROC.png)

The final model showed an accuracy of 70% with an AUC of ~0.79. This was a drastic improvement from the simple CNN.

####Fine tuning
