# Introduction
With the advent of an increasing elderly population and breast cancer awareness campaigns, the demand for routine
mammograms and early screening has increased dramatically. However, a 2009 RadiologyToday article discusses the the looming shortage of trained mammographers and a workforce struggling to keep up. Additionally, according to UCHealth, breast cancer is accurately diagnosed through mammography at a rate of 78%. These challenges raise new questions for what can be done to aid the healthcare industry to provide the quality care people need.

# Objectives
Typically, we see that masses with and irregular shape have a higher likelihood of being malignant. This can be supported by the plot below, this the the mass shape classification given by a mammographer prior to the release of this data.

![](https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/path_common_mass%20copy.png)

This plot suggests that inspecting mass shape could be a good method of discerning between malignant and benign masses.

My goal is to apply both machine and deep learning, specifically convolutional neural networks, methods on cropped mammogram images to create a model that can guide physicians and radiologists to high priority cases. These images contain masses that will be classified as benign or malignant. The priority images can then be reviewed by hospital staff.

# The Data
The Data comes from the Cancer Imaging Archive’s CBIS-DDSM. These are mammograms compiled from 2620 studies with 1600 cropped images. Each image contains a mass that has been classified as benign or malignant by a trained medical professional. The cropped images come from two different image views: Craniocaudal (side) and Mediolateral Oblique (below at a 45° angle). Each image is in the standard DICOM format, however, because these come from different studies and are cropped, each image has a unique shape.

![](https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/data_variability.png)

These images are just a few examples of variability in the data. The shape, resolution, noise, etc. of images clearly presents some challenges in the attempt to classify based off of mass shape. The convolutional neural net model input's will require square images, so the images will have to be reshaped prior to modeling. Since resizing a rectangular image into a square image will distort the image, I will need to preserve the mass shapes. To accomplish this I used openCV to preserve the aspect ratio of the image and zero pad the images to meet the square requirements of the networks.
