# Introduction
With the advent of an increasing elderly population and breast cancer awareness campaigns, the demand for routine
mammograms and early screening has increased dramatically. However, a 2009 RadiologyToday article discusses the the looming shortage of trained mammographers and a workforce struggling to keep up. Additionally, according to UCHealth, breast cancer is accurately diagnosed through mammography at a rate of 78%.These challenges raise new questions for what can be done to aid the healthcare industry to provide the quality care people need.

# Objectives
My goal is to apply both machine and deep learning methods on cropped mammogram images to create a model that can guide physicians and radiologists to high priority cases. These images contain masses that will be classified as benign or malignant.The priority images can then be reviewed by hospital staff.

# The Data
The Data comes from the Cancer Imaging Archive’s CBIS-DDSM.These are mammograms compiled from 2620 studies with 1600 cropped images . Each image contains a mass that has been classified as benign or malignant.The cropped images come from two different image views: Craniocaudal (side) and Mediolateral Oblique (45° angle). Each image is in the standard DICOM format, however, because these come from different studies and are cropped, each image has a unique shape.

[]! https://github.com/Clawton92/Classification_of_Mammograms_part2/blob/master/visuals/data_variability.png
