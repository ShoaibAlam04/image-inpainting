# Image Inpainting Project
# Overview
This project involves building a convolutional neural network (CNN) for image classification using TensorFlow and Keras. The goal is to classify images of strawberries as either "Strawberry_fresh" or "Strawberry_scrotch." The dataset used for training consists of images stored in the specified directory.

# Requirements
Python (>=3.6)
TensorFlow (>=2.0)
OpenCV
Matplotlib
Dataset
# The dataset is organized into two categories:

Strawberry_fresh: Images of fresh strawberries.
Strawberry_scrotch: Images of strawberries with scrotches or imperfections.
Ensure that your dataset is structured similarly, and adjust the DIRECTORY and CATAGORIES variables in the script accordingly.

# Data Preprocessing
The script loads images from the specified directory, resizes them to 100x100 pixels, and shuffles the data to ensure a balanced distribution between categories. The images are then normalized by dividing pixel values by 255.

# Model Architecture
The CNN model is constructed with two convolutional layers, each followed by max-pooling layers. The flattened output is connected to a dense layer with softmax activation for classification into the two categories.
