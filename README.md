# Image Classification: Cats vs Dogs

This repository contains code for a machine learning project that aims to classify images as either cats or dogs. The task is to develop a model that can accurately distinguish between the two classes.

## Methodology

In machine learning, there are two main types of models: supervised and unsupervised. For this project, a supervised learning approach was adopted using the Support Vector Machine (SVM) algorithm. Unlike neural networks, SVMs are a non-neural network-based classification technique.

The classification process involved the following steps:

1. **Experimenting with SVM Models:** Initially, various SVM models were explored by tuning hyperparameters to find the optimal configuration for the classifier. This involved testing different kernel functions, regularization parameters, and other settings to achieve the best performance.

2. **Feature Extraction:** To enable the SVM model to classify images effectively, a feature extractor was employed. The VGG16 pretrained model was chosen as the feature extractor. VGG16 is a popular deep convolutional neural network architecture known for its excellent performance in image classification tasks. By utilizing the learned features from VGG16, the SVM model could make more accurate predictions.

## Repository Contents

The codelab contains the following files:

- `pretrained_weights.h5`: Pretrained weights of the VGG16 model used for feature extraction.



## Acknowledgments

- The VGG16 model was pretrained on the ImageNet dataset, and the weights used in this project were obtained from [https://github.com/fchollet/deep-learning-models](https://github.com/fchollet/deep-learning-models).
- The dataset used for training and testing the model was sourced from [kaggle].

## Conclusion

Through the combination of SVM classification and the VGG16 feature extractor, this project strives to accurately classify images as cats or dogs. By leveraging the power of machine learning, we aim to develop a robust model that can generalize well and achieve high classification accuracy.

Please refer to the documentation within the code files for more detailed instructions.

**Note:** Add any relevant information, citations, or acknowledgments specific to your project.
