# MNIST-digit-classification-603-final

Over the past 10 years, there has been a rapid growth in the field of artificial intelligence and machine learning which has seen great progress in image and digit classification, that includes proposal of various models and model enhancing techniques to improve predictive performance.

In this paper, the motivation is to compare six mainstream statistical models. These models are Multinomial Logistic Regression, Convolutional Neural Network, Gradient Boosting , Support Vector Machine, Multi-class Linear Discriminant Analysis and Random Forest. 

For the purpose of this project, MNIST dataset has been provided with slight modification. In order to reduce the computation time and complexity, a random sample of 10,000 training images and 5,000 testing images are taken, without replacement, from the original data set. For each image, there are 28*28 pixels that contains a value from 0-255. Moreover, the data was already prepossessed and split into training and testing set.
Three files were used for this analysis which can be found in this repo.

After the best modeling efforts, the goal is to make prediction on the testing dataset.Two separate text files containing the prediction for datasets-- mnist_test_counts-1 and mnist_test_counts_new will be provided along with the report. In addition, the mis-sclassification rate of the models on mnist_test_counts-1 will be compared and the method used will be discussed.

More details can be found in the final report in this repo.
