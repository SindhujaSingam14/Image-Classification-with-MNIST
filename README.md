# Image-Classification-with-MNIST
This repository contains code snippets and explanations for implementing a deep learning model using PyTorch to perform image classification on the MNIST dataset. The MNIST dataset consists of grayscale images of handwritten digits (0-9) and is commonly used as a benchmark for image classification tasks.

## Setup and Dependencies
To run the code in this repository make sure to install this versions:

Python (version 3.6 or above)
PyTorch (version 1.7.1 or above)
torchvision (version 0.8.2 or above)
matplotlib (version 3.3.4 or above)
numpy (version 1.19.2 or above)
pandas (version 1.2.3 or above)
scikit-learn (version 0.24.1 or above).

### Loading and Preprocessing the Data
The MNIST dataset is loaded using the torchvision.datasets.MNIST class. The data is transformed into tensors using transforms.ToTensor(), which converts the PIL image data to PyTorch tensors. The data is split into training and testing sets.

### Model Architecture
The model architecture is implemented in the MultilayerPerceptron class. It is a multilayer perceptron with an input size of 784 (flattened image size), two hidden layers of sizes 120 and 84, and an output size of 10 (number of classes).

### Training the Model
The model is trained using the training dataset for the specified number of epochs. The training loop iterates over the training batches, applies the model to the batch, computes the loss using the CrossEntropyLoss criterion, updates the model's parameters using the Adam optimizer, and tracks the training accuracy.

### Evaluating the Model
After training, the model is evaluated on the testing dataset. The testing loop iterates over the testing batches, applies the trained model to the batch, and calculates the number of correct predictions. The testing accuracy is computed based on the number of correct predictions.

### Monitoring Training Progress
The training and validation losses, as well as the training and validation accuracies, are plotted at the end of each epoch using matplotlib.pyplot. The plots provide insights into the model's performance and the progress of training.

### Results
The final test accuracy achieved by the trained model is displayed. This metric indicates the model's performance on unseen test data and provides an assessment of its effectiveness in classifying handwritten digits.
