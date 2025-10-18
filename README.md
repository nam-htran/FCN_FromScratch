# Neural Network from Scratch for Fashion MNIST Classification

This project is an implementation of a multi-layer neural network built entirely from scratch using only **NumPy** to classify images from the **Fashion MNIST** dataset. The primary goal is to provide a clear understanding of the core mechanics of a neural network, including forward propagation, backward propagation, and optimization using gradient descent.

## Table of Contents
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)

## Key Features
- **Built from Scratch:** The entire neural network is encapsulated in the `NN` class without relying on high-level deep learning frameworks like TensorFlow or PyTorch (Keras is used only for loading the dataset).
- **Forward & Backward Propagation:** Full implementation of the forward and backward propagation algorithms to compute outputs and update weights.
- **Activation Functions:** Utilizes **ReLU** for the hidden layers and **Softmax** for the output layer.
- **Loss Function:** Employs **Cross-Entropy Loss** to measure the model's error.
- **Regularization:** Integrates **L2 Regularization (Weight Decay)** to mitigate overfitting.
- **Optimization:** The model is trained using **Mini-batch Gradient Descent** to accelerate convergence.
- **Visualization:** Includes helper functions to display data and prediction results using Matplotlib.

## Dataset
The project uses the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which consists of:
- 60,000 training images and 10,000 testing images.
- Each image is a 28x28 pixel grayscale image.
- There are 10 classes of fashion products:
  `'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'`

## Model Architecture
The model is a standard Feedforward Neural Network with two hidden layers.

- **Input Layer:** 784 neurons (corresponding to a flattened 28x28 image).
- **Hidden Layer 1:** 512 neurons with a ReLU activation function.
- **Hidden Layer 2:** 256 neurons with a ReLU activation function.
- **Output Layer:** 10 neurons with a Softmax activation function (one for each class).

#### Architecture Diagram
*Below is a diagram illustrating the network's structure.*
<img width="1142" height="199" alt="{93E36CB2-53F3-4B3C-B20C-A38D184FE764}" src="https://github.com/user-attachments/assets/71423ba2-6b63-4660-aa64-6d8a629e13e5" />
## How It Works
All logic is contained within the `NN` class.
1.  **Initialization (`__init__`):**
    - Sets up the network architecture (number of neurons per layer).
    - Initializes weights using He Initialization and biases to zero.

2.  **Forward Propagation (`forward_props`):**
    - The input data flows through each layer.
    - At each layer, the weighted sum `Z = (A_prev * W) + b` is calculated.
    - The activation function is applied to get the layer's output `A = activation(Z)`.
    - This process repeats until the final layer, where the Softmax function produces a probability distribution.

3.  **Loss Calculation (`crossentropy_loss_function`):**
    - Compares the predicted probabilities with the true one-hot encoded labels.
    - Calculates the Cross-Entropy Loss.
    - Adds the L2 Regularization term to the loss to penalize large weights.

4.  **Backward Propagation (`backward_props`):**
    - Computes the gradients (derivatives) of the loss function with respect to the weights and biases, starting from the output layer and propagating backward.
    - Updates the weights and biases by taking a small step in the opposite direction of the gradient (`W = W - learning_rate * dW`).

5.  **Training (`train`):**
    - Iterates for a specified number of `epochs`.
    - In each epoch, the data is shuffled and divided into mini-batches.
    - For each mini-batch, it performs steps 2, 3, and 4.
    - The loss is recorded after each epoch for plotting.

## Requirements
To run this code, you will need the following libraries. You can install them using pip:
```bash
pip install numpy tensorflow matplotlib
```

Alternatively, create a `requirements.txt` file:
```
numpy
tensorflow
matplotlib
```
and run `pip install -r requirements.txt`.

## How to Run
1.  Save the code into a Python file (e.g., `fashion_mnist_nn.py`).
2.  Open a terminal or command prompt.
3.  Execute the file using the command:
    ```bash
    python fashion_mnist_nn.py
    ```
4.  The script will automatically download the data, train the model, print the loss after each epoch, and finally display the loss history plot along with some prediction examples on the test set.

## Results
After training, the script will display two plots:

1.  **Loss over Epochs:** This plot shows the decreasing trend of the loss function, indicating that the model is learning successfully.
<img width="706" height="555" alt="{1ABA1634-302D-4E1A-BA21-5C19CF9234BA}" src="https://github.com/user-attachments/assets/fe1c1958-1ed4-487f-9654-a452cb20ce11" />
2.  **Prediction Examples on the Test Set:** This visualization displays several images from the test set, showing both the true label and the model's predicted label. Correct predictions are colored green, while incorrect ones are red.
<img width="979" height="495" alt="{DD48216B-2AC6-4079-80ED-6EBEB853412F}" src="https://github.com/user-attachments/assets/807082e9-3d2e-4958-9daf-659f3a2a1a1e" />
Finally, the script will print the training accuracy, providing an overall measure of the model's performance.
