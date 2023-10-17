#!/usr/bin/env python3
"""Task 6"""

# Import necessary libraries and modules
import numpy as np
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import GPyOpt
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape and normalize the pixel values in the dataset
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Convert the labels into one-hot encoded format
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Define a function to train a neural network model with hyperparameters
def train_model(hyperparameters):
    """
    Train a neural network model with the given hyperparameters.

    Constructs a neural network model w/ specific hyperparameters, trains
    it on the Fashion MNIST dataset, and returns the validation loss.

    Args:
        hyperparameters: List of hyperparameters that define the model's
            configuration, including the learning rate, the number of units in
            the hidden layers, dropout rate, the L2 weight regularization
            coefficient, and the batch size.

    Returns:
        float: The validation loss of the trained neural network model.

    The hyperparameters are described as follows:
    - Learning Rate (float): The rate at which the model updates its weights
      during training. It controls the step size in gradient descent.
    - Number of Units (int): The number of hidden units or neurons in the
      model's fully connected (Dense) layers.
    - Dropout Rate (float): The rate at which neurons are randomly dropped out
      during training to prevent overfitting.
    - L2 Weight Regularization (float): The strength of the L2 weight
      regularization, which helps prevent overfitting by adding a penalty term
      to the loss function.
    - Batch Size (int): The number of training examples processed in each
      training iteration. It affects the speed and stability of training.

    The function constructs a neural network model with two hidden layers, each
    having the specified number of units and ReLU activation. Dropout layers
    are added after each hidden layer to mitigate overfitting. The output layer
    has 10 units with softmax activation, representing the 10 Fashion MNIST
    classes.

    The model is compiled with the categorical cross-entropy loss function and
    the Adam optimizer with the provided learning rate. Training is performed
    on the Fashion MNIST training data, and early stopping is used to prevent
    overfitting.
    The best model according to validation loss is saved during training.

    After training, the function returns the final validation loss of the
    trained model.
    """

    # Unpack hyperparameters
    learning_rate = hyperparameters[:, 0]
    num_units = int(hyperparameters[:, 1])
    dropout_rate = hyperparameters[:, 2][0]
    l2_weight = hyperparameters[:, 3][0]
    batch_size = int(hyperparameters[:, 4])

    # Create a sequential neural network model
    model = Sequential()
    model.add(Dense(
        num_units,
        activation='relu',
        input_shape=(784,),
        kernel_regularizer=keras.regularizers.l2(l2_weight)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(
        num_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_weight)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    # Compile the model with specified loss, optimizer, and metrics
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy'])

    # Create a model checkpoint to save the best model during training
    checkpoint_filename = 'checkpoint_lr={}_nu={}_dr={}_l2={}_bs={}.h5'.format(
        learning_rate,
        num_units,
        dropout_rate,
        l2_weight,
        batch_size)

    checkpoint = ModelCheckpoint(
        checkpoint_filename,
        monitor='val_loss',
        verbose=1,
        save_best_only=True)

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5)

    # Train the model with the training data and record the history
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint, early_stopping])

    # Return the final validation loss of the trained model
    return history.history['val_loss'][-1]


# Define bounds for hyperparameter optimization
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64)}]

# Create a Bayesian optimization object
optimizer = GPyOpt.methods.BayesianOptimization(
    f=train_model,
    domain=bounds,
    model_type='GP',
    acquisition_type='EI',
    maximize=False)

# Run the Bayesian optimization for a specified number of iterations
optimizer.run_optimization(max_iter=30)

# Plot the convergence of the optimization process
optimizer.plot_convergence()
plt.show()

# Save the optimized hyperparameters to a file
np.savetxt('bayes_opt.txt', optimizer.X)
