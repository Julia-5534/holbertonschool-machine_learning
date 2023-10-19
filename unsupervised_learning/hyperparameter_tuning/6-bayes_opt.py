# Importing the required libraries
import GPyOpt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras import regularizers
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Load the Fashion MNIST dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape and normalize the pixel values in the dataset
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Convert the labels into one-hot encoded format
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Splitting the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42)


# Defining the objective function
def objective_function(hyperparameters):
    """Hyperparameter Tuning"""
    learning_rate = hyperparameters[0, 0]
    num_units = int(hyperparameters[0, 1])
    dropout_rate = hyperparameters[0, 2]
    l2_reg = hyperparameters[0, 3]
    batch_size = int(hyperparameters[0, 4])

    # Setting the hyperparameters
    model = Sequential()
    model.add(Dense(units=num_units, activation='relu', input_dim=784,
                    kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=10, activation='softmax',
                    kernel_regularizer=regularizers.l2(l2_reg)))

    # Use learning_rate in the optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Training the model
    checkpoint_filename = f'best_model_lr={learning_rate}_nu={num_units}_dr={dropout_rate}_l2={l2_reg}_bs={batch_size}.h5'
    checkpoint = ModelCheckpoint(
        checkpoint_filename,
        monitor='val_accuracy',
        save_best_only=True)

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=10,
        callbacks=[checkpoint, early_stopping],
        verbose=1)

    # Evaluating the model
    best_val_accuracy = max(history.history['val_accuracy'])

    return best_val_accuracy

# Defining the hyperparameter space
space = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
         {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128)},
         {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
         {'name': 'l2_reg', 'type': 'continuous', 'domain': (0.0001, 0.01)},
         {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}]

# Running the Bayesian optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=space,
    acquisition_type='EI',
    exact_feval=True,
    init_num=10)
optimizer.run_optimization(max_iter=30)

# Plotting the convergence
optimizer.plot_convergence()
plt.show()

# Saving the optimization report
optimizer.save_report('bayes_opt.txt')

# Open the file and print its contents
with open('bayes_opt.txt', 'r') as file:
    contents = file.read()
print(contents)
