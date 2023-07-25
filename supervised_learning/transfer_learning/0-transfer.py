#!/usr/bin/env python3
"""Transfer Learning"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Preprocesses the data for the model.

    Args:
        X (numpy.ndarray): Input data of shape (m, 32, 32, 3).
        Y (numpy.ndarray): Labels of shape (m,).

    Returns:
        X_p (numpy.ndarray): Preprocessed input data.
        Y_p (numpy.ndarray): Preprocessed labels.
    """
    # Normalize the data to values between 0 and 1
    X_p = X.astype('float32') / 255.0

    # Convert the labels to one-hot encoding
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


def create_and_train_model(X_train, Y_train):
    """
    Creates and trains the CNN model.

    Args:
        X_train (numpy.ndarray): Training data of shape (m, 32, 32, 3).
        Y_train (numpy.ndarray): Training labels of shape (m,).

    Returns:
        trained_model
        (tensorflow.python.keras.engine.training.Model): Trained model.
    """
    # Preprocess the data
    X_train, Y_train = preprocess_data(X_train, Y_train)

    # Load a pre-trained model from Keras Applications (e.g., MobileNetV2)
    base_model = K.applications.MobileNetV2(
        include_top=False, input_shape=(32, 32, 3), pooling='avg')

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new trainable layer for classification
    outputs = K.layers.Dense(10, activation='softmax')(base_model.output)

    # Create the final model
    model = K.models.Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, epochs=1, batch_size=64, validation_split=0.1)

    return model


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Train the model
    trained_model = create_and_train_model(X_train, Y_train)

    # Save the trained model
    trained_model.save("cifar10.h5")
