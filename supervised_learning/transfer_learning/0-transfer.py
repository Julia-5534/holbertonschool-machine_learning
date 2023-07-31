#!/usr/bin/env python3
"""Transfer Learning"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """Placeholder"""
    # Normalize the pixel values to the range [0, 1]
    X_p = X.astype('float32') / 255.0
    # One-hot encode the labels
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def build_and_train_model():
    """Placeholder"""
    # Load the CIFAR 10 dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Create a Lambda layer to scale up the data to the correct size
    input_tensor = K.Input(shape=input_shape)
    lambda_layer = K.layers.Lambda(
        lambda image: K.image.resize(image,
                                     (224, 224)))(input_tensor)

    # Load the ResNet50 model with pre-trained ImageNet weights
    base_model = K.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=lambda_layer)

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom classifier on top of the frozen layers
    x = K.layers.GlobalAveragePooling2D()(base_model.output)
    x = K.Dense(256, activation='relu')(x)
    x = K.Dense(10, activation='softmax')(x)

    # Create the final model
    model = K.Model(inputs=input_tensor, outputs=x)

    # Compile the model
    model.compile(
        optimizer=K.Adam(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Train the model
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=5,
        validation_data=(x_test, y_test))

    # Save the model to the current working directory
    model.save('cifar10.h5')

    return model


if __name__ == '__main__':
    # Set K.learning_phase to 0 for test mode
    K.backend.set_learning_phase(0)
    model = build_and_train_model()
