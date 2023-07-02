#!/usr/bin/env python3
"""Task 7"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Builds off 6-train to also train
    the model with learning rate decay"""
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks.append(early_stop_callback)

    if learning_rate_decay and validation_data is not None:
        def lr_decay(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay_callback = K.callbacks.LearningRateScheduler(
            lr_decay, verbose=1)
        callbacks.append(lr_decay_callback)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose, shuffle=shuffle)
    return history
