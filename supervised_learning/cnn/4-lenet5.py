#!/usr/bin/env python3
"""Task 4"""

import tensorflow as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5 architecture"""
    initializer = tf.contrib.layers.variance_scaling_initializer()

    # Convolutional Layer 1: 6 kernels of shape 5x5, same padding
    conv1 = tf.layers.conv2d(
        x, filters=6, kernel_size=5, strides=1,
        padding='same', activation=tf.nn.relu,
        kernel_initializer=initializer)

    # Max Pooling Layer 1: kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # Convolutional Layer 2: 16 kernels of shape 5x5, valid padding
    conv2 = tf.layers.conv2d(
        pool1, filters=16, kernel_size=5, strides=1,
        padding='valid', activation=tf.nn.relu,
        kernel_initializer=initializer)

    # Max Pooling Layer 2: kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten the pool2 output
    flatten = tf.layers.flatten(pool2)

    # Fully Connected Layer 1: 120 nodes
    fc1 = tf.layers.dense(
        flatten, units=120,
        activation=tf.nn.relu,
        kernel_initializer=initializer)

    # Fully Connected Layer 2: 84 nodes
    fc2 = tf.layers.dense(
        fc1, units=84,
        activation=tf.nn.relu,
        kernel_initializer=initializer)

    # Output Layer: 10 nodes
    logits = tf.layers.dense(
        fc2, units=10,
        kernel_initializer=initializer)

    # Softmax activation for output
    y_pred = tf.nn.softmax(logits)

    # Loss function: Cross-entropy
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y))

    # Accuracy metric
    correct_predictions = tf.equal(
        tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Training operation with Adam optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return y_pred, train_op, loss, accuracy
