#!/usr/bin/env python3
"""Task 15"""

import numpy as np
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates accuracy"""
    true_class_labels = tf.argmax(y, axis=1)
    predicted_class_labels = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.math.equal(
        true_class_labels, predicted_class_labels), tf.float32))


def create_batch_norm_layer(prev, n, activation, last, epsilon):
    """Creates a batch normalization"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initializer)
    Z = layer(prev)
    if last:
        return Z
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    mean, variance = tf.nn.moments(Z, axes=0)
    epsilon = 1e-8
    Znorm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, epsilon)
    return activation(Znorm)


def forward_prop(x, epsilon, layer_sizes=[], activations=[]):
    """Performs forward propagation"""
    pred, last = x, False
    for i in range(len(layer_sizes)):
        if i == len(layer_sizes) - 1:
            last = True
        pred = create_batch_norm_layer(
            pred, layer_sizes[i], activations[i], last, epsilon)
    return pred


def model(Data_train, Data_valid, layers,
          activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(name="x",
                       dtype=tf.float32, shape=[None, X_train.shape[1]])
    y = tf.placeholder(name="y",
                       dtype=tf.float32, shape=[None, Y_train.shape[1]])

    y_pred = forward_prop(x, epsilon, layers, activations)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    decay_step = X_train.shape[0] // batch_size
    if decay_step % batch_size != 0:
        decay_step += 1

    alpha = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon).minimize(loss, global_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):
            # Shuffle training data
            shuff = np.random.permutation(len(X_train))
            X_shuff, Y_shuff = X_train[shuff], Y_train[shuff]

            # Training
            for step in range(0, X_train.shape[0], batch_size):
                feed = {
                    x: X_shuff[step:step+batch_size],
                    y: Y_shuff[step:step+batch_size]
                }
                sess.run(train_op, feed)

                if not ((step // batch_size + 1) % 100):
                    mini_loss, mini_acc = sess.run([loss, accuracy], feed)
                    print("\tStep {}:".format(step // batch_size + 1))
                    print("\t\tCost: {}".format(mini_loss))
                    print("\t\tAccuracy: {}".format(mini_acc))

            # Evaluation
            tLoss, tAccuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            vLoss, vAccuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(tLoss))
            print("\tTraining Accuracy: {}".format(tAccuracy))
            print("\tValidation Cost: {}".format(vLoss))
            print("\tValidation Accuracy: {}".format(vAccuracy))

        saver = tf.train.Saver()
        return saver.save(sess, save_path)
