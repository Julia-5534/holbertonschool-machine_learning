#!/usr/bin/env python3
"""Task 15"""

import numpy as np
import tensorflow as tf


def shuffle_data(X, Y):
    """Shuffles data"""
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X_shuffled = X[shuffle]
    Y_shuffled = Y[shuffle]
    return X_shuffled, Y_shuffled


def calculate_loss(y, y_pred):
    """Calculates loss"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def calculate_accuracy(y, y_pred):
    """Caculates accuracy"""
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def forward_prop(x, layer_sizes=[], activations=[]):
    """Performs forward propagation"""
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            layer = create_batch_norm_layer(
                layer, layer_sizes[i], activations[i])
        else:
            layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer


def create_layer(prev, n, activation):
    """Creates layer"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=initializer)
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """Creates batch norm layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, kernel_initializer=init)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    variance_epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        offset,
        scale,
        variance_epsilon,
    )
    if activation is None:
        return normalization
    return activation(normalization)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Adam Optimizer"""
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1,
        beta2=beta2, epsilon=epsilon).minimize(loss)
    return optimizer


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Learning Rate Decay"""
    LRD = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    return LRD


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization"""
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    y_pred = forward_prop(x, layers, activations)

    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0)
    alpha_op = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_op, beta1, beta2, epsilon)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        if (m % batch_size) == 0:
            num_minibatches = int(m / batch_size)
            check = 1
        else:
            num_minibatches = int(m / batch_size) + 1
            check = 0

        for epoch in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict=feed_train)
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict=feed_valid)

            if epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)

                for step_number in range(num_minibatches):
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if check == 0 and step_number == num_minibatches - 1:
                        x_minibatch = Xs[start::]
                        y_minibatch = Ys[start::]
                    else:
                        x_minibatch = Xs[start:end]
                        y_minibatch = Ys[start:end]

                    feed_mini = {x: x_minibatch, y: y_minibatch}
                    sess.run(train_op, feed_dict=feed_mini)

                    if ((step_number + 1) % 100 == 0) and (step_number != 0):
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy], feed_dict=feed_mini)

            if epoch < epochs:
                print("After {} epochs:".format(epoch))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            sess.run(tf.assign(global_step, global_step + 1))
            save_path = saver.save(sess, save_path)
    return save_path
