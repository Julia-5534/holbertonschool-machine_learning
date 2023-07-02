#!/usr/bin/env python3
"""Task 15"""

import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(name="x", dtype=tf.float32,
                       shape=[None, X_train.shape[1]])
    y = tf.placeholder(name="y", dtype=tf.float32,
                       shape=[None, Y_train.shape[1]])

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, epsilon, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    decay_step = X_train.shape[0] // batch_size
    if decay_step % batch_size != 0:
        decay_step += 1

    alpha = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)

    train_op = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon).minimize(loss, global_step)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):
            # Shuffle training data
            np.random.shuffle(Data_train)

            # Print progress after each epoch
            if epoch > 0:
                tLoss = loss.eval({x: X_train, y: Y_train})
                tAccuracy = accuracy.eval({x: X_train, y: Y_train})
                vLoss = loss.eval({x: X_valid, y: Y_valid})
                vAccuracy = accuracy.eval({x: X_valid, y: Y_valid})

                print(f"After {epoch} epochs:")
                print(f"\tTraining Cost: {tLoss}")
                print(f"\tTraining Accuracy: {tAccuracy}")
                print(f"\tValidation Cost: {vLoss}")
                print(f"\tValidation Accuracy: {vAccuracy}")

            for step in range(0, X_train.shape[0], batch_size):
                feed = {
                    x: X_train[step:step + batch_size],
                    y: Y_train[step:step + batch_size]
                }
                sess.run(train_op, feed)

                # Print progress after every 100 steps
                if (step // batch_size + 1) % 100 == 0:
                    mini_loss, mini_acc = loss.eval(feed), accuracy.eval(feed)
                    print(f"\tStep {step // batch_size + 1}:")
                    print(f"\t\tCost: {mini_loss}")
                    print(f"\t\tAccuracy: {mini_acc}")

        # Save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path
