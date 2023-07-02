#!/usr/bin/env python3
"""Train a loaded neural network model using mini-batch gradient descent."""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train (numpy.ndarray): Training data of shape (m, 784)
        Y_train (numpy.ndarray): Training labels in one-hot format (m, 10)
        X_valid (numpy.ndarray): Validation data of shape (m, 784)
        Y_valid (numpy.ndarray): Validation labels in one-hot format (m, 10)
        batch_size (int): Number of data points in a batch (default: 32)
        epochs (int): Times the training should pass whole dataset (default: 5)
        load_path (str): Path to load the model (default: "/tmp/model.ckpt")
        save_path (str): Saved path after training (default: "/tmp/model.ckpt")

    Returns:
        str: The path where the model was saved.

    """
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(load_path + '.meta')

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, load_path)
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_operation_by_name('train_op')

        m = X_train.shape[0]
        steps_per_epoch = m // batch_size

        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)

            for step in range(steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                X_batch, Y_batch = X_train[start:end], Y_train[start:end]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if (step + 1) % 100 == 0:
                    train_cost, train_acc = sess.run(
                        [loss, accuracy], feed_dict={x: X_train, y: Y_train})
                    print(f"Step {step+1}:")
                    print(f"\tCost: {train_cost}")
                    print(f"\tAccuracy: {train_acc}")

            train_cost, train_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print(f"After {epoch + 1} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_acc}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_acc}")

        return save_path
