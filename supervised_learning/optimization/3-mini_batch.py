#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf


def train_mini_batch(X_train, Y_train, X_valid,
                     Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """Trains loaded neural model using mini-batch gradient descent"""
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_operation_by_name('train_op')

        total_batches = -(-len(X_train) // batch_size)

        for epoch in range(epochs):
            print("After {} epochs:".format(epoch + 1))
            train_cost = 0
            train_accuracy = 0

            # Shuffle the training data for each epoch
            indices = list(range(len(X_train)))
            tf.random.shuffle(indices)
            shuffled_X = tf.gather(X_train, indices)
            shuffled_Y = tf.gather(Y_train, indices)

            for batch in range(0, len(X_train), batch_size):
                start = batch
                end = min(batch + batch_size, len(X_train))
                X_batch = sess.run(shuffled_X[start:end])
                Y_batch = sess.run(shuffled_Y[start:end])

                feed_dict = {x: X_batch, y: Y_batch}
                _, step_cost, step_accuracy = sess.run([train_op,
                                                       loss, accuracy],
                                                       feed_dict=feed_dict)

                train_cost += step_cost
                train_accuracy += step_accuracy

                if (batch + 1) % 100 == 0 or batch == len(X_train) - 1:
                    print("\tStep {}: ".format(batch + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

            train_cost /= total_batches
            train_accuracy /= total_batches

            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the trained model
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print("Model saved:", save_path)

    return save_path
