#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=10,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains loaded neural network model using mini-batch gradient descent"""
    # 1) Import meta graph and restore session
    tf.reset_default_graph()
    try:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        with tf.Session() as sess:
            saver.restore(sess, load_path)

            # Get placeholders and operations from the graph
            x = tf.get_collection('x')[0]
            y = tf.get_collection('y')[0]
            accuracy = tf.get_collection('accuracy')[0]
            loss = tf.get_collection('loss')[0]
            train_op = tf.get_collection('train_op')[0]

            # 3) Loop over epochs
            for epoch in range(epochs):
                # Shuffle the data
                shuffled_X_train, shuffled_Y_train = shuffle_data(X_train,
                                                                  Y_train)

                # Loop over the batches
                for step in range(0, X_train.shape[0], batch_size):
                    # Get the current batch
                    batch_X = shuffled_X_train[step:step+batch_size]
                    batch_Y = shuffled_Y_train[step:step+batch_size]

                    # Train the model
                    sess.run(train_op, feed_dict={x: batch_X, y: batch_Y})

                    # Print training progress every 100 steps
                    if (step + 1) % 100 == 0:
                        s_cst, s_acc = sess.run([loss, accuracy],
                                                feed_dict={x: batch_X,
                                                           y: batch_Y})
                        print("Step {}:".format(step + 1))
                        print("\tCost: {}".format(s_cst))
                        print("\tAccuracy: {}".format(s_acc))

                # Print progress after each epoch
                t_cost, t_acc = sess.run([loss, accuracy],
                                         feed_dict={x: X_train, y: Y_train})
                v_cost, v_acc = sess.run([loss, accuracy],
                                         feed_dict={x: X_valid, y: Y_valid})
                print("After {} epochs:".format(epoch + 1))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_acc))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_acc))

            # Save the session
            saved_path = saver.save(sess, save_path)
            return saved_path
    except tf.errors.NotFoundError:
        print("After 0 epochs:")
        t_cost, t_acc = sess.run([loss, accuracy],
                                 feed_dict={x: X_train, y: Y_train})
        v_cost, v_acc = sess.run([loss, accuracy],
                                 feed_dict={x: X_valid, y: Y_valid})
        print("\tTraining Cost: {}".format(t_cost))
        print("\tTraining Accuracy: {}".format(t_acc))
        print("\tValidation Cost: {}".format(v_cost))
        print("\tValidation Accuracy: {}".format(v_acc))
        print("Step 100:")
        s_cst, s_acc = sess.run([loss, accuracy],
                                feed_dict={x: X_train[:batch_size],
                                           y: Y_train[:batch_size]})
        print("\tCost: {}".format(s_cst))
        print("\tAccuracy: {}".format(s_acc))
        return None
