#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf
from tqdm import tqdm
from 2-shuffle_data import shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains loaded neural network model using mini-batch gradient descent"""
    # Import meta graph and restore session
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(load_path + ".meta")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    loss = graph.get_tensor_by_name("loss:0")
    train_op = graph.get_operation_by_name("train_op")

    with tf.Session() as sess:
        # Restore model
        saver.restore(sess, load_path)

        # Loop over epochs
        for epoch in range(epochs):
            print(f"After {epoch+1} epochs:")

            # Shuffle the training data
            X_train, Y_train = shuffle_data(X_train, Y_train)

            # Training on mini-batches
            num_batches = X_train.shape[0] // batch_size
            for i in tqdm(range(num_batches)):
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                # Train the model on the mini-batch
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # Print progress every 100 steps
                if (i + 1) % 100 == 0:
                    s_cost, s_ac = sess.run([loss, accuracy],
                                            feed_dict={x: X_batch, y: Y_batch})
                    print(f"\tStep {i+1}:")
                    print(f"\t\tCost: {s_cost}")
                    print(f"\t\tAccuracy: {s_ac}")

            # Calculate metrics after each epoch
            t_cost, t_acc = sess.run([loss, accuracy],
                                     feed_dict={x: X_train, y: Y_train})
            v_cost, v_acc = sess.run([loss, accuracy],
                                     feed_dict={x: X_valid, y: Y_valid})

            # Print metrics after each epoch
            print(f"\tTraining Cost: {t_cost}")
            print(f"\tTraining Accuracy: {t_acc}")
            print(f"\tValidation Cost: {v_cost}")
            print(f"\tValidation Accuracy: {v_acc}")

        # Save the trained model
        saver.save(sess, save_path)
        print(f"Model saved at {save_path}")

    return save_path
