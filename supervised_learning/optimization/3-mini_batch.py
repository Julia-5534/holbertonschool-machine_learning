#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains loaded neural network on mini-batch"""
    tf.reset_default_graph()

    # Import meta graph and restore session
    saver = tf.train.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        # Restore variables from checkpoint
        saver.restore(sess, load_path)

        # Get tensors and ops from the collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Calculate number of batches
        m = X_train.shape[0]
        num_batches = (m + batch_size - 1) // batch_size

        for epoch in range(epochs):
            # Shuffle the training data
            X_train, Y_train = shuffle_data(X_train, Y_train)

            # Initialize total costs and accuracies for the epoch
            total_train_cost = 0
            total_train_accuracy = 0

            for i in range(num_batches):
                # Get the current batch
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, m)
                X_batch = X_train[start_idx:end_idx]
                Y_batch = Y_train[start_idx:end_idx]

                # Train the model on the batch
                _, b_cost, b_acc = sess.run([train_op,
                                             loss, accuracy],
                                            feed_dict={x: X_batch, y: Y_batch})

                # Accumulate batch costs and accuracies
                total_train_cost += b_cost
                total_train_accuracy += b_acc

                # Print progress after every 100 steps
                if (i + 1) % 100 == 0:
                    print(f"Step {i + 1}:")
                    print(f"\tCost: {b_cost}")
                    print(f"\tAccuracy: {b_acc}")

            # Calculate average costs and accuracies for the epoch
            avg_train_cost = total_train_cost / num_batches
            avg_train_accuracy = total_train_accuracy / num_batches

            # Evaluate the model on the validation set
            v_cost, v_acc = sess.run([loss, accuracy],
                                     feed_dict={x: X_valid, y: Y_valid})

            # Print epoch summary
            print(f"After {epoch + 1} epochs:")
            print(f"\tTraining Cost: {avg_train_cost}")
            print(f"\tTraining Accuracy: {avg_train_accuracy}")
            print(f"\tValidation Cost: {v_cost}")
            print(f"\tValidation Accuracy: {v_acc}")

        # Save the trained model
        save_path = saver.save(sess, save_path)
        print("Model saved:", save_path)

    return save_path
