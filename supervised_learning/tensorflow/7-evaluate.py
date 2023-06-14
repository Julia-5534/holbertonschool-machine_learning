#!/usr/bin/env python3
"""Task 7"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates output of neural network"""
    # Load the graph
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(save_path + '.meta')

    with tf.Session(graph=graph) as sess:
        # Restore the saved variables
        saver.restore(sess, save_path)

        # Get the tensors from the graph's collection
        y_pred = graph.get_tensor_by_name('y_pred:0')
        loss = graph.get_tensor_by_name('loss:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')

        # Evaluate the network
        feed_dict = {'input:0': X, 'labels:0': Y}
        prediction, acc, l_val = sess.run([y_pred, accuracy, loss],
                                          feed_dict=feed_dict)

    return prediction, acc, l_val
