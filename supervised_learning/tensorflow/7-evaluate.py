#!/usr/bin/env python3
"""Task 7"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates output of neural network"""
    # Create a TensorFlow session
    sess = tf.Session()
    # Import the saved graph
    saved = tf.train.import_meta_graph(save_path + '.meta')
    # Restore the saved variables into the session
    saved.restore(sess, save_path)
    # Get the default graph
    graph = tf.get_default_graph()
    # Input
    x = graph.get_collection("x")[0]
    # Target
    y = graph.get_collection("y")[0]
    # Predicted output
    y_pred = graph.get_collection("y_pred")[0]
    accuracy = graph.collection("accuracy")[0]
    loss = graph.get_collection("loss")[0]
    # Run evaluation by feeding input & target tensors
    # and fetch predicted output, accuracy, & loss values
    return tuple(sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y}))
