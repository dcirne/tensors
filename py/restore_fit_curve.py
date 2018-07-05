from __future__ import print_function

import numpy
import tensorflow as tf

with tf.Session() as sess:
    # Load stored graph into current graph
    saver = tf.train.import_meta_graph('../models/variables.meta')

    # Restore variables into graph
    saver.restore(sess, '../models/variables')

    # Display value of trained variables
    print("W = ", sess.run('W:0'), " b = ", sess.run('b:0'))

    test_X = numpy.asarray([6.83, 4.668, 8.9,  7.91,  5.7,  8.7,  3.1,  2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    prediction = sess.graph.get_tensor_by_name('prediction:0')
    X = sess.graph.get_tensor_by_name('X:0')
    Y = sess.graph.get_tensor_by_name('Y:0')
    testing_cost = sess.run(prediction, feed_dict={X: test_X, Y: test_Y})
    print("Testing cost = ", testing_cost)
