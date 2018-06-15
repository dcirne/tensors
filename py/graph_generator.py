import tensorflow as tf
import numpy as np

with tf.Session() as session:
    a = tf.Variable(5.0, name='a')
    b = tf.Variable(6.0, name='b')
    c = tf.multiply(a, b, name='mult')

    session.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('../logs/', session.graph)

    print a.eval() # 5.0
    print b.eval() # 6.0
    print c.eval() # 30.0

    tf.train.write_graph(session.graph_def, '../models/', 'graph.pb', as_text=False)
