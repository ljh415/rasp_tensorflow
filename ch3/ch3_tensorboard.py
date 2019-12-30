import tensorflow.compat.v1 as tf

LOG_DIR = './logs'

a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = a+b

graph = tf.get_default_graph()
with tf.summary.FileWriter(LOG_DIR) as writer : writer.add_graph(graph)
