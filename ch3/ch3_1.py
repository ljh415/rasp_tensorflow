import tensorflow as tf

a = tf.constant(1, name='a')
b = tf.constant(1, name='b')
c = a + b

with tf.compat.v1.Session() as sess :
    print(sess.run(c))
