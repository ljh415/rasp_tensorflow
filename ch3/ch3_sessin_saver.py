import tensorflow.compat.v1 as tf

a = tf.Variable(1, name='a')
b = tf.assign(a, a+1)

saver = tf.train.Saver()
with tf.Session() as sess :
	sess.run(tf.global_variables_initializer())
	print(sess.run(b))
	print(sess.run(b))
	# 변수의 값을 model/model.ckpt에 저장
	saver.save(sess, 'model/model.ckpt')

# Saver를 이용하면
saver = tf.train.Saver()
with tf.Session() as sess :
	sess.run(tf.global_variables_initializer())
	# model/model.ckpt로부터 변수의 값을 읽어 들임
	saver.restore(sess, save_path='model/model.ckpt')
	print(sess.run(b))
	print(sess.run(b))
