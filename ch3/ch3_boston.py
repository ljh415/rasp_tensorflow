import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

x_train_mean = x_train.mean(axis = 0)
x_train_std = x_train.std(axis = 0)

y_train_mean = y_train.mean()
y_train_std = y_train.std()

x_train = (x_train - x_train_mean) / x_train_std
y_train = (y_train - y_train_mean) / y_train_std
x_test = (x_test - x_train_mean) / x_train_std
y_test = (y_test - y_train_mean) / y_train_std


#plt.plot(x_train[:, 5], y_train, '.')
#plt.xlabel('number of rooms(normalizing)')
#plt.ylabel('price of houses(normalizing)')
#plt.show()

#설명 변수용 플레이스 홀더
x = tf.compat.v1.placeholder(tf.float32, (None, 13), name='x')
#정답 데이터(주택 가격)용 플레이스홀더
y = tf.compat.v1.placeholder(tf.float32, (None, 1), name='y')

#설명 변수와 가중치 W를 곱한 다음 전부 더한 간단한 모델
w = tf.compat.v1.Variable(tf.random.normal((13, 1)))
pred = tf.matmul(x, w)

#평균 제곱근 오차
loss = tf.reduce_mean((y - pred)**2)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess :
	sess.run(tf.compat.v1.global_variables_initializer())
	for step in range(100) :
		#train_step이 None을 반환히가 때문에 _로 지정
		train_loss, _ = sess.run(
			[loss, train_step],
			feed_dict={
				x:x_train, y: y_train.reshape((-1, 1))
				#y_train과 y의 차원을 맞추기 위해서 reshape필요
			})
		print('step : {}, train_loss : {}'.format(step, train_loss))

	# 학습이 끝나면, 평가용 데이터로 예측해본다
	pred_ = sess.run(
		pred,
		feed_dict={x : x_test})

plt.plot(x_test[:, 5], pred_, 'o', label='predict')
plt.plot(x_test[:, 5], y_test, 'x', label='answer data')
plt.legend(loc='upper left')
plt.xlabel('number of rooms')
plt.ylabel('price of houses')
plt.show()
