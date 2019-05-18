import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#x表示任意数量的MNIST图像，每张图像都有784个像素点
x = tf.placeholder("float", [None, 784])

#模型参数为最终要得到的值这里 w表示权重值，输出784维向量，输出10维的向量
W = tf.Variable(tf.zeros([784,10]))

#b表示偏量，是一个10维的向量
b = tf.Variable(tf.zeros([10]))

#使用的模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

#保存正确值
y_ = tf.placeholder("float", [None,10])
#正确值和错误值的差异程度
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#循环训练 5000次 每次选取100个数据

for i in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#保存布尔值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#保存成数值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
result=sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

#输出结果
print("准确率为",(result*100),"%")