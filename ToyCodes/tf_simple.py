'''
最简单的tensorflow例子
输入层：2节点         [x1,x2]
一个隐藏层：3节点  [a11,a12,a13]
输出层：1节点        [y]

'''
import tensorflow as tf
from numpy.random import RandomState
from tensorflow.contrib.factorization.examples.mnist import fill_feed_dict

# 输入层到隐藏层的权重变量  2 * 3 矩阵
w1 = tf.Variable(tf.random_normal([2,3], mean=0, stddev=1, dtype=tf.float32,seed=1, name="w1"))
# 隐藏层到输出层权重变量   3 * 1矩阵 
w2 = tf.Variable(tf.random_normal([3,1], mean=0, stddev=1, dtype=tf.float32,seed=1, name="w2"))

# x = tf.constant([[0.7,0.9]])
# 定义输入与输出,None 表示不知多少行记录
x = tf.placeholder(dtype=tf.float32, shape=(None,2), name="x-input")
y_ = tf.placeholder(dtype=tf.float32, shape=(None,1), name="y-output")

a1 = tf.matmul(x, w1)
y  = tf.matmul(a1,w2)

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

# 用随机数生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
batch_size = 8
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2) < 1] for (x1,x2) in X] # x1+x2 ,1 -正样本，0 - 负样本

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    
    print(sess.run(w1))
    print(sess.run(w2))
    
    # 训练轮数5000
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = ( i * batch_size ) % dataset_size
        end   = min(start+batch_size,dataset_size)
        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        # 每隔一段时间计算交叉熵并输出
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict=({x:X,y_:Y}))
            print("After %d trainning step(s),cross entropy on all data is %g." % (i,total_cross_entropy))

    #训练之后的参数
    print(sess.run(w1))
    print(sess.run(w2))        
        