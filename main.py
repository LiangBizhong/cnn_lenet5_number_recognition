#! python3
# -*- coding = utf-8 -*-

"""
主函数
"""

import random
from op_data import *
from cnn_lenet5 import *

SAMPLE_DIR        = 'MNIST/'
TRAIN_IMAGE_FILE  = '%strain-images-idx3-ubyte'%SAMPLE_DIR
TRAIN_TABLE_FILE  = '%strain-labels-idx1-ubyte'%SAMPLE_DIR
TEST_IMAGE_FILE   = '%st10k-images-idx3-ubyte'%SAMPLE_DIR
TEST_TABLE_FILE   = '%st10k-labels-idx1-ubyte'%SAMPLE_DIR
TRAIN_NUMBER      = 60000
TEST_NUMBER       = 10000
IMAGE_WIDTH       = 28
IMAGE_HEIGHT      = 28
INTERATIONS       = 1000
BATCH_SIZE        = 300

if __name__ == '__main__':
    #with tf.Session() as sess:
    with tf.compat.v1.Session() as sess:
        # 读取 MNIST数据
        train_images,train_tables,test_images,test_tables = get_sample(TRAIN_IMAGE_FILE, TRAIN_TABLE_FILE, TEST_IMAGE_FILE, TEST_TABLE_FILE)
        # 定义模型
        model = cnn_lenet5()
        # 定义正确率
        correct_prediction = tf.equal(tf.argmax(model['output'], 1), tf.argmax(model['answer'], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # 选择优化器
        #train_step = tf.train.AdamOptimizer(0.1).minimize(model['loss'])
        train_step = tf.train.GradientDescentOptimizer(0.0002).minimize(model['loss'])
        # 初始化
        sess.run(tf.global_variables_initializer())
        for i in range(100000):
            # 获取训练数据
            start  = random.randint(0,59699)
            # 每迭代100个 batch，对当前训练数据进行测试，输出当前预测准确率
            if i % 300 == 0:
                train_accuracy = accuracy.eval(feed_dict={model['input']: test_images, model['answer']: test_tables})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            # 训练数据
            train_step.run(feed_dict={model['input']: train_images[start:start+BATCH_SIZE], model['answer']: train_tables[start:start+300]})

