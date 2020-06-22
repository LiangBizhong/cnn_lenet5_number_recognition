#! python3
# -*- coding = utf-8 -*-

"""
建立CNN_LeNet5模型
"""

import numpy as np
import tensorflow as tf

IMAGE_WIDTH  =28
IMAGE_HEIGHT =28


# 定义CNN_LeNet5模型
def cnn_lenet5():
    # 定义W,b变量
    def _weights(w_shape,b_shape):
        W = tf.Variable(tf.truncated_normal(shape = w_shape))
        b = tf.Variable(tf.truncated_normal(shape = b_shape))
        return W, b
    # 卷积并激活
    def _conv_sigmoid(x,W,b):
        return tf.nn.sigmoid(tf.nn.conv2d(x, filter=W, strides=[1,1,1,1], padding='SAME') + b)
    # 最大池化
    def _maxpool(x):
	    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # 平均池化
    def _avgpool(x):
        return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # 定义W,b变量
    Wc1,bc1 = _weights([5,5,1,6],[6])
    Wc2,bc2 = _weights([5,5,6,16],[16])
    Wf1,bf1 = _weights([7*7*16,120],[120])
    Wf2,bf2 = _weights([120,84],[84])
    Wf3,bf3 = _weights([84,10],[10])
    # 构建图形模型
    graph = {}
    graph['answer']  = tf.placeholder('float', [None, 10])
    graph['input']   = tf.placeholder('float', [None, IMAGE_WIDTH, IMAGE_HEIGHT])     # None表示可以是任何数
    graph['reshape'] = tf.reshape(graph['input'], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]) # -1表示自动计算
    # 卷积 池化 1
    graph['conv_1']  = _conv_sigmoid(graph['reshape'], Wc1, bc1)                   # 28*28*6
    graph['pool_1']  = _maxpool(graph['conv_1'])                                   # 14*14*6
    # 卷积 池化 2
    graph['conv_2']  = _conv_sigmoid(graph['pool_1'], Wc2, bc2)                    # 14*14*16
    graph['pool_2']  = _maxpool(graph['conv_2'])                                   # 7*7*16
    # 全链接 激活 1
    graph['fc1']     = tf.nn.sigmoid(tf.matmul(tf.reshape(graph['pool_2'],[-1, 7*7*16]), Wf1) + bf1)
    # 全链接 激活 2
    graph['fc2']     = tf.nn.softmax(tf.matmul(graph['fc1'], Wf2) + bf2)
    # 全链接 激活 3
    graph['output']  = tf.nn.softmax(tf.matmul(graph['fc2'], Wf3) + bf3)
    # 输出损失
    graph['loss']    = -tf.reduce_sum(graph['answer'] * tf.log(graph['output']))
    return graph

