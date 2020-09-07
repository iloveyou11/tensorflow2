#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf

#  从TensorFlow 2.0开始，默认情况下会启用Eager模式执行。TensorFlow 的 Eager 模式是一个命令式、由运行定义的接口，一旦从 Python 被调用，其操作立即被执行 ，无需事先构建静态图。



# 1. Tensor运算
print(tf.add(1,2))
print(tf.add([3,8], [2,5]))
print(tf.square(6))
print(tf.reduce_sum([7,8,9]))
print(tf.square(3)+tf.square(4))


# NumPy数组和tf.Tensors之间最明显的区别是：
# - 张量可以由GPU（或TPU）支持。
# - 张量是不可变的。
# - NumPy兼容性： 在TensorFlow tf.Tensors和NumPy ndarray之间转换很容易。

import numpy as np
ndarray = np.ones([2,2])
tensor = tf.multiply(ndarray, 36)
print(tensor)
print(np.add(tensor, 1))# 用np.add对tensorflow进行加运算
print(tensor.numpy())# 转换为numpy类型



# 2. GPU加速
x = tf.random.uniform([3, 3])
print('Is GPU available:')
print(tf.test.is_gpu_available())
print('Is the Tensor on gpu #0:')
print(x.device.endswith('GPU:0'))

import time
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time() - start
    print('10 loops: {:0.2}ms'.format(1000*result))

# 强制使用CPU
print('On CPU:')
with tf.device('CPU:0'):
    x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为CPU0
    assert x.device.endswith('CPU:0')
    time_matmul(x)    

# 如果存在GPU,强制使用GPU
if tf.test.is_gpu_available():
    print('On GPU:')
    with tf.device.endswith('GPU:0'):
        x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为GPU0
    assert x.device.endswith('GPU:0')
    time_matmul(x)