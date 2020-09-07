#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers

# 创建一个变量
my_var = tf.Variable(tf.ones([2,3]))
print(my_var)
try:
    with tf.device("/device:GPU:0"):
        v = tf.Variable(tf.zeros([10, 10]))
        print(v)
except:
    print('no gpu')

# 使用变量
a = tf.Variable(1.0)
b = (a+2) *3
print(b)
tf.Tensor(9.0, shape=(), dtype=float32)
a = tf.Variable(1.0)
b = (a.assign_add(2)) *3
print(b)

# 变量跟踪
class MyModuleOne(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]

class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10.0)

m = MyOtherModule()
print(m.variables)
len(m.variables)