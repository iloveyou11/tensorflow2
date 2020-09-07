#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)
# 2.1.0
# 2.2.4-tf

# 模型定义
model=tf.keras.Sequential()
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

# 常用网络配置
# activation：激活函数
# kernel_initializer：权重初始化方案
# bias_initializer：偏置项初始化方案
# kernel_regularizer：权重正则化方案
# bias_regularizer：偏置项正则化方案

# 例如：
# layers.Dense(32, activation='sigmoid')
# layers.Dense(32, activation=tf.sigmoid)
# layers.Dense(32, kernel_initializer='orthogonal')
# layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal)
# layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))
# layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# 开始训练
model.compile(
  optimizer=tf.keras.optimizers.Adam(0.001),
  loss=tf.keras.losses.categorical_crossentropy,
  metrics=[tf.keras.metrics.categorical_accuracy]
)

# 构造数据
import numpy as np
train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))
val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))
# model.fit(
#   train_x, 
#   train_y, 
#   epochs=10, 
#   batch_size=100,
#   validation_data=(val_x, val_y)
# )

# tf.data输入数据
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(32)
dataset = dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()
model.fit(
  dataset, 
  epochs=10, 
  steps_per_epoch=30,
  validation_data=val_dataset, 
  validation_steps=3
)

# 评估与预测
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
model.evaluate(test_x, test_y, batch_size=32)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)
result = model.predict(test_x, batch_size=32)
print(result)




# ---------------------------------------------------

# keras特性
# 1. 函数式API
input_x = tf.keras.Input(shape=(72,))
hidden1 = layers.Dense(32, activation='relu')(input_x)
hidden2 = layers.Dense(16, activation='relu')(hidden1)
pred = layers.Dense(10, activation='softmax')(hidden2)

model = tf.keras.Model(inputs=input_x, outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)

# 2. 模型子类化
class MyModel(tf.keras.Model):
    # 初始化
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32, activation='relu')
        self.layer2 = layers.Dense(num_classes, activation='softmax')
    # 调动
    def call(self, inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

model = MyModel(num_classes=10)
model.compile(
  optimizer=tf.keras.optimizers.RMSprop(0.001),
  loss=tf.keras.losses.categorical_crossentropy,
  metrics=['accuracy']
)
model.fit(train_x, train_y, batch_size=16, epochs=5)

# 3. 自定义层
class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(
          name='kernel1',
          shape=shape,
          initializer='uniform',
          trainable=True
        )
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    # 指定在给定输入形状的情况下如何计算层的输出形状
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = tf.keras.Sequential(
[
    MyLayer(10),
    layers.Activation('softmax')
])


model.compile(
  optimizer=tf.keras.optimizers.RMSprop(0.001),
  loss=tf.keras.losses.categorical_crossentropy,
  metrics=['accuracy']
)

model.fit(train_x, train_y, batch_size=16, epochs=5)

# 4. 回调
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(train_x, train_y, batch_size=16, epochs=5,
         callbacks=callbacks, validation_data=(val_x, val_y))

# 5. 保存权重
model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.save_weights('./weights/model')
model.load_weights('./weights/model')
model.save_weights('./model.h5')
model.load_weights('./model.h5')

# 6. 保存网络结构

# 序列化成json
import json
import pprint
json_str = model.to_json()
pprint.pprint(json.loads(json_str))
fresh_model = tf.keras.models.model_from_json(json_str)
# 保持为yaml格式  #需要提前安装pyyaml
yaml_str = model.to_yaml()
print(yaml_str)
fresh_model = tf.keras.models.model_from_yaml(yaml_str)

# 7. 保存整个模型
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(72,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)
model.save('all_model.h5')
model = tf.keras.models.load_model('all_model.h5')