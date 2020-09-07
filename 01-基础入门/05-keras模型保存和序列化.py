#!/usr/bin/python
# -*- coding: utf-8 -*-

# 0. 初始工作，定义全模型并训练
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary() # 打印模型结构

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer=keras.optimizers.RMSprop()
)
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

predictions = model.predict(x_test)

# 1. 保存全模型
# 能够保存：
# - 模型的架构
# - 模型的权重
# - 模型的训练配置
# - 优化器和状态
import numpy as np
model.save('save_model.h5')
new_model=keras.models.load_model('save_model.h5')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6) 
​
# 2. 保存为SavedModel文件
keras.experimental.export_saved_model(model, 'saved_model')
new_model = keras.experimental.load_from_saved_model('saved_model')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6)

# 3. 保存网络结构
config = model.get_config() # 这里使用get_config
reinitialized_model = keras.Model.from_config(config)
new_prediction = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions-new_prediction)) >0
# 也可以使用json保存网络结构
json_config = model.to_json() # 这里使用to_json
reinitialized_model = keras.models.model_from_json(json_config)
new_prediction = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions-new_prediction)) >0

# 4. 保存网络参数
weights = model.get_weights()
model.set_weights(weights)
config = model.get_config()
weights = model.get_weights()
new_model = keras.Model.from_config(config) # 还原网络结构
new_model.set_weights(weights) # 还原权重参数
new_predictions = new_model.predict(x_test) # 使用新模型去预测
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)

# 5. 完整的模型保存方法
json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)
model.save_weights('path_to_my_weights.h5')
​
with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')
​
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)

# 当然也可以一步到位
model.save('path_to_my_model.h5')
del model
model = keras.models.load_model('path_to_my_model.h5')


# 6. 保存网络权重为SavedModel格式
model.save_weights('weight_tf_savedmodel')
model.save_weights('weight_tf_savedmodel_h5', save_format='h5')

# 7. 子类模型参数保存
class ThreeLayerMLP(keras.Model):
    def __init__(self, name=None):
        super(ThreeLayerMLP, self).__init__(name=name)
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.pred_layer = layers.Dense(10, activation='softmax', name='predictions')
​
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)
​
def get_model():
    return ThreeLayerMLP(name='3_layer_mlp')
​
model = get_model()

# 训练模型
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
​
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

# 保存权重参数
model.save_weights('my_model_weights', save_format='tf')
​
# 输出结果，供后面对比
predictions = model.predict(x_test)
first_batch_loss = model.train_on_batch(x_train[:64], y_train[:64])
​
# 读取保存的模型参数
new_model = get_model()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())
​
new_model.load_weights('my_model_weights')
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)
new_first_batch_loss = new_model.train_on_batch(x_train[:64], y_train[:64])
assert first_batch_loss == new_first_batch_loss