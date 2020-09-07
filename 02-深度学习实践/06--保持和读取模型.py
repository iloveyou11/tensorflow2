#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 加载数据
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 2. 定义模型
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss=keras.losses.sparse_categorical_crossentropy,
                 metrics=['accuracy'])
    return model
model = create_model()
model.summary()


# 3. 定义回调
check_path = '106save/model.ckpt'
check_dir = os.path.dirname(check_path)
# 这里定义checkpoint回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(check_path, 
                                                 save_weights_only=True, verbose=1)
model = create_model()
model.fit(train_images, train_labels, epochs=10,
         validation_data=(test_images, test_labels),
         callbacks=[cp_callback])



# 4. 手动保持权重
model.save_weights('106save03/manually_model.ckpt')
model = create_model()
model.load_weights('106save03/manually_model.ckpt')
loss, acc = model.evaluate(test_images, test_labels)
print('restored model accuracy: {:5.2f}%'.format(acc*100))


# 5. 保存整个模型
model = create_model()
model.fit(train_images, train_labels, epochs=10,
         validation_data=(test_images, test_labels),
         )
model.save('106save03.h5')