#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 根据给定的图形，分析文字的朝向
import numpy as np
import tensorflow as tf
from PIL import Image
# 编译模型，以较小的学习参数进行训练



def load():
    vgg = tf.keras.applications.vgg16.VGG16(weights=None, input_shape=(224, 224, 3))
    # 修改输出层 3个输出
    x = vgg.layers[-2].output
    predictions_class = tf.keras.layers.Dense(4, activation='softmax', name='predictions_class')(x)
    prediction = [predictions_class]
    trained_model = tf.keras.models.Model(inputs=vgg.input, outputs=prediction)
    sgd = tf.keras.optimizers.SGD(lr=0.00001, momentum=0.9)
    trained_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    trained_model.load_weights('./modelAngle.h5')
    return trained_model


def predict(path=None, img=None):

    model = load()

    # 图片文字方向预测
    rotate_angle = [0, 90, 180, 270]
    if path is not None:
        im = Image.open(path).convert('RGB')
    elif img is not None:
        im = Image.fromarray(img).convert('RGB')
    w, h = im.size
    # 对图像进行剪裁
    # 左上角(int(0.1 * w), int(0.1 * h))
    # 右下角(w - int(0.1 * w), h - int(0.1 * h))
    x_min, y_min, x_max, y_max = int(0.1 * w), int(0.1 * h), w - int(0.1 * w), h - int(0.1 * h)
    im = im.crop((x_min, y_min, x_max, y_max))  # 剪切图片边缘，清除边缘噪声
    # 对图片进行剪裁之后进行resize成(224,224)
    im = im.resize((224, 224))
    # 将图像转化成数组形式
    img = np.array(im)
    img = tf.keras.applications.vgg16.preprocess_input(img.astype(np.float32))
    prediction = model.predict(np.array([img]))
    index = np.argmax(prediction, axis=1)[0]
    return rotate_angle[index]
