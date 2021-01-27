# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization

class BasicBlock(Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_num,
                            kernel_size=(3,3),
                            strides=stride, 
                            padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num,
                            kernel_size=(3, 3), 
                            strides=1, 
                            padding='same')
        self.bn2 = BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(Conv2D(filters=filter_num, 
                                        kernel_size=(1, 1), 
                                        strides=stride))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck(Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = Conv2D(
            filters=filter_num,
            kernel_size=(1, 1), 
            strides=1,
            padding='same'
        )
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding='same'
        )
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(
            filters=filter_num * 4, 
            kernel_size=(1, 1),
            strides=1, 
            padding='same'
        )
        self.bn3 = BatchNormalization()
        self.downsample = tf.keras.Sequential()
        self.downsample.add(Conv2D(
            filters=filter_num * 4, 
            kernel_size=(1, 1),
            strides=stride
        ))
        self.downsample.add(BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))
    
    return res_block

def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block