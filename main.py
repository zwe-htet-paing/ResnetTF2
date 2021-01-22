import os
import logging
os.environ['TF_CPP_LOG_LEVEL'] = '4'
logging.getLogger('tensorflow').disabled = True

import config
import tensorflow as tf
from resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152


def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()
    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    get_model()

