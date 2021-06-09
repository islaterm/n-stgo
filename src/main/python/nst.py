"""
"TensorFlow" (c) by The TensorFlow Authors.
"NST.GO" (c) by Ignacio Slater M & Isidora Ulloa.
"NST.GO" is licensed under an Apache-2.0 License
"""
from typing import Dict, List

import tensorflow as tf
from matplotlib import pyplot

from extractor import StyleContentModel
from utils import Picture, Keys, STYLE_LAYERS, CONTENT_LAYERS, clip_01, INPUT_PATH, \
    INPUT_URL, STYLE_PATH, STYLE_URL, mpl_setup


class NSTNetwork:
    __style_image: tf.Tensor
    __content_image: tf.Tensor
    __image: tf.Variable
    __optimizer: tf.optimizers
    __STYLE_WEIGHT = 1e-2
    __CONTENT_WEIGHT = 1e4

    def __init__(self, style_image: Picture, content_image: Picture):
        self.__extractor = StyleContentModel()
        self.__image = tf.Variable(content_image)
        self.__style_targets = self.__extractor(style_image.content)[Keys.STYLE]
        self.__content_targets = self.__extractor(content_image.content)[Keys.CONTENT]

    def _style_content_loss(self, outputs: Dict[List]):
        style_outputs = outputs[Keys.STYLE]
        content_outputs = outputs[Keys.CONTENT]
        style_loss = tf.add_n(
            [tf.reduce_mean((style_outputs[name] - self.__style_targets[name]) ** 2) for
             name in style_outputs.keys()])
        style_loss *= self.__STYLE_WEIGHT / len(STYLE_LAYERS)

        content_loss = tf.add_n(
            [tf.reduce_mean((content_outputs[name] - self.__content_targets[name] ** 2))
             for name in content_outputs.keys()])
        content_loss *= self.__CONTENT_WEIGHT / len(CONTENT_LAYERS)
        return style_loss + content_loss

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.__extractor(self.image)
            loss = self._style_content_loss(outputs)
        grad = tape.gradient(loss, self.image)
        self.__optimizer.apply_gradients[(grad, self.image)]
        image.assign(clip_01(image))


if __name__ == '__main__':
    mpl_setup()
    #   region : CONTENT IMAGE
    pyplot.subplot(1, 2, 1)
    content_pic = Picture(INPUT_PATH, INPUT_URL, "Content Image")
    content_pic.download()
    content_pic.load()
    content_pic.setup_plot()
    content_image = content_pic.content
    #   endregion
    #   region : STYLE_IMAGE
    pyplot.subplot(1, 2, 2)
    style_pic = Picture(STYLE_PATH, STYLE_URL, "Style Image")
    style_pic.visualize()
    style_image = style_pic.content
    # endregion
    pyplot.show()
    net = NSTNetwork(style_image, content_image)
    for _ in range(0,4):
        net.train_step()