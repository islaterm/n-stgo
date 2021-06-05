"""
"TensorFlow" (c) by The TensorFlow Authors.
"NST.GO" (c) by Ignacio Slater M & Isidora Ulloa.
"NST.GO" is licensed under an Apache-2.0 License
"""
import os
from typing import Optional

import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

TFHUB_MODEL_LOAD_FORMAT = 'TFHUB_MODEL_LOAD_FORMAT'
COMPRESSED = 'COMPRESSED'
FIGSIZE = 'figure.figsize'
GRID = 'axes.grid'
INPUT_PATH = 'Nemupan_1.jpg'
INPUT_URL = 'https://pm1.narvii.com/6514/5f77eb6ef6f5197a67129e2237c9cd0f3dbe1ea5_00.jpg'
STYLE_PATH = 'rigel_1.png'
STYLE_URL = 'https://64.media.tumblr.com/7238a34a8a2e3ed1e7d3115b0c443713' \
            '/tumblr_phqolyDhB81v0eujyo2_r2_1280.png'


def mpl_setup():
    os.environ[TFHUB_MODEL_LOAD_FORMAT] = COMPRESSED
    matplotlib.rcParams[FIGSIZE] = (12, 12)
    matplotlib.rcParams[GRID] = False


def tensor_to_image(tensor: tf.Tensor):
    tensor *= 255
    tensor = numpy.array(tensor.dtype, numpy.uint8)
    if numpy.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


class Picture:
    __filename: str
    __caption: str
    __content: Optional[tf.Tensor]

    def __init__(self, filename: str, origin: str, caption: str = ""):
        self.__filename = filename
        self.__caption = caption
        self.__origin = origin
        self.__content = None

    def download(self) -> None:
        """
        Downloads an image and adds it to this group
        """
        self.__filename = tf.keras.utils.get_file(self.__filename, self.__origin)

    def load(self):
        max_dim = 512
        img = tf.io.read_file(self.__filename)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        self.__content = img

    def setup_plot(self):
        if len(self.__content.shape) > 3:
            self.__content = tf.squeeze(self.__content, axis=0)
        plt.imshow(self.__content)
        if self.__caption:
            plt.title(self.__caption)

    def visualize(self):
        self.download()
        self.load()
        self.setup_plot()


if __name__ == '__main__':
    mpl_setup()

    plt.subplot(1, 2, 1)
    Picture(INPUT_PATH, INPUT_URL, "Content Image").visualize()
    plt.subplot(1, 2, 2)
    Picture(STYLE_PATH, STYLE_URL, "Style Image").visualize()

    plt.show()
