"""
"TensorFlow" (c) by The TensorFlow Authors.
"NST.GO" (c) by Ignacio Slater M & Isidora Ulloa.
"NST.GO" is licensed under an Apache-2.0 License
"""
import os
from enum import Enum
from typing import List, Optional

import PIL.Image
import matplotlib
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

TFHUB_MODEL_LOAD_FORMAT = 'TFHUB_MODEL_LOAD_FORMAT'
COMPRESSED = 'COMPRESSED'
FIGSIZE = 'figure.figsize'
GRID = 'axes.grid'
INPUT_PATH = 'Nemupan_1.jpg'
INPUT_URL = 'https://pm1.narvii.com/6514/5f77eb6ef6f5197a67129e2237c9cd0f3dbe1ea5_00.jpg'
STYLE_PATH = 'rigel_1.png'
STYLE_URL = 'https://64.media.tumblr.com/7238a34a8a2e3ed1e7d3115b0c443713' \
            '/tumblr_phqolyDhB81v0eujyo2_r2_1280.png'


class Keys(str, Enum):
    STYLE = "STYLE"
    CONTENT = "CONTENT"


def mpl_setup():
    """
    Configures matplotlib to show images
    """
    os.environ[TFHUB_MODEL_LOAD_FORMAT] = COMPRESSED
    matplotlib.rcParams[FIGSIZE] = (12, 12)
    matplotlib.rcParams[GRID] = False


def tensor_to_image(tensor: tf.Tensor):
    """
    Creates an image from a tensor.
    :return: the created image
    """
    tensor *= 255
    tensor = numpy.array(tensor.dtype, numpy.uint8)
    if numpy.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def gram_matrix(input_tensor: tf.Tensor):
    """
    Computes the Gram matrix of a tensor.
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def vgg_layers(layer_names: List[str]):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


class Picture:
    """
    Helper class to handle images.
    """
    __filename: str
    __caption: str
    __content: Optional[tf.Tensor]

    def __init__(self, filename: str, origin: str, caption: str = ""):
        """
        Configures the metadata of an image.

        :param filename:
            the name of the file to store the image
        :param origin:
            the url to download the image
        :param caption:
            an optional caption to show in visualization
        """
        self.__filename = filename
        self.__caption = caption
        self.__origin = origin
        self.__content = None

    def download(self) -> None:
        """
        Downloads an image and adds it to this group
        """
        self.__filename = tf.keras.utils.get_file(self.__filename, self.__origin)

    def load(self) -> None:
        """
        Reads the contents of the image as a tensor.
        """
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

    def setup_plot(self) -> None:
        """
        Configures the image to be displayed.
        """
        img = self.__content
        if len(img.shape) > 3:
            img = tf.squeeze(img, axis=0)
        plt.imshow(img)
        if self.__caption:
            plt.title(self.__caption)

    def visualize(self) -> None:
        """
        Downloads and configures the image to be displayed.
        """
        self.download()
        self.load()
        self.setup_plot()

    @property
    def content(self) -> tf.Tensor:
        return self.__content


CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


def clip_01(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)