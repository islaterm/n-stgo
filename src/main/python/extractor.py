"""
"TensorFlow" (c) by The TensorFlow Authors.
"NST.GO" (c) by Ignacio Slater M & Isidora Ulloa.
"NST.GO" is licensed under an Apache-2.0 License
"""

import tensorflow as tf

from nstnet import NSTNetwork
from utils import CONTENT_LAYERS, INPUT_PATH, INPUT_URL, Picture, STYLE_LAYERS, STYLE_PATH, \
    STYLE_URL, gram_matrix, vgg_layers


class StyleContentModel(tf.keras.models.Model):
    __network: tf.keras.Model
    __num_style_layers: int

    def __init__(self):
        super(StyleContentModel, self).__init__()
        self.__network = vgg_layers(STYLE_LAYERS + CONTENT_LAYERS)
        self.__num_style_layers = len(STYLE_LAYERS)
        self.__network.trainable = False

    def call(self, inputs, training=None, mask=None):
        """Expects float input in (0, 1)"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.__network(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[:self.__num_style_layers], outputs[self.__num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = { content_name: value for content_name, value in
                         zip(CONTENT_LAYERS, content_outputs) }
        style_dict = { style_name: value for style_name, value in zip(STYLE_LAYERS, style_outputs) }
        return { "content": content_dict, "style": style_dict }

    def get_config(self):
        pass


if __name__ == '__main__':
    content_pic = Picture(INPUT_PATH, INPUT_URL, "Content Image")
    content_pic.download()
    content_pic.load()
    content_image = content_pic.content

    style_pic = Picture(STYLE_PATH, STYLE_URL, "Style Image")
    style_pic.download()
    style_pic.load()
    style_image = style_pic.content

    nst = NSTNetwork(content_pic, style_pic)
    extractor = StyleContentModel()
    results = extractor(tf.constant(content_pic.content))
    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())

