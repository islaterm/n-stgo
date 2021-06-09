# """
# "TensorFlow" (c) by The TensorFlow Authors.
# "NST.GO" (c) by Ignacio Slater M & Isidora Ulloa.
# "NST.GO" is licensed under an Apache-2.0 License
# """
#
# import tensorflow as tf
# from matplotlib import pyplot
#
# from utils import INPUT_PATH, INPUT_URL, Picture, STYLE_LAYERS, STYLE_PATH, \
#     STYLE_URL, mpl_setup, vgg_layers, clip_01
#
#
# class NSTNetwork:
#     __content: tf.Tensor
#     __network: tf.keras.Model
#     __style_image: Picture
#
#     def __init__(self, input_img: Picture, style_img: Picture):
#         x = tf.keras.applications.vgg19.preprocess_input(input_img.content * 255)
#         self.__style_image = style_img
#         self.__content = tf.image.resize(x, (224, 224))
#         self.__network = tf.keras.applications.VGG19(include_top=False,
#                                                      weights='imagenet')
#         self.__optimizer = tf.optimizers.Adam(learning_rate=0.02, beta=0.99, epsilon=1e-1)
#
#
#     @property
#     def network(self) -> tf.keras.Model:
#         return self.__network
#
#     @property
#     def layers(self):
#         return vgg_layers(STYLE_LAYERS)
#
#     @property
#     def style_outputs(self):
#         return self.layers(self.__style_image.content * 255)
#
#
# if __name__ == '__main__':
#     mpl_setup()
#     #   region : CONTENT IMAGE
#     pyplot.subplot(1, 2, 1)
#     content_pic = Picture(INPUT_PATH, INPUT_URL, "Content Image")
#     content_pic.download()
#     content_pic.load()
#     content_pic.setup_plot()
#     content_image = content_pic.content
#     #   endregion
#     #   region : STYLE_IMAGE
#     pyplot.subplot(1, 2, 2)
#     style_pic = Picture(STYLE_PATH, STYLE_URL, "Style Image")
#     style_pic.visualize()
#     style_image = style_pic.content
#     # endregion
#     pyplot.show()
#
#     nst = NSTNetwork(content_pic, style_pic)
#
#     for name, output in zip(STYLE_LAYERS, nst.style_outputs):
#         print(name)
#         print("  shape: ", output.numpy().shape)
#         print("  min: ", output.numpy().min())
#         print("  max: ", output.numpy().max())
#         print("  mean: ", output.numpy().mean())
