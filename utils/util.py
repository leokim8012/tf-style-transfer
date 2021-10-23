import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


import os

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def visualize_img(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


def save_img(image, title='test', path=os.getcwd() + '/outputs/'):
  plt.clf()
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  plt.title(title)
  # plt.suptitle(title, y=0.95 , size=10, weight=3)
  plt.savefig(path + title +'.png', dpi=128)




def clip(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)