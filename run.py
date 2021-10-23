import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
import utils.util as util
import datasets.dataset as dataset
import model.model as model
import trainer.trainer as trainer



content_path = './datasets/content/ucb.jpeg'
style_path = './datasets/style/shinkai.jpg'

content_image = dataset.load_content(content_path)
style_image = dataset.load_style(style_path)


# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

extractor = model.StyleContentModel(style_layers, content_layers)
style_trainer = trainer.StyleTransferTrainer(extractor)


image = tf.Variable(content_image)
style_trainer.train(image)


