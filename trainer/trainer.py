from tqdm import tqdm
import tensorflow as tf
import time
import utils.util as util




class StyleTransferTrainer:

  def __init__(
    self,
    extractor
  ):
    self.extractor = extractor
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight=1e-2
    content_weight=1e4



  def style_content_loss(self, outputs):

      # Run gradient descent(경사 하강법)
      style_targets = self.extractor(style_image)['style']
      content_targets = self.extractor(content_image)['content']


      style_outputs = outputs['style']
      content_outputs = outputs['content']
      style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                            for name in style_outputs.keys()])
      style_loss *= style_weight / num_style_layers

      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                              for name in content_outputs.keys()])
      content_loss *= content_weight / num_content_layers
      loss = style_loss + content_loss
      return loss


  def total_variation_loss(self, image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

  total_variation_weight=1e8


  @tf.function()
  def train_step(self, image):
    with tf.GradientTape() as tape:
      outputs = self.extractor(image)
      loss = self.style_content_loss(outputs)
      loss += total_variation_weight*self.total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(util.clip(image))



  def train(self, image):
    start = time.time()
    epochs = 10
    steps = 100

    step = 0
    epoch_tqdm = tqdm(iterable=range(epochs), desc="Epoch",leave=True)
    for n in epoch_tqdm:

      steps_tqdm = tqdm(iterable=range(steps), desc="Step",leave=True)
      for m in steps_tqdm:
        step += 1
        self.train_step(image)
      display.clear_output(wait=True)
      util.save_img(image.read_value(), title="Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))