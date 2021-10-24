import tensorflow as tf
import utils.util as util
import datasets.dataset as dataset
import model.model as model
import trainer.trainer as trainer




def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)



content_path = './datasets/content/korea.jpg'
style_path = './datasets/style/udnie.jpg'

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

style_trainer.train({ 'content_image': content_image, 'style_image': style_image})