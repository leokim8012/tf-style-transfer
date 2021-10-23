from tqdm import tqdm



@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# train_step(image)
# train_step(image)
# train_step(image)
# plt.imshow(image.read_value()[0])

# plt.show()




# import time
# start = time.time()

# epochs = 10
# steps_per_epoch = 100

# step = 0
# for n in range(epochs):
#   for m in range(steps_per_epoch):
#     step += 1
#     train_step(image)
#     print(".", end='')
#   display.clear_output(wait=True)
#   util.visualize_img(image.read_value())
#   plt.title("Train step: {}".format(step))
#   plt.show()

# end = time.time()
# print("Total time: {:.1f}".format(end-start))