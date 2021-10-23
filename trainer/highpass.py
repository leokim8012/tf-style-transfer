

def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var


# x_deltas, y_deltas = high_pass_x_y(content_image)

# plt.figure(figsize=(14,10))
# plt.subplot(2,2,1)
# imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

# plt.subplot(2,2,2)
# imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

# x_deltas, y_deltas = high_pass_x_y(image)

# plt.subplot(2,2,3)
# imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

# plt.subplot(2,2,4)
# imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")



# plt.figure(figsize=(14,10))

# sobel = tf.image.sobel_edges(content_image)
# plt.subplot(1,2,1)
# imshow(clip_0_1(sobel[...,0]/4+0.5), "Horizontal Sobel-edges")
# plt.subplot(1,2,2)
# imshow(clip_0_1(sobel[...,1]/4+0.5), "Vertical Sobel-edges")