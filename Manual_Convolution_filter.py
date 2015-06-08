__author__ = 'paul'

'''
This program can apply filters to an image
The current filter creates a blur effect.
'''

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy
import numpy as np
import pylab
from PIL import Image
import matplotlib.pyplot as plt

# open random image of dimensions 639x516
img = Image.open(open('photo3.bmp'))
img.show()
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 255.
img = img.transpose(2, 0, 1)

# put image in 4D tensor of shape (1, 3, height, width)
# img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)

filter1 = np.array([[[-10,-10,-10],[-10,8,-10],[-10,-10,-10]],
                    [[-10,-10,-10],[-10,8,-10],[-10,-10,-10]],
                    [[-10,-10,-10],[-10,8,-10],[-10,-10,-10]]])

filter2_ = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# Create RGB filter:
filter2 = np.array([filter2_,filter2_,filter2_])

shape_img = img.shape
shape_filter1 = filter1.shape
shape_filter2 = filter2.shape
# start_filter_i_pixel = filter1.shape[0]/2
# start_filter_j_pixel = filter1.shape[1]/2
# end_filter_i_pixel = img.

# Determine size of the output after convolution:
output_height = shape_img[1] - shape_filter2[1] + 1
output_width = shape_img[2] - shape_filter2[2] + 1

# Initialize the ouput of the convolution:
convolution_output = np.zeros((output_height, output_width))

#Now run a double loop to apply a filer:
for i in range(output_height):
    for j in range(output_width):
        pixel_dot_product = sum(sum(sum(np.multiply(filter2, img[:,i:i+shape_filter2[1],j:j+shape_filter2[2]]))))
        convolution_output[i,j] = pixel_dot_product

plt.gray()
plt.imshow(convolution_output)
plt.show()

