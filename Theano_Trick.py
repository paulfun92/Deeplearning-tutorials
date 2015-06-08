import numpy
import theano
import theano.tensor as T
import pylab
from PIL import Image

x = T.dscalar('x')

y = 2*x
f = theano.function([x],y)

a = [f(i) for i in xrange(10)]
print(a)
print(numpy.mean(a))


# Reshape an array in numpy example:
x = numpy.array([1,2,3])
print x.reshape(1,3)
print x.reshape(3)
print x.reshape(1,1,3)

# Take an image input as a 3 dimensional array: height*width*RGB layer (entry in the matrix is pixel RGB value)
# open random image of dimensions 639x516
img = Image.open(open('3wolfmoon.jpg'))
#img.show()
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64')
print img.shape