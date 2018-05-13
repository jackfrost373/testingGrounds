
import numpy as np
import scipy as sp
#import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import filters, transform


image = np.zeros((100,100),int)
np.fill_diagonal(image,5)
image[:,15] = 1
image[70,:] = 3

#plt.matshow(image)
#plt.show()

edges = filters.sobel(image)
plt.matshow(edges)
plt.show()

h, theta, d = transform.hough_line(image)




#from skimage.transform import (hough_line, hough_line_peaks,
                               #probabilistic_hough_line)
#from skimage.feature import canny
#from skimage import data

