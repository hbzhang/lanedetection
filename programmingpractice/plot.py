import numpy as np
import cv2
import matplotlib.pyplot as plt
a = np.array([[[10,10], [100,10], [10,100], [100,100]]], dtype=np.int32)
#im = np.zeros([100.100], dtype=np.uint8)
im = np.zeros([240,320],dtype=np.uint8)
cv2.fillPoly(im, a, 255)
plt.imshow(im)
plt.show()
