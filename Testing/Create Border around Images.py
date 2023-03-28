# Python program to explain cv2.copyMakeBorder() method

# importing cv2
import cv2
from cv2 import resize

# path
path = r'C:\Mugesh Rao\JIT\CANSAT\New folder\Python-Learning\CANSAT\neww.png'

# Reading an image in default mode
image = cv2.imread(path)
resizer =cv2.resize(image, (1000, 610))

# Window name in which image is displayed
window_name = 'Image'

# Using cv2.copyMakeBorder() method
resizer = cv2.copyMakeBorder(resizer, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)

# Displaying the image
cv2.imshow(window_name, resizer)
cv2.waitKey(0)

