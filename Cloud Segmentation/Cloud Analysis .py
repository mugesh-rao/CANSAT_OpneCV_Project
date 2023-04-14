import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('C:/Mugesh Rao/JIT/CANSAT/CANSAT_OpneCV_Project/Clouds_img/cloud4.jpg')


# Apply Gaussian blur to remove noise and smooth the edges
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Convert the image to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image to binarize it
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Display the segmented image
plt.imshow(img)
plt.show()
