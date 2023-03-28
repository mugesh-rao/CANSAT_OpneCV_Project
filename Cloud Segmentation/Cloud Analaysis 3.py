import cv2
import numpy as np
import matplotlib.pyplot as plt

# This Segement Images of Clouds and field and Mountains

img = cv2.imread('C:/Mugesh Rao/JIT/CANSAT/CANSAT_OpneCV_Project/Clouds_img/cloud6.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:  # adjust the threshold to filter smaller contours
        filtered_contours.append(cnt)

# Create a blank mask for the clouds
mask_clouds = np.zeros_like(thresh)

# Create a blank mask for the mountains
mask_mountains = np.zeros_like(thresh)

# Identify the contours for clouds and mountains
for cnt in filtered_contours:
    # Find the bounding box of the contour
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    
    # If the aspect ratio of the bounding box is greater than a threshold, it's likely a mountain
    if aspect_ratio > 2:
        cv2.drawContours(mask_mountains, [cnt], -1, 255, -1)
    # Otherwise, it's likely a cloud
    else:
        cv2.drawContours(mask_clouds, [cnt], -1, 255, -1)

# Apply morphological operations to remove noise and fill gaps in the clouds
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(mask_clouds, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Draw the filtered contours on the original image
img_masked = cv2.bitwise_and(img, img, mask=mask_mountains)
img_masked = cv2.bitwise_or(img_masked, img, mask=closing)

# Display the segmented image
plt.imshow(img_masked)
plt.show()
