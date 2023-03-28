import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('C:/Mugesh Rao/JIT/CANSAT/CANSAT_OpneCV_Project/Clouds_img/cloud4.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper color thresholds for clouds
lower_cloud = np.array([0, 0, 200])
upper_cloud = np.array([180, 50, 255])

# Create a binary mask for the clouds using the color thresholds
mask = cv2.inRange(hsv, lower_cloud, upper_cloud)

# Apply morphological operations to remove noise and fill gaps in the clouds
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours in the binary mask
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:  # adjust the threshold to filter smaller contours
        filtered_contours.append(cnt)

# Draw the filtered contours on the original image
cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 3)

# Display the segmented image
plt.imshow(img)
plt.show()
