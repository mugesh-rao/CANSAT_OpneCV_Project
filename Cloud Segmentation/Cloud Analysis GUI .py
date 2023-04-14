import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()


file_paths = filedialog.askopenfilenames()


nrows = int(np.ceil(len(file_paths) / 2))
fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 10))

# Loop over each image path and display the image in the corresponding axes
for i, file_path in enumerate(file_paths):
    # Load the image
    img = cv2.imread(file_path)

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

    # Display the segmented image in the corresponding axes
    row_idx = int(np.floor(i / 2))
    col_idx = i % 2
    axs[row_idx, col_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[row_idx, col_idx].axis('off')

plt.show()
