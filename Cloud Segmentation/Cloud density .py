import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# Satellite cloud images Classification for Environmental Analysis using Image Processing with Python Opencv


# Load the cloud images and their corresponding labels

cloud_images = []
labels = []

for i in range(1, 11):
    image = cv2.imread('C:/Mugesh Rao/JIT/CANSAT/CANSAT_OpneCV_Project/Clouds_img/cloud4.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cloud_images.append(image_gray)
    if i <= 5:
        labels.append(0)  # Label 0 for clear skies
    else:
        labels.append(1)  # Label 1 for cloudy skies
# Preprocess the images by resizing them and flattening them into 1D arrays
resized_images = []
for image in cloud_images:
    resized_image = cv2.resize(image, (50, 50))
    resized_image = resized_image.flatten()
    resized_images.append(resized_image)

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    resized_images, labels, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors (KNN) classifier on the training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict the labels of the testing set and evaluate the accuracy of the classifier
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classify a new cloud image using the trained classifier
new_image = cv2.imread('C:/Mugesh Rao/JIT/CANSAT/CANSAT_OpneCV_Project/Clouds_img/cloud1.png')
new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
resized_new_image = cv2.resize(new_image_gray, (50, 50))
resized_new_image = resized_new_image.flatten()
prediction = knn.predict([resized_new_image])[0]
if prediction == 0:
    print('The new cloud image has clear skies.')
else:
    print('The new cloud image has cloudy skies.')
