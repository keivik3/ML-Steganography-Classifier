import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog

from sklearn.impute import SimpleImputer

def extract_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    hist_hue = cv2.calcHist([hue_channel], [0], None, [256], [0, 256])
    hist_hue /= hist_hue.sum()
    feature_hist_hue = hist_hue.flatten()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True)

    brightness_mean = np.mean(gray_image)
    brightness_std = np.std(gray_image)
    saturation_mean = np.mean(hsv_image[:, :, 1])
    saturation_std = np.std(hsv_image[:, :, 1])

    max_length = 1000

    feature_hist_hue_resized = cv2.resize(feature_hist_hue, (max_length, 1))

    hog_features_resized = cv2.resize(hog_features, (max_length, 1))

    brightness_mean_resized = np.full((max_length, 1), brightness_mean)
    brightness_std_resized = np.full((max_length, 1), brightness_std)
    saturation_mean_resized = np.full((max_length, 1), saturation_mean)
    saturation_std_resized = np.full((max_length, 1), 0.0)  # Specify the fill value

    all_features = np.concatenate((feature_hist_hue_resized.flatten(), hog_features_resized.flatten(),
                                   brightness_mean_resized.flatten(), brightness_std_resized.flatten(),
                                   saturation_mean_resized.flatten(), saturation_std_resized.flatten()))

    return all_features


clean = 'clear'
clean_path = []

stego = 'stego_us'
stego_path = []

for filename in os.listdir(clean):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        file_path = os.path.join(clean, filename)
        clean_path.append(file_path)

for filename in os.listdir(stego):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        file_path = os.path.join(stego, filename)
        stego_path.append(file_path)


clean_features = [extract_features(cv2.imread(image_path)) for image_path in clean_path]

stego_features = [extract_features(cv2.imread(image_path)) for image_path in stego_path]


X_train = np.vstack(clean_features + stego_features)


y_train = ['clean'] * len(clean_features) + ['stego'] * len(stego_features)

k = 3
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(X_train, y_train)

result = 'yo'
for filename in os.listdir(result):
    if filename.endswith('.jpg') or filename.endswith('.png')or filename.endswith('.jpeg'):
        file_path = os.path.join(result, filename)
        new_image = cv2.imread(file_path)
        new_features = extract_features(new_image)
        predicted_class = classifier.predict([new_features])[0]
        #print("Predicted class:", predicted_class)

image_path = input("Enter the path to the image: ")
new_image = cv2.imread(image_path)
new_features = extract_features(new_image)
predicted_class = classifier.predict([new_features])[0]
print("Predicted class:", predicted_class)