import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

image_size = (224, 224)

data = []
labels = []

dataset_dir = 'D:/Projects/Dataset'

for denomination in os.listdir(dataset_dir):
    denomination_path = os.path.join(dataset_dir, denomination)
    label = int(denomination)

    for image_file in os.listdir(denomination_path):
        image_path = os.path.join(denomination_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        image = image / 255.0
        data.append(image)
        labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

output_dir_test = 'D:/Projects/TEST and train/test'
output_dir_train = 'D:/Projects/TEST and train/train'

os.makedirs(output_dir_test, exist_ok=True)
os.makedirs(output_dir_train, exist_ok=True)

for i, image in enumerate(X_train):
    filename = os.path.join(output_dir_train, f'image_{i}.png')
    cv2.imwrite(filename, image)

for i, image in enumerate(X_test):
    filename = os.path.join(output_dir_test, f'image_{i}.png')
    cv2.imwrite(filename, image)
