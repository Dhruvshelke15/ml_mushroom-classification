# proj2_test.py - Evaluate the mushroom classifier on test set
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score

IMG_SIZE = (160, 160)

# Load the trained model
model = tf.keras.models.load_model("mushroom_model.h5")

# Load test image filenames and true labels
test_folder = "mushroom_test"
csv_path = "mushroom_test.csv"
df = pd.read_csv(csv_path)

# Map class names to indices (same as training order)
class_names = sorted(['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma',
                      'Hygrocybe', 'Lactarius', 'Russula', 'Suillus'])
class_to_index = {name: i for i, name in enumerate(class_names)}

# Prepare test data
true_labels = []
predicted_labels = []

for i, row in df.iterrows():
    img_path = os.path.join(test_folder, row['file'])
    label_str = row['label']
    label_idx = class_to_index[label_str]
    true_labels.append(label_idx)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    predicted_labels.append(np.argmax(pred))

# Calculate accuracy
acc = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {acc * 100:.2f}%")
