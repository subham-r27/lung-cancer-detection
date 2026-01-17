import os
import random
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ===== PATHS =====
DATASET_PATH = "datasets/val"   # use validation as test
MODEL_PATH = "models/cnn-100epochs.h5"

# ===== LOAD MODEL =====
cnn = load_model(MODEL_PATH)

# ===== CLASS LABELS (AUTO-SAFE) =====
labels = list(cnn.class_names) if hasattr(cnn, "class_names") else ["Benign", "Malignant", "Normal"]

# ===== PICK RANDOM IMAGE =====
random_class = random.choice(os.listdir(DATASET_PATH))
class_path = os.path.join(DATASET_PATH, random_class)

random_image = random.choice(os.listdir(class_path))
image_path = os.path.join(class_path, random_image)

# ===== PREPROCESS IMAGE =====
test_image = image.load_img(image_path, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)

# ===== PREDICTION =====
result = cnn.predict(test_image)
predicted_index = np.argmax(result)
predicted_class = labels[predicted_index]

# ===== OUTPUT =====
print("Image path:", image_path)
print("Probabilities:", result)
print("Predicted class:", predicted_class)
