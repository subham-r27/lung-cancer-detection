import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

dataset_path = "datasets"

# -------- TRAIN DATA --------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    f"{dataset_path}/train",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

# -------- VALIDATION DATA --------
val_datagen = ImageDataGenerator(rescale=1./255)

val_set = val_datagen.flow_from_directory(
    f"{dataset_path}/val",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

# -------- CNN MODEL --------
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(128, 128, 3)))
cnn.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D(2, 2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(3, activation="softmax"))

cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"]
)

cnn.summary()

cnn.fit(
    training_set,
    validation_data=val_set,
    epochs=50
)

print("Class indices:", training_set.class_indices)

# -------- SAVE MODEL --------
cnn.save("models/cnn_model.keras")
