# Lightweight Transfer Learning Implementation
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications 
import VGG16, ResNet50
from tensorflow.keras.layers import (
 Input,Dense,Concatenate,
 GlobalAveragePooling2D,Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load CIFAR10 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Reduce Dataset Size (IMPORTANT)
x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]
# Resize Images to Smaller Size
x_train = tf.image.resize(x_train, (128,128))
x_test = tf.image.resize(x_test, (128,128))
# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0
# Convert Labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Input Layer
input_layer = Input(shape=(128,128,3))
# a. Feature Extraction using 2 Pretrained Models
base_model1 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
base_model2 = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze Models
model1.trainable = False
model2.trainable = False
# Extract Features
feat1 = GlobalAveragePooling2D()(model1.output)
feat2 = GlobalAveragePooling2D()(model2.output)
# b. Neural Network using Extracted Features
combined = Concatenate()([feat1, feat2])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output)
# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# c. Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
# Train Model
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)
# d. Accuracy Score
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print("\nAccuracy Score :", accuracy)
# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("\nTest Loss :", loss)
print("Test Accuracy :", acc)
OUTPUT
Epoch 1/50
accuracy: 0.5455 - loss: 1.3217 - val_accuracy: 0.7030 - val_loss: 0.8662
Epoch 2/50
accuracy: 0.7103 - loss: 0.8309 - val_accuracy: 0.7560 - val_loss: 0.7380
Epoch 3/50
accuracy: 0.5548 - loss: 0.7066 - val_accuracy: 0.7630 - val_loss: 0.7052
Epoch 4/50
accuracy: 0.7832 - loss: 0.6327 - val_accuracy: 0.7220 - val_loss: 0.8017
Epoch 5/50
accuracy: 0.8175 - loss: 0.5190 - val_accuracy: 0.7640 - val_loss: 0.7094
Epoch 6/50
accuracy: 0.5295 - loss: 0.4805 - val_accuracy: 0.7590 - val_loss: 0.7366
Accuracy Score : 0.753
accuracy: 0.5530 - loss: 0.6904
