import tensorflow as tf
from tensorflow import keras
import os
import random
import numpy as np
import PIL
from tqdm import tqdm
import pathlib
from pathlib import Path
from PIL import Image
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef

seed = 42
np.random.seed = seed

img_width = 256
img_height = 256
batch_size=32
img_channels = 3
epochs=10
brainimages = os.path.join (r"C:\Users\Nasir\PycharmProjects\my\Braindata\pics")
brainmasks = os.path.join (r"C:\Users\Nasir\PycharmProjects\my\Braindata\masks")

brainimages=pathlib.Path(brainimages)
brainmasks=pathlib.Path(brainmasks)

imagelen=len(os.listdir(brainimages))
imaglist=os.listdir(brainimages)
count=0
pic=[]
mask=[]
for images in glob.iglob(f'{brainimages}/*'):


    if (images.endswith(".png")):
        count += 1

        pic.append(images)
print('Total number of images are:', count)

for images in glob.iglob(f'{brainmasks}/*'):


    if (images.endswith(".png")):
        mask.append(images)
training_images= np.zeros((count, img_height, img_width, img_channels))
masked_images = np.zeros((count, img_height, img_width, 1))
for i in range (count):

    x = cv2.imread(pic[i], cv2.IMREAD_COLOR)

    img=x
    newimage=resize(img,(img_width, img_height))

    training_images[i]=newimage
    img2 = cv2.imread(mask[i], cv2.IMREAD_GRAYSCALE)
    img2 = np.expand_dims(img2, axis=-1)
    img2= resize(img2,(img_width, img_height))
    masked_images[i]=img2



train_x, valid_x = train_test_split(training_images, test_size=0.2, random_state=42)
train_y, valid_y = train_test_split(masked_images, test_size=0.2, random_state=42)


inputs = tf.keras.layers.Input((img_height, img_width, img_channels))
normalized = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(normalized)
tf.keras.layers.BatchNormalization()
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
tf.keras.layers.BatchNormalization()
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
tf.keras.layers.BatchNormalization()
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
tf.keras.layers.BatchNormalization()
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
tf.keras.layers.BatchNormalization()
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
tf.keras.layers.BatchNormalization()
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
tf.keras.layers.BatchNormalization()
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
tf.keras.layers.BatchNormalization()
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
tf.keras.layers.BatchNormalization()
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

history= model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

test_loss, test_acc = model.evaluate(valid_x,  valid_y, verbose=1)



test_img_number = np.random.randint(0, len(valid_x))
x= valid_x[test_img_number]
test_img_input  = np.expand_dims(x, 0)
prediction = np.squeeze(model.predict(test_img_input))



imshow(prediction)
plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

test_img_number = np.random.randint(0, len(valid_x))
x= valid_x[test_img_number]
test_img_input  = np.expand_dims(x, 0)
prediction = np.squeeze(model.predict(test_img_input))

print('The test acc is:',test_acc)


plt.figure(figsize=(16, 16))
ax = plt.subplot(1, 3, 1)
imshow(x)
plt.title('Actual Test Image')
plt.axis("off")
ax = plt.subplot(1, 3, 2)
imshow(valid_y[test_img_number])
plt.title('Actual Mask')
plt.axis("off")
ax = plt.subplot(1, 3, 3)
imshow(prediction)
plt.title('Predicted segmentation')
plt.axis("off")
plt.show()

for x, y in (valid_x, valid_y):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (img_width, img_height))
    image = np.expand_dims(image, axis=0)

    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_width, img_height))
    x= np.expand_dims(x, 0)
    y_pred = np.squeeze(model.predict(x))
    y_pred = y_pred >= 0.1
    y_pred = y_pred.flatten()

    mask = (mask > 0.1).flatten()


f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
