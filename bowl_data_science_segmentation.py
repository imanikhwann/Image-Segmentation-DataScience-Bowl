# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:16:21 2022

@author: owner
"""

import tensorflow as tf
import numpy as np
import os
from scipy import io

file_directory_train = r"C:\Users\owner\Desktop\SHRDC MIDA AIML\Deep Learning\Git Repo\image_segmentation_bowl\data-science-bowl-2018-2\train"
file_directory_test = r"C:\Users\owner\Desktop\SHRDC MIDA AIML\Deep Learning\Git Repo\image_segmentation_bowl\data-science-bowl-2018-2\test"
#%%
import cv2

train_images = []
train_masks = []
test_images = []
test_masks = []

train_image_dir = os.path.join(file_directory_train, "inputs")
for train_image in os.listdir(train_image_dir):
    img = cv2.imread(os.path.join(train_image_dir, train_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    train_images.append(img)
    
train_masks_dir = os.path.join(file_directory_train, "masks")
for train_mask in os.listdir(train_masks_dir):
    mask = cv2.imread(os.path.join(train_masks_dir, train_mask), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128,128))
    train_masks.append(mask)
#%%

test_image_dir = os.path.join(file_directory_test, 'inputs')
for test_image in os.listdir(test_image_dir):
    img = cv2.imread(os.path.join(test_image_dir, test_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    test_images.append(img)
    
test_masks_dir = os.path.join(file_directory_test, "masks")
for test_mask in os.listdir(test_masks_dir):
    mask = cv2.imread(os.path.join(test_masks_dir, test_mask), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128,128))
    test_masks.append(mask)

#%%

train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)

test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3, i)
    img_plot = train_images[i]
    plt.imshow(img_plot)
    plt.axis("off")
plt.show()

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3, i)
    mask_plot = train_masks[i]
    plt.imshow(mask_plot, cmap="gray")
    plt.axis("off")
plt.show()

#%%

train_masks_np_exp = np.expand_dims(train_masks_np, axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np, axis=-1)
#%%

train_conv_masks = np.round(train_masks_np_exp/255)
train_conv_masks = 1 - train_conv_masks

test_conv_masks = np.round(test_masks_np_exp/255)
test_conv_masks = 1 - test_conv_masks

#%%

train_conv_images = train_images_np/255.0
test_conv_images = test_images_np/255.0

#%%

train_x = tf.data.Dataset.from_tensor_slices(train_conv_images)
test_x = tf.data.Dataset.from_tensor_slices(test_conv_images)
train_y = tf.data.Dataset.from_tensor_slices(train_conv_masks)
test_y = tf.data.Dataset.from_tensor_slices(test_conv_masks)

#%%

train = tf.data.Dataset.zip((train_x, train_y))
test = tf.data.Dataset.zip((test_x, test_y))
#%%

BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEP_PER_EPOCH = 800 // BATCH_SIZE
VALIDATION_STEPS = 200 // BATCH_SIZE
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size = AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#%%
from tensorflow_examples.models.pix2pix import pix2pix

base_model = tf.keras.applications.MobileNetV2(input_shape = [128,128,3], include_top = False)

layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 2
model = unet_model(output_channels= OUTPUT_CLASSES)

#%%

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.summary()
#%%
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()
    
for images, masks in train.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])
    
#%%
from IPython.display import clear_output

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)[0]])
        else:
            display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis,...]))[0]])
            
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print("\n Sample prediction after epoch {}\n".format(epoch+1))
        
import datetime

log_dir = r"C:\Users\owner\Desktop\SHRDC MIDA AIML\Deep Learning\TensorFlow\Tensorboard\day_last\day_last_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1, profile_batch = 0)


EPOCH = 20

history = model.fit(train, epochs = EPOCH, steps_per_epoch = STEP_PER_EPOCH, validation_steps= VALIDATION_STEPS, validation_data= test, callbacks=[DisplayCallback(), tb_callback]) 

#%%

show_predictions(test, 3)
#%% 