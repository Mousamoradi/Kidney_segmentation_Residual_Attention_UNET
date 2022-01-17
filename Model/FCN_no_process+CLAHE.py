# -*- coding: utf-8 -*-
"""
This Code developed by Mousa Moradi, Ph.D student of BME, UMASS Amherst.
Last Checkpoint: 02/18/2021
Last Update: 01/14/2022

"""

# For image processing opencv>4.4 is needed. (Tensorflow cpu only)
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices() \
      else "NOT AVAILABLE")
tf.config.list_physical_devices()
# Importing the required library and modules
import os
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#plt.style.use("ggplot")
%matplotlib inline

import cv2
from PIL import Image,ImageDraw,ImageFont
import sklearn as sk
from tqdm import tqdm
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Loading cropped Images before image processing from directory
X_data=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\split\x\*.jpg')
y_data=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\split\y\*.jpg')

# If image processing with CLAHE
# Loading cropped Images
#X_data=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\split\x\*.jpg')
#y_data=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\split\y\*.jpg')

#Sanity checking
rows,cols=3,3
fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    fig.add_subplot(rows,cols,i)
    img_path=X_data[i]
    msk_path=y_data[i]
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    msk=cv2.imread(msk_path)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.55)
plt.show()
# Create dataframe
df = pd.DataFrame(data={"filename": X_data, 'mask' : y_data})
df_train, df_test = train_test_split(df,test_size = 0.1)
df_train, df_val = train_test_split(df_train,test_size = 0.2)
print(df_train.values.shape)
print(df_val.values.shape)
print(df_test.values.shape)
# Data Augmentation
def train_generator(data_frame, batch_size, aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

# Define segmentation metrics
smooth=100
def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Define Model
def create_model_fcn32(n, input_w, input_h):
   
    """
    This function will create a Fully Convolutional Network(FCN32)
    and returns the model for training.

    -------------------------------------
    Arguments:
        n: number of classes to be detected
        input_w: width of the image
        input_h: height of the image
    --------------------------------------    
    """

    input = Input(shape=(input_w, input_h, 3))

    # initialize feature extractor excuding fully connected layers
    # here we use VGG model, with pre-trained weights. 
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input)
    # create further network
    x = Conv2D(4096, kernel_size=(7,7), use_bias=False, 
               activation='relu', padding="same")(vgg.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, kernel_size=(1,1), use_bias=False, 
               activation='relu', padding="same")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(n, kernel_size=(1,1), use_bias=False,  
               padding="same")(x)
    # upsampling to image size
    x = Conv2DTranspose(n ,  kernel_size=(64,64), 
                        strides=(32,32), use_bias=False, padding='same')(x)
    x = Activation('sigmoid')(x)
    model = Model(input, x)
    return model
fcn = create_model_fcn32(1,256,256)
fcn.summary()
# Hyperprameters
EPOCHS = 50
BATCH_SIZE = 32
learning_rate = 1e-4
im_width = 256
im_height = 256

#early stopping callback
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
csv_logger = CSVLogger(r'C:\Users\MOUSAMORADI\Desktop\newtry\log\fcn_no_process_{}.log'.format(datetime.now().strftime("%Y_%m_%d-%H_%M")))
train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')
train_gen = train_generator(df_train, BATCH_SIZE,
                                train_generator_args,
                                target_size=(im_height, im_width))
    
test_gener = train_generator(df_val, BATCH_SIZE,
                                dict(),
                                target_size=(im_height, im_width))
    
model = create_model_fcn32(1, im_width, im_height)


decay_rate = learning_rate / EPOCHS
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
model.compile(optimizer=opt, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef,tf.keras.metrics.Recall(),
                                                           tf.keras.metrics.Precision()])

callbacks = [ModelCheckpoint('fcn_no_process.hdf5', verbose=1, save_best_only=True), es, csv_logger]

history = model.fit(train_gen,
                    steps_per_epoch=len(df_train) / BATCH_SIZE, 
                    epochs=EPOCHS, 
                    callbacks=callbacks,
                    validation_data = test_gener,
                    validation_steps=len(df_val) / BATCH_SIZE)
%matplotlib qt5

a = history.history

list_traindice = a['dice_coef']
list_testdice = a['val_dice_coef']

list_trainjaccard = a['iou']
list_testjaccard = a['val_iou']

list_trainloss = a['loss']
list_testloss = a['val_loss']
plt.figure(1)
plt.plot(list_testloss, 'b-')
plt.plot(list_trainloss,'r-')
plt.xlabel('iteration')
plt.ylabel('Dice Loss')
plt.title('Loss graph', fontsize = 15)
plt.figure(2)
plt.plot(list_traindice, 'r-')
plt.plot(list_testdice, 'b-')
plt.xlabel('iteration')
plt.ylabel('Dice Coefficient')
plt.title('Accuracy graph', fontsize = 15)
plt.show()

# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting
import pandas as pd    
fcn_history_df = pd.DataFrame(history.history) 
with open('fcn_history_df.csv', mode='w') as f:
    fcn_history_df.to_csv(f)
    
model = load_model('fcn_no_process.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
# Testing results
test_gen = train_generator(df_test, BATCH_SIZE,
                                dict(),
                                target_size=(im_height, im_width))
results = model.evaluate(test_gen, steps=len(df_test) / BATCH_SIZE)
print("Test lost: ",results[0])
print("Test IOU: ",results[1])
print("Test Dice Coefficent: ",results[2])
#overlay
%matplotlib qt5

for i in range(30):
    index=np.random.randint(1,len(df_test.index))
    img1 = cv2.imread(df_test['filename'].iloc[index])
    img = cv2.resize(img1 ,(im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,4,1)
    plt.imshow(np.squeeze(img1))
    plt.title('Original Image')
    plt.subplot(1,4,2)
    mask = cv2.imread(df_test['mask'].iloc[index])
    plt.imshow(mask)
    plt.title('Original Mask')
    plt.subplot(1,4,3)
    pre=np.squeeze(pred) > .5
    plt.imshow(pre)
    plt.title('Prediction Mask')
    pre=np.uint8(pre)
    pre=cv2.cvtColor(pre,cv2.COLOR_GRAY2RGB)
    plt.subplot(1,4,4)
    p=np.where(pre>0)
    m=np.where(mask>128)
    pre[p[0:2]]=[0, 255, 0]
    mask[m[0:2]]=[255, 0, 0]    
    dst1 = cv2.addWeighted(pre,1,mask,1,0)
    dst2 = cv2.addWeighted(img1,1,dst1,1,0)
    plt.imshow(dst2)
    plt.title('Overlay')
    plt.show()