# -*- coding: utf-8 -*-
"""
Spyder Editor

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
from PIL import Image
import cv2
import matplotlib.pyplot as plt
#from keras.preprocessing.image import ImageDataGenerator
import glob
import os
import sys
import numpy as np
from tqdm import tqdm
%matplotlib qt5
from shutil import copyfile
from skimage import morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import label
from PIL import Image,ImageDraw,ImageFont
import pandas as pd 
from datetime import datetime
#from keras import utils as np_utils
from tensorflow.keras import utils as np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib qt5
plt.style.use("dark_background")

# Loading original Images from directory
X_data=glob.glob(r'C:\Users\MOUSAMORADI\kidney_dataset\ori\*.jpg') #Kidney images
y_data=glob.glob(r'C:\Users\MOUSAMORADI\kidney_dataset\kidney\*.jpg')# Groudtruth

df = pd.DataFrame()
df['train'] = X_data
df['ground_truth'] = y_data
df.head()
# Number of images with their corresponding groundtruth.
len(df)
# check whether all the images have it's corresponding ground_truth
len(df['train']) == len(df['ground_truth'])
# Load annotation
ann=pd.read_csv('patientNames.csv')
print(ann.shape)
print(ann.size)
ann.head()
# The ground_truth image has area covered in white where the PCT lumen is present and black elsewhere.
i = cv2.imread(df['ground_truth'].iloc[1])
plt.imshow(i)
%%time
import time

df['diag'] = 3
for i in range(len(df)):
  img = df['ground_truth'].iloc[i]
  value = np.max(cv2.imread(img))
  if value > 0: 
    df['diag'].iloc[i] = 1
  else: 
    df['diag'].iloc[i] = 0
df.head()

df['diag'].value_counts().plot(kind='bar')
# We have a total of 14497 images with segmented PCT lumen and 2 images with no segmented PCT lumen.
df['diag'].value_counts()

# Visualizing- see original/ground_truth and segmented area
fig=plt.figure(figsize=(10,10))

#plotting the image
fig.add_subplot(1,3,1)
img = cv2.imread(df['train'].iloc[125])
plt.imshow(img)

#plotting the ground truth
fig.add_subplot(1,3,2)
msk = cv2.imread(df['ground_truth'].iloc[125])
plt.imshow(msk)

#plotting the mask overlayed on the image
fig.add_subplot(1,3,3)
#identify the edges using the CannyEdgeDetector
edged = cv2.Canny(msk, 10, 250)
#find the contours on the the image(edge detected)
(cnts, _)= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#draw the contours
for cnt in cnts:
    cv2.drawContours(img,[cnt],0,(255,0,0),2)
plt.imshow(img)

# Image Enhancement using CLAHE+ OTSU thresholding
def enh_img():
    for i in tqdm(range(len(df))):
        img = cv2.imread(df['train'].iloc[i])
        mask = cv2.imread(df['ground_truth'].iloc[i])

        #set the clip value and the gridsize changing these values will give different output
        clahe = cv2.createCLAHE(clipLimit=6, \
                                tileGridSize=(16,16))
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ1=clahe.apply(grayimg)
        ret,thre1=cv2.threshold(equ1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Set aditional threshold by OTSU method
        equ = cv2.cvtColor(thre1,cv2.COLOR_BGR2RGB)
        imgc = grayimg.copy()

        #Now we can now crop the image.
        #to crop the image we'll first find the edges in the image.
        edged = cv2.Canny(grayimg, 10, 250)

        #Once we have the edges now we'll perform dilate operation to remove any small regions of noises
        dilate = cv2.dilate(edged, None, iterations=1)

        #Now we can find the contours in the image
        (cnts, thres) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #for c in cnts:
             #peri = cv2.arcLength(c, True)
             #eps = 0.01 * peri
             #approx = cv2.approxPolyDP(c, eps, True)
             #cv2.drawContours(imgc, [approx], -1, (0, 255, 0), 2)

#Now we'll get the extreme points in the contours and crop the image
#https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_image = equ[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    
        cv2.imwrite(os.path.join('C:/Users/MOUSAMORADI/Mydata/train', 'train' + str(i) + '.jpg'), new_image)
        cv2.imwrite(os.path.join('C:/Users/MOUSAMORADI/Mydata/groundtruth', 'mask' + str(i) + '.jpg'), mask)    
enh_img()
# Loading enhanced Images
data_x_enh=glob(r'C:\Users\MOUSAMORADI\Mydata\train\*.jpg')
data_y_enh=glob(r'C:\Users\MOUSAMORADI\Mydata\groundtruth\*.jpg')
# resize to 1536*256
for i,j in tqdm(zip(data_x_enh,data_y_enh)):
    Image.open(i).resize((1536,256)).save(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\resized\x\{}'.format(os.path.basename(i)))
    Image.open(j).resize((1536,256)).save(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\resized\y\{}'.format(os.path.basename(j)))
# Loading resized Images
data_x_res=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\resized\x\*.jpg')
data_y_res=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\resized\y\*.jpg')

df = pd.DataFrame()
df['train'] = data_x_res
df['ground_truth'] = data_y_res
df.head()
# # Splitting into 6 sets (1536>>6*256)

def split():
    j=0
    for i in tqdm(range(len(df))):
        img = cv2.imread(df['train'].iloc[i])
        mask = cv2.imread(df['ground_truth'].iloc[i])
        gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_msk = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        height,width= gray_im.shape
        CROP_W_SIZE=6
        for iw in range(CROP_W_SIZE ):
            x = width // CROP_W_SIZE*iw
            w = (width //CROP_W_SIZE )
            t1 = gray_im[:, x:x+256]
            m1= gray_msk[:, x:x+256]
            cv2.imwrite(os.path.join('C:/Users/MOUSAMORADI/Desktop/newtry/enh/split/x','org' + str(j) + '.jpg'), t1)
            cv2.imwrite(os.path.join('C:/Users/MOUSAMORADI/Desktop/newtry/enh/split/y', 'mask' + str(j) + '.jpg'), m1)
            j=j+1
split()
# Loading cropped Images
X_data=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\split\x\*.jpg')
y_data=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\enh\split\y\*.jpg')

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






