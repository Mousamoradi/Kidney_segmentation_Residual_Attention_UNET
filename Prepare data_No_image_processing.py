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
# Loading original Images from directory- If no image processing
data_x=glob(r'C:\Users\MOUSAMORADI\kidney_dataset\ori\*.jpg')
data_y=glob(r'C:\Users\MOUSAMORADI\kidney_dataset\kidney\*.jpg')
# resize to 1536*256
for i,j in tqdm(zip(data_x,data_y)):
    Image.open(i).resize((1536,256)).save(r'C:\Users\MOUSAMORADI\Desktop\newtry\resized\x\{}'.format(os.path.basename(i)))
    Image.open(j).resize((1536,256)).save(r'C:\Users\MOUSAMORADI\Desktop\newtry\resized\y\{}'.format(os.path.basename(j)))
# Loading resized Images
data_x_res=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\resized\x\*.jpg')
data_y_res=glob(r'C:\Users\MOUSAMORADI\Desktop\newtry\resized\y\*.jpg')
# Create dataframe
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
            cv2.imwrite(os.path.join('C:/Users/MOUSAMORADI/Desktop/newtry/split/x','org' + str(j) + '.jpg'), t1)
            cv2.imwrite(os.path.join('C:/Users/MOUSAMORADI/Desktop/newtry/split/y', 'mask' + str(j) + '.jpg'), m1)
            j=j+1
split()




