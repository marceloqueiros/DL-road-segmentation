"""
Road segmentation using Deep Learning.

Author: bbarbosa
Date: 22-03-2018

Network architecture
https://devblogs.nvidia.com/solving-spacenet-road-detection-challenge-deep-learning/

Model trained with data from
https://www.cs.toronto.edu/~vmnih/data/
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import time
#import platform
import argparse
sys.path.append('/usr/local/lib/python3.5/site-packages')
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
import signal
import keras
#from keras_tqdm import TQDMNotebookCallback
from keras.layers import Input
from keras.optimizers import Adam, RMSprop
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.constraints import maxnorm 
from keras.optimizers import SGD 
from keras.utils import np_utils 
from keras import backend as K 
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import keras
K.set_image_dim_ordering('tf') #ordem 'th' ou 'tf' 
import numpy as np
from numpy import genfromtxt
import math 

from timeit import default_timer as timer
from time import time as tick
import matplotlib.pyplot as plt 
import pickle 
from os import listdir
from PIL import Image, ImageOps
from os.path import isfile, join
import os
from scipy.misc	import toimage 
from scipy import misc, ndimage
import scipy.fftpack as pack
import scipy.misc
from scipy.ndimage import rotate
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize

from sklearn.metrics import roc_auc_score
import pandas as pd
import pathlib

from keras.layers import (Activation, Conv2D, Conv2DTranspose, Dense, Dropout,
                          Flatten, Input, MaxPooling2D, concatenate,
                          GlobalAveragePooling2D)

# fixar random seed para se puder reproduzir os resultados 
seed = 1 
np.random.seed(seed)










# argument parser
custom_formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position=2000)
parser = argparse.ArgumentParser(description="Road segmentation using Deep Learning.", 
                                 prefix_chars='-',
                                 formatter_class=custom_formatter_class) 
# optional arguments
parser.add_argument("--test_dir", required=False, help="Path to testing images", type=str)
parser.add_argument("--model_path", required=False, help="Path to trained model", type=str)
parser.add_argument("--width", required=False, help="Images width (default=256)", default=256, type=int)
parser.add_argument("--height", required=False, help="Images height (default=256)", default=256, type=int)
parser.add_argument("--save_path", required=False, help="Save output directory", type=str)
parser.add_argument("--delay", required=False, help="Delay between frames (default=0)", default=0, type=int)

# parse arguments
args = parser.parse_args()

# print commands line arguments
print(args, "\n")

# images properties
HEIGHT   = args.height
WIDTH    = args.width
CHANNELS = 3

# ////////////////////////////////////////////////
#                 Static paths
# ////////////////////////////////////////////////
if(args.test_dir is None): args.test_dir = "./images/"
if(args.model_path is None): args.model_path = "./model/unet_road_best_best.h5" 
# ////////////////////////////////////////////////


# ////////////////////////////////////////////////
#           Try to load test images
# ////////////////////////////////////////////////
try: 
    print("[INFO] Testing images folder:", args.test_dir)
    X, _ = image_dirs_to_samples(args.test_dir, resize=(WIDTH, HEIGHT), filetypes=[".png", ".jpg", ".bmp"])

    # images / ground truth split
    split = len(X) // 2
    Xim = X[:split]
    OXgt = X[split:] 

    # convert ground-truth images from GS to RGB
    OXgt = np.array(OXgt)
    OXgt = np.reshape(OXgt, (-1, HEIGHT, WIDTH, 1))
    Xgt = []

    for index, elem in enumerate(OXgt):
        Xgt.append(cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB))
    
    Xgt = np.array(Xgt)
    Xgt = np.reshape(Xgt, (-1, HEIGHT, WIDTH, 3))
    
    print("[INFO] Images: ", len(Xim), Xim[0].shape)
    print("[INFO] Ground: ", len(Xgt), Xgt[0].shape)
    print("[INFO] Testing images loaded!\n", "green")

except Exception as load_data_exception:
    print("[EXCEPTION] %s" % load_data_exception, "red")
    sys.exit("[EXCEPTION] Error loading test images!")

# computational resources definition
tflearn.init_graph(num_cores=4)

# ////////////////////////////////////////////////
#       Network architecture definition
# ////////////////////////////////////////////////
network = input_data(shape=[None, HEIGHT, WIDTH, CHANNELS])

# build tiramisu segmentation architecture
n=5
Ni=8

#Pool1
network_1 = conv_2d(network, Ni, 3, regularizer='L2', weight_decay=0.0001) # 256
network_1 = residual_block(network_1, n, Ni)
network_1 = residual_block(network_1, 1, Ni)     
pool_1 = max_pool_2d(network_1, 2)              # downsampling 2x - 128
#Pool2
network_2 = residual_block(pool_1, n-1, 2*Ni)           
network_2 = residual_block(network_2, 1, Ni*2)   
pool_2 = max_pool_2d(network_2, 2)              # downsampling 4x - 64
#Pool3
network_3 = residual_block(pool_2, n-1, 4*Ni)           
network_3 = residual_block(network_3, 1, Ni*4)   
pool_3 = max_pool_2d(network_3, 2)              # downsampling 8x - 32
#Pool4
network_4 = residual_block(pool_3, n-1, 8*Ni)           
network_4 = residual_block(network_4, 1, Ni*8)   
pool_4 = max_pool_2d(network_4, 2)              # downsampling 16x - 16
#Pool5
network_5 = residual_block(pool_4, n-1, 16*Ni)           
network_5 = residual_block(network_5, 1, 16*Ni)

# --------------------------------------------------

Unpool1 = conv_2d_transpose(network_5, 8*Ni, 3, strides=2, output_shape=[HEIGHT // 8, WIDTH // 8, 8*Ni]) 
merge1 = merge([Unpool1, network_4], mode='concat', axis=3) # merge 
merge1 = conv_2d(merge1, 8*Ni, 3, activation='relu')           
merge1 = conv_2d(merge1, 8*Ni, 3, activation='relu')   #

Unpool2 = conv_2d_transpose(merge1, 4*Ni, 3, strides=2, output_shape=[HEIGHT // 4, WIDTH // 4, 4*Ni]) 
merge2 = merge([Unpool2, network_3], mode='concat', axis=3) # merge 
merge2 = conv_2d(merge2, 4*Ni, 3, activation='relu')           
merge2 = conv_2d(merge2, 4*Ni, 3, activation='relu')   

Unpool3 = conv_2d_transpose(merge2, 2*Ni, 3, strides=2, output_shape=[HEIGHT // 2, WIDTH // 2, 2*Ni]) 
merge3 = merge([Unpool3, network_2], mode='concat', axis=3) # merge 
merge3 = conv_2d(merge3, 2*Ni, 3, activation='relu')
merge3 = conv_2d(merge3, 2*Ni, 3, activation='relu')
        
Unpool4 = conv_2d_transpose(merge3, Ni, 3, strides=2, output_shape=[HEIGHT, WIDTH, Ni])
merge4 = merge([Unpool4, network_1], mode='concat', axis=3) # merge 
merge4 = conv_2d(merge4, Ni, 3, activation='relu')
merge4 = conv_2d(merge4, Ni, 3, activation='relu')
   
final_merge = conv_2d(merge4, 3, 1, activation='relu')

network = tflearn.regression(final_merge, optimizer='adam', loss='mean_square') 

# ////////////////////////////////////////////////
#             Create model object 
# ////////////////////////////////////////////////    
model = tflearn.DNN(network)

try:
    print("")
    print("[INFO] Loading pre-trained model for testing...")  
    model.load(args.model_path)
    print("[INFO] Model:", args.model_path)
    print("[INFO] Trained model loaded!\n") 
except Exception as load_model_exception:
    print("[EXCEPTION] %s" % load_model_exception)
    sys.exit("[EXCEPTION] Error loading test images!")

# image index
n_images = len(Xim)
image_id = 0

# flag to upsample prediction image show
upsample = False

spacing = 50 
cv2.namedWindow("Original")
cv2.moveWindow("Original", spacing, spacing)
cv2.namedWindow("Ground Truth")
cv2.moveWindow("Ground Truth", spacing + WIDTH + 20, spacing)
cv2.namedWindow("Predicted Mask")
cv2.moveWindow("Predicted Mask", spacing, spacing + HEIGHT + spacing)
cv2.namedWindow("Predicted")
cv2.moveWindow("Predicted", spacing + WIDTH + 20, spacing + HEIGHT + spacing)

# while loop to constantly load images 
try:
    
    while image_id < n_images:
        # start measuring time
        ctime = time.time()

        try:
            frame = Xim[image_id]
        except Exception as load_frame_exception:
            print("[EXCEPTION] %s" % load_frame_exception)
            print("[EXCEPTION] Error loading frame!")
            continue

        # reshape test image to NHWC tensor format
        test_image = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        copy_test_image = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        test_image = np.reshape(test_image, (1, HEIGHT, WIDTH, CHANNELS)) 

        try:
            # output mask prediction
            prediction = model.predict(test_image)
            prediction = np.reshape(prediction[0], (HEIGHT, WIDTH, 3))
        except Exception as predict_exception:
            print("[EXCEPTION] %s" % predict_exception)
            print("[EXCEPTION] Error loading frame!")
            continue

        # original image 
        original_bgr = cv2.cvtColor(copy_test_image, cv2.COLOR_RGB2BGR)

        print("[INFO] Show original", original_bgr.shape)
        cv2.imshow("Original", original_bgr)

        # predicted segmentation 
        try:
            prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
            prediction_bgr = np.absolute(prediction)
        except Exception as convert_pred_colorspace:
            print("[EXCEPTION] %s" % convert_pred_colorspace)
            print("[EXCEPTION] Prediction can't be converted to BGR colorspace!")
            prediction_bgr = prediction.copy()
            continue

        print("[INFO] Show predicted mask", prediction_bgr.shape)
        cv2.imshow("Predicted Mask", prediction)

        # ground truth
        try:
            gtruth = cv2.cvtColor(Xgt[image_id], cv2.COLOR_RGB2BGR)
        except Exception as convert_gt_colorspace:
            print("[EXCEPTION]  %s" % convert_gt_colorspace)
            print("[EXCEPTION] Ground truth can't be converted to BGR colorspace!")
            gtruth = Xgt[image_id]
        annotations = 0.5 * original_bgr + 0.5 * gtruth

        print("[INFO] Show ground truth", annotations.shape)
        cv2.imshow("Ground Truth", annotations)

        # prediction overlay 
        overlay = 0.5 * original_bgr + 0.5 * prediction_bgr
        if(upsample):
            overlay = cv2.resize(overlay, (WIDTH*2, HEIGHT*2), interpolation=cv2.INTER_CUBIC)

        print("[INFO] Show predicted", overlay.shape)
        cv2.imshow("Predicted", overlay)

        if(args.save_path is not None):
            cv2.imwrite("%soutput-%d.png" % (args.save_path, args.image_id), overlay)

        ctime = time.time() - ctime

        print("[INFO] Image %d of %d | Time %.3f seconds" % (image_id+1, n_images, ctime), end='\n\n')

        image_id += 1

        key = cv2.waitKey(args.delay)
        if(key == 27):
            # pressed ESC
            print("[INFO] Pressed ESC")
            break

except KeyboardInterrupt:
    print("[INFO] Pressed 'Ctrl+C'")

print("\n[INFO] All done!\a")
