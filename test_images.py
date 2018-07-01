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

import architectures
import os
import sys
import time
#import platform
import argparse
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
import signal
import keras
#from keras_tqdm import TQDMNotebookCallback
from keras.layers import (Activation, Conv2D, Conv2DTranspose, Dense, Dropout,
                          Flatten, Input, MaxPooling2D, concatenate,
                          GlobalAveragePooling2D, Input)
from keras.optimizers import Adam, RMSprop
from keras.models import load_model, Model, Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm 
from keras.optimizers import SGD 
from keras.utils import np_utils 
from keras import backend as K 
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import keras
from keras.utils.generic_utils import CustomObjectScope
K.set_image_dim_ordering('tf')
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
from sklearn.metrics import roc_auc_score
from skimage.io import imread
from skimage.transform import resize

import pandas as pd
import pathlib

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
if(args.model_path is None): args.model_path = "./model/unet.h5" 
if(args.width is None): args.width = 256
if(args.height is None): args.height = 256
if(args.delay is None): args.delay = 1
# ////////////////////////////////////////////////

##########################################################################
##########################################################################
##########################################################################
# ////////////////////////////////////////////////
#           Try to load test images
# ////////////////////////////////////////////////
try: 
    print("[INFO] Testing images folder:", args.test_dir)
    X = []

    for path, _, files in os.walk(args.test_dir + 'input/'):
        for file in files:
            X.append(np.array(Image.open(args.test_dir + 'input/' + file)))

    for path, _, files in os.walk(args.test_dir + 'output/'):
        for file in files:
            X.append(np.array(Image.open(args.test_dir + 'output/' + file)))
            
 
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
    print("[INFO] Testing images loaded!\n")

except Exception as load_data_exception:
    print("[EXCEPTION] %s" % load_data_exception)
    sys.exit("[EXCEPTION] Error loading test images!")

    
##########################################################################
##########################################################################
##########################################################################
try:
    print("[INFO] Loading pre-trained model for testing...") 
    with CustomObjectScope({
                    "relu6":
                    keras.applications.mobilenet.relu6,
                    "DepthwiseConv2D":
                    keras.applications.mobilenet.DepthwiseConv2D,
                    "acc_conf":
                    architectures.custom_acc(threshold=0.75),
                    "dice_coef_loss":
                    architectures.dice_coef_loss,
                    "dice_coef":
                    architectures.dice_coef,
            }):
		
                model = load_model(
                    filepath=args.model_path, custom_objects=None, compile=True)
except Exception as load_model_exception:
    print("[EXCEPTION] %s" % load_model_exception)
    sys.exit("[EXCEPTION] Error loading test images!")
##########################################################################
##########################################################################
##########################################################################
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



##########################################################################
##########################################################################
##########################################################################
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
        cv2.imshow("Ground Truth", annotations/255)

        # prediction overlay 
        overlay = 0.5 * original_bgr/255 + 0.5 * prediction_bgr
        if(upsample):
            overlay = cv2.resize(overlay, (WIDTH*2, HEIGHT*2), interpolation=cv2.INTER_CUBIC)

        print("[INFO] Show predicted", overlay.shape)
        cv2.imshow("Predicted", overlay)
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
