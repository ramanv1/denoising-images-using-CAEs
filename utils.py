#*****************************************************************************80
# project: Denoising images using Convolutional Autoencoders
# author : Vinay Raman, PhD
# date   : 08/26/2019
#*****************************************************************************80
import sys
import os
import math
import matplotlib.pyplot as plt
import h5py
import numpy as np 
import pandas as pd
import random 
import copy 
import seaborn as sns
from photutils.datasets import apply_poisson_noise # for generating noisy images
import zipfile
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import backend as K
import horovod.keras as hvd
import zipfile
from models import *
from utils import *
from train import *
from keras.preprocessing.image import ImageDataGenerator
import glob
from PIL import Image



#*****************************************************************************80
def extract_data(source_dir, dest_dir):
  """ 
  description: extracts files from zip file
  input: source directory where zip file resides
  output: directory containing extracted files
  
  """
  with zipfile.ZipFile(source_dir, 'r') as zip_ref: 
    zip_ref.extractall(dest_dir)

#*****************************************************************************80
def rgb2gray(image):
  """ 
  description: converts from rgb to gray scale 
  input: rgb image
  output: gray-scale image
  """

  return np.dot(image[:, :, :3], [0.2125,  0.7154, 0.0721])

#*****************************************************************************80

def tf_norm_crop_resize_image(image, resize_dim=(64,64)):
    """
    description: rescales the image to [0., 1.]
    input: image
    output: re-scaled image 
    """
    image = tf.cast(image, tf.float32)/255.
    return image

#*****************************************************************************80
def make_noisy_images(image):
  """ applies poisson noise to image """
  return apply_poisson_noise(image, random_state=12345)

#*****************************************************************************80

def get_image_with_poisson_noise(image):
  """ obtains images after cropping 
     and applying poisson noise"""
  img = tf_norm_crop_resize_image(image, resize_dim=(64,64))
  noisy_img = np.clip(make_noisy_images(img*255.)/255., 0., 1.)
  return noisy_img

#*****************************************************************************80
def create_data_generators(train_dir, validation_dir, test_dir):

  """
     description: creates generators for training, validation and testing
     inputs: 
             train_dir - directory containing training zip file
             validation_dir - directory containing validation zip file
             test)dir - directory containing testing zip file
     outputs:
             train_generator - image-data generator for training
             val_generator  - image-data generator for validation
             test_generator - image-data generator for testing
  """
  extract_data(train_dir,'./train_data')    
  extract_data(test_dir, './test_data')
  extract_data(validation_dir,'./validation_data')
  seed = 1
  trainx_gen = ImageDataGenerator(preprocessing_function=
                                  get_image_with_poisson_noise)
  trainy_gen = ImageDataGenerator(preprocessing_function=
                                tf_norm_crop_resize_image)
  train_generator = zip(trainx_gen.flow_from_directory('./train_data',
                                                color_mode='rgb',
                                                target_size=(64,64),
                                                batch_size=64,
                                                class_mode=None, seed=seed
                                                ),
                        trainy_gen.flow_from_directory('./train_data',
                                                color_mode='rgb',
                                                target_size=(64,64),
                                                batch_size=64,
                                                class_mode=None,
                                                seed =seed
                                                )) 
 #*****************************************************************************80                       
  valx_gen = ImageDataGenerator(preprocessing_function=
                                get_image_with_poisson_noise)
  valy_gen = ImageDataGenerator(preprocessing_function=t
                                f_norm_crop_resize_image)
  val_generator = zip(valx_gen.flow_from_directory('./validation_data',
                                                color_mode='rgb',
                                                target_size=(64,64),
                                                batch_size=64,
                                                class_mode=None, seed=seed
                                                ),
                      valy_gen.flow_from_directory('./validation_data',
                                                color_mode='rgb',
                                                target_size=(64,64),
                                                batch_size=64,
                                                class_mode=None,
                                                seed =seed
                                                )) 

  testx_gen = ImageDataGenerator(preprocessing_function=
                                 get_image_with_poisson_noise)
  testy_gen = ImageDataGenerator(preprocessing_function=
                                 tf_norm_crop_resize_image)
  test_generator = zip(testx_gen.flow_from_directory('./test_data',
                                                color_mode='rgb',
                                                target_size=(64,64),
                                                batch_size=50,
                                                class_mode=None, seed=seed
                                                ),
                      testy_gen.flow_from_directory('./test_data',
                                                color_mode='rgb',
                                                target_size=(64,64),
                                                batch_size=50,
                                                class_mode=None,
                                                seed =seed
                                                )) 
  return train_generator, val_generator, test_generator
#*****************************************************************************80
