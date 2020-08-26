#*****************************************************************************80
# project: Denoising images using Convolutional Autoencoders
# author : Vinay Raman, PhD
# date   : 08/26/2019
#*****************************************************************************80
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import r2_score
from keras.utils.vis_utils import plot_model
import tensorflow_datasets as tfds
from keras import backend as K
import horovod.keras as hvd

#*****************************************************************************80

def make_denoiser_model(hvd, 
                        name='denoiser', 
                        img_size = (64,64,3), 
                        lr = 1e-3, decay = 1e-6):
  """
   description: creates a DNN comprising of two encoder-decoder networks
                with different compression ratios, the outputs of both are 
                concatenated to get the final output
   inputs: 
           hvd - initialized horovod object (for distributed computing)
           name - model name
           img_size - image size 
           lr  - learning rate for the Adam optimizer 
           decay - learning rate decay for the Adam optimizer
   outputs:
           model - compiled model
  """

  input_layer = tf.keras.layers.Input(shape=img_size)

  #lower compression ratio autoencoding: branch 1

  # encoder-decoder network start:
  uconv1 = tf.keras.layers.Conv2D(32, 
                                  kernel_size= (4, 4),
                                  strides = (2, 2),
                                  padding ='same',
                                  activation = 'relu',
                                  ) (input_layer)

  uconv1b = tf.keras.layers.BatchNormalization()(uconv1)
  
  uconv2 = tf.keras.layers.Conv2D(16, 
                                  kernel_size= (4, 4),
                                  strides = (2, 2),
                                  padding ='same',
                                  activation = 'relu',
                                  ) (uconv1b)
 
  ubottleneck = tf.keras.layers.Flatten()(uconv2)
  
  udenselayer1  = tf.keras.layers.Dense(125,
                                       activation='relu')(ubottleneck)

  
  udenselayer2 = tf.keras.layers.Dense(4096, activation='relu')(udenselayer1)


  ulatentlayer = tf.keras.layers.Reshape((16, 16, 16))(udenselayer2)

  uconv1t = tf.keras.layers.Conv2DTranspose(filters = 32, 
                                            kernel_size= (4, 4), 
                                            strides = (2, 2),
                                            padding ='same',
                                            activation = 'relu')(ulatentlayer)
  uadd = tf.keras.layers.Add()([uconv1t, uconv1])

  uoutput = tf.keras.layers.Conv2DTranspose(3, 
                                            kernel_size=(4,4),
                                            strides = (2,2),
                                            padding ='same',
                                            activation ='linear')(uadd)
  #end encoder-docoder network

#*****************************************************************************80                                            
  #higher compression ratio autoencoding: branch 2

  # encoder-decoder network start
  lconv1 = tf.keras.layers.Conv2D(32,
                                  kernel_size=(4,4),
                                  strides =(2,2),
                                  padding ='same',
                                  activation='relu')(input_layer)
  lconv1b = tf.keras.layers.BatchNormalization()(lconv1)                               
  lconv2 = tf.keras.layers.Conv2D(16, 
                                  kernel_size=(4,4),
                                  strides=(2,2),
                                  padding='same',
                                  activation='relu')(lconv1b) 
  lconv3 = tf.keras.layers.Conv2D(8, 
                                  kernel_size=(4,4),
                                  strides=(2,2),
                                  padding='same',
                                  activation='relu')(lconv2)

  lbottleneck = tf.keras.layers.Flatten()(lconv3)

  ldenselayer1 = tf.keras.layers.Dense(125, activation='relu')(lbottleneck)

  
  ldenselayer2 = tf.keras.layers.Dense(512, activation='relu')(ldenselayer1)

  l_latentlayer = tf.keras.layers.Reshape((8, 8, 8))(ldenselayer2)
  
  lconv1t = tf.keras.layers.Conv2DTranspose(16,
                                            kernel_size=(4,4),
                                            strides=(2,2),
                                            padding="same",
                                            activation='relu')(l_latentlayer)
  ladd1 = tf.keras.layers.Add()([lconv1t, lconv2])

  lconv2t = tf.keras.layers.Conv2DTranspose(32, 
                                            kernel_size=(4,4),
                                            strides=(2,2),
                                            padding='same',
                                            activation='relu')(ladd1)
  ladd2 = tf.keras.layers.Add()([lconv2t, lconv1])
  
  loutput = tf.keras.layers.Conv2DTranspose(3, 
                                            kernel_size=(4,4),
                                            strides=(2,2),
                                            padding='same',
                                            activation='linear')(ladd2)
  #encoder- decoder network end

#*****************************************************************************80 

  # concatenation of branch 1 and branch 2:
  output_layer = tf.keras.layers.Add()([loutput, uoutput])

  model = tf.keras.models.Model(inputs = input_layer,
                                outputs= output_layer,
                                name = name)
 #*****************************************************************************80 
 # horovod code for running on multiple GPUs

  if (hvd.size()>1): # if multiple-GPUs found  
    
    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.keras.optimizers.Adam(learning_rate=lr*hvd.size(),
                                      decay = decay)

    # Horovod: add Horovod Distributed Optimizer.
    optimizer = hvd.DistributedOptimizer(opt)

  else:
 
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr, decay = decay)
 
  # mean-squared error between clean image and reconstructed image 
  loss = tf.keras.losses.mse

  model.compile(optimizer=optimizer,loss = loss)
  
  return model
  #*****************************************************************************80