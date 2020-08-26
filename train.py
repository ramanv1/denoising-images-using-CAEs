#*****************************************************************************80
# project: Denoising images using Convolutional Autoencoders
# author : Vinay Raman, PhD
# date   : 08/26/2019
#*****************************************************************************80
#TensorFlow + Keras imports

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
import math
#*****************************************************************************80


def train(hvd,
          model, 
          train_generator, 
          validation_generator,
          epochs = 1, 
          steps_per_epoch = 100,
          validation_steps = 50):
  
  """
    description: function to perform training of the DNN
    inputs: 
           hvd - initialized horovod object (for distributed training on
                                             multiple GPUs)
           model - compile model
           train_generator - image generator for training
           validation_generator - image generator for validation
           epochs - number of epochs for training
           steps_per_epoch - number of steps taken in training generator/epoch
           validation_steps - number of steps taken in validation generator
    outputs:
           history - stats of training (training loss, accuracy; 
                                        validation loss, accuracy)
           model - trained model     
  """

  epochs = int(math.ceil(epochs / hvd.size()))
  earlystopping = [tf.keras.callbacks.EarlyStopping(min_delta=1e-4, 
                                                 patience = 5,
                                                 monitor='val_loss')]


  callbacks = [
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      hvd.callbacks.BroadcastGlobalVariablesCallback(0), 
      earlystopping
  ]

  # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
  if hvd.rank() == 0:
      callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
   
  history = model.fit(train_generator,
                      epochs = epochs,
		                  steps_per_epoch = steps_per_epoch,
                      verbose=1,
                      callbacks= callbacks,
                      validation_data = validation_generator,
                      validation_steps = validation_steps)

  return history, model