# Denoising images using Convolutional Autoencoders
  We remove Poisson noise in the image using Convolutional Autoencoders. We use two encoder-decoder networks, that work in tandem.
  The two encoder-decoder networks have different compression ratios. The encode-decoder network having a lower compression ratio preserves
  the finer details of the image whereas the one with higher compression ratio removes the noise in the image. 
  We use horovod for training the DNN on multiple GPUs. We use image data generators working on custom preprocessing functions to create 
  the pipeline for handling large datasets. Currently we have performed simulations using Adam optimizer. The code can be further customized
  to include any other optimzer as well. 
  The code is structed as follows:
  1. utils.py - file containing all the utility functions required for creating data pipelines
  2. models.py - file containing the function that creates the model using tensorflow/keras functional API
  3. train.py - file containing the function that trains the compiled model using horovod package
  
  The results of the network have been evalulated using peak signal to noise ratio (PSNR) as the metric. We see an improvement in PSNR for the reconstructed images (over the noisy images).

### Dependencies:
1. TensorFlow and Keras 
2. Horovod for TensorFlow/Keras:
   this can be installed using pip install horovod[tensorflow,keras]
3. photutils:
   this is a package used for applying poisson noise to image
4. glob, zipfile:
   use for file-handling
5. numpy, pandas, matplotlib, seaborn:
   for data-processing, and plotting

