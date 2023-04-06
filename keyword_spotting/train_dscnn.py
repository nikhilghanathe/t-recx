#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os, sys
import argparse
from tensorflow import keras
import tensorflow as tf
import keras_model as models
import get_dataset as kws_data
import kws_util

num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None



#custom callback to transfer weights from early exit to final exit before each train batch
class weight_transf_callback(tf.keras.callbacks.Callback):
  def on_train_batch_begin(self, batch, logs=None):
      for layer in self.model.layers:
          if layer.name=='depth_conv_ee_1':
              conv_layer = layer
          if layer.name=='depth_conv_eefinal_out':
              depthconv_eefinal_layer = layer
      #transfer weights
      weights = conv_layer.get_weights()
      depthconv_eefinal_layer.set_weights(weights)

#custom callback to convey epoch info to SDN_loss endpoint layer
class sdn_callback(tf.keras.callbacks.Callback):
  def __init__(self, epochs):
    self.epochs = epochs
    
  def on_epoch_begin(self,epoch, logs=None):
    for layer in self.model.layers:
      if layer.name in ['ee_1_loss', 'ee_2_loss', 'ee_3_loss']:
        layer.epoch = epoch
  def on_train_begin(self, logs=None):
    for layer in self.model.layers:
      if layer.name in ['ee_1_loss', 'ee_2_loss', 'ee_3_loss']:
        layer.epochs = self.epochs



if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()

  print('We will download data to {:}'.format(Flags.data_dir))
  print('We will train for {:} epochs'.format(Flags.epochs))

  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
  print("Done getting data")


  #append ffmpeg path to system path
  # pwd = os.getcwd()
  # sys.path.append(pwd+'/ffmpeg')
  # print(sys.path)
  

  # this is taken from the dataset web page.
  # there should be a better way than hard-coding this
  train_shuffle_buffer_size = 85511
  val_shuffle_buffer_size = 10102
  test_shuffle_buffer_size = 4890

  ds_train = ds_train.shuffle(train_shuffle_buffer_size)
  ds_val = ds_val.shuffle(val_shuffle_buffer_size)
  ds_test = ds_test.shuffle(test_shuffle_buffer_size)
  
  
  if Flags.model_init_path is None:
    print("Starting with untrained model")
    model = models.get_model(args=Flags)
  else:
    print(f"Starting with pre-trained model from {Flags.model_init_path}")
    model = keras.models.load_model(Flags.model_init_path)

  model.summary()

  callbacks = kws_util.get_callbacks(args=Flags)
  epochs = Flags.epochs


  if Flags.isTrecx:#if using trecx tecnhiques
    if Flags.isEV:#use weight transfer callback
      train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks+[weight_transf_callback()])
    else:
      train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks)
  else:
    if Flags.model_architecture=='ds_cnn_sdn':#add a callback for sdn loss computation in the endpoint layer
      train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks+[sdn_callback(epochs)])
    else:
      train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks)

  #strip the endpoint layer and the targets input and save model for trecx models
  kws_util.save_trecx_model(model, Flags.model_save_name, Flags.model_architecture)
  
  # if Flags.run_test_set:
  #   test_scores = model.evaluate(ds_test)
  #   print("Test loss:", test_scores[0])
  #   print("Test accuracy:", test_scores[1])
