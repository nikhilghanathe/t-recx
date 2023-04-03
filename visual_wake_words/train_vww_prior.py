# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os
import argparse
from absl import app
import vww_model_prior as models

import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')

def custom_generator_train(gen):
    num_classes = 2
    while True:
        (x, y) =gen.next()
        BS = x.shape[0]
        if BS<BATCH_SIZE:
          # for i in range(BS, 100):
          x_0, x_1 = x, y
          dummy_x_tensor = tf.constant(0, dtype='float32', shape=[BATCH_SIZE-BS, 96,96,3])
          dummy_y_tensor =  tf.zeros(shape=[BATCH_SIZE-BS,2],  dtype='float32')
          x_0 = tf.concat([x_0, dummy_x_tensor], axis=0)
          x_1= tf.concat([x_1, dummy_y_tensor], axis=0)
          yield (x_0, x_1), x_1
          print(x.shape, y, BS)
        else:
          yield (x, y), y

def main(argv):
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--model_save_name', type=str, default="""pretrained_vww_default""", 
  #     help="""Specify the name with which the trained model is saved """)
  # parser.add_argument('--get_trace', type=bool, default=False, 
  #     help="""If set to true, get trace data """)
  # args = parser.parse_args()
  model_name = argv[1]
  model_arch = argv[2]

  if model_arch=='sdn':
    model = models.mobilenet_v1_sdn()
  elif model_arch=='branchynet':
    model = models.mobilenet_v1_branchynet()
  else:
    raise 'model architecture not defined. Select from [sdn, branchnet]'

  model.summary()

  batch_size = 50
  validation_split = 0.1

  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.05,
      height_shift_range=0.05,
      zoom_range=.1,
      horizontal_flip=True,
      validation_split=validation_split,
      rescale=1. / 255)
  train_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='training',
      color_mode='rgb')
  val_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      subset='validation',
      color_mode='rgb')
  print(train_generator.class_indices)

  if model_arch=='sdn':
    model = train_epochs_sdn(model, train_generator, val_generator, 20, 0.001, model_name, 0)
    model = train_epochs_sdn(model, train_generator, val_generator, 10, 0.0005, model_name, 1)
    model = train_epochs_sdn(model, train_generator, val_generator, 20, 0.00025, model_name, 2)
  else:
    model = train_epochs_branchynet(model, train_generator, val_generator, 20, 0.001, model_name, 0)
    model = train_epochs_branchynet(model, train_generator, val_generator, 10, 0.0005, model_name, 1)
    model = train_epochs_branchynet(model, train_generator, val_generator, 20, 0.00025, model_name, 2)
  # # Save model HDF5
  # if len(argv) >= 3:
  #   model.save(argv[2])
  # else:
  #   model.save('trained_models/vww_96.h5')
  model.save(model_name)
  model.save(model_name+'.h5')




#custom callback to set weights of convolution from ee-1 to conv just before ee-final classification
class customCallback(tf.keras.callbacks.Callback):
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




def train_epochs_sdn(model, train_generator, val_generator, epoch_count,
                 learning_rate, model_name, train_count):

  model.summary()


  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'], loss_weights=[0,0,0,1], run_eagerly=False)


  history_fine = model.fit(
      custom_generator_train(train_generator),
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=custom_generator_train(val_generator),
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE, callbacks=[customCallback(epoch_count)])
  # model_name = 'test_on_batch_begin_ch64'
  model.save(model_name)

  return model

def train_epochs_branchynet(model, train_generator, val_generator, epoch_count,
                 learning_rate, model_name, train_count):

  model.summary()

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy', 
      metrics=['accuracy'], loss_weights=[1,0.3, 1], run_eagerly=False)


  history_fine = model.fit(
      (train_generator),
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=(val_generator),
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE, callbacks=[])
  
  model.save(model_name)

  return model


if __name__ == '__main__':
  app.run(main)
