'''
Project: t-recx
Subproject: Visual wake words (human face detection) on VWW dataset
File:train_mobnet.py 
desc: loads data, trains and saves model, 
Train the mobilenetv1 model with EV-assistance and without assistance

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
'''

import os
import argparse
from absl import app
import vww_model as models

import tensorflow as tf
assert tf.__version__.startswith('2')
from helpers import save_trecx_model 
import Config


BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')

# custom datagen for trecx models
def custom_generator_train(gen):
    num_classes = 2
    while True:
        (x, y) =gen.next()
        BS = x.shape[0]
        if BS<Config.BATCH_SIZE:
          x_0, x_1 = x, y
          dummy_x_tensor = tf.constant(0, dtype='float32', shape=[Config.BATCH_SIZE-BS, 96,96,3])
          dummy_y_tensor =  tf.zeros(shape=[Config.BATCH_SIZE-BS,2],  dtype='float32')
          x_0 = tf.concat([x_0, dummy_x_tensor], axis=0)
          x_1= tf.concat([x_1, dummy_y_tensor], axis=0)
          yield (x_0, x_1), x_1
        else:
          yield (x, y), y

def main(argv):
  if len(argv) >= 3:
    model_save_name = argv[1]
    model_architecture = argv[2]
  else:
    print('Please provide model_save_name and model_architecture (choose from [mobnet_ev, mobnet_noev])  in cmdl; Usage: python train_mobnet.py <model_save_name> <model_architecture>')


  # load uninitialized model
  if model_architecture=='mobnet_ev':
    model = models.mobilenet_v1_ev(Config.W_aux)
  elif model_architecture=='mobnet_noev':
    model = models.mobilenet_v1_noev(Config.W_aux)
  else:
    raise ValueError("Model architecture {:} not supported".format(model_architecture))

  model.summary()


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
      target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
      batch_size=Config.BATCH_SIZE,
      subset='training',
      color_mode='rgb')
  val_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
      batch_size=Config.BATCH_SIZE,
      subset='validation',
      color_mode='rgb')
  print(train_generator.class_indices)

  # train model in 3 iterations
  # model = train_epochs(model, train_generator, val_generator, Config.epochs_0, 0.001, model_save_name, 0)
  # model = train_epochs(model, train_generator, val_generator, Config.epochs_1, 0.0005, model_save_name, 1)
  # model = train_epochs(model, train_generator, val_generator, Config.epochs_2, 0.00025, model_save_name, 2)

  # Save model
  save_trecx_model(model, model_save_name, model_architecture)




##custom callback to transfer weights from early exit to final exit before each train batch
# this train routine trains the model in 3 steps. Allow atleast 5 epochs for finetuning after weight transfer
class weight_transfer_callback(tf.keras.callbacks.Callback):
  def __init__(self, epochs, train_count):
        #not using 4/5 because the train is done in 3 stages. So this will esentially result in 4/5 of all train epochs        
        # self. epoch_threshold = (epochs//3) *2 
        # self. epoch_threshold_max = (epochs//5) *4 
        self.train_count = train_count
        self.isFrozen = False
        self.no_transfer = False

  def on_train_batch_begin(self, batch, logs=None):
    if not self.no_transfer:
        for layer in self.model.layers:
            if layer.name=='depth_conv_ee_1':
                dw_ee_conv_layer = layer
            if layer.name=='depth_conv_eefinal_out':
                depthconv_eefinal_layer = layer

        weights = dw_ee_conv_layer.get_weights()
        depthconv_eefinal_layer.set_weights(weights)


    def on_epoch_end(self, epoch, logs=None):
      #hardcoding the epoch after which weight transfer stops such that atleast 5 epochs for finetuning 
      if self.train_count==0 and epoch==15:
        self.no_transfer = True
      if self.train_count==1 and epoch==5:
        self.no_transfer = True
      if self.train_count==2 and epoch==15:
        self.no_transfer = True


def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate, model_name, train_count):

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'], loss_weights=None, run_eagerly=False)


  history_fine = model.fit(
      custom_generator_train(train_generator),#use trecx custom datagen
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=custom_generator_train(val_generator),
      validation_steps=len(val_generator),
      batch_size=Config.BATCH_SIZE, callbacks=[weight_transfer_callback(epoch_count, train_count)])
  
  return model


if __name__ == '__main__':
  app.run(main)