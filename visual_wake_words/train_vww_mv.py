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
import vww_model_test as models

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


  model = models.mobilenet_v1_mv()

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

  model = train_epochs(model, train_generator, val_generator, 20, 0.001, model_name, 0)
  # model = train_epochs(model, train_generator, val_generator, 10, 0.0005, model_name, 1)
  # model = train_epochs(model, train_generator, val_generator, 20, 0.00025, model_name, 2)

  # # Save model HDF5
  # if len(argv) >= 3:
  #   model.save(argv[2])
  # else:
  #   model.save('trained_models/vww_96.h5')
  model.save(model_name)




#custom callback to set weights of convolution from ee-1 to conv just before ee-final classification
class customCallback(tf.keras.callbacks.Callback):
  def __init__(self, epochs, train_count):
        self. epoch_threshold = (epochs//3) *2 
        self. epoch_threshold_max = (epochs//5) *4 #not using 4/5 because the train is done in 3 stages. So this will esentially result in 4/5 of all train epochs
        self.train_count = train_count
        self.isFrozen = False
        self.no_transfer = False

  def on_train_batch_begin(self, batch, logs=None):
    if not self.no_transfer:
        for layer in self.model.layers:
            if layer.name=='dw_ee_1':
                conv_layer = layer
            if layer.name=='endpoint':
                endpoint_layer = layer
            if layer.name == 'activation_6':
                act_6_layer = layer
            if layer.name=='depth_conv_eefinal_out':
                depthconv_eefinal_layer = layer

        weights = conv_layer.get_weights()
        in_fmaps = act_6_layer.output
        
        # endpoint_layer.weigths_conv_ee_1.assign(weights)
        # print((np.array(weights, dtype='float32')).shape)

        # depthconv_eefinal_layer.trainable=False
        depthconv_eefinal_layer.set_weights(weights)


    def on_epoch_end(self, epoch, logs=None):
      if self.train_count==0 and epoch==15:
        self.no_transfer = True
      if self.train_count==1 and epoch==5:
        self.no_transfer = True
      if self.train_count==2 and epoch==15:
        self.no_transfer = True
      # if epoch > self.epoch_threshold_max:
      #     self.no_transfer = True


def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate, model_name, train_count):

  model.summary()

  #model checkpoints
  checkpoint_filepath = os.getcwd()+'/checkpoints'+'/'+model_name[15:]
  if not os.path.exists(checkpoint_filepath):
      os.makedirs(checkpoint_filepath, exist_ok=True)
  print(checkpoint_filepath)
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=False,
      monitor='loss',
      verbose=1,
      # monitor = 'val_dense_accuracy',
      #monitor=['val_ee_1_out_accuracy', 'val_dense_accuracy', 'val_loss', 'val_ee_1_out_loss', 'val_dense_loss'],
      mode='max')

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'], loss_weights=None, run_eagerly=False)


  history_fine = model.fit(
      custom_generator_train(train_generator),
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=custom_generator_train(val_generator),
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE, callbacks=[customCallback(epoch_count, train_count)])
  # model_name = 'test_on_batch_begin_ch64'
  model.save(model_name)

  return model


if __name__ == '__main__':
  app.run(main)