'''
Project: t-recx
Subproject: Image classification on cifar10
File:train_resnet.py 
desc: loads data, trains and saves model, plots training metrics
Train the resnet8 model with EV-assistance and without assistance

'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import keras_model as keras_model

import datetime
import json
import sys
import argparse
from helpers import get_loss_weights, save_trecx_model
import Config



EPOCHS = Config.epochs
BS = Config.batch_size

# get date ant time to save model
dt = datetime.datetime.today()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute
# NUM_SAMPLES = 1008
# NUM_SAMPLES = int(sys.argv[1])
"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""



#learning rate schedule
def lr_schedule(epoch):
    # initial_learning_rate = 0.001
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print('Learning rate = %f'%lrate)
    return lrate

lr_scheduler = LearningRateScheduler(lr_schedule)

#optimizer
optimizer = tf.keras.optimizers.Adam()

#define data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    #brightness_range=(0.9, 1.2),
    #contrast_range=(0.9, 1.2),
    validation_split=0.2
)




def unpickle(file):
    """load the cifar-100 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data



def convert(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return map(convert, data)
    return data



#---------------------------------------------------
# USER helper functions
#---------------------------------------------------
def get_random_img():
    img = []
    for i in range(0,3072):
        img.append(np.random.randint(0,255))
    return img





def load_cifar_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    Add 1000 random samples from the cifar-100 dataset
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000 (cifar-10)
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    
    cifar_label_names = meta_data_dict[b'label_names']+ [b'uncertain']
    cifar_label_names = np.array(cifar_label_names)
  
    
    # training data
    cifar_train_data, cifar100_train_data_dict = None, None
    cifar_train_filenames = []
    cifar_train_labels = []
    cifar_train_data_dict = {}
    debug_dict = {}

    print('----------------------------------------------------')
    print('Loading cifar-10 data...')
    print('----------------------------------------------------')

    #load cifar-10 data
    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_filenames += cifar_train_data_dict[b'filenames']
   
    
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)
    
    
    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)


    
    print('DONE!\n')
    
    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)


    return cifar_train_data, cifar_train_filenames, to_categorical(cifar_train_labels), \
        cifar_test_data, cifar_test_filenames, to_categorical(cifar_test_labels), cifar_label_names


# custom datagen for trecx models
def custom_generator_val(gen):
    while True:
        (x, y) =gen.next()
        curr_BS = y.shape[0]
        if curr_BS !=BS:#if not equal to batch size, append dummy tensors
            x_0, x_1 = x, y
            dummy_x_tensor = tf.constant(0, dtype='float32', shape=[BS-curr_BS, 32,32,3])
            dummy_y_tensor =  tf.zeros(shape=[BS-curr_BS, 10], dtype='float32')
            x_0 = tf.concat([x_0, dummy_x_tensor], axis=0)
            x_1= tf.concat([x_1, dummy_y_tensor], axis=0)
            yield (x_0, x_1), x_1
        else:
            yield [x,y], y


#-----------------Callbacks----------------------------------------------------------------
#custom callback to transfer weights from early exit to final exit before each train batch
class weight_transfer_callback(tf.keras.callbacks.Callback):
    def __init__(self):
        self. epoch_threshold = (EPOCHS//3) *2 
        self. epoch_threshold_max = (EPOCHS//5) *4
        self.no_transfer = False


    def on_train_batch_begin(self, batch, logs=None):
        if not self.no_transfer:
            for layer in self.model.layers:
                if layer.name=='depth_conv_ee_1':
                    ee_conv_layer = layer
                if layer.name=='depth_conv_eefinal_out':
                    depthconv_eefinal_layer = layer
            #transfer weights
            weights = ee_conv_layer.get_weights()
            depthconv_eefinal_layer.set_weights(weights)

    #transfer for only ~80% of epochs
    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.epoch_threshold_max:
            self.no_transfer = True

#custom callback to convey epoch info to SDN_loss endpoint layer
class sdn_callback(tf.keras.callbacks.Callback):
  def __init__(self, epochs):
    self.epochs = epochs
    
  def on_epoch_begin(self,epoch, logs=None):
    for layer in self.model.layers:
      if layer.name in ['ee_1_loss', 'ee_2_loss']:
        layer.epoch = epoch
  def on_train_begin(self, logs=None):
    for layer in self.model.layers:
      if layer.name in ['ee_1_loss', 'ee_2_loss']:
        layer.epochs = self.epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_name', type=str, default="""trainedResnet""", 
        help="""Specify the model architecture to use from: [resnet_ev, resnet_noev, resnet_sdn, resnet_branchynet] """)
    parser.add_argument('--model_architecture', type=str, default="""resnet_ev""", 
        help="""Specify the name with which the trained model is saved """)
    parser.add_argument('--W_aux', type=float, default=0.5, 
        help="""Specify the weight of the auxiliary loss at the early exit. The paper uses a value of 0.5 """)
    parser.add_argument('--isEV', action="store_true", 
        help=""" Specify whether to use EV architecture. To use EV-assistance use this flag in command line (--isEV). Exclude for no EV-assistance """)
    parser.add_argument('--isTrecx', action="store_true", 
        help=""" Specify whether to use custom datagen for T-recx models. HAVE TO USE this: --isTrecx on cmdl for all T-recx models. Exclude to use normal datagen """)



    args = parser.parse_args()
    model_save_name = args.model_save_name
    W_aux = float(args.W_aux)
    isEV = args.isEV
    isTrecx = args.isTrecx
    model_architecture = args.model_architecture

    """load cifar10 data and train model"""
    cifar_10_dir = 'cifar-10-batches-py'
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)



    #load uninitialized model
    if isTrecx:
        if isEV:
            new_model = keras_model.resnet_v1_eembc_with_EV(W_aux)
        else:
            new_model = keras_model.resnet_v1_eembc_with_noEV(W_aux)            
    else:
        if model_architecture=='resnet_sdn':
            new_model = keras_model.resnet_v1_sdn()            
        elif model_architecture=='resnet_branchynet':
            new_model = keras_model.resnet_v1_branchynet() 
        else:
            raise ValueError("Model architecture {:} not supported".format(model_architecture))           
    new_model.summary()

 

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(train_data)

    
    #------compile the model--------
    #get loss weights depending on model_arch
    loss_weights = get_loss_weights(model_architecture)
    new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy', 
            loss_weights=loss_weights,weighted_metrics=None, run_eagerly=False)
    
    # fits the model on batches with real-time data augmentation:
    print('----------------------------------------------------')
    print('STARTING TRAINING ....')
    print('----------------------------------------------------')

    #custom datagen
    train_gen = custom_generator_val(datagen.flow( train_data, train_labels,  batch_size=BS))
    if isTrecx:
        if isEV:#use weight transfer callback
            History = new_model.fit(train_gen, steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler,  weight_transfer_callback()])
        else:
            History = new_model.fit(train_gen, steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler])
    else:
        if model_architecture=='resnet_sdn':
            History = new_model.fit(train_gen, steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler,  sdn_callback(EPOCHS)])
        else:
            History = new_model.fit(datagen.flow( train_data, train_labels,  batch_size=BS), steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler])
    
    print('DONE!')

    print('----------------------------------------------------')
    print('Saving Final Model....')
    print('----------------------------------------------------')
    #strip the endpoint layer and the targets input and save model for trecx models and SDN model
    save_trecx_model(new_model, model_save_name, model_architecture)
    
    print('DONE!')


