'''
Project: t-recx
Subproject: Image classification on cifar10
File:train_prior.py 
desc: Applies the early-exit techniques presented in the branchynet and the SDN papers on the resnet model 
Train the resnet8 model with early exit techniques of Branchynet and SDN

'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import keras_model_prior as keras_model

import datetime
import json
import sys
import argparse

EPOCHS = 500
# EPOCHS = 1
BS = 32

# get date ant time to save model
dt = datetime.datetime.today()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute



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


    # # add 16 more samples so that its completely divisible by BS
    # for i in range(0,16):
    #     rand_loc = np.random.randint(0,49999)
    #     img = cifar_train_data[rand_loc]
    #     img = np.reshape(img, [1,32,32,3])
    #     label = cifar_train_labels[rand_loc]
    #     cifar_train_data =  np.append(cifar_train_data,img, axis=0)
    #     cifar_train_labels =  np.append(cifar_train_labels, label) 

    return cifar_train_data, cifar_train_filenames, to_categorical(cifar_train_labels), \
        cifar_test_data, cifar_test_filenames, to_categorical(cifar_test_labels), cifar_label_names
# def custom_generator_train(gen):
#     while True:
#         (x, y) =gen.next()
#         yield [x,y], y
def custom_generator_val(gen):
    while True:
        (x, y) =gen.next()
        curr_BS = y.shape[0]
        if curr_BS !=BS:
            x_0, x_1 = x, y
            dummy_x_tensor = tf.constant(0, dtype='float32', shape=[BS-curr_BS, 32,32,3])
            dummy_y_tensor =  tf.zeros(shape=[BS-curr_BS, 10], dtype='float32')
            x_0 = tf.concat([x_0, dummy_x_tensor], axis=0)
            x_1= tf.concat([x_1, dummy_y_tensor], axis=0)
            yield (x_0, x_1), x_1
        else:
            yield [x,y], y


#custom callback to set weights of convolution from ee-1 to conv just before ee-final classification
class customCallback(tf.keras.callbacks.Callback):
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
    # parser.add_argument('--model_save_name', type=str, default="""pretrainedResnet_default""", 
    #     help="""Specify the name with which the trained model is saved """)
    parser.add_argument('--model_architecture', type=str, default=False, 
        help="""Model architecture to be used. Choose from: ['branchynet', 'sdn'] """)

    args = parser.parse_args()
    model_arch = args.model_architecture
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



    
    #uninitialized model
    if model_arch=='branchynet':
        new_model = keras_model.resnet_v1_branchynet()
        model_save_name = 'trained_models/model_branchynet'
    elif model_arch=='sdn':
        new_model = keras_model.resnet_v1_sdn()
        model_save_name = 'trained_models/model_sdn'
    new_model.summary()

 
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(train_data)
    
    if model_arch=='sdn':
        new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy', loss_weights=[0,0,1],
            weighted_metrics=None, run_eagerly=False)
    elif model_arch=='branchynet':
        new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy', loss_weights=[1,0.3,1],
            weighted_metrics=None, run_eagerly=False)
    
    # fits the model on batches with real-time data augmentation:
    print('----------------------------------------------------')
    print('STARTING TRAINING ....')
    print('----------------------------------------------------')


    train_gen = datagen.flow( train_data, train_labels,  batch_size=BS)        

    if model_arch=='sdn':
        train_gen = custom_generator_val(train_gen)
        History = new_model.fit(train_gen, steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler, customCallback(EPOCHS)])    
    elif model_arch=='branchynet':
        History = new_model.fit(train_gen, steps_per_epoch=len(train_data) / BS, epochs=EPOCHS, callbacks=[lr_scheduler])
    
    print('DONE!')
    
    print('----------------------------------------------------')
    print('Saving Final Model....')
    print('----------------------------------------------------')
    if model_arch=='sdn':
        #strip the endpoint layer and the targets input and save model
        cnt = 0
        for layer in new_model.layers:
            if layer.name=='ee_1':
                ee1_layer_num = cnt
            if layer.name=='ee_2':
                ee2_layer_num = cnt
            if layer.name=='ef_out':
                ef_layer_num = cnt
            cnt+=1
        #construct new model
        final_model = tf.keras.models.Model(inputs=new_model.inputs[0], outputs=[new_model.layers[ee1_layer_num].output, 
            new_model.layers[ee2_layer_num].output, new_model.layers[ef_layer_num].output])
        final_model.save("trained_models/" + model_save_name)
    else:
        new_model.save("trained_models/" + model_save_name)
    print('DONE!')

    