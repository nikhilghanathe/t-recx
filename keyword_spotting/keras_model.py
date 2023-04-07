#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape, Softmax
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
import Config
from kws_util import get_loss_weights
import Config


# =========================keras model for t-recx=============================
# class Endpoint_ee(tf.keras.layers.Layer):
#     def __init__(self, name=None, W_aux=0.3, num_classes=12):
#         super().__init__(name=name)
#         self.batch_size = Config.BATCH_SIZE

#     @tf.function
#     def loss_fn(self, ee_1, targets):
#         scce = tf.keras.losses.SparseCategoricalCrossentropy()
#         cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#         W_aux = 0.3
#         P_aux = 1.0
#         num_classes=12

#         self.batch_size = targets.shape[0]
#         y_true = targets
#         y_true_transformed_ee1, y_true_transformed_eefinal = [], []
#         y_pred_ee1_transformed = []
#         y_pred_eefinal_transformed = []
#         y_pred_ee1 = ee_1
        
#         if self.batch_size==None:
#             self.batch_size=100
#         loss_ee1, loss_eefinal =0.0, 0.0

#         #for EE-1
#         for i in range(0, self.batch_size):
#             arg_max_true = targets[i]
#             arg_max_true = tf.cast(arg_max_true, dtype='int32')
#             prob_list = y_pred_ee1[i]
#             values, indices =  tf.math.top_k(prob_list, k=3)            
#             [score_max_1, score_max_2, score_max_3] = tf.split(values, num_or_size_splits=3)
#             [arg_max_1, arg_max_2, arg_max_3] = tf.split(indices, num_or_size_splits=3)
#             arg_max_true = tf.reshape(arg_max_true, [1])
#             if tf.math.equal(arg_max_true, arg_max_1):
#               if tf.math.less_equal(tf.math.subtract(score_max_1, score_max_2), tf.constant(0.3)):
#                 y_uncrtn = tf.one_hot([arg_max_true], depth=num_classes, on_value=1., off_value=0.0, dtype='float32')
#               else:
#                 y_uncrtn = tf.one_hot([arg_max_true], depth=num_classes, on_value=P_aux, off_value=0.0, dtype='float32')
#             else:
#                 y_uncrtn = tf.one_hot([arg_max_true], depth=num_classes, on_value=P_aux, off_value=0.0, dtype='float32')
#             y_true_transformed_ee1.append(y_uncrtn)
#         y_true_transformed_ee1 = tf.reshape(y_true_transformed_ee1, [self.batch_size,num_classes])
        
#         loss_cce =  cce(y_true_transformed_ee1, y_pred_ee1) 
#         return tf.multiply(W_aux, loss_cce)

# #define endpoint layer for loss calculation
class Endpoint_ee(tf.keras.layers.Layer):
    def __init__(self, name=None, W_aux=0.3, num_classes=12):
        super().__init__(name=name)
        self.batch_size = 1
        self.W_aux = W_aux
        self.num_classes = num_classes

    @tf.function
    def loss_fn(self, softmax_output, targets):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.batch_size = targets.shape[0]
        
        y_true, y_true_transformed = [], []
        if self.batch_size==None:
            self.batch_size=Config.BATCH_SIZE
        #compute the one hot 'y_true' vector for loss computation
        for i in range(0, self.batch_size):
            arg_max_true = tf.keras.backend.argmax(targets[i])
            arg_max_true = tf.cast(arg_max_true, dtype='int32')
            arg_max_true = tf.reshape(arg_max_true, [1])
            y_true.append(tf.one_hot([arg_max_true], depth=self.num_classes, on_value=1., off_value=0.0, dtype='float32'))
        
        #compute loss for whole batch
        y_true_transformed = tf.reshape(y_true, [self.batch_size,self.num_classes])
        loss_cce =  cce(y_true_transformed, softmax_output) 
      
        return tf.multiply(self.W_aux, loss_cce)
        

    def call(self, softmax_output, targets=None,   sample_weight=None):
        if targets is not None:
            loss = self.loss_fn(softmax_output, targets)
            self.add_loss(loss)
            self.add_metric(loss, name='aux_loss', aggregation='mean')
        #for inference
        return softmax_output


#custom loss function that uses tau
#tau starts from 0.01 and steadily increases to relative inference cost (0.3, 0.6, 0.8) as epochs go by
#define endpoint layer for loss calculation
class SDN_loss(tf.keras.layers.Layer):
  def __init__(self, name=None, max_tau=0.3):
    super().__init__(name=name)
    self.batch_size = 100
    self.tau = 0.01
    self.epoch = 0 #set on epoch_begin
    self.epochs = 20 #set on train_begin
    self.max_tau = max_tau
    self.tau_incr = self.max_tau/self.epochs
 

  @tf.function
  def loss_fn(self, ee, targets):
      sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
      W_aux = self.tau + (self.epoch * self.tau_incr)
      loss_cce = sce( targets, ee)
      return tf.multiply(W_aux, loss_cce)

  def call(self, ee, targets=None, sample_weight=None):
      if targets is not None:
          loss = self.loss_fn(ee, targets)
          self.add_loss(loss)
          self.add_metric(loss, name='aux_loss_'+self.name, aggregation='mean')
      return ee







def prepare_model_settings(label_count, args):
  """Calculates common settings needed for all models.
  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.
  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(args.sample_rate * args.clip_duration_ms / 1000)
  if args.feature_type == 'td_samples':
    window_size_samples = 1
    spectrogram_length = desired_samples
    dct_coefficient_count = 1
    window_stride_samples = 1
    fingerprint_size = desired_samples
  else:
    dct_coefficient_count = args.dct_coefficient_count
    window_size_samples = int(args.sample_rate * args.window_size_ms / 1000)
    window_stride_samples = int(args.sample_rate * args.window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
      spectrogram_length = 0
    else:
      spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
      fingerprint_size = args.dct_coefficient_count * spectrogram_length
  return {
    'desired_samples': desired_samples,
    'window_size_samples': window_size_samples,
    'window_stride_samples': window_stride_samples,
    'feature_type': args.feature_type, 
    'spectrogram_length': spectrogram_length,
    'dct_coefficient_count': dct_coefficient_count,
    'fingerprint_size': fingerprint_size,
    'label_count': label_count,
    'sample_rate': args.sample_rate,
    'background_frequency': 0.8, # args.background_frequency
    'background_volume_range_': 0.1,
    'W_aux':float(args.W_aux)#Aux loss weight for EE; default is 0.3
  }



def get_model(args):
  model_name = args.model_architecture

  label_count=12
  model_settings = prepare_model_settings(label_count, args)

  if model_name=="fc4":
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(model_settings['spectrogram_length'],
                                             model_settings['dct_coefficient_count'])),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(model_settings['label_count'], activation="softmax")
    ])

  elif model_name == 'ds_cnn_ev':
    print("DS CNN model weith EV-assist invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='ee_1_fmaps')(x)


    #add EE
    ee1_fmaps = x

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)


    #add EE
    slice_ind_depthconv_eefinal = 32
    conv_ee1_fmaps = tf.keras.layers.DepthwiseConv2D(name='depth_conv_ee_1',
                   kernel_size=[3,3],
                   strides=2,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_regularizer=regularizer)(ee1_fmaps[:,:,:,:slice_ind_depthconv_eefinal])
    pool_size = (2,2)
    conv_ee1_fmaps = AveragePooling2D(pool_size=pool_size)(conv_ee1_fmaps)
    y_ee_1 = Flatten(name='flatten_ee1')(conv_ee1_fmaps)
    ee_1_logits = Dense(model_settings['label_count'])(y_ee_1)
    ee_1 = Softmax(name='ee_out')(ee_1_logits)

    
    #EV-assist DCONV
    depth_conv_eefinal_out = tf.keras.layers.DepthwiseConv2D(name='depth_conv_eefinal_out',
                   kernel_size=[3,3],
                   strides=1,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_regularizer=regularizer)(x[:,:,:,:slice_ind_depthconv_eefinal])


    x_pooled = AveragePooling2D(pool_size=final_pool_size)(x)
    depth_conv_eefinal_out_pooled = AveragePooling2D(pool_size=(13,3))(depth_conv_eefinal_out)

    #combine early and final view
    y_depthconv_eefinal = Flatten(name='flatten_y_depthconv_eefinal')(depth_conv_eefinal_out_pooled[:,:,:,:slice_ind_depthconv_eefinal])
    y = Flatten()(x_pooled)
    y_combined = tf.keras.layers.concatenate([y, y_depthconv_eefinal])
    ee_final = Dense(model_settings['label_count'], activation='softmax',name='ef_out')(y_combined)


    #add endpoint layer
    targets = Input(shape=[1], name='input_2')
    ee_1 = Endpoint_ee(name='endpoint', W_aux=model_settings['W_aux'], num_classes=model_settings['label_count'])(ee_1, targets)
    # Instantiate model.
    model = Model(inputs=[inputs, targets], outputs=[ee_1,ee_final])

  

  
  elif model_name =='ds_cnn_noev':
    print("DS CNN model without EV-assist invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='ee_1_fmaps')(x)

    #add EE
    ee1_fmaps = x


    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)


    #add EE
    slice_ind_depthconv_eefinal = 32
    conv_ee1_fmaps = tf.keras.layers.DepthwiseConv2D(name='depth_conv_ee_1',
                   kernel_size=[3,3],
                   strides=2,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_regularizer=regularizer)(ee1_fmaps[:,:,:,:32])
    pool_size = (2,2)
    conv_ee1_fmaps = AveragePooling2D(pool_size=pool_size)(conv_ee1_fmaps)
    y_ee_1 = Flatten(name='flatten_ee1')(conv_ee1_fmaps)
    ee_1_logits = Dense(model_settings['label_count'])(y_ee_1)
    ee_1 = Softmax(name='ee_out')(ee_1_logits)

    
    x_pooled = AveragePooling2D(pool_size=final_pool_size)(x)
    y = Flatten()(x_pooled)
    ee_final = Dense(model_settings['label_count'], activation='softmax',name='ef_out')(y)


    #add endpoint layer
    targets = Input(shape=[1], name='input_2')
    ee_1 = Endpoint_ee(name='endpoint', W_aux=model_settings['W_aux'], num_classes=model_settings['label_count'])(ee_1, targets)
    model = Model(inputs=[inputs, targets], outputs=[ee_1,ee_final])


  elif model_name == 'ds_cnn_sdn':
    print("DS CNN model with SDN invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_ee1 = x 

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_ee2 = x
    
    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_ee3 = x
    
    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)

    #Early-exits
    x_ee1_pooled = AveragePooling2D(pool_size=[4,4])(x_ee1)
    x_ee2_pooled = AveragePooling2D(pool_size=[4,4])(x_ee2)
    x_ee3_pooled = AveragePooling2D(pool_size=[4,4])(x_ee3)

    x_ee1_flatten = Flatten()(x_ee1_pooled)
    x_ee2_flatten = Flatten()(x_ee2_pooled)
    x_ee3_flatten = Flatten()(x_ee3_pooled)

    ee_1 = Dense(model_settings['label_count'], activation='softmax', name='ee_1')(x_ee1_flatten)
    ee_2 = Dense(model_settings['label_count'], activation='softmax', name='ee_2')(x_ee2_flatten)
    ee_3 = Dense(model_settings['label_count'], activation='softmax', name='ee_3')(x_ee3_flatten)

    #custom loss for early-exits according to sdn training
    targets = Input(shape=[1], name='input_2')
    ee_1 = SDN_loss(name='ee_1_loss', max_tau=0.25)(ee_1, targets)
    ee_2 = SDN_loss(name='ee_2_loss', max_tau=0.5)(ee_2, targets)
    ee_3 = SDN_loss(name='ee_3_loss', max_tau=0.75)(ee_3, targets)



    x = AveragePooling2D(pool_size=final_pool_size)(x)
    x = Flatten()(x)
    outputs = Dense(model_settings['label_count'], activation='softmax', name='ef_out')(x)

    # Instantiate model.
    model = Model(inputs=[inputs, targets], outputs=[ee_1, ee_2, ee_3, outputs])

  elif model_name == 'ds_cnn_branchynet':
    print("DS CNN model with Branchynet invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_ee1 = x 
    num_filters_ee1 = filters

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_ee2 = x
    num_filters_ee2 = filters

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)


    #Early-exits  
    x_ee1_conv = Conv2D(num_filters_ee1//2,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x_ee1)
    x_ee2_conv = Conv2D(num_filters_ee2//2,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x_ee2)

    x_ee1_pooled = AveragePooling2D(pool_size=final_pool_size)(x_ee1_conv)
    x_ee2_pooled = AveragePooling2D(pool_size=final_pool_size)(x_ee2_conv)

    x_ee1_flatten = Flatten()(x_ee1_pooled)
    x_ee2_flatten = Flatten()(x_ee2_pooled)
    ee_1 = Dense(model_settings['label_count'], activation='softmax', name='ee_1')(x_ee1_flatten)
    ee_2 = Dense(model_settings['label_count'], activation='softmax', name='ee_2')(x_ee2_flatten)

    x = AveragePooling2D(pool_size=final_pool_size)(x)
    x = Flatten()(x)
    outputs = Dense(model_settings['label_count'], activation='softmax', name='ef_out')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[ee_1, ee_2, outputs])

  elif model_name == 'ds_cnn_eefmaps_concat':
    print("DS CNN model with EE-fmaps concat with final fmaps invoked")
    input_shape = [model_settings['spectrogram_length'], model_settings['dct_coefficient_count'],1]
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='ee_1_fmaps')(x)
    #add EE
    ee1_fmaps = x

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)


    #add EE
    slice_ind_depthconv_eefinal = 32
    conv_ee1_fmaps = tf.keras.layers.DepthwiseConv2D(name='depth_conv_ee_1',
                   kernel_size=[3,3],
                   strides=2,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_regularizer=regularizer)(ee1_fmaps[:,:,:,:slice_ind_depthconv_eefinal])
    # print(conv_ee1_fmaps.shape)
    pool_size = (2,2)
    conv_ee1_fmaps_pooled = AveragePooling2D(pool_size=pool_size)(conv_ee1_fmaps)
    y_ee_1 = Flatten(name='flatten_ee1')(conv_ee1_fmaps_pooled)
    ee_1_logits = Dense(model_settings['label_count'])(y_ee_1)
    ee_1 = Softmax(name='ee_out')(ee_1_logits)

    #concatenate the ee-fmaps with final-fmaps at the final exit
    x_pooled = AveragePooling2D(pool_size=final_pool_size)(x)
    y = Flatten()(x_pooled)
    y_ee_pooled = AveragePooling2D(pool_size=(13,3))(conv_ee1_fmaps)
    y_ee = Flatten()(y_ee_pooled[:,:,:,:slice_ind_depthconv_eefinal])
    y_combined = tf.keras.layers.concatenate([y, y_ee])
    ee_final = Dense(model_settings['label_count'], activation='softmax',name='ef_out')(y_combined)

    #add endpoint layer
    targets = Input(shape=[1], name='input_2')
    ee_1 = Endpoint_ee(name='endpoint', W_aux=model_settings['W_aux'], num_classes=model_settings['label_count'])(ee_1, targets)
    # Instantiate model.
    model = Model(inputs=[inputs, targets], outputs=[ee_1,ee_final])
  

  else:
    raise ValueError("Model name {:} not supported".format(model_name))


  #loss weights vary for each architecture
  loss_weights = get_loss_weights(model_name)
  model.compile(
      #optimizer=keras.optimizers.RMSprop(learning_rate=args.learning_rate),  # Optimizer
      optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),  # Optimizer
      # Loss function to minimize
      loss=keras.losses.SparseCategoricalCrossentropy(),
      loss_weights = loss_weights,
      # List of metrics to monitor
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

  return model
