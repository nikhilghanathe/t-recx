'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

keras_model.py: CIFAR10_ResNetv1 from eembc
'''

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

#get model
def get_model_name():
    if os.path.exists("trained_models/trainedResnet.h5"):
        return "trainedResnet"
    else:
        return "pretrainedResnet"

def get_quant_model_name():
    if os.path.exists("trained_models/trainedResnet.h5"):
        return "trainedResnet"
    else:
        return "pretrainedResnet"

#define model
def resnet_v1_eembc():
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)



    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Fourth stack.
    # While the paper uses four stacks, for cifar10 that leads to a large increase in complexity for minor benefits
    # Uncomments to use it

    ## Weight layers
    # num_filters = 128
    # y = Conv2D(num_filters,
    #              kernel_size=3,
    #              strides=2,
    #              padding='same',
    #              kernel_initializer='he_normal',
    #              kernel_regularizer=l2(1e-4))(x)
    # y = BatchNormalization()(y)
    # y = Activation('relu')(y)
    # y = Conv2D(num_filters,
    #              kernel_size=3,
    #              strides=1,
    #              padding='same',
    #              kernel_initializer='he_normal',
    #              kernel_regularizer=l2(1e-4))(y)
    # y = BatchNormalization()(y)

    # # Adjust for change in dimension due to stride in identity
    # x = Conv2D(num_filters,
    #              kernel_size=1,
    #              strides=2,
    #              padding='same',
    #              kernel_initializer='he_normal',
    #              kernel_regularizer=l2(1e-4))(x)

    # # Overall residual, connect weight layer and identity paths
    # x = tf.keras.layers.add([x, y])
    # x = Activation('relu')(x)
    
    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model







# =========================keras model for t-recx=============================
#define endpoint layer for loss calculation
class Endpoint_ee(tf.keras.layers.Layer):
    def __init__(self, name=None, W_aux=0.5, num_classes=10):
        super().__init__(name=name)
        self.batch_size = 32

    @tf.function
    def loss_fn(self, ee_1, ef_out, targets):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        W_aux = 0.5
        P_aux = 1.0
        num_classes=10

        self.batch_size = targets.shape[0]
        y_true = targets
        y_true_transformed_ee1 = []
        y_pred_ee1 = ee_1
        
        if self.batch_size==None:
            self.batch_size=32
        loss_ee1, loss_eefinal =0.0, 0.0

        #for EE-1
        for i in range(0, self.batch_size):
            arg_max_true = tf.keras.backend.argmax(targets[i])
            arg_max_true = tf.cast(arg_max_true, dtype='int32')
            prob_list = y_pred_ee1[i]
            values, indices =  tf.math.top_k(prob_list, k=3)            
            [score_max_1, score_max_2, score_max_3] = tf.split(values, num_or_size_splits=3)
            [arg_max_1, arg_max_2, arg_max_3] = tf.split(indices, num_or_size_splits=3)
            arg_max_true = tf.reshape(arg_max_true, [1])
            if tf.math.equal(arg_max_true, arg_max_1):
              if tf.math.less_equal(tf.math.subtract(score_max_1, score_max_2), tf.constant(0.3)):
                y_uncrtn = tf.one_hot([arg_max_true], depth=num_classes, on_value=1., off_value=0.0, dtype='float32')
              else:
                y_uncrtn = tf.one_hot([arg_max_true], depth=num_classes, on_value=P_aux, off_value=0.0, dtype='float32')
            else:
                y_uncrtn = tf.one_hot([arg_max_true], depth=num_classes, on_value=P_aux, off_value=0.0, dtype='float32')
            y_true_transformed_ee1.append(y_uncrtn)
        y_true_transformed_ee1 = tf.reshape(y_true_transformed_ee1, [self.batch_size,num_classes])
        
        loss_cce =  cce(y_true_transformed_ee1, y_pred_ee1) 
        return tf.multiply(W_aux, loss_cce)


# #define endpoint layer for loss calculation
# class Endpoint_ee(tf.keras.layers.Layer):
#     def __init__(self, name=None, W_aux=0.5, num_classes=10):
#         super().__init__(name=name)
#         self.batch_size = 1
#         self.W_aux = W_aux
#         self.num_classes = num_classes

#     @tf.function
#     def loss_fn(self, softmax_output, ef_out, targets):
#         cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#         self.batch_size = targets.shape[0]
        
#         y_true, y_true_transformed = [], []
#         if self.batch_size==None:
#             self.batch_size=32
#         #compute the one hot 'y_true' vector for loss computation
#         for i in range(0, self.batch_size):
#             arg_max_true = tf.keras.backend.argmax(targets[i])
#             arg_max_true = tf.cast(arg_max_true, dtype='int32')
#             arg_max_true = tf.reshape(arg_max_true, [1])
#             print(tf.one_hot([arg_max_true], depth=self.num_classes, on_value=1., off_value=0.0, dtype='float32'))
#             y_true.append(tf.one_hot([arg_max_true], depth=self.num_classes, on_value=1., off_value=0.0, dtype='float32'))
        
#         #compute loss for whole batch
#         y_true_transformed = tf.reshape(y_true, [self.batch_size,self.num_classes])
#         loss_cce =  cce(y_true_transformed, softmax_output) 
      
#         return tf.multiply(self.W_aux, loss_cce)
        

    def call(self, softmax_output, ef_out, targets=None,   sample_weight=None):
        if targets is not None:
            loss = self.loss_fn(softmax_output, ef_out, targets)
            self.add_loss(loss)
            self.add_metric(loss, name='aux_loss', aggregation='mean')
        #for inference
        return softmax_output, ef_out







#define model without early-view assistance
def resnet_v1_eembc_with_noEV(W_aux):
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10  # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,name='conv2d',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Activation('relu', name='activation')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,name='conv2d_1',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization(name='batch_normalization_1')(y)
    y = Activation('relu', name='activation_1')(y)
    y = Conv2D(num_filters,name='conv2d_2',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization(name='batch_normalization_2')(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y], name='add')
    x = Activation('relu', name='activation_2')(x)
    x_ee_1_fmaps = x

    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,name='conv2d_3',
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization(name='batch_normalization_3')(y)
    y = Activation('relu', name='activation_3')(y)
    y = Conv2D(num_filters,name='conv2d_4',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization(name='batch_normalization_4')(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,name='conv2d_5',
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y], name='add_1')
    x = Activation('relu', name='activation_4')(x)


    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,name='conv2d_6',
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization(name='batch_normalization_5')(y)
    y = Activation('relu', name='activation_5')(y)
    y = Conv2D(num_filters,name='conv2d_7',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization(name='batch_normalization_6')(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,name='conv2d_8',
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y], name='add_2')
    x = Activation('relu', name='activation_6')(x)


    #add early exit
    #extend the channels of x_ee_1 to match that of the output from the last stack
    x_ee_1 = Conv2D(32,
                 name='pointwise_conv_ee_1',
                 kernel_size=1,
                 strides=1,
                 padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(1e-4))(x_ee_1_fmaps)
    
    # Add a depthwise conv without batch norm and depth_multiplier=1
    dconv_ee = tf.keras.layers.DepthwiseConv2D(name='depth_conv_ee_1',
                   kernel_size=[2,2],
                   strides=2,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x_ee_1)
    
    pool_size = (4,4)
    pool_ee = AveragePooling2D(pool_size=pool_size, name='average_pooling_2d_1')(dconv_ee)
    y_ee_1 = Flatten(name='flatten_1')(pool_ee)
    ee_1 = Dense(num_classes, name='ee_out',
                    activation='softmax',
                    kernel_initializer='he_normal')(y_ee_1)

    


    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x_pooled = AveragePooling2D(pool_size=pool_size, name='average_pooling_2d')(x)
    
    y = Flatten(name='flatten_x_pooled')(x_pooled)
    
    outputs = Dense(num_classes, name='ef_out',
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    #define a endpoint layer
    targets =  Input(shape=[num_classes])
    ee_1, outputs  = Endpoint_ee(name='endpoint', W_aux=W_aux, num_classes=num_classes)(ee_1, outputs, targets)


    # Instantiate model.
    model = Model(inputs=[inputs, targets], outputs=[ee_1, outputs])
    return model










#define model with early-view assistance
def resnet_v1_eembc_with_EV(W_aux):
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10  # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,name='conv2d',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Activation('relu', name='activation')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,name='conv2d_1',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization(name='batch_normalization_1')(y)
    y = Activation('relu', name='activation_1')(y)
    y = Conv2D(num_filters,name='conv2d_2',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization(name='batch_normalization_2')(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y], name='add')
    x = Activation('relu', name='activation_2')(x)
    x_ee_1_fmaps = x

    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,name='conv2d_3',
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization(name='batch_normalization_3')(y)
    y = Activation('relu', name='activation_3')(y)
    y = Conv2D(num_filters,name='conv2d_4',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization(name='batch_normalization_4')(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,name='conv2d_5',
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y], name='add_1')
    x = Activation('relu', name='activation_4')(x)


    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,name='conv2d_6',
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization(name='batch_normalization_5')(y)
    y = Activation('relu', name='activation_5')(y)
    y = Conv2D(num_filters,name='conv2d_7',
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization(name='batch_normalization_6')(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,name='conv2d_8',
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y], name='add_2')
    x = Activation('relu', name='activation_6')(x)


    #add early exit
    #extend the channels of x_ee_1 to match that of the output from the last stack
    x_ee_1 = Conv2D(32,
                 name='pointwise_conv_ee_1',
                 kernel_size=1,
                 strides=1,
                 padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(1e-4))(x_ee_1_fmaps)
    
    # Add a depthwise conv without batch norm and depth_multiplier=1
    dconv_ee = tf.keras.layers.DepthwiseConv2D(name='depth_conv_ee_1',
                   kernel_size=[2,2],
                   strides=2,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x_ee_1)
    
    pool_size = (4,4)
    pool_ee = AveragePooling2D(pool_size=pool_size, name='average_pooling_2d_1')(dconv_ee)
    y_ee_1 = Flatten(name='flatten_1')(pool_ee)
    ee_1 = Dense(num_classes, name='ee_out',
                    activation='softmax',
                    kernel_initializer='he_normal')(y_ee_1)

    

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x_pooled = AveragePooling2D(pool_size=pool_size, name='average_pooling_2d')(x)
    
    slice_ind_depthconv_eefinal = 32
    depth_conv_eefinal_out = tf.keras.layers.DepthwiseConv2D(name='depth_conv_eefinal_out',
                   kernel_size=[2,2],
                   strides=2,
                   padding='same',
                   depth_multiplier=1,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x[:,:,:,:slice_ind_depthconv_eefinal])

    y_depthconv_eefinal = Flatten(name='flatten_y_depthconv_eefinal')(depth_conv_eefinal_out[:,:,:,:slice_ind_depthconv_eefinal])

    y = Flatten(name='flatten_x_pooled')(x_pooled)
    y_combined = tf.keras.layers.concatenate([y, y_depthconv_eefinal])

    outputs = Dense(num_classes, name='ef_out',
                    activation='softmax',
                    kernel_initializer='he_normal')(y_combined)

    #define a endpoint layer
    targets =  Input(shape=[num_classes])
    ee_1, outputs  = Endpoint_ee(name='endpoint', W_aux=W_aux, num_classes=num_classes)(ee_1, outputs, targets)




    # Instantiate model.
    model = Model(inputs=[inputs, targets], outputs=[ee_1, outputs])
    return model








# ------------------keras models for branchynet and SDN-------------------
#define model
def resnet_v1_branchynet():
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    x_ee1 = x
    num_filters_ee1 = num_filters
    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    x_ee2 = x
    num_filters_ee2 = num_filters

    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    #Early-exits  
    x_ee1_conv = Conv2D(num_filters_ee1,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x_ee1)
    x_ee2_conv = Conv2D(num_filters_ee1,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x_ee2)


    pool_size = [4,4]                  
    x_ee1_pooled = AveragePooling2D(pool_size=pool_size)(x_ee1_conv)
    x_ee2_pooled = AveragePooling2D(pool_size=pool_size)(x_ee2_conv)

    x_ee1_flatten = Flatten()(x_ee1_pooled)
    x_ee2_flatten = Flatten()(x_ee2_pooled)
    targets = Input(shape=[1], name='input_2')
    ee_1 = Dense(num_classes, activation='softmax', name='ee_1', kernel_initializer='he_normal')(x_ee1_flatten)
    ee_2 = Dense(num_classes, activation='softmax', name='ee_2', kernel_initializer='he_normal')(x_ee2_flatten)


    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax',name='ef_out',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[ee_1, ee_2, outputs])
    return model
















#custom loss function that uses tau - As mentioned in the SDN paper
#tau starts from 0.01 and steadily increases to relative inference cost (0.33, 0.66) as epochs go by - two early exits
#define endpoint layer for loss calculation
class SDN_loss(tf.keras.layers.Layer):
  def __init__(self, name=None, max_tau=0.3):
    super().__init__(name=name)
    self.batch_size = 32
    self.tau = 0.01
    self.epoch = 0 #set on epoch_begin
    self.epochs = 500 #set on train_begin
    self.max_tau = max_tau
    self.tau_incr = self.max_tau/self.epochs
 

  @tf.function
  def loss_fn(self, ee, targets):
      #tf.print(self.epoch, self.epochs)
      cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
      W_aux = self.tau + (self.epoch * self.tau_incr)
      loss_cce = cce(targets, ee)
      return tf.multiply(W_aux, loss_cce)

  def call(self, ee, targets=None, sample_weight=None):
      if targets is not None:
          loss = self.loss_fn(ee, targets)
          self.add_loss(loss)
          self.add_metric(loss, name='aux_loss_'+self.name, aggregation='mean')
      return ee


#define model
def resnet_v1_sdn():
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    x_ee1 = x
    num_filters_ee1 = num_filters
    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    x_ee2 = x
    num_filters_ee2 = num_filters

    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)



    #Early-exits
    x_ee1_pooled = AveragePooling2D(pool_size=[4,4])(x_ee1)
    x_ee2_pooled = AveragePooling2D(pool_size=[4,4])(x_ee2)
    
    x_ee1_flatten = Flatten()(x_ee1_pooled)
    x_ee2_flatten = Flatten()(x_ee2_pooled)
    
    ee_1 = Dense(num_classes, activation='softmax', name='ee_1', kernel_initializer='he_normal')(x_ee1_flatten)
    ee_2 = Dense(num_classes, activation='softmax', name='ee_2', kernel_initializer='he_normal')(x_ee2_flatten)

    #custom loss for early-exits according to sdn training
    targets = Input(shape=[num_classes,], name='input_2')
    ee_1 = SDN_loss(name='ee_1_loss', max_tau=0.33)(ee_1, targets)
    ee_2 = SDN_loss(name='ee_2_loss', max_tau=0.66)(ee_2, targets)


    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax', name='ef_out',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=[inputs, targets], outputs=[ee_1, ee_2, outputs])
    return model



# Model with baseline EE of simple average pooling at early exit
def resnet_v1_baselineEE():
    # Resnet parameters
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    ee_1_fmaps =x
    


    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    ee_2_fmaps =x


    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)


    #First EE without uncertain class
    # pool_size = int(np.amin(x.shape[1:3]))
    pool_size = (4,4)
    x_ee_1 = AveragePooling2D(pool_size=pool_size)(ee_1_fmaps)
    y_ee_1 = Flatten()(x_ee_1)
    ee_1 = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal', name='ee_out')(y_ee_1)
    
    # pool_size = (4,4)
    # x_ee_2 = AveragePooling2D(pool_size=pool_size)(ee_2_fmaps)
    # y_ee_2 = Flatten()(x_ee_2)
    # ee_2 = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y_ee_2)
    
    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal', name='ef_out')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[ee_1, outputs])
    return model
