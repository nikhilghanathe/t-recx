import os
import argparse
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras
import Config


def parse_command():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'data'),
      help="""\
      Where to download the speech training data to. Or where it is already saved.
      """)
  parser.add_argument(
      '--bg_path',
      type=str,
      default=os.path.join(os.getenv('PWD')),
      help="""\
      Where to find background noise folder.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=20.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--feature_type',
      type=str,
      default="mfcc",
      choices=["mfcc", "lfbe", "td_samples"],
      help='Type of input features. Valid values: "mfcc" (default), "lfbe", "td_samples"',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=10,
      help='How many MFCC or log filterbank energy features')
  parser.add_argument(
      '--epochs',
      type=int,
      default=36,
      help='How many epochs to train',)
  parser.add_argument(
      '--num_train_samples',
      type=int,
      default=-1, # 85511,
    help='How many samples from the training set to use',)
  parser.add_argument(
      '--num_val_samples',
      type=int,
      default=-1, # 10102,
    help='How many samples from the validation set to use',)
  parser.add_argument(
      '--num_test_samples',
      type=int,
      default=-1, # 4890,
    help='How many samples from the test set to use',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--num_bin_files',
      type=int,
      default=1000,
      help='How many binary test files for benchmark runner to create',)
  parser.add_argument(
      '--bin_file_path',
      type=str,
      default=os.path.join(os.getenv('HOME'), 'kws_test_files'),
      help="""\
      Directory where plots of binary test files for benchmark runner are written.
      """)
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='ds_cnn_ev',
      help='What model architecture to use from: [ds_cnn_ev, ds_cnn_noev, ds_cnn_sdn, ds_cnn_branchynet, ds_cnn_eefmaps_concat]')
  parser.add_argument(
      '--run_test_set',
      type=bool,
      default=True,
      help='In train.py, run model.eval() on test set if True')
  parser.add_argument(
      '--saved_model_path',
      type=str,
      default='trained_models/kws_model.h5',
      help='In quantize.py, path to load pretrained model from; in train.py, destination for trained model')
  parser.add_argument(
      '--model_init_path',
      type=str,
      default=None,
      help='Path to load pretrained model for evaluation or starting point for training')
  parser.add_argument(
      '--tfl_file_name',
      default='trained_models/kws_model.tflite',
      help='File name to which the TF Lite model will be saved (quantize.py) or loaded (eval_quantized_model)')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.00001,
      help='Initial LR',)
  parser.add_argument(
      '--lr_sched_name',
      type=str,
      default='step_function',
      help='lr schedule scheme name to be picked from lr.py')  
  parser.add_argument(
      '--plot_dir',
      type=str,
      default='./plots',
      help="""\
      Directory where plots of accuracy vs Epochs are stored
      """)
  parser.add_argument(
      '--target_set',
      type=str,
      default='test',
      help="""\
      For eval_quantized_model, which set to measure.
      """)
  parser.add_argument(
      '--model_save_name',
      type=str,
      default='trained_models/kws_model.h5',
      help='The model name under which the trained model is saved after training ')
  parser.add_argument(
      '--W_aux',
      type=float,
      default=0.3,
      help='Specify the weight of the auxiliary loss at the early exit. The paper uses a value of 0.3 ')
  parser.add_argument('--isEV', action="store_true", 
        help=""" Specify whether to use EV architecture. To use EV-assistance use this flag in command line (--isEV). Exclude for no EV-assistance """)
  parser.add_argument('--isTrecx', action="store_true", 
        help=""" Specify whether to use custom datagen for T-recx models. HAVE TO USE this: --isTrecx on cmdl for all T-recx models. Exclude to use normal datagen """)


  Flags, unparsed = parser.parse_known_args()
  return Flags, unparsed


def plot_training(plot_dir,history):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.subplot(2,1,1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(plot_dir+'/acc.png')

def step_function_wrapper(batch_size):
  
    def step_function(epoch, lr):
        if (epoch < 12):
            return 0.0005
        elif (epoch < 24):
            return 0.0001
        elif (epoch < 36):
            return 0.00002
        else:
            return 0.00001
    return step_function

def get_callbacks(args):
    lr_sched_name = args.lr_sched_name
    batch_size = args.batch_size
    initial_lr = args.learning_rate
    callbacks = None
    if(lr_sched_name == "step_function"):
        callbacks = [keras.callbacks.LearningRateScheduler(step_function_wrapper(batch_size),verbose=1)]
    return callbacks


    


# ----trecx util functions-------------------
# strip the endpoint layer, recompile and save model for testing
def save_trecx_model(model, model_save_name, model_arch):
  if model_arch=='ds_cnn_ev' or model_arch=='ds_cnn_noev' or model_arch=='ds_cnn_eefmaps_concat':
    cnt = 0
    for layer in model.layers:
        if layer.name=='ee_out':
            ee_layer_num = cnt
        if layer.name=='ef_out':
            ef_layer_num = cnt
        cnt+=1
    #construct new model
    final_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[ee_layer_num].output, model.layers[ef_layer_num].output])
    final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Optimizer
        loss=keras.losses.SparseCategoricalCrossentropy(),
        loss_weights=None,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
    final_model.save('trained_models/'+ model_save_name)
  elif model_arch=='ds_cnn_sdn':
    cnt = 0
    for layer in model.layers:
        if layer.name=='ee_1':
            ee1_layer_num = cnt
        if layer.name=='ee_2':
            ee2_layer_num = cnt
        if layer.name=='ee_3':
            ee2_layer_num = cnt
        if layer.name=='ef_out':
            ef_layer_num = cnt
        cnt+=1
    #construct new model
    final_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[ee1_layer_num].output, model.layers[ee2_layer_num].output, 
      model.layers[ee3_layer_num].output, model.layers[ef_layer_num].output])
    final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Optimizer
        loss=keras.losses.SparseCategoricalCrossentropy(),
        loss_weights=None,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
    final_model.save('trained_models/'+ model_save_name)
  else:
    model.save('trained_models/'+ model_save_name)




def get_loss_weights(model_arch):
  if model_arch=='ds_cnn_sdn':
    return Config.loss_weights_sdn
  elif model_arch=='ds_cnn_branchynet':
    return Config.loss_weights_branchynet
  else:
    return None
