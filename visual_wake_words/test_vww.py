# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os
import sys
from absl import app
from vww_model_test import mobilenet_v1_no_mv

import tensorflow as tf
import numpy as np
import json
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'minival')
  
# def customgen(gen):
#   while True:
#     (x, y) =gen.next()
#     yield x,y 

def main(argv):
  if len(argv) >= 2:
    model = tf.keras.models.load_model(argv[1])
  else:
    model = mobilenet_v1()
  model_name = argv[1]
  model.summary()

  batch_size = 50
  validation_split = 0.1

  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.05,
      height_shift_range=0.05,
      zoom_range=.1,
      horizontal_flip=True,
      # validation_split=validation_split,
      rescale=1. / 255)
  # train_generator = datagen.flow_from_directory(
  #     BASE_DIR,
  #     target_size=(IMAGE_SIZE, IMAGE_SIZE),
  #     batch_size=BATCH_SIZE,
  #     subset='training',
  #     color_mode='rgb')
  val_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      # subset='validation',
      color_mode='rgb')

  cnt=0
  for layer in model.layers:
    if layer.name=='ee_1_out':
      ee_layer_num = cnt
    if layer.name=='dense_1':
      eefinal_layer_num = cnt
    cnt+=1

  new_model = tf.keras.Model(inputs=model.inputs, outputs=[model.layers[ee_layer_num].output, model.layers[eefinal_layer_num].output ])

  #get flops data
  from keras_flops import get_flops
  # calc FLOPS
  flops = get_flops(new_model, batch_size=1)
  print(f"FLOPS: {flops / 10 ** 9:.03} G")
  # ss



  ref_model_name = 'trained_models/vww_96.h5'
  ref_model = tf.keras.models.load_model(ref_model_name)
  #test accuracy of ref model on val set
  # test_gen = customgen(val_generator)
  # test_metrics = ref_model.evaluate(val_generator, batch_size=BATCH_SIZE, verbose=1, return_dict=True)
  isBreak = False
  EE_correct, correct, count = 0, 0, 0
  prediction_dict_ee1, prediction_dict_eefinal = {}, {}
  for val_batch in val_generator:
    (test_sample, test_label) = val_batch[0], val_batch[1]
    #generator loop indefintely, so add manual break when you reach dataset size
    if isBreak:
      break

    BS = test_sample.shape[0]
    prediction = model.predict([test_sample, test_label], batch_size=BS)
    prediction_ee1, prediction_eefinal = prediction[0], prediction[1]
    for i in range(0, BS):
      if count==4448:#generator loop indefintely, so add manual break when you reach dataset size
        isBreak = True
        break
      truth = np.argmax(test_label[i])

      prob_list = prediction_ee1[i]
      arg_max_1 = np.argmax(prob_list)
      score_max_1 = max(prob_list)

      if int(truth) == int(arg_max_1):
            isCorrect = True
            EE_correct +=1
      else:
          isCorrect = False
      prediction_dict_ee1.update({str(count): {'probability':[str(i) for i in list(prob_list)],  'prediction':str(arg_max_1), 'truth':str(truth), 'isCorrect': str(isCorrect), 'score_max_1' : str(score_max_1), 'arg_max_1' : str(arg_max_1),  'isEE': 'True' }})            
       

      prob_list = prediction_eefinal[i]
      arg_max_1 = np.argmax(prob_list)
      score_max_1 = max(prob_list)

      if int(truth) == int(arg_max_1):
            isCorrect = True
            correct +=1
      else:
          isCorrect = False
      prediction_dict_eefinal.update({str(count): {'probability':[str(i) for i in list(prob_list)],  'prediction':str(arg_max_1), 'truth':str(truth), 'isCorrect': str(isCorrect), 'score_max_1' : str(score_max_1), 'arg_max_1' : str(arg_max_1)}})            

      count+=1

  #dump trace data
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1_ref.json', 'w') as fp:
     json.dump(prediction_dict_ee1, fp, indent=4)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_eefinal_ref.json', 'w') as fp:
     json.dump(prediction_dict_eefinal, fp, indent=4)

  print('The number of times we are correct is = ', correct)
  # print('The number of times we EE = ', EE_cnt)
  print(' correct predictions from EE = ', EE_correct)
  # print(' incorrect predictions from EE = ', EE_cnt-EE_correct)
  # print(' EE accuarcy = ', (EE_correct/EE_cnt)*100)
  print(' total accuarcy of EE-1 is = ', (EE_correct/count) *100)
  print(' total accuarcy of EE-final is = ', (correct/(count)) *100)

if __name__ == '__main__':
  # app.run(main)
  main(sys.argv)