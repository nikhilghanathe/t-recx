import os
import sys
from absl import app

import tensorflow as tf
import numpy as np
import json

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'minival')



def analyse(model_name):

  with open('trace_data/'+'trace_data_'+model_name+'_ee1_ref.json', 'r') as fp:
    pred_ee1_dict = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name+'_eefinal_ref.json', 'r') as fp:
    pred_eefinal_dict = json.load(fp)

  #metrics for ee-1
  EE_final_keys = []
  EE_cnt, EE_correct, correct = 0, 0, 0
  for num, pred in pred_ee1_dict.items():
    prob_list = [float(x) for x in pred['probability']]
    truth = int(pred['truth'])
    arg_max_1 = np.argmax(prob_list)
    arg_max_2 = np.argsort(prob_list)[-2]
    isCorrect = truth==arg_max_1
    prob_list = [float(x) for x in pred['probability']]
    score_max_1, score_max_2 = float(pred['score_max_1']), min(prob_list)

    if score_max_1>=1:
      EE_cnt+=1
      if isCorrect:
        EE_correct +=1
    else:
      EE_final_keys.append(num)


  #metrics for EE-final
  for num in EE_final_keys:
    pred = pred_eefinal_dict[num]
    prob_list = [float(x) for x in pred['probability']]
    truth = int(pred['truth'])
    arg_max_1 = np.argmax(prob_list)
    arg_max_2 = np.argsort(prob_list)[-2]
    isCorrect = truth==arg_max_1
    prob_list = [float(x) for x in pred['probability']]
    score_max_1, score_max_2 = float(pred['score_max_1']), min(prob_list)

    pred_ee1 = pred_ee1_dict[num]
    arg_max_1_ee1 = pred_ee1['arg_max_1']
    score_max_1_ee1 = float(pred_ee1['score_max_1'])
    
    # if score_max_1_ee1 > score_max_1:
    #   isCorrect = truth==arg_max_1_ee1
    # else:
    #   isCorrect = truth==arg_max_1

    if isCorrect:
      correct+=1


  total = len(pred_eefinal_dict.keys())

  print('Total samples = ', total)
  print('The number of times we EE = ', EE_cnt)
  print('EE % = ', EE_cnt/total *100)
  print('The number of times we EE is Incorrect = ', EE_cnt-EE_correct)
  print('The number of times we EE is Correct = ', EE_correct)

  print('Accuracy of EE-1 = ', EE_correct/EE_cnt)
  print('The number of times we go to EE-final = ', len(EE_final_keys))
  print('Accuracy of EE-final =', correct/len(EE_final_keys))
  print('total accuracy = ', (EE_correct+correct)/total  )



if __name__ == '__main__':
  model_name = sys.argv[1]
  # model = tf.keras.models.load_model(model_name)
  # model.summary()
  

  # generate_trace(Flags.model_init_path)  

  analyse(model_name[15:])