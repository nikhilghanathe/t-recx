'''
Project: t-recx
Subproject: Image classification on cifar10
File:helpers.py 
desc: helper functions to plot benefit curve, generate trace data and to report standalone accuracies

'''

import os, sys, json
import matplotlib.pyplot as plt	
import numpy as np
from keras_flops import get_flops
import tensorflow as tf
import Config


#read the trace data 
# takes rho as an argument. Reads the trace data and calculates the overall accuracy of the model based on Eq1 & Eq2 from the paper
def calc_accuracy(model_name, rho, total_samples):
	#read trace data of model
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee.json', 'r') as fp:
		predict_dict_ee1 = json.load(fp)
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ef.json', 'r') as fp:
		predict_dict_eefinal = json.load(fp)

	EE_cnt, EE_1_correct, EE_final_correct = 0,0,0	
	for num, pred in predict_dict_ee1.items():
		truth = int(pred['truth'])
		arg_max_1 = int(pred['arg_max_1'])
		isCorrect = truth==arg_max_1
		score_max_1 = float(pred['score_max_1'])
		if score_max_1>rho:#if score > rho then early-exit
			if isCorrect:
				EE_1_correct +=1
			EE_cnt +=1
		else:
			pred_eefinal = predict_dict_eefinal[num]
			arg_max_1 = int(pred_eefinal['arg_max_1'])
			isCorrect = truth==arg_max_1
			if isCorrect:
				EE_final_correct+=1
	return (EE_1_correct/total_samples)*100, (EE_final_correct/total_samples)*100, (EE_cnt/total_samples), ((EE_1_correct+EE_final_correct)*100)/total_samples

#return flops when using 1)early exit and 2) final exit
def get_flops_ee_ef(model_name):
	model = tf.keras.models.load_model(model_name)
	#get flops of ee and ef 
	exit_list = ['ee_out', 'ef_out']
	flops = []
	for i in range(0, len(exit_list)):
		new_model = get_ee_model(model, exit_list[:i+1])
		flops.append(get_flops(new_model))
	return flops[0], flops[1]


# return points for scatter plot
def calculate_scatter_points(model_name, total_samples):
	#get flops count when using ee only and when using ee+ef
	latency_ee, latency_ef = get_flops_ee_ef(model_name)
	x_axis_accuracy, y_axis_flops =[], []
	#vary the ee exit confidence criteria (rho from Eq1)  from 0.0 to 1.0 in steps of 0.01
	for rho in list(np.linspace(0.01,1.0, 101)):
		ee_1_accuracy, ee_final_accuracy, ee_percent, total_accuracy = calc_accuracy(model_name, rho, total_samples)
		flops_ee1 = latency_ee* ee_percent
		flops_eefinal = (latency_ef) * (1-ee_percent)
		flops_total = flops_ee1+flops_eefinal
		flops_total = flops_total/1000000 #divide by 1.0E+6 
		x_axis_accuracy.append(total_accuracy)
		y_axis_flops.append(flops_total)
	return x_axis_accuracy, y_axis_flops

def collect_pred_metrics(prob_list):
	score_max_1, score_max_2 = sorted(prob_list)[-1], sorted(prob_list)[-2]
	arg_max_1, arg_max_2 = np.argmax(prob_list), np.argsort(prob_list)[-2]
	return score_max_1, score_max_2, arg_max_1, arg_max_2

#generate trace data and dump into json files for faster access to prediction behavior
def generate_trace(val_generator, model_name):
	if os.path.exists('trace_data/'+'trace_data_'+model_name[15:]+'_ee.json'):
		print('Trace data already exists for ', model_name)
		return
	#load model
	model = tf.keras.models.load_model(model_name)
  #generate predictions and dump into json file
	count = 0
	prediction_dict_ee1, prediction_dict_eefinal = {}, {}
	for val_batch in val_generator:
	  (test_sample, test_label) = val_batch[0], val_batch[1]
	  BS = test_sample.shape[0]
	  prediction = model.predict(test_sample, batch_size=BS)#batch prediction
	  prediction_ee1, prediction_eefinal = prediction[0], prediction[1]
	  #collect trace for each pred in batch
	  for i in range(0, BS):
	  	truth = test_label.numpy()[i]
	  	arg_max_1_ee1, arg_max_1_eefinal  = np.argmax(prediction_ee1[i]), np.argmax(prediction_eefinal[i])
	  	score_max_1_ee1, score_max_1_eefinal = max(prediction_ee1[i]), max(prediction_eefinal[i])

	  	isCorrect = int(truth) == int(arg_max_1_ee1)
	  	prediction_dict_ee1.update({str(count): {'prediction':str(arg_max_1_ee1), 'truth':str(truth), 'isCorrect': str(isCorrect), 
	      'score_max_1' : str(score_max_1_ee1), 'arg_max_1' : str(arg_max_1_ee1) }})
	  	isCorrect = int(truth) == int(arg_max_1_eefinal)
	  	prediction_dict_eefinal.update({str(count): {'prediction':str(arg_max_1_eefinal), 'truth':str(truth), 'isCorrect': str(isCorrect), 
			  'score_max_1' : str(score_max_1_eefinal), 'arg_max_1' : str(arg_max_1_eefinal) }})

	  	count+=1

	if not os.path.exists('trace_data/'):
		os.makedirs('trace_data')
	#dump trace data
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee.json', 'w') as fp:
	 json.dump(prediction_dict_ee1, fp, indent=4)
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ef.json', 'w') as fp:
	 json.dump(prediction_dict_eefinal, fp, indent=4)




# -----------------------------------------------------------------------------------------
# -------------Helper functions for comparison with branchynet and SDN---------------------
# -----------------------------------------------------------------------------------------
def get_ee_model(model, exit_names):
	cnt = 0
	output_layers = []
	for layer in model.layers:
		if layer.name in exit_names:
		 	output_layers.append(model.layers[cnt].output)
		cnt+=1
	new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=output_layers)
	new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return new_model

def get_flops_prior(model_name, model_arch):
	model = tf.keras.models.load_model(model_name)
	flops = []
	if model_arch=='branchynet':
		exit_list = ['ee_1', 'ee_2', 'ef_out']
		for i in range(0, len(exit_list)):
			new_model = get_ee_model(model, exit_list[:i+1])
			flops.append(get_flops(new_model))
		return flops[0], flops[1], flops[2]
	else:
		exit_list = ['ee_1', 'ee_2', 'ee_3', 'ef_out']
		for i in range(0, len(exit_list)):
			new_model = get_ee_model(model, exit_list[:i+1])
			flops.append(get_flops(new_model))
		return flops[0], flops[1], flops[2], flops[3]




# =========collect trace data =================================
def generate_trace_prior(val_generator, model_name, model_arch):
	if os.path.exists('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json'):
		print('Trace data already exists for ', model_name)
		return
	model = tf.keras.models.load_model(model_name)
	#generate predictions and dump into json file
	count = 0
	if model_arch=='branchynet':
		prediction_dict_ee1, prediction_dict_ee2, prediction_dict_eefinal = {}, {}, {}
		for val_batch in val_generator:
		  (test_sample, test_label) = val_batch[0], val_batch[1]
		  BS = test_sample.shape[0]
		  prediction = model.predict(test_sample, batch_size=BS)#batch prediction
		  prediction_ee1, prediction_ee2,  prediction_eefinal = prediction[0], prediction[1], prediction[2]
		  #collect trace for each pred in batch
		  for i in range(0, BS):
		  	truth = test_label.numpy()[i]
		  	arg_max_1_ee1, arg_max_1_ee2, arg_max_1_eefinal  = np.argmax(prediction_ee1[i]), np.argmax(prediction_ee2[i]), np.argmax(prediction_eefinal[i])
		  	score_max_1_ee1, score_max_1_ee2, score_max_1_eefinal = max(prediction_ee1[i]), max(prediction_ee2[i]), max(prediction_eefinal[i])

		  	isCorrect = int(truth) == int(arg_max_1_ee1)
		  	prediction_dict_ee1.update({str(count): {'prediction':str(arg_max_1_ee1), 'truth':str(truth), 'isCorrect': str(isCorrect), 
		      'score_max_1' : str(score_max_1_ee1), 'arg_max_1' : str(arg_max_1_ee1) }})
		  	isCorrect = int(truth) == int(arg_max_1_ee2)
		  	prediction_dict_ee2.update({str(count): {'prediction':str(arg_max_1_ee2), 'truth':str(truth), 'isCorrect': str(isCorrect), 
		      'score_max_1' : str(score_max_1_ee2), 'arg_max_1' : str(arg_max_1_ee2) }})
		  	isCorrect = int(truth) == int(arg_max_1_eefinal)
		  	prediction_dict_eefinal.update({str(count): {'prediction':str(arg_max_1_eefinal), 'truth':str(truth), 'isCorrect': str(isCorrect), 
				  'score_max_1' : str(score_max_1_eefinal), 'arg_max_1' : str(arg_max_1_eefinal) }})
		  	count+=1

  	#dump trace data
		with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json', 'w') as fp:
		 json.dump(prediction_dict_ee1, fp, indent=4)
		with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2.json', 'w') as fp:
		 json.dump(prediction_dict_ee2, fp, indent=4)
		with open('trace_data/'+'trace_data_'+model_name[15:]+'_ef.json', 'w') as fp:
		 json.dump(prediction_dict_eefinal, fp, indent=4)
	else:
		prediction_dict_ee1, prediction_dict_ee2, prediction_dict_ee3, prediction_dict_eefinal = {}, {}, {}, {}
		for val_batch in val_generator:
		  (test_sample, test_label) = val_batch[0], val_batch[1]
		  BS = test_sample.shape[0]
		  prediction = model.predict(test_sample, batch_size=BS)#batch prediction
		  prediction_ee1, prediction_ee2, prediction_ee3,  prediction_eefinal = prediction[0], prediction[1], prediction[2], prediction[3]
		  #collect trace for each pred in batch
		  for i in range(0, BS):
		  	truth = test_label.numpy()[i]
		  	arg_max_1_ee1, arg_max_1_ee2, arg_max_1_ee3, arg_max_1_eefinal  = np.argmax(prediction_ee1[i]), np.argmax(prediction_ee2[i]), np.argmax(prediction_ee3[i]), np.argmax(prediction_eefinal[i])
		  	score_max_1_ee1, score_max_1_ee2, score_max_1_ee3, score_max_1_eefinal = max(prediction_ee1[i]), max(prediction_ee2[i]), max(prediction_ee3[i]), max(prediction_eefinal[i])

		  	isCorrect = int(truth) == int(arg_max_1_ee1)
		  	prediction_dict_ee1.update({str(count): {'prediction':str(arg_max_1_ee1), 'truth':str(truth), 'isCorrect': str(isCorrect), 
		      'score_max_1' : str(score_max_1_ee1), 'arg_max_1' : str(arg_max_1_ee1) }})
		  	isCorrect = int(truth) == int(arg_max_1_ee2)
		  	prediction_dict_ee2.update({str(count): {'prediction':str(arg_max_1_ee2), 'truth':str(truth), 'isCorrect': str(isCorrect), 
		      'score_max_1' : str(score_max_1_ee2), 'arg_max_1' : str(arg_max_1_ee2) }})
		  	isCorrect = int(truth) == int(arg_max_1_ee3)
		  	prediction_dict_ee3.update({str(count): {'prediction':str(arg_max_1_ee3), 'truth':str(truth), 'isCorrect': str(isCorrect), 
		      'score_max_1' : str(score_max_1_ee3), 'arg_max_1' : str(arg_max_1_ee3) }})
		  	isCorrect = int(truth) == int(arg_max_1_eefinal)
		  	prediction_dict_eefinal.update({str(count): {'prediction':str(arg_max_1_eefinal), 'truth':str(truth), 'isCorrect': str(isCorrect), 
				  'score_max_1' : str(score_max_1_eefinal), 'arg_max_1' : str(arg_max_1_eefinal) }})
		  	count+=1
  	#dump trace data
		with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json', 'w') as fp:
		 json.dump(prediction_dict_ee1, fp, indent=4)
		with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2.json', 'w') as fp:
		 json.dump(prediction_dict_ee2, fp, indent=4)
		with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee3.json', 'w') as fp:
		 json.dump(prediction_dict_ee3, fp, indent=4)
		with open('trace_data/'+'trace_data_'+model_name[15:]+'_ef.json', 'w') as fp:
		 json.dump(prediction_dict_eefinal, fp, indent=4)


	


# return points for scatter plot
def calculate_scatter_points_prior(model_name, model_arch, total_samples):
	#get flops count when using ee1 only, ee1+ee2 and when using ee1+ee2+ef
	if model_arch=='branchynet':
		flops_ee1, flops_ee2, flops_eefinal = get_flops_prior(model_name, model_arch)
	else:
		flops_ee1, flops_ee2, flops_ee3, flops_eefinal = get_flops_prior(model_name, model_arch)
	x_axis_accuracy, y_axis_flops =[], []
	#vary the ee exit confidence criteria (rho from Eq1)  from 0.0 to 1.0 in steps of 0.01
	for rho in list(np.linspace(0.01,1.0, 101)):
		if model_arch=='branchynet':
			EE_1_cnt,EE_2_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_final_correct = calc_accuracy_branchynet(model_name, rho)
			total_accuracy = ((EE_1_correct+EE_2_correct+ EE_final_correct)*100)/total_samples
			flops_total = (flops_ee1*EE_1_cnt + flops_ee2*EE_2_cnt + flops_eefinal*EE_final_cnt )/total_samples
		else:
			EE_1_cnt,EE_2_cnt,EE_3_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_3_correct, EE_final_correct = calc_accuracy_sdn(model_name, rho)
			total_accuracy = ((EE_1_correct+EE_2_correct+EE_2_correct+EE_final_correct)*100)/total_samples
			flops_total = (flops_ee1*EE_1_cnt + flops_ee2*EE_2_cnt + flops_ee3*EE_3_cnt + flops_eefinal*EE_final_cnt )/total_samples
		flops_total = flops_total/1000000 #divide by 1.0E+6 
		x_axis_accuracy.append(total_accuracy)
		y_axis_flops.append(flops_total)
	return x_axis_accuracy, y_axis_flops


#benefit curve
def calc_accuracy_branchynet(model_name, rho):
  #read trace data of model
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json', 'r') as fp:
    predict_dict_ee1 = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2.json', 'r') as fp:
    predict_dict_ee2 = json.load(fp)
  with open('trace_data/''trace_data_'+model_name[15:]+'_ef.json', 'r') as fp:
    predict_dict_eefinal = json.load(fp)

  EE_1_correct, EE_2_correct, EE_final_correct = 0,0,0
  EE_1_cnt, EE_2_cnt, EE_final_cnt = 0,0,0

  for num, pred in predict_dict_ee1.items():
    truth = int(pred['truth'])
    arg_max_1, score_max_1 = int(pred['arg_max_1']), float(pred['score_max_1'])
    if score_max_1>=rho:#if score> rho then early-exit here; else move to next exit
      if truth==arg_max_1:
        EE_1_correct +=1
      EE_1_cnt +=1
    else:
      pred_ee2 = predict_dict_ee2[num]
      arg_max_1, score_max_1 = int(pred_ee2['arg_max_1']), float(pred_ee2['score_max_1'])
      if score_max_1>=rho:
        if truth==arg_max_1:
          EE_2_correct+=1
        EE_2_cnt +=1
      else:
        pred_eefinal = predict_dict_eefinal[num]
        arg_max_1, score_max_1 = int(pred_eefinal['arg_max_1']), float(pred_eefinal['score_max_1'])
        if truth==arg_max_1:
          EE_final_correct+=1
        EE_final_cnt +=1
  return  EE_1_cnt,EE_2_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_final_correct



#benefit curve
def calc_accuracy_sdn(model_name, rho):
  #read trace data of model
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json', 'r') as fp:
    predict_dict_ee1 = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2.json', 'r') as fp:
    predict_dict_ee2 = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee3.json', 'r') as fp:
    predict_dict_ee3 = json.load(fp)
  with open('trace_data/''trace_data_'+model_name[15:]+'_ef.json', 'r') as fp:
    predict_dict_eefinal = json.load(fp)

  EE_1_correct, EE_2_correct, EE_3_correct, EE_final_correct = 0,0,0,0
  EE_1_cnt, EE_2_cnt, EE_3_cnt, EE_final_cnt = 0,0,0,0

  for num, pred in predict_dict_ee1.items():
    truth = int(pred['truth'])
    arg_max_1, score_max_1 = int(pred['truth']), float(pred['score_max_1'])
    if score_max_1>=rho:
      if truth==arg_max_1:
        EE_1_correct +=1
      EE_1_cnt +=1
    else:
      pred_ee2 = predict_dict_ee2[num]
      arg_max_1, score_max_1 = int(pred_ee2['arg_max_1']), float(pred_ee2['score_max_1'])
      if score_max_1>=rho:
        if truth==arg_max_1:
          EE_2_correct+=1
        EE_2_cnt +=1
      else:
        pred_ee3 = predict_dict_ee3[num]
        arg_max_1, score_max_1 = int(pred_ee3['arg_max_1']), float(pred_ee3['score_max_1'])
        if score_max_1>=rho:
          if truth==arg_max_1:
            EE_3_correct+=1
          EE_3_cnt +=1
        else:
          pred_eefinal = predict_dict_eefinal[num]
          arg_max_1, score_max_1 = int(pred_eefinal['arg_max_1']), float(pred_eefinal['score_max_1'])
          if truth==arg_max_1:
            EE_final_correct+=1
          EE_final_cnt +=1

  return  EE_1_cnt,EE_2_cnt,EE_3_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_3_correct, EE_final_correct





