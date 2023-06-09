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
import sys, os

label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck','uncertain']


def get_ground_truth(id, labels):
    return labels[id]

def isCorrectPred(pred):
    truth, prediction = pred['truth'], pred['prediction']
    return int(truth)==int(prediction)


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
		flops.append(get_flops(new_model, batch_size=1))
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

#generate trace data and dump into json files for faster access to prediction behavior
def generate_trace(val_generator, model_name):
	if os.path.exists('trace_data/'+'trace_data_'+model_name[15:]+'_ee.json'):
		print('Trace data already exists for ', model_name)
		return
	#load model
	model = tf.keras.models.load_model(model_name)
  #generate predictions and dump into json file

	isBreak = False
	count = 0
	prediction_dict_ee1, prediction_dict_eefinal = {}, {}
	for val_batch in val_generator:
	  (test_sample, test_label) = val_batch[0], val_batch[1]
	  if isBreak:
	  	break
	  prediction = model.predict(test_sample, batch_size=Config.test_batch_size)#batch prediction
	  prediction_ee1, prediction_eefinal = prediction[0], prediction[1]
	  #collect trace for each pred in batch
	  for i in range(0, Config.test_batch_size):
	  	if count==Config.total_samples:#generator loop indefintely, so add manual break when you reach dataset size
	  		isBreak = True
	  		break
	  	truth = np.argmax(test_label[i])
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

def get_flops_prior(model_name):
	model = tf.keras.models.load_model(model_name)
	exit_list = ['ee_1', 'ee_2', 'ef_out']
	flops = []
	for i in range(0, len(exit_list)):
		new_model = get_ee_model(model, exit_list[:i+1])
		flops.append(get_flops(new_model, batch_size=1))
	return flops[0], flops[1], flops[2]

# =========collect trace data =================================
def generate_trace_prior(val_generator, model_name):
	if os.path.exists('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json'):
		print('Trace data already exists for ', model_name)
		return
	model = tf.keras.models.load_model(model_name)
	isBreak = False
	count = 0
	prediction_dict_ee1, prediction_dict_ee2, prediction_dict_ee3, prediction_dict_eefinal = {}, {}, {}, {}
	for val_batch in val_generator:
		(test_sample, test_label) = val_batch[0], val_batch[1]
		if isBreak:
			break
		prediction = model.predict(test_sample, batch_size=Config.test_batch_size)#batch prediction
		prediction_ee1, prediction_ee2,  prediction_eefinal = prediction[0], prediction[1], prediction[2]
		#collect trace for each pred in batch
		for i in range(0, Config.test_batch_size):
			if count==Config.total_samples:#generator loop indefintely, so add manual break when you reach dataset size
				isBreak = True
				break
			truth = np.argmax(test_label[i])
			arg_max_1_ee1, arg_max_1_ee2, arg_max_1_eefinal = np.argmax(prediction_ee1[i]), np.argmax(prediction_ee2[i]), np.argmax(prediction_eefinal[i])
			score_max_1_ee1, score_max_1_ee2, score_max_1_eefinal = max(prediction_ee1[i]), max(prediction_ee2[i]), max(prediction_eefinal[i])

			if int(truth) == int(arg_max_1_ee1): isCorrect = True
			else: isCorrect = False
			prediction_dict_ee1.update({str(count): {'prediction':str(arg_max_1_ee1), 'truth':str(truth), 'isCorrect': str(isCorrect), 
			'score_max_1' : str(score_max_1_ee1), 'arg_max_1' : str(arg_max_1_ee1) }})            

			if int(truth) == int(arg_max_1_ee2): isCorrect = True
			else: isCorrect = False
			prediction_dict_ee2.update({str(count): {'prediction':str(arg_max_1_ee2), 'truth':str(truth), 'isCorrect': str(isCorrect), 
			'score_max_1' : str(score_max_1_ee2), 'arg_max_1' : str(arg_max_1_ee2) }})            


			if int(truth) == int(arg_max_1_eefinal): isCorrect = True
			else: isCorrect = False
			prediction_dict_eefinal.update({str(count): {'prediction':str(arg_max_1_eefinal), 'truth':str(truth), 'isCorrect': str(isCorrect), 
			'score_max_1' : str(score_max_1_eefinal), 'arg_max_1' : str(arg_max_1_eefinal) }})            

			count+=1

	#dump trace data
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json', 'w') as fp:
		json.dump(prediction_dict_ee1, fp, indent=4)
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2.json', 'w') as fp:
		json.dump(prediction_dict_ee2, fp, indent=4)
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_eefinal.json', 'w') as fp:
		json.dump(prediction_dict_eefinal, fp, indent=4)


# return points for scatter plot
def calculate_scatter_points_prior(model_name, total_samples):
	#get flops count when using ee1 only, ee1+ee2 and when using ee1+ee2+ef
	flops_ee1, flops_ee2, flops_eefinal = get_flops_prior(model_name)
	x_axis_accuracy, y_axis_flops =[], []
	#vary the ee exit confidence criteria (rho from Eq1)  from 0.0 to 1.0 in steps of 0.01
	for rho in list(np.linspace(0.01,1.0, 101)):
		EE_1_cnt,EE_2_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_final_correct = calc_accuracy_prior(model_name, rho)
		total_accuracy = ((EE_1_correct+EE_2_correct+ EE_final_correct)*100)/total_samples
		flops_total = (flops_ee1*EE_1_cnt + flops_ee2*EE_2_cnt + flops_eefinal*EE_final_cnt )/total_samples
		flops_total = flops_total/1000000 #divide by 1.0E+6 
		x_axis_accuracy.append(total_accuracy)
		y_axis_flops.append(flops_total)
	return x_axis_accuracy, y_axis_flops


#benefit curve
def calc_accuracy_prior(model_name, rho):
  #read trace data of model
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1.json', 'r') as fp:
    predict_dict_ee1 = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2.json', 'r') as fp:
    predict_dict_ee2 = json.load(fp)
  with open('trace_data/''trace_data_'+model_name[15:]+'_eefinal.json', 'r') as fp:
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






# ----trecx train util functions-------------------
def save_trecx_model(model, model_save_name, model_arch):
	if model_arch=='resnet_ev' or model_arch=='resnet_noev':
		cnt = 0
		for layer in model.layers:
			if layer.name=='ee_out':
			    ee_layer_num = cnt
			if layer.name=='ef_out':
			    ef_layer_num = cnt
			cnt+=1
	    #construct new model
		final_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[ee_layer_num].output, model.layers[ef_layer_num].output])
		final_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy', 
		    loss_weights=None,weighted_metrics=None, run_eagerly=False)
		final_model.save("trained_models/" + model_save_name)
	elif model_arch=='resnet_sdn':
		cnt = 0
		for layer in model.layers:
		    if layer.name=='ee_1':
		        ee1_layer_num = cnt
		    if layer.name=='ee_2':
		        ee2_layer_num = cnt
		    if layer.name=='ef_out':
		        ef_layer_num = cnt
		    cnt+=1
		#construct new model
		final_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[ee1_layer_num].output, model.layers[ee2_layer_num].output, model.layers[ef_layer_num].output])
		final_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy', 
		    loss_weights=None,weighted_metrics=None, run_eagerly=False)
		final_model.save("trained_models/" + model_save_name)
	else:
		model.save('trained_models/'+ model_save_name)



def get_loss_weights(model_arch):
	if model_arch=='resnet_sdn':
		return Config.loss_weights_sdn
	elif model_arch=='resnet_branchynet':
		return Config.loss_weights_branchynet
	else:
		return None
