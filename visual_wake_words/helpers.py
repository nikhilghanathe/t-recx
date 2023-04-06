'''
Project: t-recx
Subproject: Visual wake words (human face detection) on VWW dataset
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
		score_max_1, score_max_2 = float(pred['score_max_1']), float(pred['score_max_2'])
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



def collect_pred_metrics(prob_list):
	score_max_1, score_max_2 = sorted(prob_list)[-1], sorted(prob_list)[-2]
	arg_max_1, arg_max_2 = np.argmax(prob_list), np.argsort(prob_list)[-2]
	return score_max_1, score_max_2, arg_max_1, arg_max_2


#generate trace data and dump into json files for faster access to prediction behavior
def generate_trace(val_generator, model_name):
	if os.path.exists('trace_data/'+'trace_data_'+model_name[15:]+'_ee.json'):
		print('Trace data already exists for ', model_name)
		return
	model = tf.keras.models.load_model(model_name)
	prediction_dict_ee, prediction_dict_ef = {}, {}
	count = 0
	isBreak = False
	for val_batch in val_generator:
		(test_sample, test_label) = val_batch[0], val_batch[1]
		#generator loop indefintely, so add manual break when you reach dataset size
		if isBreak:
			break
		BS = test_sample.shape[0]
		prediction = model.predict(test_sample, batch_size=BS)
		prediction_ee, prediction_ef = prediction[0][0], prediction[1][0]	
		for i in range(0, BS):#loop on batch
		    if count==Config.total_samples:#generator loop indefintely, so add manual break when you reach dataset size
		    	isBreak = True
		    	break
		    truth = np.argmax(test_label[i])
		    prob_list = prediction_ee[i]
		    score_max_1, _, arg_max_1, _ = collect_pred_metrics(prob_list)
		    isCorrect = truth == arg_max_1
		    prediction_dict_ee.update({str(count): {'truth':str(truth), 'prediction':str(arg_max_1),  'isCorrect': str(isCorrect), 
	    				'score_max_1' : str(score_max_1), 'arg_max_1' : str(arg_max_1)}})

		    prob_list = prediction_ef[i]
		    score_max_1, _, arg_max_1, _ = collect_pred_metrics(prob_list)
		    isCorrect = truth == arg_max_1
		    prediction_dict_ef.update({str(count): {'truth':str(truth), 'prediction':str(arg_max_1),  'isCorrect': str(isCorrect), 
			    'score_max_1' : str(score_max_1), 'arg_max_1' : str(arg_max_1)}})

		    count+=1


	if not os.path.exists('trace_data/'):
		os.makedirs('trace_data')
	#dump trace data
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee.json', 'w') as fp:
		json.dump(prediction_dict_ee, fp, indent=4)
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ef.json', 'w') as fp:
		json.dump(prediction_dict_ef, fp, indent=4)





# ----trecx train util functions-------------------
def save_trecx_model(model, model_save_name, model_arch):
	if model_arch=='mobnet_ev' or model_arch=='mobnet_noev':
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
	else:
		model.save('trained_models/'+ model_save_name)



