import numpy as np
import os, sys
import matplotlib.pyplot as plt
import json



# label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck','uncertain']
label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
NUM_SAMPLES = 10000

isHIGHC = True
isMEDC = True
isLOWC = True

isORG_v_UNCRTN = True


def update_plot(plot_dict:dict, probability:list):
	for label in label_list:
		obj_l = plot_dict[label]
		obj_l.append(probability.pop(0))	
		plot_dict.update({label: obj_l})
	return plot_dict

def isCorrect(pred, truth):
	return int(pred)==int(truth)


if __name__ == "__main__":
	fname_org = 'trace_data/trace_data_orig_ee1_plus_eefinal_ee1.json'
	# fname_org = 'trace_data_trainedResnet_random_noise_cifar100_2000_ee_aggr_ee2.json'
	model_name = sys.argv[1]
	fname = 'trace_data/trace_data_'+model_name[15:]+'_ee1_ref.json'
	#load trace data
	with open(fname_org, 'r') as fp:
		predict_dict_org = json.load(fp)	
	with open(fname, 'r') as fp:
		predict_dict = json.load(fp)

	# print(predict_dict.keys())

	#count the number of times the difference between first and second major is more in retrained vs original
	diff_orginal, diff_retrained = [], []
	for num, pred in predict_dict_org.items():
		prob_list = [float(x) for x in pred['probability']]
		first_max_prob, second_max_prob = sorted(prob_list)[-1], sorted(prob_list)[-2]
		diff_orginal.append(first_max_prob - second_max_prob)

	for num, pred in predict_dict.items():
		prob_list = [float(x) for x in pred['probability']][:-2]
		first_max_prob, second_max_prob = sorted(prob_list)[-1], sorted(prob_list)[-2]
		diff_retrained.append(first_max_prob - second_max_prob)

	num_gap_higher, num_gap_lower = 0, 0
	diff_original_lowC, diff_retrained_lowC = [], []
	for i in range(0, NUM_SAMPLES):
		# if diff_orginal[i] >=0.65 and diff_orginal[i] <0.9 :
		if diff_orginal[i] >=0.9:
			if diff_orginal[i] > diff_retrained[i]:
				num_gap_lower +=1
			else:
				num_gap_higher +=1
			diff_original_lowC.append(diff_orginal[i])
			diff_retrained_lowC.append(diff_retrained[i])

	print('The number of times the difference is high in retrained model is = ', num_gap_higher)
	print('The number of times the difference is high in original model is = ', num_gap_lower)
	plt.scatter([x for x in range(0,num_gap_lower+num_gap_higher)], diff_original_lowC, label='diff-original', color='blue' )
	plt.scatter([x for x in range(0,num_gap_lower+num_gap_higher)], diff_retrained_lowC, label='diff-retrained', color='grey' )
	plt.legend()
	plt.title('Plot showing difference between first and second-majors of original and retraiend model of '+sys.argv[1])
	plt.show()
		


	


	# plot for model with uncertain 
	plot_dict_highC = {x:[] for x in label_list}
	plot_dict_lowC = {x:[] for x in label_list}
	plot_dict_medC = {x:[] for x in label_list}
	#sort the prediction numbers
	cnt_highC, cnt_medC, cnt_lowC = 0, 0, 0
	#get first $NUMS_SAMPLES predictions from the testing stage and plot high, low and medium confidence plots
	for num, pred in predict_dict.items():
		# if isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth']) or not isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth'])::
		if isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth']):
			pred_id = np.argmax([float(x) for x in pred['probability'] ])
			pred_label = label_list[pred_id]
			# plot_dict_ = update_plot(plot_dict, [float(x) for x in pred['probability']])			
			probability_list = [float(x) for x in pred['probability'] ]
			#high confidence i.e. 90+
			if float(np.max([float(x) for x in pred['probability']])) >=0.9:
				# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_highC)
				if cnt_highC<NUM_SAMPLES: 
					plot_dict_highC = update_plot(plot_dict_highC, [float(x) for x in pred['probability']])	
					# print(plot_dict_highC)
					cnt_highC+=1			
				
			#med confidence 65-90
			elif float(np.max([float(x) for x in pred['probability']])) >=0.65 and float(np.max([float(x) for x in pred['probability']])) <0.9:
				# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_medC)
				if cnt_medC<NUM_SAMPLES: 
					plot_dict_medC = update_plot(plot_dict_medC, [float(x) for x in pred['probability']])
					# print(plot_dict_medC)				
					cnt_medC+=1

			#low confidence <65
			elif float(np.max([float(x) for x in pred['probability']])) <0.65 :
				# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_lowC)
				if cnt_lowC<NUM_SAMPLES: 
					plot_dict_lowC = update_plot(plot_dict_lowC, [float(x) for x in pred['probability']])
					cnt_lowC+=1

			if cnt_highC>NUM_SAMPLES-1 and cnt_medC>NUM_SAMPLES-1 and cnt_lowC>NUM_SAMPLES-1:
				break






	if not isORG_v_UNCRTN:
		if isHIGHC:
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['airplane'], label="airplane", color='black')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['automobile'], label="automobile", color='bisque')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['bird'], label="bird", color='r')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['cat'], label="cat", color='g')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['deer'], label="deer", color='b')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['dog'], label="dog", color='c')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['frog'], label="frog", color='lawngreen')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['horse'], label="horse", color='pink')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['ship'], label="ship", color='purple')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['truck'], label="truck", color='teal')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)



			# high confidence plot
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['airplane'], label="airplane", color='black')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['automobile'], label="automobile", color='bisque')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['bird'], label="bird", color='r')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['cat'], label="cat", color='g')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['deer'], label="deer", color='b')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['dog'], label="dog", color='c')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['frog'], label="frog", color='lawngreen')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['horse'], label="horse", color='pink')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['ship'], label="ship", color='purple')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['truck'], label="truck", color='teal')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)


			plt.yscale('linear')
			plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  HIGH CONFIDENCE (90+ probability)')
			plt.xlabel('Classes')
			plt.ylabel('Probability')
			plt.legend()
			plt.savefig('high_confidence.png')
			plt.show()


		if isLOWC:
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['airplane'], label="airplane", color='black')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['automobile'], label="automobile", color='bisque')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['bird'], label="bird", color='r')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['cat'], label="cat", color='g')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['deer'], label="deer", color='b')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['dog'], label="dog", color='c')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['frog'], label="frog", color='lawngreen')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['horse'], label="horse", color='pink')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['ship'], label="ship", color='purple')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['truck'], label="truck", color='teal')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)

			plt.yscale('linear')
			plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  LOW CONFIDENCE (<65 probability)')
			plt.xlabel('Classes')
			plt.ylabel('Probability')
			plt.legend()
			plt.show()

		if isMEDC:
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['airplane'], label="airplane", color='black')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['automobile'], label="automobile", color='bisque')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['bird'], label="bird", color='r')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['cat'], label="cat", color='g')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['deer'], label="deer", color='b')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['dog'], label="dog", color='c')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['frog'], label="frog", color='lawngreen')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['horse'], label="horse", color='pink')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['ship'], label="ship", color='purple')
			plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['truck'], label="truck", color='teal')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)



			plt.yscale('linear')
			plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  MEDIUM CONFIDENCE (65-90 probability)')
			plt.xlabel('Classes')
			plt.ylabel('Probability')
			plt.legend()
			plt.show()
	else:
		# plot the first-major and second major and uncertain for comparison btwn original model scores and retrained model scores
				#plot for the original model vs uncertain model
		#get first $NUMS_SAMPLES predictions from the testing stage and plot high, low and medium confidence plots
		plot_dict_highC_org = {'first-major': [], 'second-major':[]}
		plot_dict_lowC_org = {'first-major': [], 'second-major':[]}
		plot_dict_medC_org = {'first-major': [], 'second-major':[]}

		plot_dict_highC_uncrtn = {'first-major': [], 'second-major':[], 'uncertain':[]}
		plot_dict_lowC_uncrtn = {'first-major': [], 'second-major':[], 'uncertain':[]}
		plot_dict_medC_uncrtn = {'first-major': [], 'second-major':[], 'uncertain':[]}

		#sort the prediction numbers for original model
		cnt_highC, cnt_medC, cnt_lowC = 0, 0, 0
		for num, pred in predict_dict_org.items():
			# if isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth']) or not isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth'])::
			if not isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth']):
				
				#high confidence i.e. 90+
				if float(np.max([float(x) for x in pred['probability']])) >=0.9:
					# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_highC)
					if cnt_highC<NUM_SAMPLES: 
						maxProb = float(np.max([float(x) for x in pred['probability']]))
						prob_list = [float(x) for x in pred['probability']]
						second_max_prob = sorted(prob_list)[-2]
						# uncertain_prob = float(pred['probability'][-1])
						plot_dict_highC_org.update({'first-major': plot_dict_highC_org['first-major']+ [maxProb], 'second-major': plot_dict_highC_org['second-major'] + [second_max_prob]}) 
						# print(plot_dict_highC_org)
						cnt_highC+=1			
					
				#med confidence 65-90
				elif float(np.max([float(x) for x in pred['probability']])) >=0.65 and float(np.max([float(x) for x in pred['probability']])) <0.9:
					# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_medC)
					if cnt_medC<NUM_SAMPLES: 
						maxProb = float(np.max([float(x) for x in pred['probability']]))
						prob_list = [float(x) for x in pred['probability']]
						second_max_prob = sorted(prob_list)[-2]
						# uncertain_prob = float(pred['probability'][-1])
						plot_dict_medC_org.update({'first-major': plot_dict_medC_org['first-major']+ [maxProb], 'second-major': plot_dict_medC_org['second-major'] + [second_max_prob]}) 
						# print(plot_dict_highC_org)
						cnt_medC+=1

				#low confidence <65
				elif float(np.max([float(x) for x in pred['probability']])) <0.65 :
					# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_lowC)
					if cnt_lowC<NUM_SAMPLES: 
						maxProb = float(np.max([float(x) for x in pred['probability']]))
						prob_list = [float(x) for x in pred['probability']]
						second_max_prob = sorted(prob_list)[-2]
						# uncertain_prob = float(pred['probability'][-1])
						plot_dict_lowC_org.update({'first-major': plot_dict_lowC_org['first-major']+ [maxProb], 'second-major': plot_dict_lowC_org['second-major'] + [second_max_prob]}) 
						# print(plot_dict_highC_org)
						cnt_lowC+=1

				if cnt_highC>NUM_SAMPLES-1 and cnt_medC>NUM_SAMPLES-1 and cnt_lowC>NUM_SAMPLES-1:
					break

		#sort for uncertain models
		cnt_highC, cnt_medC, cnt_lowC = 0, 0, 0
		for num, pred in predict_dict.items():
			# if isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth']) or not isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth'])::
			if not isCorrect(np.argmax([float(x) for x in pred['probability']]), pred['truth']):
				
				#high confidence i.e. 90+
				if float(np.max([float(x) for x in pred['probability']])) >=0.9:
					# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_highC)
					if cnt_highC<NUM_SAMPLES: 
						maxProb = float(np.max([float(x) for x in pred['probability']]))
						prob_list = [float(x) for x in pred['probability']][:-2]
						second_max_prob = sorted(prob_list)[-2]
						uncertain_prob = float(pred['probability'][-1])
						plot_dict_highC_uncrtn.update({'first-major': plot_dict_highC_uncrtn['first-major']+ [maxProb], 'second-major': plot_dict_highC_uncrtn['second-major'] + [second_max_prob], 'uncertain': plot_dict_highC_uncrtn['uncertain']+[uncertain_prob]}) 
						# print(plot_dict_highC_org)
						cnt_highC+=1			
					
				#med confidence 65-90
				elif float(np.max([float(x) for x in pred['probability']])) >=0.65 and float(np.max([float(x) for x in pred['probability']])) <0.9:
					# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_medC)
					if cnt_medC<NUM_SAMPLES: 
						maxProb = float(np.max([float(x) for x in pred['probability']]))
						prob_list = [float(x) for x in pred['probability']][:-2]
						second_max_prob = sorted(prob_list)[-2]
						uncertain_prob = float(pred['probability'][-1])
						plot_dict_medC_uncrtn.update({'first-major': plot_dict_medC_uncrtn['first-major']+ [maxProb], 'second-major': plot_dict_medC_uncrtn['second-major'] + [second_max_prob], 'uncertain': plot_dict_medC_uncrtn['uncertain']+[uncertain_prob]}) 
						# print(plot_dict_highC_org)
						cnt_medC+=1

				#low confidence <65
				elif float(np.max([float(x) for x in pred['probability']])) <0.65 :
					# print('here', num, float(np.max([float(x) for x in pred['probability']])), pred['probability'],  cnt_lowC)
					if cnt_lowC<NUM_SAMPLES: 
						maxProb = float(np.max([float(x) for x in pred['probability']]))
						prob_list = [float(x) for x in pred['probability']][:-2]
						second_max_prob = sorted(prob_list)[-2]
						uncertain_prob = float(pred['probability'][-1])
						plot_dict_lowC_uncrtn.update({'first-major': plot_dict_lowC_uncrtn['first-major']+ [maxProb], 'second-major': plot_dict_lowC_uncrtn['second-major'] + [second_max_prob], 'uncertain': plot_dict_lowC_uncrtn['uncertain']+[uncertain_prob]}) 
						# print(plot_dict_highC_org)
						cnt_lowC+=1

				if cnt_highC>NUM_SAMPLES-1 and cnt_medC>NUM_SAMPLES-1 and cnt_lowC>NUM_SAMPLES-1:
					break


		if isHIGHC:
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['airplane'], label="airplane", color='black')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['automobile'], label="automobile", color='bisque')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['bird'], label="bird", color='r')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['cat'], label="cat", color='g')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['deer'], label="deer", color='b')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['dog'], label="dog", color='c')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['frog'], label="frog", color='lawngreen')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['horse'], label="horse", color='pink')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['ship'], label="ship", color='purple')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['truck'], label="truck", color='teal')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)



			# high confidence plot
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC_org['first-major'], label="first-major-org", color='black', marker='^')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC_org['second-major'], label="second-major-org", color='red', marker='^')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC_uncrtn['first-major'], label="first-major-uncrtn", color='black', marker='o')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC_uncrtn['second-major'], label="second-major-uncrtn", color='red', marker='o')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_highC_uncrtn['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)

			# plt.yscale('linear')
			# plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  HIGH CONFIDENCE (90+ probability)')
			# plt.xlabel('Num Predictions')
			# plt.ylabel('Probability')
			# plt.legend()
			# plt.savefig('high_confidence.png')
			# plt.show()
		
			#plot the difference
			fig, ax = plt.subplots()
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], list(map(lambda x,y: x-y, plot_dict_highC_org['first-major'], plot_dict_highC_org['second-major'])), label='diff-org', color='black')
			ax.violinplot(list(map(lambda x,y: x-y, plot_dict_highC_org['first-major'],  plot_dict_highC_org['second-major'])))
			ax.set_xticks([1,2])
			xticklabels=['Baseline', 'T-RecX']
			ax.set_xticklabels(xticklabels)
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], list(map(lambda x,y: x-y, plot_dict_highC_uncrtn['first-major'], plot_dict_highC_uncrtn['second-major'])), label='diff-uncrtn', color='red')
			data = [list(map(lambda x,y: x-y, plot_dict_highC_org['first-major'],  plot_dict_highC_org['second-major'])), list(map(lambda x,y: x-y, plot_dict_highC_uncrtn['first-major'], plot_dict_highC_uncrtn['second-major']))]
			ax.violinplot( data,  showmedians=True)
			ax.yaxis.grid(True)

			plt.yscale('linear')
			# plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  LOW CONFIDENCE (<65 probability)')
			# plt.title("Violin Plot comparing the difference between first-major and second-major for Original Model  vs "+ sys.argv[1] + ' --HIGH CONFIDENCE (90+ probability)')
			# plt.xlabel('Num Predictions')
			# plt.ylabel('Probability')
			plt.ylabel('Confidence')
			plt.legend()
			plt.show()

			#get metrics for difference btwn first and second major
			num_gap_higher, num_gap_lower = 0, 0
			# print(len(plot_dict_highC_org['first-major']), len(plot_dict_highC_uncrtn['first-major']))
			# print(len(plot_dict_medC_org['first-major']), len(plot_dict_medC_uncrtn['first-major']))
			# print(len(plot_dict_lowC_org['first-major']), len(plot_dict_lowC_uncrtn['first-major']))
			
			# for i in range(0, len(plot_dict_highC_org['first-major'])):
			# 	if plot_dict_highC_org['first-major'][i] - plot_dict_highC_org['second-major'][i] > plot_dict_highC_uncrtn['first-major'][i] - plot_dict_highC_uncrtn['second-major'][i]:
			# 		num_gap_lower +=1
			# 	else:
			# 		num_gap_higher +=1
			# 	print(num_gap_higher, num_gap_lower)
			# print(' The number of times the gap between first-major and second-major is higher in uncrtn is = ', num_gap_higher)
			# print(' The number of times the gap between first-major and second-major is higher in org is = ', num_gap_lower)



		if isLOWC:
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC_org['first-major'], label="first-major-org", color='black', marker='s')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC_org['second-major'], label="second-major-org", color='red', marker='s')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC_uncrtn['first-major'], label="first-major-uncrtn", color='black', marker='o', alpha=0.5, s=500)
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC_uncrtn['second-major'], label="second-major-uncrtn", color='red', marker='o', alpha=0.5, s=500)
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_lowC_uncrtn['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)


			#plot the difference
			fig, ax = plt.subplots()
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], list(map(lambda x,y: x-y, plot_dict_lowC_org['first-major'], plot_dict_lowC_org['second-major'])), label='diff-org', color='black')
			ax.violinplot(list(map(lambda x,y: x-y, plot_dict_lowC_org['first-major'],  plot_dict_lowC_org['second-major'])))
			ax.set_xticks([1,2])
			xticklabels=['original', 'retrained']
			ax.set_xticklabels(xticklabels)
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], list(map(lambda x,y: x-y, plot_dict_lowC_uncrtn['first-major'], plot_dict_lowC_uncrtn['second-major'])), label='diff-uncrtn', color='red')
			data = [list(map(lambda x,y: x-y, plot_dict_lowC_org['first-major'],  plot_dict_lowC_org['second-major'])), list(map(lambda x,y: x-y, plot_dict_lowC_uncrtn['first-major'], plot_dict_lowC_uncrtn['second-major']))]
			ax.violinplot( data, showmeans=True, showmedians=True)
			ax.yaxis.grid(True)

			plt.yscale('linear')
			# plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  LOW CONFIDENCE (<65 probability)')
			plt.title("Violin Plot comparing the difference between first-major and second-major for Original Model  vs "+ sys.argv[1] + ' --LOW CONFIDENCE (<65 probability)')
			# plt.xlabel('Num Predictions')
			# plt.ylabel('Probability')
			plt.ylabel('Difference bwtn first-major and second-major')
			plt.legend()
			plt.show()

			#get metrics for difference btwn first and second major
			num_gap_higher, num_gap_lower = 0, 0
			# print(plot_dict_lowC_org['first-major'])
			# for i in range(0, len(plot_dict_lowC_org['first-major'])):
			# 	if plot_dict_lowC_org['first-major'][i] - plot_dict_lowC_org['second-major'][i] > plot_dict_lowC_uncrtn['first-major'][i] - plot_dict_lowC_uncrtn['second-major'][i]:
			# 		num_gap_lower +=1
			# 	else:
			# 		num_gap_higher +=1
			# print(' The number of times the gap between first-major and second-major is higher in uncrtn is = ', num_gap_higher)
			# print(' The number of times the gap between first-major and second-major is higher in org is = ', num_gap_lower)


		if isMEDC:
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC_org['first-major'], label="first-major-org", color='black', marker='^')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC_org['second-major'], label="second-major-org", color='red', marker='^')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC_uncrtn['first-major'], label="first-major-uncrtn", color='black', marker='o')
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC_uncrtn['second-major'], label="second-major-uncrtn", color='red', marker='o')
			# # plt.scatter([x for x in range(0,NUM_SAMPLES)], plot_dict_medC_uncrtn['uncertain'], label="uncertain", color='peru', alpha=0.5, s=500)


			# plt.yscale('linear')
			# plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  MEDIUM CONFIDENCE (65-90 probability)')
			# plt.xlabel('Num Predictions')
			# plt.ylabel('Probability')
			# plt.legend()
			# plt.show()


			#plot the difference
			fig, ax = plt.subplots()
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], list(map(lambda x,y: x-y, plot_dict_medC_org['first-major'], plot_dict_medC_org['second-major'])), label='diff-org', color='black')
			ax.violinplot(list(map(lambda x,y: x-y, plot_dict_medC_org['first-major'],  plot_dict_medC_org['second-major'])))
			ax.set_xticks([1,2])
			xticklabels=['original', 'retrained']
			ax.set_xticklabels(xticklabels)
			# plt.scatter([x for x in range(0,NUM_SAMPLES)], list(map(lambda x,y: x-y, plot_dict_medC_uncrtn['first-major'], plot_dict_medC_uncrtn['second-major'])), label='diff-uncrtn', color='red')
			data = [list(map(lambda x,y: x-y, plot_dict_medC_org['first-major'],  plot_dict_medC_org['second-major'])), list(map(lambda x,y: x-y, plot_dict_medC_uncrtn['first-major'], plot_dict_medC_uncrtn['second-major']))]
			ax.violinplot( data, showmeans=True, showmedians=True)
			ax.yaxis.grid(True)

			plt.yscale('linear')
			# plt.title("Confidence Plot for Cifar-10 predictions of  "+ sys.argv[1] + ' samples  LOW CONFIDENCE (<65 probability)')
			plt.title("Violin Plot comparing the difference between first-major and second-major for Original Model  vs "+ sys.argv[1] + ' --MED CONFIDENCE (90-65 probability)')
			# plt.xlabel('Num Predictions')
			# plt.ylabel('Probability')
			plt.ylabel('Difference bwtn first-major and second-major')
			plt.legend()
			plt.show()

			#get metrics for difference btwn first and second major
			num_gap_higher, num_gap_lower = 0, 0
			# print(plot_dict_medC_org['first-major'])
			# for i in range(0, len(plot_dict_medC_org['first-major'])):
			# 	if plot_dict_medC_org['first-major'][i] - plot_dict_medC_org['second-major'][i] > plot_dict_medC_uncrtn['first-major'][i] - plot_dict_medC_uncrtn['second-major'][i]:
			# 		num_gap_lower +=1
			# 	else:
			# 		num_gap_higher +=1
			# print(' The number of times the gap between first-major and second-major is higher in uncrtn is = ', num_gap_higher)
			# print(' The number of times the gap between first-major and second-major is higher in org is = ', num_gap_lower)

	

	ss

	#calc std deviation
	cnt = 0
	max_prob_dist, uncertain_prob_dist, second_max_prob_dist = [], [], []
	for num, pred in predict_dict.items():
		pred_id = np.argmax([float(x) for x in pred['probability'] ])
		pred_label = label_list[pred_id]
		maxProb = float(np.max([float(x) for x in pred['probability']]))
		prob_list = [float(x) for x in pred['probability']][:-2]
		second_max_prob = sorted(prob_list)[-2]
		uncertain_prob = float(pred['probability'][-1])
		max_prob_dist.append(maxProb)
		uncertain_prob_dist.append(uncertain_prob)
		second_max_prob_dist.append(second_max_prob)
		if cnt==NUM_SAMPLES:
			break	
		else: cnt+=1



	# print(uncertain_prob_dist)
	# plt.plot([x for x in range(0,len(predict_dict.keys()))], max_prob_dist, label="max probability distribution", color='b', ls='dotted')
	# plt.plot([x for x in range(0,len(predict_dict.keys()))], uncertain_prob_dist, label="max probability distribution", color='grey')
	plt.plot([x for x in range(0,NUM_SAMPLES+1)], max_prob_dist, label="max probability distribution", color='b', ls='dotted')
	plt.plot([x for x in range(0,NUM_SAMPLES+1)], uncertain_prob_dist, label="uncertain probability distribution", color='grey')
	plt.plot([x for x in range(0,NUM_SAMPLES+1)], second_max_prob_dist, label="Second-max probability distribution", color='red')
	# plt.hist(uncertain_prob_dist)
	plt.yscale('linear')
	plt.legend()
	plt.title('Probability distribution')
	plt.xlabel('Num of Predictions')
	plt.ylabel('Probability')
	plt.show()






