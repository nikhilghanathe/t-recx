import os, sys, json
import csv
from graphviz import Digraph
# import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt	
from networkx.drawing.nx_agraph import graphviz_layout
import itertools
import numpy as np



#class defn for simple graph of DNN model
class Node:
	def __init__(self, id:int, name:str, shape:list, flops:float, isRoot=False):
		self.name = name
		self.edges = []
		self.isRoot = isRoot
		self.shape = shape
		self.flops = flops
		self.id = id

	def setEdges(self, edge_list):
		for edge in edge_list:
			if edge not in self.edges:
				self.edges.append(edge)
	def getEdges(self):
		return self.edges


graph = {}

def setEdges(src, dest):
	for d in dest:
		if d=='':
			continue
		node = graph[d]
		node.setEdges([src])
		graph.update({d:node})

#method to evaluate/plot the tradeoff between the benefit and the overhead of using the EE technique
def readData(model_name, graph_fname):
	#read csv file and create graph
    # with open('flops_calc_resnet8.csv', 'r') as fp:
    with open(graph_fname, 'r') as fp:
        reader =csv.reader(fp)
        for row in reader:
        	# print(row)
        	id, name, shape_str, edge_str, flops_32, flops = int(row[0]), row[1], row[2], row[3], float(row[4]), float(row[5])
        	shape_str = shape_str.replace(' ', '')
        	shape_str = shape_str.replace('(', '')
        	shape_str = shape_str.replace(')', '')
        	shape_str = shape_str.split(',')
        	shape = [int(x) for x in shape_str[1:]]
        	if edge_str=='':
        		node = Node(id, name, shape, flops, True)
        	else:
        		node = Node(id, name, shape, flops, False)
        	#update edge and build graph 
        	edge_str =edge_str.replace(' ', '')
        	edges =edge_str.split(',')

        	graph.update({name: node})
        	setEdges(name, edges)
    # print(graph)
#TODO
def drawGraph():
	#draw graph
	G=Digraph('G', 'graph')
	# for node in graph.keys():
	# 	G.add_node(node)
	for src, node in graph.items():
		for e in node.edges:
			G.edge(src, e)
	G.view()
	# nx.draw(G, 'graph.pdf')
	# nx.draw(G, pos=graphviz_layout(G), node_size=1600, cmap=plt.cm.Blues, node_color=range(len(G)),prog='dot')
	plt.show()
	# print(graph)


def sortTopological(vertex,isVisited,topological_nodeList, DFG): 
	# Mark the current vertex as visited. 
    isVisited[vertex] = True

    for name in DFG[vertex].edges: 
        if isVisited[vertex] == False: 
            sortTopological(name,visited,topological_nodeList) 
    
	#if all adjacent vertices visited, then append node
    topological_nodeList.append(vertex) 

def sortNodesTopological(DFG):
	numVertices = len(DFG.keys())
    # Initialize all nodes as not visited
	isVisited = dict.fromkeys(list(DFG.keys()),False)
	topological_nodeList =[] 
  
    # Recursively start sorting all the vertices one by one
	for vertex in DFG.keys(): 
		if isVisited[vertex] == False: 
			sortTopological(vertex,isVisited,topological_nodeList, DFG) 
  
	return topological_nodeList

#get all paths from src to dest
def getPath(src, dest, visited, path, paths):
	visited[src] = True
	path.append(src)
	# print(path, src, dest, graph[src].getEdges())
	# print('3 ' ,paths)
	if src==dest:
		paths.append(list(path))
		# return path
	else:
		for edge in graph[src].getEdges():
			getPath(edge, dest, visited, path, paths)

	path.pop()
	visited[src] = False


def calc_accuracy(model_name, EE_1_conf_param):
	#read trace data of model
	with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1_ref.json', 'r') as fp:
		predict_dict_ee1 = json.load(fp)
	with open('trace_data/''trace_data_'+model_name[15:]+'_eefinal_ref.json', 'r') as fp:
		predict_dict_eefinal = json.load(fp)

	EE_cnt, EE_1_correct, EE_final_correct = 0,0,0
	EE_final_keys = []
	total_samples = 4448
	
	for num, pred in predict_dict_ee1.items():
		prob_list = [float(x) for x in pred['probability']]
		truth = int(pred['truth'])
		uncertain = float(prob_list[-1])
		arg_max_1 = np.argmax(prob_list)
		arg_max_2 = np.argsort(prob_list)[-2]
		isCorrect = truth==arg_max_1
		score_max_1, score_max_2 = float(pred['score_max_1']), min(prob_list)
		# score_max_1 = max(prob_list)
		if score_max_1>=EE_1_conf_param:
			if isCorrect:
				EE_1_correct +=1
			EE_cnt +=1
		else:
			pred_eefinal = predict_dict_eefinal[num]
			prob_list = [float(x) for x in pred_eefinal['probability']]
			uncertain = float(prob_list[-1])
			arg_max_1 = np.argmax(prob_list)
			arg_max_2 = np.argsort(prob_list)[-2]
			isCorrect = truth==arg_max_1
			if isCorrect:
				EE_final_correct+=1
	return (EE_1_correct/total_samples)*100, (EE_final_correct/total_samples)*100, (EE_cnt/total_samples), ((EE_1_correct+EE_final_correct)*100)/total_samples



def plot_benefit_curve(model_name):
	


	#get paths for ee1 and eefinal
	topological_nodeList = sortNodesTopological(graph)
	#get all paths from 'input_1' to 'dense'
	# paths = [[]]
	visited = dict.fromkeys(list(graph.keys()), False)
	all_paths_ee1, all_paths_eefinal = [], []
	getPath('input_1', 'ee_1_out', visited, [], all_paths_ee1)
	getPath('input_1', 'dense_1', visited, [], all_paths_eefinal)
	# print(all_paths_ee1)
	# print(all_paths_eefinal)

	#------------------------SW------------------------------------------------
	#calc ee1 metrics
	#find longest path for ee1 and eefinal
	#	- assuming 1 op = 1 cycle
	latency = 0
	
	nodes_for_ee1 = (list(dict.fromkeys(list(itertools.chain.from_iterable(all_paths_ee1)))))
	nodes_for_eefinal = (list(dict.fromkeys(list(itertools.chain.from_iterable(all_paths_eefinal)))))
	#subract the nodes already visited for ee_1
	nodes_for_eefinal_ee =  list(set(nodes_for_ee1)^set(nodes_for_eefinal+['dense', 'pointwise_conv_ee1', 'depth_conv_ee_1', 'ee_1_out', 'average_pooling2d']))
	
	#-----------------------------SW---------------------------------------------------------------------
	#for the SW model, all nodes (from all paths have to be considered once)
	latency_ee1 = sum([graph[x].flops for x in nodes_for_ee1])
	latency_eefinal = sum([graph[x].flops for x in nodes_for_eefinal])
	latency_eefinal_ee = sum([graph[x].flops for x in nodes_for_eefinal_ee])
	
	
	#software model - running on microcontrollers
	total_samples = 4448
	x_axis= [5*x for x in range(0,21)]
	y_axis, y_axis_ee = [], []
	y_axis_flops_ee1, y_axis_flops_eefinal = [], []
	x_axis_accuracy_ee1, x_axis_accuracy_eefinal = [], []
	x_axis_accuracy, y_axis_flops, conf_list =[], [], []


	#REFERENCE - *100 because 100% of samples take the path; also divide by Batch size
	flops_noEE = 15.69# This is the flops count when we use the pretrainedResnet model from the tinyML repo w/o any changes to the structure
	flops_EE = 1# This is the flops count when we use the pretrainedResnet model from the tinyML repo and make the following changes. 1)add avg pooling (pool_size=4,4)  and dense layer, 2) remove all blocks after ee-1
	accuracy_noEE = 85
	accuracy_EE = 1 

	#vary the ee-1 exit confidence criteria from 0.0 to 1.0 in steps of 0.01
	for conf in list(np.linspace(0.01,1.0, 101)):
		ee_1_accuracy, ee_final_accuracy, ee_percent, total_accuracy = calc_accuracy(model_name, conf)
		# total_accuracy = ee_1_accuracy + ee_final_accuracy
		flops_ee1 = latency_ee1* ee_percent
		flops_eefinal = (latency_ee1+ latency_eefinal_ee) * (1-ee_percent)
		flops_total = flops_ee1+flops_eefinal
		x_axis_accuracy.append(total_accuracy)
		y_axis_flops.append(flops_total)
		# print(total_accuracy, flops_total)
		# ss
		conf_list.append(conf)
		# y_axis_flops_ee1.append(flops_ee1)
		# x_axis_accuracy_ee1.append(ee_1_accuracy)
		# y_axis_flops_eefinal.append(flops_eefinal)
		# x_axis_accuracy_eefinal.append(ee_final_accuracy)
	x_axis_accuracy_mv = x_axis_accuracy.copy()
	y_axis_flops_mv = y_axis_flops.copy()

	# plt.scatter(x_axis_accuracy_ee1, y_axis_flops_ee1)
	# plt.scatter(x_axis_accuracy_eefinal, y_axis_flops_eefinal)
	plt.scatter(accuracy_EE, flops_EE, label='only EE', color = 'red')
	plt.scatter(accuracy_noEE, flops_noEE, label='no EE', color = 'orange')
	# plt.plot((0, flops_EE), (accuracy_EE, flops_EE), color='red', linestyle='--', animated=False)
	plt.vlines(accuracy_EE, 0,flops_EE, linestyles='dashed', color='red')
	plt.vlines(accuracy_noEE, 0,flops_noEE, linestyles='dashed', color='orange')
	plt.hlines(flops_EE, 0,accuracy_EE, linestyles='dashed', color='red')
	plt.hlines(flops_noEE, 0,accuracy_noEE, linestyles='dashed', color='orange')

	plt.scatter(x_axis_accuracy, y_axis_flops, color='blue', label='w/ multiview info')
	plt.title('Flops vs Accuracy tradeoff for - model name: '+model_name)
	plt.ylabel('FLOPS (millions)')
	plt.xlabel('Total Accuracy (%)')
	plt.xlim([70,100])
	plt.ylim([5,20])
	
	
	# for x,y, conf in zip(x_axis_accuracy,y_axis_flops, conf_list):

	#     # label = f"({x},{y})"
	#     label = f"({conf})"


	#     plt.annotate(label, # this is the text
	#                  (x,y), # these are the coordinates to position the label
	#                  textcoords="offset points", # how to position the text
	#                  xytext=(0,10), # distance from text to points (x,y)
	#                  ha='center') # horizontal alignment can be left, right or center

	label = f"({accuracy_EE},{flops_EE})"
	plt.annotate(label, # this is the text
	                 (accuracy_EE,flops_EE), # these are the coordinates to position the label
	                 textcoords="offset points", # how to position the text
	                 xytext=(0,10), # distance from text to points (x,y)
	                 ha='center') # horizontal alignment can be left, right or center
	label = f"({accuracy_noEE},{flops_noEE})"
	plt.annotate(label, # this is the text
	                 (accuracy_noEE,flops_noEE), # these are the coordinates to position the label
	                 textcoords="offset points", # how to position the text
	                 xytext=(0,10), # distance from text to points (x,y)
	                 ha='center') # horizontal alignment can be left, right or center	

	model_name = 'trained_models/model_no_mv_ee_at_quarter_with_weighted_loss_point3'
	x_axis_accuracy, y_axis_flops = [], []
	for conf in list(np.linspace(0.01,1.0, 101)):
		ee_1_accuracy, ee_final_accuracy, ee_percent, total_accuracy = calc_accuracy(model_name, conf)
		# total_accuracy = ee_1_accuracy + ee_final_accuracy
		flops_ee1 = (latency_ee1 - graph['depth_conv_eefinal_out'].flops )* ee_percent
		flops_eefinal = (latency_ee1+ latency_eefinal_ee) * (1-ee_percent)
		flops_total = flops_ee1+flops_eefinal
		x_axis_accuracy.append(total_accuracy)
		y_axis_flops.append(flops_total)
		# conf_list.append(conf)
	x_axis_accuracy_no_mv = x_axis_accuracy.copy()
	y_axis_flops_no_mv = y_axis_flops.copy()


	#create data file for tikz graph
	with open('benefit_curve_data_VWW_mv.dat', 'w') as fp:
		fp.write('accuracy\tflops\n')
		for i in range(len(x_axis_accuracy_mv)):
			fp.write(str(x_axis_accuracy_mv[i])+'\t'+str(y_axis_flops_mv[i])+'\n')
	with open('benefit_curve_data_VWW_no_mv.dat', 'w') as fp:
		fp.write('accuracy\tflops\n')
		for i in range(len(x_axis_accuracy_no_mv)):
			fp.write(str(x_axis_accuracy_no_mv[i])+'\t'+str(y_axis_flops_no_mv[i])+'\n')

	plt.scatter(x_axis_accuracy, y_axis_flops, color='black', label='w/o multiview info ')
	plt.legend()

	os.chdir('EE_results')
	fig = plt.gcf()
	fig.set_size_inches((20, 15), forward=False)
	fig.savefig('benefit_curve_'+model_name[15:]+'.png', dpi=1000)
	# plt.savefig('benefit_curve_'+model_name[15:]+'.png')
	os.chdir('..')

	
	plt.show()



	

	#when no EE
	for x in x_axis:
		y_axis.append(latency_eefinal)#*total_samples)

	#when EE
	for x in x_axis:
		y_axis_ee.append( ( (x/100)*latency_ee1 + (100-x)/100 * (latency_ee1+ latency_eefinal_ee) ))# *total_samples)

	#------------------------------HW------------------------------------------
	#hardware model - choose the longest path for both ee1 and eefinal
	latency_ee1_hw, latency_eefinal_hw = 0, 0
	nodes_for_ee1_hw, nodes_for_eefinal_hw = 0, 0
	nodes_for_eefinal_ee_hw = []
	for path in all_paths_ee1:#for ee1
		latency = sum([graph[x].flops for x in path])
		if latency>latency_ee1_hw:
			nodes_for_ee1 = path
			latency_ee1_hw = latency
	for path in all_paths_eefinal: #for eefinal
		latency = sum([graph[x].flops for x in path])
		if latency>latency_eefinal_hw:
			nodes_for_eefinal_hw = path
			latency_eefinal_hw = latency
	#subract the nodes already visited for ee_1
	# nodes_for_eefinal_ee_hw =  list(set(nodes_for_ee1_hw)^set(nodes_for_eefinal_hw+['dense', 'average_pooling2d']))
	# subtract the flops/cycles for work done for ee1 because the branch for eefinal will also be runnnig in parallel while ee1 is computed
	latency_eefinal_ee_hw = latency_eefinal_hw - sum([graph[x].flops for x in nodes_for_ee1])#(graph['dense'].flops + graph['average_pooling2d'].flops)  
	y_axis, y_axis_ee_hw = [], []
	#when no EE
	for x in x_axis:
		y_axis.append(latency_eefinal_hw)#*total_samples)

	#when EE
	for x in x_axis:
		y_axis_ee_hw.append( ( (x/100)*latency_ee1_hw + (100-x)/100 * (latency_ee1_hw+ latency_eefinal_ee_hw) ))# *total_samples)

	#------------------------------------------------------------------------


	# diff_sw_hw = []
	# for i in range (0, len(x_axis)):
	# 	diff_sw_hw.append(y_axis_ee_hw[i] - y_axis_ee[i])
	# plt.plot(x_axis, diff_sw_hw , label='diff-ee')
	# plt.xlabel('EE%')
	# plt.ylabel('average Diff Latency for 10000 samples (Million cycles) ')
	# plt.legend()
	# for i in range(0, len(x_axis)):
	# 	label = "{:.2f}".format(y_axis_ee[i])
	# 	plt.annotate(label, (x_axis[i],y_axis_ee[i]), textcoords="offset points", xytext=(0,10), ha='center') 
	# plt.show()
	
if __name__ == "__main__":
	model_name = sys.argv[1]
	graph_fname = 'flops.csv'

	readData(model_name, graph_fname)
	# drawGraph()
	plot_benefit_curve(model_name)
	# plt.plot(x_axis, y_axis, label='no-ee')
	# plt.plot(x_axis, y_axis_ee, label='with-ee')
	# plt.xlabel('EE%')
	# plt.ylabel('Total Latency for 10000 samples (Million cycles) ')
	# plt.legend()
	# for i in range(0, len(x_axis)):
	# 	label = "{:.2f}".format(y_axis_ee[i])
	# 	plt.annotate(label, # this is the text
	# (x_axis[i],y_axis_ee[i]), # these are the coordinates to position the label
	# textcoords="offset points", # how to position the text
	# xytext=(0,10), # distance from text to points (x,y)
	# ha='center') # horizontal alignment can be left, right or center

	# plt.plot(x_axis, y_axis, label='no-ee')
 	