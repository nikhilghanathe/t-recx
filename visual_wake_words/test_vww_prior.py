# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os, csv
import sys
from absl import app
from vww_model_prior import mobilenet_v1_sdn, mobilenet_v1_branchynet, mobilenet_v1, mobilenet_v1_sdn_incr_pool

from keras_flops import get_flops
import tensorflow as tf
import numpy as np
import json
assert tf.__version__.startswith('2')

from graphviz import Digraph
# import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt 
from networkx.drawing.nx_agraph import graphviz_layout
import itertools


IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'minival')
  

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




#trecx util funcs

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
  #   G.add_node(node)
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


# -----------------------------------------------------------------------------------






def test_sdn(model, val_generator):
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2', 'ee_3', 'dense']:
      print(layer.name)
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output, model.layers[output_layer_nums[2]].output, model.layers[output_layer_nums[3]].output])
  new_model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  test_metrics = new_model.evaluate(val_generator, batch_size=BATCH_SIZE, verbose=1, return_dict=True)    


def test_branchynet(model, val_generator):
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2', 'dense']:
      print(layer.name)
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output, model.layers[output_layer_nums[2]].output])
  new_model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  test_metrics = new_model.evaluate(val_generator, batch_size=BATCH_SIZE, verbose=1, return_dict=True)    


def get_flops_sdn(model_name):
  model = tf.keras.models.load_model(model_name)
  #get flops of ee_1
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output])
  new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  flops_ee_1 = get_flops(new_model, batch_size=1)
  print(flops_ee_1)

  #get flops of ee_1 and ee_2
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output])
  new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  flops_ee_2 = get_flops(new_model, batch_size=1)
  print(flops_ee_2)

  #get flops of ee_1 and ee_2 and ee_3
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2', 'ee_3']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output, model.layers[output_layer_nums[2]].output])
  new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  flops_ee_3 = get_flops(new_model, batch_size=1)
  print(flops_ee_3)


  #get flops of ee_1 and ee_2 and ee_3 and dense (whole model)
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2', 'ee_3', 'dense']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output, model.layers[output_layer_nums[2]].output, model.layers[output_layer_nums[3]].output])
  new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  flops_final = get_flops(new_model, batch_size=1)
  print(flops_final)

  return flops_ee_1, flops_ee_2, flops_ee_3, flops_final


def get_flops_branchynet(model_name):
  model = tf.keras.models.load_model(model_name)
  #get flops of ee_1
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output])
  new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  flops_ee_1 = get_flops(new_model, batch_size=1)
  print(flops_ee_1)

  #get flops of ee_1 and ee_2
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output])
  new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  flops_ee_2 = get_flops(new_model, batch_size=1)
  print(flops_ee_2)

  #get flops of ee_1 and ee_2 and dense (whole model)
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2', 'dense']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output, model.layers[output_layer_nums[2]].output])
  new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy', metrics=['accuracy'])
  flops_final = get_flops(new_model, batch_size=1)
  print(flops_final)

  return flops_ee_1, flops_ee_2, flops_final






def get_trace_data_sdn(model, val_generator, model_name):
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2', 'ee_3', 'dense']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output, model.layers[output_layer_nums[2]].output, model.layers[output_layer_nums[3]].output])
  new_model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])

  #generate predictions and dump into json file
  isBreak = False
  count = 0
  prediction_dict_ee1, prediction_dict_ee2, prediction_dict_ee3, prediction_dict_eefinal = {}, {}, {}, {}
  for val_batch in val_generator:
    (test_sample, test_label) = val_batch[0], val_batch[1]
    #generator loop indefintely, so add manual break when you reach dataset size
    if isBreak:
      break

    BS = test_sample.shape[0]
    prediction = new_model.predict(test_sample, batch_size=BS)
    # print(prediction)
    prediction_ee1, prediction_ee2, prediction_ee3, prediction_eefinal = prediction[0], prediction[1], prediction[2], prediction[3]
    for i in range(0, BS):
      if count==4448:#generator loop indefintely, so add manual break when you reach dataset size
        isBreak = True
        break
      truth = np.argmax(test_label[i])
      prob_list_ee1 = prediction_ee1[i]
      prob_list_ee2 = prediction_ee2[i]
      prob_list_ee3 = prediction_ee3[i]
      prob_list_eefinal = prediction_eefinal[i]
      arg_max_1_ee1 = np.argmax(prob_list_ee1)
      arg_max_1_ee2 = np.argmax(prob_list_ee2)
      arg_max_1_ee3 = np.argmax(prob_list_ee3)
      arg_max_1_eefinal = np.argmax(prob_list_eefinal)
      score_max_1_ee1 = max(prob_list_ee1)
      score_max_1_ee2 = max(prob_list_ee2)
      score_max_1_ee3 = max(prob_list_ee3)
      score_max_1_eefinal = max(prob_list_eefinal)

      if int(truth) == int(arg_max_1_ee1): isCorrect = True
      else: isCorrect = False
      prediction_dict_ee1.update({str(count): {'probability':[str(i) for i in list(prob_list_ee1)],  'prediction':str(arg_max_1_ee1), 'truth':str(truth), 'isCorrect': str(isCorrect), 
        'score_max_1' : str(score_max_1_ee1), 'arg_max_1' : str(arg_max_1_ee1) }})            
       
      if int(truth) == int(arg_max_1_ee2): isCorrect = True
      else: isCorrect = False
      prediction_dict_ee2.update({str(count): {'probability':[str(i) for i in list(prob_list_ee2)],  'prediction':str(arg_max_1_ee2), 'truth':str(truth), 'isCorrect': str(isCorrect), 
        'score_max_1' : str(score_max_1_ee2), 'arg_max_1' : str(arg_max_1_ee2) }})            

      if int(truth) == int(arg_max_1_ee3): isCorrect = True
      else: isCorrect = False
      prediction_dict_ee3.update({str(count): {'probability':[str(i) for i in list(prob_list_ee3)],  'prediction':str(arg_max_1_ee3), 'truth':str(truth), 'isCorrect': str(isCorrect), 
        'score_max_1' : str(score_max_1_ee3), 'arg_max_1' : str(arg_max_1_ee3) }})            


      if int(truth) == int(arg_max_1_eefinal): isCorrect = True
      else: isCorrect = False
      prediction_dict_eefinal.update({str(count): {'probability':[str(i) for i in list(prob_list_eefinal)],  'prediction':str(arg_max_1_eefinal), 'truth':str(truth), 'isCorrect': str(isCorrect), 
        'score_max_1' : str(score_max_1_eefinal), 'arg_max_1' : str(arg_max_1_eefinal) }})            

      count+=1
      

  #dump trace data
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1_ref.json', 'w') as fp:
   json.dump(prediction_dict_ee1, fp, indent=4)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2_ref.json', 'w') as fp:
   json.dump(prediction_dict_ee2, fp, indent=4)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee3_ref.json', 'w') as fp:
   json.dump(prediction_dict_ee3, fp, indent=4)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_eefinal_ref.json', 'w') as fp:
   json.dump(prediction_dict_eefinal, fp, indent=4)



def get_trace_data_branchynet(model, val_generator, model_name):
  output_layer_nums = []
  cnt = 0
  for layer in model.layers:
    if layer.name in ['ee_1', 'ee_2', 'dense']:
      output_layer_nums.append(cnt)
    cnt+=1
  new_model = tf.keras.models.Model(inputs=model.inputs[0], outputs=[model.layers[output_layer_nums[0]].output, model.layers[output_layer_nums[1]].output, model.layers[output_layer_nums[2]].output])
  new_model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])

  #generate predictions and dump into json file
  isBreak = False
  count = 0
  prediction_dict_ee1, prediction_dict_ee2, prediction_dict_eefinal = {}, {}, {}
  for val_batch in val_generator:
    (test_sample, test_label) = val_batch[0], val_batch[1]
    #generator loop indefintely, so add manual break when you reach dataset size
    if isBreak:
      break

    BS = test_sample.shape[0]
    prediction = new_model.predict(test_sample, batch_size=BS)
    # print(prediction)
    prediction_ee1, prediction_ee2, prediction_eefinal = prediction[0], prediction[1], prediction[2]
    for i in range(0, BS):
      if count==4448:#generator loop indefintely, so add manual break when you reach dataset size
        isBreak = True
        break
      truth = np.argmax(test_label[i])
      prob_list_ee1 = prediction_ee1[i]
      prob_list_ee2 = prediction_ee2[i]
      prob_list_eefinal = prediction_eefinal[i]
      arg_max_1_ee1 = np.argmax(prob_list_ee1)
      arg_max_1_ee2 = np.argmax(prob_list_ee2)
      arg_max_1_eefinal = np.argmax(prob_list_eefinal)
      score_max_1_ee1 = max(prob_list_ee1)
      score_max_1_ee2 = max(prob_list_ee2)
      score_max_1_eefinal = max(prob_list_eefinal)

      if int(truth) == int(arg_max_1_ee1): isCorrect = True
      else: isCorrect = False
      prediction_dict_ee1.update({str(count): {'probability':[str(i) for i in list(prob_list_ee1)],  'prediction':str(arg_max_1_ee1), 'truth':str(truth), 'isCorrect': str(isCorrect), 
        'score_max_1' : str(score_max_1_ee1), 'arg_max_1' : str(arg_max_1_ee1) }})            
       
      if int(truth) == int(arg_max_1_ee2): isCorrect = True
      else: isCorrect = False
      prediction_dict_ee2.update({str(count): {'probability':[str(i) for i in list(prob_list_ee2)],  'prediction':str(arg_max_1_ee2), 'truth':str(truth), 'isCorrect': str(isCorrect), 
        'score_max_1' : str(score_max_1_ee2), 'arg_max_1' : str(arg_max_1_ee2) }})            

      if int(truth) == int(arg_max_1_eefinal): isCorrect = True
      else: isCorrect = False
      prediction_dict_eefinal.update({str(count): {'probability':[str(i) for i in list(prob_list_eefinal)],  'prediction':str(arg_max_1_eefinal), 'truth':str(truth), 'isCorrect': str(isCorrect), 
        'score_max_1' : str(score_max_1_eefinal), 'arg_max_1' : str(arg_max_1_eefinal) }})            

      count+=1
      

  #dump trace data
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1_ref.json', 'w') as fp:
   json.dump(prediction_dict_ee1, fp, indent=4)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2_ref.json', 'w') as fp:
   json.dump(prediction_dict_ee2, fp, indent=4)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_eefinal_ref.json', 'w') as fp:
   json.dump(prediction_dict_eefinal, fp, indent=4)








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





#benefit curve
def calc_accuracy_sdn(model_name, conf):
  ee1_th, ee2_th, ee3_th = conf, conf, conf
  #read trace data of model
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1_ref.json', 'r') as fp:
    predict_dict_ee1 = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2_ref.json', 'r') as fp:
    predict_dict_ee2 = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee3_ref.json', 'r') as fp:
    predict_dict_ee3 = json.load(fp)
  with open('trace_data/''trace_data_'+model_name[15:]+'_eefinal_ref.json', 'r') as fp:
    predict_dict_eefinal = json.load(fp)

  EE_1_correct, EE_2_correct, EE_3_correct, EE_final_correct = 0,0,0,0
  EE_1_cnt, EE_2_cnt, EE_3_cnt, EE_final_cnt = 0,0,0,0

  EE_final_keys = []
  total_samples = 4448
  
  for num, pred in predict_dict_ee1.items():
    prob_list = [float(x) for x in pred['probability']]
    truth = int(pred['truth'])
    arg_max_1 = np.argmax(prob_list)
    arg_max_2 = np.argsort(prob_list)[-2]
    isCorrect = truth==arg_max_1
    score_max_1, score_max_2 = float(pred['score_max_1']), min(prob_list)
    # score_max_1 = max(prob_list)
    # if score_max_1>=ee1_th:
    if False:
      if isCorrect:
        EE_1_correct +=1
      EE_1_cnt +=1
    else:
      pred_ee2 = predict_dict_ee2[num]
      arg_max_1, score_max_1 = int(pred_ee2['arg_max_1']), float(pred_ee2['score_max_1'])
      isCorrect = truth==arg_max_1
      if score_max_1>=ee2_th:
      # if False:
        if isCorrect:
          EE_2_correct+=1
        EE_2_cnt +=1
      else:
        pred_ee3 = predict_dict_ee3[num]
        arg_max_1, score_max_1 = int(pred_ee3['arg_max_1']), float(pred_ee3['score_max_1'])
        isCorrect = truth==arg_max_1
        if score_max_1>=ee3_th:
          if isCorrect:
            EE_3_correct+=1
          EE_3_cnt +=1
        else:
          pred_eefinal = predict_dict_eefinal[num]
          arg_max_1, score_max_1 = int(pred_eefinal['arg_max_1']), float(pred_eefinal['score_max_1'])
          isCorrect = truth==arg_max_1
          if isCorrect:
            EE_final_correct+=1
          EE_final_cnt +=1



  return  EE_1_cnt,EE_2_cnt,EE_3_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_3_correct, EE_final_correct

#benefit curve
def calc_accuracy_branchynet(model_name, conf):
  ee1_th, ee2_th = conf, conf
  #read trace data of model
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee1_ref.json', 'r') as fp:
    predict_dict_ee1 = json.load(fp)
  with open('trace_data/'+'trace_data_'+model_name[15:]+'_ee2_ref.json', 'r') as fp:
    predict_dict_ee2 = json.load(fp)
  with open('trace_data/''trace_data_'+model_name[15:]+'_eefinal_ref.json', 'r') as fp:
    predict_dict_eefinal = json.load(fp)

  EE_1_correct, EE_2_correct, EE_final_correct = 0,0,0
  EE_1_cnt, EE_2_cnt, EE_final_cnt = 0,0,0

  EE_final_keys = []
  total_samples = 4448
  
  for num, pred in predict_dict_ee1.items():
    prob_list = [float(x) for x in pred['probability']]
    truth = int(pred['truth'])
    arg_max_1 = np.argmax(prob_list)
    arg_max_2 = np.argsort(prob_list)[-2]
    isCorrect = truth==arg_max_1
    score_max_1, score_max_2 = float(pred['score_max_1']), min(prob_list)
    # score_max_1 = max(prob_list)
    if score_max_1>=ee1_th:
    # if False:
      if isCorrect:
        EE_1_correct +=1
      EE_1_cnt +=1
    else:
      pred_ee2 = predict_dict_ee2[num]
      arg_max_1, score_max_1 = int(pred_ee2['arg_max_1']), float(pred_ee2['score_max_1'])
      isCorrect = truth==arg_max_1
      if score_max_1>=ee2_th:
      # if False:
        if isCorrect:
          EE_2_correct+=1
        EE_2_cnt +=1
      else:
        pred_eefinal = predict_dict_eefinal[num]
        arg_max_1, score_max_1 = int(pred_eefinal['arg_max_1']), float(pred_eefinal['score_max_1'])
        isCorrect = truth==arg_max_1
        if isCorrect:
          EE_final_correct+=1
        EE_final_cnt +=1



  return  EE_1_cnt,EE_2_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_final_correct





def calc_benefit_curve_sdn(model_name):
  #REFERENCE - *100 because 100% of samples take the path; also divide by Batch size
  flops_noEE = 15.69# This is the flops count when we use the pretrainedResnet model from the tinyML repo w/o any changes to the structure
  accuracy_noEE = 85



  flops_ee1, flops_ee2, flops_ee3, flops_eefinal = get_flops_sdn(model_name)
  total_samples = 4448
  
  x_axis_accuracy, y_axis_flops, conf_list= [], [], []
  #vary the ee-1 exit confidence criteria from 0.0 to 1.0 in steps of 0.01
  for conf in list(np.linspace(0.01,1.0, 101)):
    EE_1_cnt,EE_2_cnt,EE_3_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_3_correct, EE_final_correct = calc_accuracy_sdn(model_name, conf)
    # print(EE_1_cnt,EE_2_cnt,EE_3_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_3_correct, EE_final_correct)
    total_accuracy = ((EE_1_correct+EE_2_correct+ EE_3_correct+ EE_final_correct)*100)/total_samples
    flops_total = (flops_ee1*EE_1_cnt + flops_ee2*EE_2_cnt + flops_ee3*EE_3_cnt + flops_eefinal*EE_final_cnt )/total_samples
    flops_total = flops_total/1000000
    # print(total_accuracy, flops_total)
    
    x_axis_accuracy.append(total_accuracy)
    y_axis_flops.append(flops_total)
    conf_list.append(conf)
  x_axis_accuracy_mv = x_axis_accuracy.copy()
  y_axis_flops_mv = y_axis_flops.copy()

  plt.scatter(accuracy_noEE, flops_noEE, label='no EE', color = 'orange')
  # plt.plot((0, flops_EE), (accuracy_EE, flops_EE), color='red', linestyle='--', animated=False)
  plt.vlines(accuracy_noEE, 0,flops_noEE, linestyles='dashed', color='orange')
  plt.hlines(flops_noEE, 0,accuracy_noEE, linestyles='dashed', color='orange')

  plt.scatter(x_axis_accuracy, y_axis_flops, color='blue', label='sdn-method')
  plt.title('Flops vs Accuracy tradeoff for - model name: '+model_name)
  plt.ylabel('FLOPS (millions)')
  plt.xlabel('Total Accuracy (%)')
  plt.xlim([70,100])
  plt.ylim([3,18])
  
  
  # for x,y, conf in zip(x_axis_accuracy,y_axis_flops, conf_list):

  #     # label = f"({x},{y})"
  #     label = f"({conf})"


  #     plt.annotate(label, # this is the text
  #                  (x,y), # these are the coordinates to position the label
  #                  textcoords="offset points", # how to position the text
  #                  xytext=(0,10), # distance from text to points (x,y)
  #                  ha='center') # horizontal alignment can be left, right or center

  # label = f"({accuracy_EE},{flops_EE})"
  # plt.annotate(label, # this is the text
  #                  (accuracy_EE,flops_EE), # these are the coordinates to position the label
  #                  textcoords="offset points", # how to position the text
  #                  xytext=(0,10), # distance from text to points (x,y)
  #                  ha='center') # horizontal alignment can be left, right or center




  label = f"({accuracy_noEE},{flops_noEE})"
  plt.annotate(label, # this is the text
                   (accuracy_noEE,flops_noEE), # these are the coordinates to position the label
                   textcoords="offset points", # how to position the text
                   xytext=(0,10), # distance from text to points (x,y)
                   ha='center') # horizontal alignment can be left, right or center 






  #compare with trecx
  model_name_trecx = 'trained_models/model_mv_ee_at_quarter_with_weighted_loss_point3'
  graph_fname = 'flops.csv'
  readData(model_name_trecx, graph_fname)
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
  # - assuming 1 op = 1 cycle
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

  
  x_axis_accuracy_trecx, y_axis_flops_trecx = [], []
  for conf in list(np.linspace(0.01,1.0, 101)):
    ee_1_accuracy, ee_final_accuracy, ee_percent, total_accuracy = calc_accuracy(model_name_trecx, conf)
    # total_accuracy = ee_1_accuracy + ee_final_accuracy
    flops_ee1 = (latency_ee1 - graph['depth_conv_eefinal_out'].flops )* ee_percent
    flops_eefinal = (latency_ee1+ latency_eefinal_ee) * (1-ee_percent)
    flops_total = flops_ee1+flops_eefinal
    x_axis_accuracy_trecx.append(total_accuracy)
    y_axis_flops_trecx.append(flops_total)
    # conf_list.append(conf)
  x_axis_accuracy_no_mv = x_axis_accuracy_trecx.copy()
  y_axis_flops_no_mv = y_axis_flops_trecx.copy()


  #create data file for tikz graph
  with open('benefit_curve_data_VWW_sdn.dat', 'w') as fp:
    fp.write('accuracy\tflops\n')
    for i in range(len(x_axis_accuracy_mv)):
      fp.write(str(x_axis_accuracy_mv[i])+'\t'+str(y_axis_flops_mv[i])+'\n')
  with open('benefit_curve_data_VWW_trecx.dat', 'w') as fp:
    fp.write('accuracy\tflops\n')
    for i in range(len(x_axis_accuracy_no_mv)):
      fp.write(str(x_axis_accuracy_no_mv[i])+'\t'+str(y_axis_flops_no_mv[i])+'\n')

  plt.scatter(x_axis_accuracy_trecx, y_axis_flops_trecx, color='black', label='trecx')
  plt.legend()

  os.chdir('EE_results')
  fig = plt.gcf()
  fig.set_size_inches((20, 15), forward=False)
  fig.savefig('benefit_curve_'+model_name[15:]+'.png', dpi=1000)
  # plt.savefig('benefit_curve_'+model_name[15:]+'.png')
  os.chdir('..')

  
  plt.show()





def calc_benefit_curve_branchynet(model_name):
  #REFERENCE - *100 because 100% of samples take the path; also divide by Batch size
  flops_noEE = 15.69# This is the flops count when we use the pretrainedResnet model from the tinyML repo w/o any changes to the structure
  accuracy_noEE = 85



  flops_ee1, flops_ee2, flops_eefinal = get_flops_branchynet(model_name)
  total_samples = 4448
  
  x_axis_accuracy, y_axis_flops, conf_list= [], [], []
  #vary the ee-1 exit confidence criteria from 0.0 to 1.0 in steps of 0.01
  for conf in list(np.linspace(0.01,1.0, 101)):
    EE_1_cnt,EE_2_cnt, EE_final_cnt, EE_1_correct,EE_2_correct,  EE_final_correct = calc_accuracy_branchynet(model_name, conf)
    # print(EE_1_cnt,EE_2_cnt,EE_3_cnt, EE_final_cnt, EE_1_correct,EE_2_correct, EE_3_correct, EE_final_correct)
    total_accuracy = ((EE_1_correct+EE_2_correct+  EE_final_correct)*100)/total_samples
    flops_total = (flops_ee1*EE_1_cnt + flops_ee2*EE_2_cnt + flops_eefinal*EE_final_cnt )/total_samples
    flops_total = flops_total/1000000
    # print(total_accuracy, flops_total)
    
    x_axis_accuracy.append(total_accuracy)
    y_axis_flops.append(flops_total)
    conf_list.append(conf)
  x_axis_accuracy_mv = x_axis_accuracy.copy()
  y_axis_flops_mv = y_axis_flops.copy()

  plt.scatter(accuracy_noEE, flops_noEE, label='no EE', color = 'orange')
  # plt.plot((0, flops_EE), (accuracy_EE, flops_EE), color='red', linestyle='--', animated=False)
  plt.vlines(accuracy_noEE, 0,flops_noEE, linestyles='dashed', color='orange')
  plt.hlines(flops_noEE, 0,accuracy_noEE, linestyles='dashed', color='orange')

  plt.scatter(x_axis_accuracy, y_axis_flops, color='blue', label='sdn-method')
  plt.title('Flops vs Accuracy tradeoff for - model name: '+model_name)
  plt.ylabel('FLOPS (millions)')
  plt.xlabel('Total Accuracy (%)')
  plt.xlim([70,100])
  plt.ylim([5,30])
  
  
  # for x,y, conf in zip(x_axis_accuracy,y_axis_flops, conf_list):

  #     # label = f"({x},{y})"
  #     label = f"({conf})"


  #     plt.annotate(label, # this is the text
  #                  (x,y), # these are the coordinates to position the label
  #                  textcoords="offset points", # how to position the text
  #                  xytext=(0,10), # distance from text to points (x,y)
  #                  ha='center') # horizontal alignment can be left, right or center

  # label = f"({accuracy_EE},{flops_EE})"
  # plt.annotate(label, # this is the text
  #                  (accuracy_EE,flops_EE), # these are the coordinates to position the label
  #                  textcoords="offset points", # how to position the text
  #                  xytext=(0,10), # distance from text to points (x,y)
  #                  ha='center') # horizontal alignment can be left, right or center




  label = f"({accuracy_noEE},{flops_noEE})"
  plt.annotate(label, # this is the text
                   (accuracy_noEE,flops_noEE), # these are the coordinates to position the label
                   textcoords="offset points", # how to position the text
                   xytext=(0,10), # distance from text to points (x,y)
                   ha='center') # horizontal alignment can be left, right or center 






  #compare with trecx
  model_name_trecx = 'trained_models/model_mv_ee_at_quarter_with_weighted_loss_point3'
  graph_fname = 'flops.csv'
  readData(model_name_trecx, graph_fname)
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
  # - assuming 1 op = 1 cycle
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

  
  x_axis_accuracy_trecx, y_axis_flops_trecx = [], []
  for conf in list(np.linspace(0.01,1.0, 101)):
    ee_1_accuracy, ee_final_accuracy, ee_percent, total_accuracy = calc_accuracy(model_name_trecx, conf)
    # total_accuracy = ee_1_accuracy + ee_final_accuracy
    flops_ee1 = (latency_ee1 - graph['depth_conv_eefinal_out'].flops )* ee_percent
    flops_eefinal = (latency_ee1+ latency_eefinal_ee) * (1-ee_percent)
    flops_total = flops_ee1+flops_eefinal
    x_axis_accuracy_trecx.append(total_accuracy)
    y_axis_flops_trecx.append(flops_total)
    # conf_list.append(conf)
  x_axis_accuracy_no_mv = x_axis_accuracy_trecx.copy()
  y_axis_flops_no_mv = y_axis_flops_trecx.copy()


  #create data file for tikz graph
  with open('benefit_curve_data_VWW_sdn.dat', 'w') as fp:
    fp.write('accuracy\tflops\n')
    for i in range(len(x_axis_accuracy_mv)):
      fp.write(str(x_axis_accuracy_mv[i])+'\t'+str(y_axis_flops_mv[i])+'\n')
  with open('benefit_curve_data_VWW_trecx.dat', 'w') as fp:
    fp.write('accuracy\tflops\n')
    for i in range(len(x_axis_accuracy_no_mv)):
      fp.write(str(x_axis_accuracy_no_mv[i])+'\t'+str(y_axis_flops_no_mv[i])+'\n')

  plt.scatter(x_axis_accuracy_trecx, y_axis_flops_trecx, color='black', label='trecx')
  plt.legend()

  os.chdir('EE_results')
  fig = plt.gcf()
  fig.set_size_inches((20, 15), forward=False)
  fig.savefig('benefit_curve_'+model_name[15:]+'.png', dpi=1000)
  # plt.savefig('benefit_curve_'+model_name[15:]+'.png')
  os.chdir('..')

  
  plt.show()












def main(argv):
  if len(argv) >= 2:
    model = tf.keras.models.load_model(argv[1])
  else:
    model = mobilenet_v1()
  model_name = argv[1]
  model_arch = argv[2]
  model.summary()





  if model_arch=='sdn':
    # test_sdn(model, val_generator)
    # get_trace_data_sdn(model, val_generator, model_name)
    # model_1 = mobilenet_v1_sdn_incr_pool()
    # output_layer_nums = []
    # cnt = 0
    # for layer in model.layers:
    #   if layer.name in ['ee_1', 'ee_2', 'ee_3', 'dense']:
    #     print(layer.name)
    #     output_layer_nums.append(cnt)
    #   cnt+=1
    # new_model = tf.keras.models.Model(inputs=model_1.inputs[0], outputs=[model_1.layers[output_layer_nums[0]].output, model_1.layers[output_layer_nums[1]].output, model_1.layers[output_layer_nums[2]].output, model_1.layers[output_layer_nums[3]].output])
    # new_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy'])
    # new_model.summary()
    # flops = get_flops(new_model, batch_size=1)
    # ss
    calc_benefit_curve_sdn(model_name)
  else:
    # get_trace_data_branchynet(model, val_generator, model_name)
    calc_benefit_curve_branchynet(model_name)



if __name__ == '__main__':
  # app.run(main)
  main(sys.argv)