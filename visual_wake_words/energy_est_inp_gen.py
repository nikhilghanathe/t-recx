'''

Input Formatting for energy est tool https://energyestimation.mit.edu/

0: layer index
1-6: height, width, num_channels, num_filts, num_zero_entries, bitwidth (ifmap)
7-12: height, width, num_channels, num_filts, num_zero_entries, bitwidth (filt)
13-18: height, width, num_channels, num_filts, num_zero_entries, bitwidth (ofmap)
19, 20: stride -tuple
21-24: padding (top, botton, left, right)


Layer Index: the index of the layer, from 1 to the number of layers. It should be the same as the line number.
Conf_IfMap, Conf_Filt, Conf_OfMap: the configuration of the input feature maps, the filters and the output feature maps. The configuration of each of the three data types is in the format of "height width number_of_channels number_of_maps_or_filts number_of_zero_entries bitwidth_in_bits".
Stride: the stride of this layer. It is in the format of "stride_y stride_x".
Pad: the amount of input padding. It is in the format of "pad_top pad_bottom pad_left pad_right".
'''

import os, sys
import tensorflow as tf
import numpy as np


#load model
model_name= sys.argv[1]
model = tf.keras.models.load_model(model_name)
model.summary()


#create config file
config_dict = {}
cnt= 1
for layer in model.layers:
	if layer.name[:4]=='conv' or layer.name[:14]=='depthwise_conv': #support for conv and dw
		input_shape = layer.input_shape
		output_shape = layer.output_shape
		weight_shape = np.array(layer.get_weights()[0]).shape
		# config = [cnt] +   [x for x in input_shape[1:]] + [1, 0, 16]  +   [x for x in weight_shape[0:2]] + [input_shape[3], weight_shape[2]] +[0, 16] +   [x for x in output_shape[1:]] + [1, 0, 16]
		config = [cnt] +   [x for x in input_shape[1:]] + [1, 0, 16]  +  [x for  x in weight_shape]+ [0, 16] + [x for x in output_shape[1:]] + [1, 0, 16]
		strides = layer.strides
		padding = 1 if weight_shape[1]==3 else 0
		config += [x for x in strides] + 4*[padding]
		config_dict.update({cnt:config})
	# elif layer.name[:5]=='dense': #support for conv and FC layer
	# 	input_shape = layer.input_shape
	# 	output_shape = layer.output_shape
	# 	weight_shape = np.array(layer.get_weights()[0]).shape
	# 	config = [cnt] +   [x for x in input_shape[1:]] + [0, 16]  +   [x for x in weight_shape[1:]] + [0, 16] +   [x for x in output_shape[1:]] + [0, 16]
	# 	strides = layer.strides
	# 	padding = 1 if weight_shape[1]==3 else 0
	# 	config += [x for x in strides] + 4*[padding]
		print('\n', cnt)
		print(layer.name)
		print(input_shape)
		print(output_shape)
		print(weight_shape)
		print(layer.strides)
		print(layer.padding)
		print(padding)
		print(config)
		
		cnt+=1


#write config file
with open(model_name[15:-3]+'.txt', 'w') as fp:
	for num, config in config_dict.items():
		config_str = ",".join(str(x) for x in config)
		fp.write(config_str+'\n')



