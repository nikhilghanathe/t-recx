
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from absl import app

import tensorflow as tf
import numpy as np
import json
assert tf.__version__.startswith('2')

import helpers, Config

BASE_DIR = os.path.join(os.getcwd(), 'minival')




# ===============================================================================
#plot the benefit curve with and without weight transfer (Fig 4c from the paper)
def plot_benefit_curve(model_name_ev, model_name_noev, total_samples):
    #calc benefit curve with EV-assistance
    x_axis_accuracy_ev, y_axis_flops_ev = helpers.calculate_scatter_points(model_name_ev, total_samples)
    #calc benefit curve without EV-assistance
    x_axis_accuracy_noev, y_axis_flops_noev = helpers.calculate_scatter_points(model_name_noev, total_samples)
    

    #plot benefit curve
    plt.scatter(Config.accuracy_noEE, Config.flops_noEE, label='no EE', color = 'red')
    plt.vlines(Config.accuracy_noEE, 0,Config.flops_noEE, linestyles='dashed', color='orange')
    plt.hlines(Config.flops_noEE, 0,Config.accuracy_noEE, linestyles='dashed', color='orange')
    plt.scatter(x_axis_accuracy_ev, y_axis_flops_ev, color='blue', label='w/ weight transfer')
    plt.scatter(x_axis_accuracy_noev, y_axis_flops_noev, color='black', label='w/o weight transfer ')
    plt.title('Flops vs Accuracy tradeoff (Benefit curve for Resnet) Fig4a')
    plt.ylabel('FLOPS (millions)')
    plt.xlabel('Total Accuracy (%)')
    plt.xlim([50,90])
    plt.ylim([0,30])

    #annotate the plot with accuracy and flops of the base model
    label = f"({Config.accuracy_noEE},{Config.flops_noEE})"
    plt.annotate(label, # this is the text
                     (Config.accuracy_noEE,Config.flops_noEE), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center   

    plt.legend()

    if not os.path.exists('results'):
        os.makedirs('results')
    os.chdir('results')
    fig = plt.gcf()
    fig.set_size_inches((20, 15), forward=False)
    fig.savefig('Fig4a.png', dpi=1000)
    os.chdir('..')
    # plt.show()



def main(argv):
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1. / 255)
  val_generator = datagen.flow_from_directory(
      BASE_DIR,
      target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
      batch_size=Config.BATCH_SIZE,
      color_mode='rgb')

  print('Done getting data!\n')


  #=========Generate Fig 4c=============================
    #generate trace_data for EV-assist and noEV-assist models
    print('=====================================')
    print('Generating trace data. This may take several minutes (20-30min) to complete...')
    print('=====================================')
    helpers.generate_trace(val_generator, Config.model_name_ev)
    helpers.generate_trace(val_generator, Config.model_name_noev)
    print('DONE!')
    #plot the benefit curve
    print('=====================================')
    print('Plotting benefit curve...The image will be saved in results/Fig4c.png')
    print('=====================================')
    plot_benefit_curve(Config.model_name_ev, Config.model_name_noev, total_samples=Config.total_samples)
    print('DONE!\n\n')
    # ====================================================


