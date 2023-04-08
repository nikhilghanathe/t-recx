'''
Project: t-recx
Subproject: keyword spotting on Speech Command dataset
File:test_dscnn.py 
desc: Loads data, generates trace of all predictions and stores in a json file. 
    Next, it generates all plots from paper and saves to results/ 

'''

import matplotlib.pyplot as plt
import numpy as np
import os, csv
import argparse
from tensorflow import keras

import tensorflow as tf
import keras_model as models
import get_dataset as kws_data
import kws_util
import json
import itertools
from keras_flops import get_flops
import helpers, Config





# ===============================================================================
#plot the benefit curve with and without weight transfer (Fig 4b from the paper)
def plot_benefit_curve(model_name_ev, model_name_noev, total_samples):
    #calc benefit curve with EV-assistance
    x_axis_accuracy_ev, y_axis_flops_ev = helpers.calculate_scatter_points(model_name_ev, total_samples)
    #calc benefit curve without EV-assistance
    x_axis_accuracy_noev, y_axis_flops_noev = helpers.calculate_scatter_points(model_name_noev, total_samples)
    

    #plot benefit curve
    fig, ax = plt.subplots()
    plt.scatter(Config.accuracy_noEE, Config.flops_noEE, label='no EE', color = 'red')
    plt.vlines(Config.accuracy_noEE, 0,Config.flops_noEE, linestyles='dashed', color='orange')
    plt.hlines(Config.flops_noEE, 0,Config.accuracy_noEE, linestyles='dashed', color='orange')
    plt.scatter(x_axis_accuracy_ev, y_axis_flops_ev, color='blue', label='w/ weight transfer')
    plt.scatter(x_axis_accuracy_noev, y_axis_flops_noev, color='purple', label='w/o weight transfer ')
    plt.title('Flops vs Accuracy tradeoff (Benefit curve for DSCNN) Fig4a')
    plt.ylabel('FLOPS (millions)')
    plt.xlabel('Total Accuracy (%)')
    plt.xlim([83,95])
    plt.ylim([2,7])

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
    fig.savefig('Fig4b.png', dpi=1000)
    os.chdir('..')
    # plt.show()



#plot the benefit curve with & without weight transfer, and with ee-fmaps concatenated with final-fmaps (Fig 6 from the paper)
def plot_benefit_curve_ev_effectiveness(model_name_ev, model_name_noev, model_name_eefmaps_concat, total_samples):
    #calc benefit curve with EV-assistance
    x_axis_accuracy_ev, y_axis_flops_ev = helpers.calculate_scatter_points(model_name_ev, total_samples)
    #calc benefit curve without EV-assistance
    x_axis_accuracy_noev, y_axis_flops_noev = helpers.calculate_scatter_points(model_name_noev, total_samples)
    #calc benefit curve with EE-fmaps-concat
    x_axis_accuracy_eefmaps_concat, y_axis_flops_eefmaps_concat = helpers.calculate_scatter_points(model_name_eefmaps_concat, total_samples)
    

    #plot benefit curve
    fig, ax = plt.subplots()
    plt.scatter(Config.accuracy_noEE, Config.flops_noEE, label='no EE', color = 'red')
    plt.vlines(Config.accuracy_noEE, 0,Config.flops_noEE, linestyles='dashed', color='orange')
    plt.hlines(Config.flops_noEE, 0,Config.accuracy_noEE, linestyles='dashed', color='orange')
    plt.scatter(x_axis_accuracy_ev, y_axis_flops_ev, color='blue', label='with weight transfer')
    plt.scatter(x_axis_accuracy_noev, y_axis_flops_noev, color='purple', label='without weight transfer ')
    plt.scatter(x_axis_accuracy_eefmaps_concat, y_axis_flops_eefmaps_concat, color='green', label='EE ')
    plt.title('Flops vs Accuracy tradeoff (Fig6: Benefit curve for DSCNN: Benefit curve for DS-CNN:\n 1) w/ weight transfer\n 2) w/o weight transfer from early to final exit\n3) concatenation of early-exit and final exit feature maps at final exit)')
    plt.ylabel('FLOPS (millions)')
    plt.xlabel('Total Accuracy (%)')
    plt.xlim([86,95])
    plt.ylim([2,7])

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
    fig.savefig('Fig6.png', dpi=1000)
    os.chdir('..')


# ----------------------------------------------------------------------------------------
# --------------Comparison with prior works-----------------------------------------------
# ----------------------------------------------------------------------------------------
# Plot the benefit curve of resnet with trecx, branchynet and SDN techniques - Fig 7b
def plot_benefit_curve_prior(model_name_ev, model_name_sdn, model_name_branchynet, total_samples):
    x_axis_accuracy_ev, y_axis_flops_ev = helpers.calculate_scatter_points(model_name_ev, total_samples)
    x_axis_accuracy_sdn, y_axis_flops_sdn = helpers.calculate_scatter_points_prior(model_name_sdn, 'sdn', total_samples)
    x_axis_accuracy_branchynet, y_axis_flops_branchynet = helpers.calculate_scatter_points_prior(model_name_branchynet, 'branchynet', total_samples)

    #plot benefit curve
    fig, ax = plt.subplots()
    plt.scatter(Config.accuracy_noEE, Config.flops_noEE, label='no EE', color = 'red')
    plt.vlines(Config.accuracy_noEE, 0,Config.flops_noEE, linestyles='dashed', color='orange')
    plt.hlines(Config.flops_noEE, 0,Config.accuracy_noEE, linestyles='dashed', color='orange')
    plt.scatter(x_axis_accuracy_ev, y_axis_flops_ev, color='blue', label='T-RECX')
    plt.scatter(x_axis_accuracy_sdn, y_axis_flops_sdn, color='black', label='SDN')
    plt.scatter(x_axis_accuracy_branchynet, y_axis_flops_branchynet, color='green', label='BRANCHYNET')
    plt.title('Flops vs Accuracy tradeoff (Comparison with Prior works) Fig7a')
    plt.ylabel('FLOPS (millions)')
    plt.xlabel('Total Accuracy (%)')
    plt.xlim([85,95])
    plt.ylim([2,15])
    plt.yscale('log')

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
    fig.savefig('Fig7b.png', dpi=1000)
    os.chdir('..')


def evaluate_models(ds_test):
    print('========================================================')
    print('Evaluating base model...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_base)
    test_metrics = model.evaluate(ds_test)
    print('Standalone accuracies are ', test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating T-Recx model with EV-assistance...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_ev)
    test_metrics = model.evaluate(ds_test)
    print('Standalone accuracies are ', test_metrics[-2], test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating T-Recx model without EV-assistance...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_noev)
    test_metrics = model.evaluate(ds_test)
    print('Standalone accuracies are ', test_metrics[-2], test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating EE-fmaps concat model...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_eefmaps_concat)
    test_metrics = model.evaluate(ds_test)
    print('Standalone accuracies are ', test_metrics[-2], test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating SDN-DSCNN...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_sdn)
    test_metrics = model.evaluate(ds_test)
    print('Standalone accuracies are ', test_metrics[-4], test_metrics[-3], test_metrics[-2], test_metrics[-1])
    print('DONE!\n')
    
    print('========================================================')
    print('Evaluating Branchynet-DSCNN...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_branchynet)
    test_metrics = model.evaluate(ds_test)
    print('Standalone accuracies are ', test_metrics[-3], test_metrics[-2], test_metrics[-1])
    print('DONE!\n\n\n')



if __name__ == "__main__":
    Flags, unparsed = kws_util.parse_command()
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    print("Done getting data")
    
    model = tf.keras.models.load_model(Flags.model_init_path)
    # model.summary()
    test_metrics = model.evaluate(ds_test)
    print('Standalone accuracies are ', test_metrics[-2], test_metrics[-1])
    ss

    #evaluate models
    evaluate_models(ds_test)
    # =========Generate Fig 4b=============================
    # generate trace_data for EV-assist and noEV-assist models
    print('=====================================')
    print('Generating trace data. This may take several minutes (10-15min) to complete...')
    print('=====================================')
    helpers.generate_trace(ds_test, Config.model_name_ev)
    helpers.generate_trace(ds_test, Config.model_name_noev)
    helpers.generate_trace(ds_test, Config.model_name_eefmaps_concat)
    print('DONE!')
    #plot the benefit curve
    print('=====================================')
    print('Plotting benefit curve...The image will be saved in results/Fig4b.png')
    print('=====================================')
    plot_benefit_curve(Config.model_name_ev, Config.model_name_noev, total_samples=Config.total_samples)
    print('DONE!\n\n')
    # ====================================================


    #=========Generate Fig 6===========================
    print('=====================================')
    print('Plotting benefit curve to compare EV-assist effectiveness ...The image will be saved in results/Fig6.png')
    print('=====================================')
    plot_benefit_curve_ev_effectiveness(Config.model_name_ev, Config.model_name_noev, Config.model_name_eefmaps_concat, total_samples=Config.total_samples)
    print('DONE!\n\n')
    # ====================================================


    #=========Generate Fig 7b===========================
    print('\n=============================================================')
    print('Generating trace data... This may take several minutes (10-15min) to complete...')
    print('===============================================================\n')
    helpers.generate_trace(ds_test, Config.model_name_ev)
    helpers.generate_trace_prior(ds_test, Config.model_name_sdn, model_arch='sdn')
    helpers.generate_trace_prior(ds_test, Config.model_name_branchynet, model_arch='branchynet')
    print('DONE!')
    #plot the benefit curve
    print('=====================================')
    print('Plotting benefit curve for prior work...The image will be saved in results/Fig7b.png')
    print('=====================================\n')
    plot_benefit_curve_prior(Config.model_name_ev, Config.model_name_sdn, Config.model_name_branchynet, total_samples=Config.total_samples)
    print('DONE!\n\n')
    # ====================================================
    
