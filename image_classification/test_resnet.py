'''
Project: t-recx
Subproject: Image classification on cifar10
File:test_resnet.py 
desc: Loads data, generates trace of all predictions and stores in a json file. 
    Next, it generates all plots from paper and saves to results/ 

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import train_resnet as train
import keras_model
import os, sys, copy, json
from tensorflow.keras.utils import to_categorical
import argparse, os
import helpers, Config

# ===============================================================================
#plot the benefit curve with and without weight transfer (Fig 4a from the paper)
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
    plt.scatter(x_axis_accuracy_noev, y_axis_flops_noev, color='black', label='w/o weight transfer ')
    plt.title('Flops vs Accuracy tradeoff (Benefit curve for Resnet) Fig4a')
    plt.ylabel('FLOPS (millions)')
    plt.xlabel('Total Accuracy (%)')
    plt.xlim([75,90])
    plt.ylim([7,30])

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
    # fig = plt.gcf()
    # fig.set_size_inches((20, 15), forward=False)
    plt.savefig('Fig4a.png', dpi=1000)
    os.chdir('..')
    # plt.show()

    # #create data file for tikz graph
    # with open('benefit_curve_data_IC_ev.dat', 'w') as fp:
    #     fp.write('accuracy\tflops\n')
    #     for i in range(len(x_axis_accuracy_mv)):
    #         fp.write(str(x_axis_accuracy_ev[i])+'\t'+str(y_axis_flops_ev[i])+'\n')
    # with open('benefit_curve_data_IC_no_ev.dat', 'w') as fp:
    #     fp.write('accuracy\tflops\n')
    #     for i in range(len(x_axis_accuracy_no_ev)):
    #         fp.write(str(x_axis_accuracy_no_ev[i])+'\t'+str(y_axis_flops_no_ev[i])+'\n')



# ----------------------------------------------------------------------------------------
# --------------Comparison with prior works-----------------------------------------------
# ----------------------------------------------------------------------------------------
# Plot the benefit curve of resnet with trecx, branchynet and SDN techniques
def plot_benefit_curve_prior(model_name_ev, model_name_sdn, model_name_branchynet, total_samples):
    x_axis_accuracy_ev, y_axis_flops_ev = helpers.calculate_scatter_points(model_name_ev, total_samples)
    x_axis_accuracy_sdn, y_axis_flops_sdn = helpers.calculate_scatter_points_prior(model_name_sdn, total_samples)
    x_axis_accuracy_branchynet, y_axis_flops_branchynet = helpers.calculate_scatter_points_prior(model_name_branchynet, total_samples)

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
    plt.xlim([70,90])
    plt.ylim([10,35])
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
    plt.savefig('Fig7a.png', dpi=1000)
    os.chdir('..')




def evaluate_models(test_gen):
    print('========================================================')
    print('Evaluating base model...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_base)
    test_metrics = model.evaluate(test_gen)
    print('Standalone accuracies are ', test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating T-Recx model with EV-assistance...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_ev)
    test_metrics = model.evaluate(test_gen)
    print('Standalone accuracies are ', test_metrics[-2], test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating T-Recx model without EV-assistance...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_noev)
    test_metrics = model.evaluate(test_gen)
    print('Standalone accuracies are ', test_metrics[-2], test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating base model with baselineEE...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_baseline_ee)
    test_metrics = model.evaluate(test_gen)
    print('Standalone accuracies are ', test_metrics[-2], test_metrics[-1])
    print('DONE!\n')

    print('========================================================')
    print('Evaluating SDN-Resnet...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_sdn)
    test_metrics = model.evaluate(test_gen)
    print('Standalone accuracies are ', test_metrics[-3], test_metrics[-2], test_metrics[-1])
    print('DONE!\n')
    
    print('========================================================')
    print('Evaluating Branchynet-Resnet...')
    print('========================================================')
    model = tf.keras.models.load_model(Config.model_name_branchynet)
    test_metrics = model.evaluate(test_gen)
    print('Standalone accuracies are ', test_metrics[-3], test_metrics[-2], test_metrics[-1])
    print('DONE!\n\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_trace', type=bool, default=False, 
        help="""If set to true, get trace data """)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    cifar_10_dir = 'cifar-10-batches-py'
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_data(cifar_10_dir)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_gen = datagen.flow(test_data, test_labels,  batch_size=Config.test_batch_size)#batch test data for faster processing

    #evaluate models and report accuracies
    evaluate_models(test_gen)

    # =========Generate Fig 4a=============================
    # generate trace_data for EV-assist and noEV-assist models
    print('=====================================')
    print('Generating trace data. This may take several minutes (10-20min) to complete...')
    print('=====================================')
    helpers.generate_trace(test_gen, Config.model_name_ev)
    helpers.generate_trace(test_gen, Config.model_name_noev)
    helpers.generate_trace(test_gen, Config.model_baseline_ee)
    print('DONE!')
    #plot the benefit curve
    print('=====================================')
    print('Plotting benefit curve...The image will be saved in results/Fig4a.png')
    print('=====================================')
    plot_benefit_curve(Config.model_name_ev, Config.model_name_noev, total_samples=int(test_labels.shape[0]))
    print('DONE!\n\n')
    # ====================================================




    #=========Generate Fig 7a===========================
    print('\n=============================================================')
    print('Generating trace data... This may take several minutes (10-15min) to complete...')
    print('===============================================================\n')
    helpers.generate_trace(test_gen, Config.model_name_ev)
    helpers.generate_trace_prior(test_gen, Config.model_name_sdn)
    helpers.generate_trace_prior(test_gen, Config.model_name_branchynet)
    print('DONE!')
    #plot the benefit curve
    print('=====================================')
    print('Plotting benefit curve for prior work...The image will be saved in results/Fig7a.png')
    print('=====================================\n')
    plot_benefit_curve_prior(Config.model_name_ev, Config.model_name_sdn, Config.model_name_branchynet, total_samples=int(test_labels.shape[0]))
    print('DONE!\n\n')
    # ====================================================
