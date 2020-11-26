import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold

import LoadConfigFiles as CONF
import TrainingDataKeras as TD
import ANN_reg as AR

###################################################################
# https://stackabuse.com/tensorflow-2-0-solving-classification-and-regression-problems/
# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
###################################################################

ANN_reg = CONF.ANN_reg()
ANN = ANN_reg.ANN_config

Dataset = CONF.Dataset()
Scaling = Dataset.Dataset_config['Scaling']

def compareStandard(save_study_path, values_type_stand, values_scaling, repetitions):
    ###############################################
    # LOAD TRAINING DATA
    ###############################################
    # train_file_path = "./databaseANN/DeltaCartesian_ErrorIncluded/trainingData_Feas_Lambert_big.txt"

    save_file_path =  "./databaseANN/Organized/cartesian/"
    train_file_path = save_file_path + 'Together.txt'

    mae = np.zeros([len(values_type_stand), len(values_scaling), repetitions])
    std_mae = np.zeros([len(values_type_stand), len(values_scaling), repetitions])

    # Evaluate the loss of each data scaling type
    for i in range(len(values_type_stand)):
        for j in range(len(values_scaling)):
            # stand_file_path = save_file_path + 'Together_' + str(values_type_stand[i]) +'_' + str(values_scaling[j]) +'.txt'
            for k in range(repetitions):
                # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
                dataset_np = TD.LoadNumpy(train_file_path, save_file_path, error= 'vector',\
                        equalize = False, \
                        standardizationType =values_type_stand[i], scaling = values_scaling[j] ,
                        plotDistribution=False, plotErrors=False, labelType = 3)
                
                traindata, testdata = TD.splitData_reg(dataset_np, samples = 500)

                ###############################################
                # CREATE AND TRAIN CLASS NETWORK
                ###############################################
                perceptron = AR.ANN_reg(dataset_np)
                perceptron.get_traintestdata(traindata, testdata)
                mae[i, j, k], std_mae[i, j, k] = perceptron.training()

    # Save to file
    np.save(save_study_path+'mae', mae)
    np.save(save_study_path+'_std_mae', std_mae)

    return save_study_path

def plot_compareStandard(save_study_path, values_type_stand, values_scaling, repetitions):
    # load file
    mae = np.load(save_study_path+'mae' +'.npy')
    mae = np.load(save_study_path+'_std_mae' +'.npy')

    labels = ['MinMax Scaler', 'Standard Scaler']
    color = ['black', 'blue', 'green']

    fig = plt.figure(figsize = (13,7))
    for plot in range(len(values_scaling)):
        ax = fig.add_subplot(1,2,plot+1)
        for k in range(repetitions):
            if k == 0:
                plt.scatter(values_type_stand, mae[:, plot, k], marker = 'o', color = color[plot],  label = labels[values_scaling[plot]])
            else:
                plt.scatter(values_type_stand, mae[:, plot, k], marker = 'o', color = color[plot])

        plt.ylabel("Loss (MSE)")
        plt.legend(loc = 'upper left')
        plt.xticks([0,1,2], ['Common', 'Input-Output', 'Input-Output-Output'])
        plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(save_study_path, dpi = 100)
    plt.show()

    fig = plt.figure(figsize = (13,7))
    for plot in range(len(values_scaling)):
        ax = fig.add_subplot(1,2,plot+1)
        mae_mean = [np.mean(mae[i, plot, :]) for i in range(len(values_type_stand))]
        mae_std = [np.std(mae[i, plot, :]) for i in range(len(values_type_stand))]
        for i in range(len(values_type_stand)):
            plt.errorbar(values_type_stand[i], mae_mean[i], mae_std[i], marker = 'o', color = color[plot])
        plt.title(labels[plot])
        plt.ylabel("Loss (MSE)")
        # plt.legend(loc = 'upper left')
        plt.xticks([0,1,2], ['Common', 'Input-Output', 'Input-Output-Output'])
        plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(save_study_path+"errorbar.png", dpi = 100)
    plt.show()

def compareInputs(repetitions, typeInputs, save_study_path): # Using same values
    mae = np.zeros([len(typeInputs), repetitions])
    std_mae = np.zeros([len(typeInputs),  repetitions])

    for i in range(len(typeInputs)):
        save_file_path =  "./databaseANN/Organized/" + typeInputs[i] +"/"
        train_file_path = save_file_path + 'Random.txt'

        # stand_file_path = save_file_path + 'Together_' + str(values_type_stand[i]) +'_' + str(values_scaling[j]) +'.txt'
        for k in range(repetitions):
            # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
            dataset_np = TD.LoadNumpy(train_file_path, save_file_path, error= 'vector',\
                    equalize = False, \
                    standardizationType = Scaling['type_stand'], scaling = Scaling['scaling'],\
                    plotDistribution=False, plotErrors=False, labelType = 3)
            
            traindata, testdata = TD.splitData_reg(dataset_np, samples = 500)

            ###############################################
            # CREATE AND TRAIN CLASS NETWORK
            ###############################################
            perceptron = AR.ANN_reg(dataset_np)
            perceptron.get_traintestdata(traindata, testdata)
            mae[i, k], std_mae[i, k] = perceptron.training()

            print("####################################################")
            print(mae, i, k )
        
    with open(save_study_path+"MAE.txt", "w") as myfile:
        np.savetxt(myfile, mae)
    myfile.close()
    with open(save_study_path+"STD_MAE.txt", "w") as myfile:
        np.savetxt(myfile, std_mae)
    myfile.close()

    return save_study_path

def plot_compareInputs(path, repetitions, typeInputs):
    mae = np.loadtxt(path+"MAE.txt")
    std_mae = np.loadtxt(path+"STD_MAE.txt")

    color = ['black', 'blue', 'green']

    for i in range(len(typeInputs)):
        plt.plot( np.arange(0,repetitions, 1), mae[i, :], marker = 'o', color = color[i], label = typeInputs[i] )
    
    plt.xticks(np.arange(0, repetitions, 1))
    plt.title("Mean Average Error")
    plt.xlabel('Repetitions')
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison.png", dpi = 100)
    plt.show()

if __name__ == "__main__":
    # COMPARE STANDARDIZATION
    values_type_stand = [0, 1, 2]
    values_scaling = [0, 1]
    repetitions = 5 # repetitions of each setting
    save_study_path =  "./Results/StudyStandardization/Results"
    # path = compareStandard(save_study_path, values_type_stand, values_scaling, repetitions)
    # plot_compareStandard(save_study_path, values_type_stand, values_scaling, repetitions)

    # COMPARE INPUTS
    repetitions = 3 # repetitions of each setting
    typeInputs = ['cartesian', 'deltakeplerian', 'deltakeplerian_planet']
    save_study_path =  "./Results/StudyInputs/"
    # compareInputs(repetitions, typeInputs, save_study_path)
    plot_compareInputs(save_study_path, repetitions, typeInputs)
    