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

Dataset_conf = CONF.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']

def compareStandard(save_study_path, values_type_stand, values_scaling, repetitions):
    ###############################################
    # LOAD TRAINING DATA
    ###############################################
    # train_file_path = "./databaseANN/DeltaCartesian_ErrorIncluded/trainingData_Feas_Lambert_big.txt"

    save_file_path =  "./databaseANN/DatabaseOptimized/deltakeplerian/"
    train_file_path = save_file_path + 'Random.txt'

    loss = np.zeros([len(values_type_stand), len(values_scaling), repetitions])
    val_loss = np.zeros([len(values_type_stand), len(values_scaling), repetitions])

    # Evaluate the loss of each data scaling type
    for i in range(len(values_type_stand)):
        for j in range(len(values_scaling)):
            # stand_file_path = save_file_path + 'Together_' + str(values_type_stand[i]) +'_' + str(values_scaling[j]) +'.txt'
            for k in range(repetitions):
                # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
                dataset_np = TD.LoadNumpy(train_file_path, save_file_path, error= 'vector',\
                        equalize = False, \
                        standardizationType =values_type_stand[i], scaling = values_scaling[j],\
                        dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
                        plotDistribution=False, plotErrors=False, labelType = False)
                
                traindata, testdata = TD.splitData_reg(dataset_np)

                print(i,j,k)
                ###############################################
                # CREATE AND TRAIN CLASS NETWORK
                ###############################################
                perceptron = AR.ANN_reg(dataset_np)
                perceptron.get_traintestdata(traindata, testdata)
                loss[i, j, k], val_loss[i, j, k] = perceptron.training()

    # Save to file
    np.save(save_study_path+'loss', loss)
    np.save(save_study_path+'_val_loss', val_loss)

    return save_study_path

def plot_compareStandard(save_study_path, values_type_stand, values_scaling, repetitions):
    # load file
    loss = np.load(save_study_path+'loss' +'.npy')
    loss = np.load(save_study_path+'_val_loss' +'.npy')

    labels = ['MinMax Scaler', 'Standard Scaler']
    color = ['black', 'blue', 'green']

    fig = plt.figure(figsize = (13,7))
    for plot in range(len(values_scaling)):
        ax = fig.add_subplot(1,2,plot+1)
        for k in range(repetitions):
            if k == 0:
                plt.scatter(values_type_stand, loss[:, plot, k], marker = 'o', color = color[plot],  label = labels[values_scaling[plot]])
            else:
                plt.scatter(values_type_stand, loss[:, plot, k], marker = 'o', color = color[plot])

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
        loss_mean = [np.mean(loss[i, plot, :]) for i in range(len(values_type_stand))]
        loss_std = [np.std(loss[i, plot, :]) for i in range(len(values_type_stand))]
        for i in range(len(values_type_stand)):
            plt.errorbar(values_type_stand[i], loss_mean[i], loss_std[i], marker = 'o', color = color[plot])
        plt.title(labels[plot])
        plt.ylabel("Loss (MSE)")
        # plt.legend(loc = 'upper left')
        plt.xticks([0,1,2], ['Common', 'Input-Output', 'Input-Output-Output'])
        plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(save_study_path+"errorbar.png", dpi = 100)
    plt.show()

def compareInputs(repetitions, typeInputs, save_study_path): # Using same values
    loss = np.zeros([len(typeInputs), repetitions])
    val_loss = np.zeros([len(typeInputs),  repetitions])

    for i in range(len(typeInputs)):
        save_file_path =  "./databaseANN/DatabaseOptimized/"  + typeInputs[i] +"/"
        train_file_path = save_file_path + 'Random.txt'

        # stand_file_path = save_file_path + 'Together_' + str(values_type_stand[i]) +'_' + str(values_scaling[j]) +'.txt'
        for k in range(repetitions):
            # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
            dataset_np = TD.LoadNumpy(train_file_path, save_file_path, error= 'vector',\
                    equalize = False, \
                    standardizationType = Scaling['type_stand'], scaling = Scaling['scaling'],\
                    dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
                    plotDistribution=False, plotErrors=False, labelType = 3)
            
            traindata, testdata = TD.splitData_reg(dataset_np, samples = 500)

            ###############################################
            # CREATE AND TRAIN CLASS NETWORK
            ###############################################
            perceptron = AR.ANN_reg(dataset_np)
            perceptron.get_traintestdata(traindata, testdata)
            loss[i, k], val_loss[i, k] = perceptron.training()

            print("####################################################")
            print(loss, i, k )
        
    with open(save_study_path+"loss.txt", "w") as myfile:
        np.savetxt(myfile, loss)
    myfile.close()
    with open(save_study_path+"val_loss.txt", "w") as myfile:
        np.savetxt(myfile, val_loss)
    myfile.close()

    return save_study_path

def plot_compareInputs(path, repetitions, typeInputs):
    loss = np.loadtxt(path+"loss.txt")
    val_loss = np.loadtxt(path+"val_loss.txt")

    color = ['black', 'blue', 'green']

    for i in range(len(typeInputs)):
        plt.plot( np.arange(0,repetitions, 1), loss[i, :], marker = 'o', color = color[i], label = typeInputs[i] )
    
    plt.xticks(np.arange(0, repetitions, 1))
    plt.title("Mean Squared Error for different types of inputs")
    plt.xlabel('Repetitions')
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison.png", dpi = 100)
    plt.show()

def compareNumSamples(save_study_path, typeSamples, repetitions):
    loss = np.zeros([len(typeSamples), repetitions])
    val_loss = np.zeros([len(typeSamples),  repetitions])

    

    for i in range(len(typeSamples)):   
        if typeSamples[i] == 500:
            save_file_path =  "./databaseANN/DatabaseOptimized/deltakeplerian/500_AU/" 
        elif typeSamples[i] == 5000:
            save_file_path =  "./databaseANN/DatabaseOptimized/deltakeplerian/500_AU/"
        
        train_file_path = save_file_path + 'Random.txt'

        # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
        dataset_np = TD.LoadNumpy(train_file_path, save_file_path, error= 'vector',\
                equalize = False, \
                standardizationType = Scaling['type_stand'], scaling = Scaling['scaling'],\
                dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
                plotDistribution=False, plotErrors=False, labelType = False)
        
        traindata, testdata = TD.splitData_reg(dataset_np)
        traindata2 = [0,0]
        for k in range(repetitions):
            traindata2[0] = traindata[0][0:typeSamples[i], :] # select only the ones to study
            traindata2[1] = traindata[1][0:typeSamples[i], :] # select only the ones to study
            ###############################################
            # CREATE AND TRAIN CLASS NETWORK
            ###############################################
            perceptron = AR.ANN_reg(dataset_np)
            perceptron.get_traintestdata(traindata2, testdata)
            loss[i, k], val_loss[i, k] = perceptron.training()

            print("####################################################")
            print(loss, i, k )
        
    with open(save_study_path+"loss.txt", "w") as myfile:
        np.savetxt(myfile, loss)
    myfile.close()
    with open(save_study_path+"val_loss.txt", "w") as myfile:
        np.savetxt(myfile, val_loss)
    myfile.close()

    return save_study_path

def plot_compareNumSamples(path, typeSamples, repetitions):
    loss = np.loadtxt(path+"loss.txt")
    val_loss = np.loadtxt(path+"val_loss.txt")

    color = ['black', 'blue', 'green', 'orange', 'yellow']

    for i in range(len(typeSamples)):
        plt.plot( np.arange(0,repetitions, 1), loss[i, :], marker = 'o', color = color[i], label = typeSamples[i] )
    
    plt.xticks(np.arange(0, repetitions, 1))
    plt.title("Mean Squared Error for different number of samples")
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
    # plot_compareInputse_study_path, repetitions, typeInputs)
    
    # COMPARE NUMBER SAMPLES: LEARNING CURVE
    repetitions = 5
    typeSamples = [500, 5000]
    save_study_path =  "./Results/StudyNSamples/DifferentFiles/"
    # compareNumSamples(save_study_path, typeSamples, repetitions)
    # plot_compareNumSamples(save_study_path, typeSamples, repetitions)