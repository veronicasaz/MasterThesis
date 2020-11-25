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


if __name__ == "__main__":

    ###############################################
    # LOAD TRAINING DATA
    ###############################################
    # train_file_path = "./databaseANN/DeltaCartesian_ErrorIncluded/trainingData_Feas_Lambert_big.txt"

    save_file_path =  "./databaseANN/Organized/cartesian/"
    train_file_path = save_file_path + 'Together.txt'
    save_study_path =  "./Results/StudyStandardization/Results.txt"

    values_type_stand = [0, 1, 2]
    values_scaling = [0, 1]
    repetitions = 5 # repetitions of each setting

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



    labels = ['MinMax Scaler', 'Standard Scaler']
    color = ['black', 'blue', 'green']
    for plot in range(len(values_scaling)):
        for k in range(repetitions):
            if k == 0:
                plt.scatter(values_type_stand, mae[:, plot, k], marker = 'o', color = color[plot],  label = labels[values_scaling[plot]])
            else:
                plt.scatter(values_type_stand, mae[:, plot, k], marker = 'o', color = color[plot])

    plt.legend()
    plt.xticks([0,1], ['Common', 'Input-Output', 'Input-Output-Output'])
    plt.tight_layout()
    plt.savefig(save_study_path, dpi = 100)
    plt.show()
