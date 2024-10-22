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
import TrainingData as TD
import ANN_reg_2 as AR

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
        save_file_path =  "./databaseANN/DatabaseFitness/"  + typeInputs[i] +"/"
        train_file_path = save_file_path + 'Together.txt'

        # stand_file_path = save_file_path + 'Together_' + str(values_type_stand[i]) +'_' + str(values_scaling[j]) +'.txt'
        for k in range(repetitions):
            # TD.plotInitialDataPandasError(train_file_path,  save_study_path + typeInputs[i] +"/",  pairplot= False, corrplot= True)
            dataset_np = TD.LoadNumpy(train_file_path,
                    scaling = Scaling['scaling'],\
                    dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
                    outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
                    output_type = Dataset_conf.Dataset_config['Outputs'],
                    plotDistribution=False, plotErrors=False, labelType = 2)

            traindata, testdata = TD.splitData_reg(dataset_np)

            ###############################################
            # CREATE AND TRAIN CLASS NETWORK
            ###############################################
            perceptron = AR.ANN_reg(dataset_np, save_path = save_file_path)
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

    # Using errorbars
    mean_loss = np.zeros((len(typeInputs),2)) # 1 column for loss and 1 for val_loss
    std_loss = np.zeros((len(typeInputs),2)) # 1 column for loss and 1 for val_loss
    for i in range(len(typeInputs)):
        mean_loss[i, 0] = np.mean(loss[i,:])
        std_loss[i, 0] = np.std(loss[i,:])
        mean_loss[i, 1] = np.mean(val_loss[i,:])
        std_loss[i, 1] = np.std(val_loss[i,:])

    plt.errorbar(np.arange(0,len(typeInputs),1), mean_loss[:,0], std_loss[:,0], linestyle='None', marker='^')
    plt.errorbar(np.arange(0,len(typeInputs),1), mean_loss[:,1], std_loss[:,1], linestyle='None', marker='o')

    plt.ylabel("Loss (MSE)")
    plt.xticks(np.arange(0, len(typeInputs), 1), typeInputs)
    plt.title("Mean Squared Error for different types of inputs")
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison2.png", dpi = 100)
    plt.show()

def compareNumSamples(save_study_path, typeSamples, repetitions):
    loss = np.zeros([len(typeSamples), repetitions])
    val_loss = np.zeros([len(typeSamples),  repetitions])

    
    for i in range(len(typeSamples)):
        base = "./databaseANN/DatabaseFitness/deltakeplerian/"   
        save_file_path = base + str(typeSamples[i]) +"/"

        train_file_path = save_file_path + 'Together.txt'

        # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
        dataset_np = TD.LoadNumpy(train_file_path,
                scaling = Scaling['scaling'],\
                dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
                outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
                output_type = Dataset_conf.Dataset_config['Outputs'],
                plotDistribution=False, plotErrors=False, labelType = 2)

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

    # Using errorbars
    mean_loss = np.zeros((len(typeSamples),2)) # 1 column for loss and 1 for val_loss
    std_loss = np.zeros((len(typeSamples),2)) # 1 column for loss and 1 for val_loss
    for i in range(len(typeSamples)):
        mean_loss[i, 0] = np.mean(loss[i,:])
        std_loss[i, 0] = np.std(loss[i,:])
        mean_loss[i, 1] = np.mean(val_loss[i,:])
        std_loss[i, 1] = np.std(val_loss[i,:])

    plt.errorbar(np.arange(0,len(typSampless),1), mean_loss[:,0], std_loss[:,0], linestyle='None', marker='^')
    plt.errorbar(np.arange(0,len(typeSamples),1), mean_loss[:,1], std_loss[:,1], linestyle='None', marker='o')

    plt.ylabel("Loss (MSE)")
    plt.xticks(np.arange(0, len(typeSamples), 1), typeSamples)
    plt.title("Mean Squared Error for different number of samples")
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison2.png", dpi = 100)
    plt.show()

def compareNumSamples_val(base, save_study_path, typeSamples, repetitions):
    loss = np.zeros([len(typeSamples), repetitions])
    val_loss = np.zeros([len(typeSamples),  repetitions])

    for i in range(len(typeSamples)):
        # base = "./databaseANN/DatabaseFitness/deltakeplerian/ComparisonNInputs_files/" 
        train_file_path = base + "Random_MBH_" +typeSamples[i] +".txt"

         
        # train_file_path = base +typeSamples[i] +".txt"

        # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
        dataset_np = TD.LoadNumpy(train_file_path,
                scaling = Scaling['scaling'],\
                dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
                outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
                output_type = Dataset_conf.Dataset_config['Outputs'],
                plotDistribution=False, plotErrors=False, labelType = 0)

        traindata, testdata = TD.splitData_reg(dataset_np)
        
        for k in range(repetitions):
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
    
def plot_compareNumSamples_val(path, typeSamples, repetitions):
    loss = np.loadtxt(path+"loss.txt")
    val_loss = np.loadtxt(path+"val_loss.txt")

    color = ['black', 'blue', 'green', 'orange', 'yellow', 'purple', 'pink']

    fig = plt.figure(figsize = (13,7))
    ax = fig.add_subplot(1,2,1)

    for i in range(len(typeSamples)):
        plt.plot( np.arange(0,repetitions, 1), loss[i, :], marker = 'o', color = color[i], label = typeSamples[i] )
    
    plt.xticks(np.arange(0, repetitions, 1))
    plt.title("Mean Squared Error for different number of samples")
    plt.xlabel('Repetitions')
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    # plt.savefig(path+"comparison.png", dpi = 100)
    # plt.show()

    ax = fig.add_subplot(1,2,2)
    for i in range(len(typeSamples)):
        plt.plot( np.arange(0,repetitions, 1), val_loss[i, :], marker = 'x', color = color[i] )

    plt.xticks(np.arange(0, repetitions, 1))
    plt.title("Mean Squared Error for different number of samples")
    plt.xlabel('Repetitions')
    plt.ylabel("Validatoin Loss (MSE)")
    
    # plt.legend()
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison.png", dpi = 100)
    plt.show()


    # Using errorbars
    mean_loss = np.zeros((len(typeSamples),2)) # 1 column for loss and 1 for val_loss
    std_loss = np.zeros((len(typeSamples),2)) # 1 column for loss and 1 for val_loss
    for i in range(len(typeSamples)):
        mean_loss[i, 0] = np.mean(loss[i,:])
        std_loss[i, 0] = np.std(loss[i,:])
        mean_loss[i, 1] = np.mean(val_loss[i,:])
        std_loss[i, 1] = np.std(val_loss[i,:])

    plt.errorbar(np.arange(0,len(typeSamples),1), mean_loss[:,0], std_loss[:,0], linestyle='None', marker='^', label = 'Train loss')
    plt.errorbar(np.arange(0,len(typeSamples),1), mean_loss[:,1], std_loss[:,1], linestyle='None', marker='o', label = 'Validation loss')
    plt.yscale('log')
    plt.legend()
    plt.ylabel("Loss (MSE)")
    plt.xticks(np.arange(0, len(typeSamples), 1), typeSamples, rotation = 70)
    plt.title("Mean Squared Error for different number of samples")
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison2.png", dpi = 100)
    plt.show()

def compareDataAugm(base, save_study_path, typeSamples, repetitions):
    loss = np.zeros([len(typeSamples), repetitions])
    val_loss = np.zeros([len(typeSamples),  repetitions])

    train_file_path = base + "Together.txt"

    for i in range(len(typeSamples)):
        dataset_np = TD.LoadNumpy(train_file_path, save_file_path=base,
                scaling = Scaling['scaling'],\
                dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
                outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
                output_type = Dataset_conf.Dataset_config['Outputs'],
                plotDistribution=False, plotErrors=False, labelType = 6,
                data_augmentation=typeSamples[i])

        traindata, testdata = TD.splitData_reg(dataset_np)
        
        for k in range(repetitions):
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

def plot_compareDataAugm(path, typeSamples, repetitions):
    loss = np.loadtxt(path+"loss.txt")
    val_loss = np.loadtxt(path+"val_loss.txt")

    color = ['black', 'blue', 'green', 'orange', 'yellow']
    for i in range(len(typeSamples)):
        plt.plot( np.arange(0,repetitions, 1), loss[i, :], marker = 'o', color = color[i], label = typeSamples[i] )

    for i in range(len(typeSamples)):
        plt.plot( np.arange(0,repetitions, 1), val_loss[i, :], marker = 'x', color = color[i], label = typeSamples[i] )
    
    plt.xticks(np.arange(0, repetitions, 1))
    plt.title("Mean Squared Error for different number of samples")
    plt.xlabel('Repetitions')
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison.png", dpi = 100)
    plt.show()

    # Using errorbars
    mean_loss = np.zeros((len(typeSamples),2)) # 1 column for loss and 1 for val_loss
    std_loss = np.zeros((len(typeSamples),2)) # 1 column for loss and 1 for val_loss
    for i in range(len(typeSamples)):
        mean_loss[i, 0] = np.mean(loss[i,:])
        std_loss[i, 0] = np.std(loss[i,:])
        mean_loss[i, 1] = np.mean(val_loss[i,:])
        std_loss[i, 1] = np.std(val_loss[i,:])

    plt.errorbar(np.arange(0,len(typeSamples),1), mean_loss[:,0], std_loss[:,0], linestyle='None', marker='^', label = 'Train loss')
    plt.errorbar(np.arange(0,len(typeSamples),1), mean_loss[:,1], std_loss[:,1], linestyle='None', marker='o', label = 'Validation loss')
    plt.yscale('log')
    plt.legend()
    plt.ylabel("Loss (MSE)")
    plt.xticks(np.arange(0, len(typeSamples), 1), typeSamples)
    plt.title("Mean Squared Error for different number of samples")
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison2.png", dpi = 100)
    plt.show()

def comparetypeLog(save_study_path, typeLog, repetitions):
    loss = np.zeros([len(typeLog), repetitions])
    val_loss = np.zeros([len(typeLog),  repetitions])

    base = "./databaseANN/DatabaseFitness/deltakeplerian/500"   
    save_file_path = base +"/"

    train_file_path = save_file_path + 'Together.txt'

    for i in range(len(typeLog)):
        if typeLog[i] == 'Log':
            logtype = True
        else:
            logtype = False

        # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
        dataset_np = TD.LoadNumpy(train_file_path,
                scaling = Scaling['scaling'],\
                dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = logtype,\
                outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
                output_type = Dataset_conf.Dataset_config['Outputs'],
                plotDistribution=False, plotErrors=False, labelType = 4)

        traindata, testdata = TD.splitData_reg(dataset_np)

        for k in range(repetitions):
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

def plot_comparetypeLog(path, typeLog, repetitions):
    loss = np.loadtxt(path+"loss.txt")
    val_loss = np.loadtxt(path+"val_loss.txt")

    color = ['black', 'blue', 'green', 'orange', 'yellow']

    for i in range(len(typeLog)):
        plt.plot( np.arange(0,repetitions, 1), loss[i, :], marker = 'o', color = color[i], label = typeLog[i] )
    
    plt.xticks(np.arange(0, repetitions, 1))
    plt.title("Mean Squared Error when applying log() to the errors")
    plt.xlabel('Repetitions')
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison.png", dpi = 100)
    plt.show()

    # Using errorbars
    mean_loss = np.zeros((len(typeLog),2)) # 1 column for loss and 1 for val_loss
    std_loss = np.zeros((len(typeLog),2)) # 1 column for loss and 1 for val_loss
    for i in range(len(typeLog)):
        mean_loss[i, 0] = np.mean(loss[i,:])
        std_loss[i, 0] = np.std(loss[i,:])
        mean_loss[i, 1] = np.mean(val_loss[i,:])
        std_loss[i, 1] = np.std(val_loss[i,:])

    plt.errorbar(np.arange(0,len(typeLog),1), mean_loss[:,0], std_loss[:,0], linestyle='None', marker='^')
    plt.errorbar(np.arange(0,len(typeLog),1), mean_loss[:,1], std_loss[:,1], linestyle='None', marker='o')

    plt.ylabel("Loss (MSE)")
    plt.xticks(np.arange(0, len(typeLog), 1), typeLog)
    plt.title("Mean Squared Error when applying log() to the errors")
    plt.grid(alpha = 0.5)
    plt.tight_layout()
    plt.savefig(path+"comparison2.png", dpi = 100)
    plt.show()

def compareObjFunctions():
    base = "./databaseANN/DatabaseFitness/deltakeplerian/ComparisonObjFunction/"
    file_path = [base+ 'fp1/Random_MBH.txt',
                base+ 'fp2/Random_MBH.txt',
                base+ 'fp5/Random_MBH.txt',
                base+ '500/Random_MBH.txt',
                base+ 'fp50/Random_MBH.txt',
                base+ 'fp100/Random_MBH.txt']
    
    file_path_together = base +'ComparisonObjFunction.txt'
    TD.join_files(file_path, file_path_together)


    dataset_np = TD.LoadNumpy(file_path_together, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            labelType = len(file_path),
            plotDistribution=False, plotErrors=False,
            # plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            plotOutputDistr = False, plotEpvsEv = True,
            data_augmentation = 'False')

if __name__ == "__main__":
    # COMPARE STANDARDIZATION
    values_type_stand = [0, 1, 2]
    values_scaling = [0, 1]
    repetitions = 5 # repetitions of each setting
    save_study_path =  "./Results/StudyStandardization/Results"
    # path = compareStandard(save_study_path, values_type_stand, values_scaling, repetitions)
    # plot_compareStandard(save_study_path, values_type_stand, values_scaling, repetitions)

    # COMPARE INPUTS
    repetitions = 5 # repetitions of each setting
    typeInputs = ['cartesian', 'deltakeplerian', 'deltakeplerian_planet']
    save_study_path =  "./Results/StudyInputs/"
    # compareInputs(repetitions, typeInputs, save_study_path)
    # plot_compareInputs(save_study_path, repetitions, typeInputs)
    

    # COMPARE NUMBER SAMPLES: LEARNING CURVE
    repetitions = 3
    typeSamples = [500, 1000, 5000]
    save_study_path =  "./Results/StudyNSamples/DifferentFiles/"
    # compareNumSamples(save_study_path, typeSamples, repetitions)
    # plot_compareNumSamples(save_study_path, typeSamples, repetitions)

    # COMPARE NUMBER SAMPLES: LEARNING CURVE. Train and val loss
    repetitions = 5
    base = "./databaseANN/3_DatabaseLast/deltakeplerian/"
    database = base +'databaseSaved_fp100/'
    save_study_path = base + 'Results/StudyNSamples/'
    typeSamples = ['500', '1000','5000']
    # typeSamples = ['500_opt', '1000_opt', '1000_eval_1000_opt', '5000_opt', '5000_1000_200', '5000_5000_1000_mixed']
    # compareNumSamples_val(database, save_study_path, typeSamples, repetitions)
    # plot_compareNumSamples_val(save_study_path, typeSamples, repetitions)


    # DATA AUGMENTATION
    repetitions = 3
    typeAugm = ['False', 'multiplication', 'noise_gauss']
    base = "./databaseANN/3_DatabaseLast/deltakeplerian/"
    database = base  
    save_study_path =  base+  "Results/StudyDataAugmentation/"
    compareDataAugm(database, save_study_path, typeAugm, repetitions)
    plot_compareDataAugm(save_study_path, typeAugm, repetitions)


    # COMPARE EFFECT OF LOG IN ERRORS
    repetitions = 5
    typeLog = ['Without log', 'Log']
    save_study_path =  "./Results/StudytypeLog/"
    # comparetypeLog(save_study_path, typeLog, repetitions)
    # plot_comparetypeLog(save_study_path, typeLog, repetitions)

    # COMPARE OBJECTIVE FUNCTIONS
    # compareObjFunctions()