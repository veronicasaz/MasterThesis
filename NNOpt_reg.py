import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
from math import floor, ceil

import time

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.model_selection import RepeatedKFold

import LoadConfigFiles as CONF
import TrainingData as DTS
from AstroLibraries import AstroLib_OPT as AL_OPT

ANN_reg = CONF.ANN_reg()
ANN = ANN_reg.ANN_config

Dataset_conf = CONF.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']

class ANN_reg:
    def __init__(self, dataset):
        self.dataset_np = dataset

        self.n_classes = int(dataset.n_outputs) # Outputs
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

    def create_model(self, dv):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()

        # dont apply regularization
        # if ANN['Training']['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
        #     for layer in range(dv[0]):
        #         model.add(keras.layers.Dense(
        #             dv[1], 
        #             activation='relu', 
        #             use_bias=True, bias_initializer='zeros',
        #             kernel_initializer = initializer,
        #             kernel_regularizer= keras.regularizers.l2(ANN['Architecture']['regularizer_value']) ))
        # else:
        for layer in range(int(dv[0])):
            model.add(keras.layers.Dense(
                int(dv[1]),
                activation='relu', 
                use_bias=True, bias_initializer='zeros',
                kernel_initializer = initializer) )

        model.add(keras.layers.Dense(self.n_classes) ) # output layer
       
        # Compile
        self.loss = 'mse'
        opt = keras.optimizers.Adam(learning_rate=dv[3])
        model.compile(loss=self.loss, optimizer =opt)

        return model
            
    def get_traintestdata(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def training(self, dv):
        self.model = self.create_model(dv)

        self.history = self.model.fit(self.traindata[0], self.traindata[1], 
                    validation_split= ANN['Training']['validation_size'],
                    epochs =int( dv[2] ) )
        
        return self.history.history['loss'], self.history.history['val_loss']


def optComplete(base, perceptron, bounds, nind):
    solution = np.zeros((nind, len(bounds) ))

    FileName1 = base+"Results/Study_RegParams/regparams.txt"

    def f(inputs):
        # Study architecture
        start_time = time.time()
        train_accuracy, val_accuracy = perceptron.training(inputs)
        t = (time.time() - start_time)

        fobj = train_accuracy + (val_accuracy - train_accuracy)
        
        vector = np.column_stack((fobj, train_accuracy))
        vector = np.column_stack((vector, val_accuracy))
        vector = np.column_stack((vector, t))
        vector = np.column_stack((vector, inputs.reshape((1,4))))
        vector = vector.flatten()

        with open(FileName1, "a") as myfile:
            for value in vector:
                if value != vector[-1]:
                    myfile.write(str(value) +" ")
                else:
                    myfile.write(str(value))
                    myfile.write("\n")
        myfile.close()

        return fobj

    # Select which inputs are forced to be integers
    int_input = np.ones(len(bounds))
    int_input[-1] = 0

    f_min, Best = AL_OPT.EvolAlgorithm_integerinput(f, bounds , x_add = False, \
        ind = 10, 
        max_iter = 3,
        max_iter_success = 3,
        elitism = 0, 
        mutation = 0.001, 
        immig = 0,
        bulk_fitness = False,
        int_input = int_input)

    print("Min", f_min,'Inputs', Best) 

def plot_complete(base, dv):
    results = np.loadtxt(base +"Results/Study_RegParams/regparams.txt")
    fitness = results[:,0]
    t_comp = results[:,3]

    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']
    
    fig = plt.figure(figsize=(12,12), constrained_layout=True)
    gs = fig.add_gridspec(1,1)

    fig.suptitle("Optimization \n Regression Network", fontsize=16)
    
    ax = fig.add_subplot(gs[0,:])
    plt.scatter(t_comp, fitness)
    plt.xlabel('Computation time (s')
    plt.ylabel('Fitness')
    plt.grid()

    # plt.tight_layout()

    plt.savefig(base+"Results/Study_RegParams/RegNet_paramOpt.png", dpi = 100)
    plt.show()

def optArch(perceptron, dv_HL, dv_NH):
    # Fix training values to study architecture
    dv2 = 500
    dv3 = 0.001

    test_accuracy_Matrix_arch_train = np.zeros((len(dv_HL), len(dv_NH)))
    test_accuracy_Matrix_arch_val = np.zeros((len(dv_HL), len(dv_NH)))
    history_arch_train = np.zeros((len(dv_HL), len(dv_NH), dv2) )
    history_arch_val = np.zeros((len(dv_HL), len(dv_NH), dv2) )

    # Study architecture
    for i, dv0 in enumerate(dv_HL):
        for j, dv1 in enumerate(dv_NH):
            print("OPT arch", dv0, dv1)
            dv = [dv0, dv1, dv2, dv3]
            test_accuracy, test_accuracy_val = perceptron.training(dv)
            
            test_accuracy_Matrix_arch_train[i, j] = test_accuracy[-1]
            test_accuracy_Matrix_arch_val[i, j] = test_accuracy_val[-1]
            history_arch_train[i, j] = test_accuracy
            history_arch_val[i, j] = test_accuracy_val

    FileName1 = "./Results/Study_RegParams/arch_test.txt"
    FileName2 = "./Results/Study_RegParams/arch_val.txt"
    FileName3 = "./Results/Study_RegParams/history_train.npy"
    FileName4 = "./Results/Study_RegParams/history_val.npy"

    mat = np.matrix(test_accuracy_Matrix_arch_train)
    mat2 = np.matrix(test_accuracy_Matrix_arch_val)

    with open(FileName1, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.6f')  
    f.close()

    with open(FileName2, "wb") as f:
        for line in mat2:  
            np.savetxt(f, line, fmt='%.6f')  
    f.close()

    np.save(FileName3, history_arch_train)
    np.save(FileName4, history_arch_val)
   

def optTra(perceptron, dv_EP):
    
    dv0 = 5
    dv1 = 100
    dv3 = 0.001

    test_accuracy_Matrix_train = np.zeros((len(dv_EP), 3 ))
    history_test_train = np.zeros((len(dv_EP), 2, dv_EP[-1]))
    
    for k, dv2 in enumerate(dv_EP):
        dv = [dv0, dv1, dv2, dv3]
        t0 = time.time()
        test_accuracy, test_accuracy_val = perceptron.training(dv)
        tf = (time.time() - t0) 
        print('Time network eval', tf)
        test_accuracy_Matrix_train[k, 0] = test_accuracy[-1]
        test_accuracy_Matrix_train[k, 1] = test_accuracy_val[-1]
        test_accuracy_Matrix_train[k, 2] = tf

        history_test_train[k, 0, 0:dv_EP[k]] = test_accuracy
        history_test_train[k, 1, 0:dv_EP[k]] = test_accuracy_val
                
    FileName1 =  "./Results/Study_RegParams/train_test.txt"
    FileName2 =  "./Results/Study_RegParams/history_train_test.npy"

    mat = np.matrix(test_accuracy_Matrix_train) # cannot save 3d matrix

    with open(FileName1, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.6f')  
    f.close()

    np.save(FileName2, history_test_train)

def loadData(): 
    FileName1 = "./Results/Study_RegParams/arch_test.txt"
    FileName2 = "./Results/Study_RegParams/arch_val.txt"
    FileName3 = "./Results/Study_RegParams/history_train.npy"
    FileName4 = "./Results/Study_RegParams/history_val.npy"

    arch_train = np.loadtxt(FileName1)
    arch_test = np.loadtxt(FileName2)
    arch_hist_train = np.load(FileName3)
    arch_hist_val = np.load(FileName4)

    FileName1 =  "./Results/Study_RegParams/train_test.txt"
    FileName2 =  "./Results/Study_RegParams/history_train_test.npy"
    
    train_train = np.loadtxt(FileName1)
    train_hist = np.load(FileName2)

    return  [arch_train, arch_test, arch_hist_train, arch_hist_val], \
        [train_train, train_hist]

def plot(dv_HL, dv_NH, dv_EP):
    arch, train = loadData()
    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow', 'red', 'green', 'blue', 'black', 'orange', 'yellow']
    markers = ['x', 'o', '^', 'x', 'o', '^', 'x', 'o', '^', 'x', 'o', '^']
    linetype = ['--','--','--', '-.', '-.', '-.', ':', ':', ':' ]

    fig = plt.figure(figsize=(12,12))
    gs = fig.add_gridspec(2,2)

    fig.suptitle("Sweep parameter \n Regression Network", fontsize=16)
    
    ax = fig.add_subplot(gs[0,0])
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, arch[0][i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training MSE for training dataset')
    plt.grid()
    plt.yscale('log')
    plt.legend(title = "Hidden layers")

    ax = fig.add_subplot(gs[0,1])
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, arch[1][i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training MSE for validation dataset')
    plt.grid()
    plt.yscale('log')
    plt.legend(title = "Epoch")

    ax = fig.add_subplot(gs[1,0])
    for i in range(len(dv_HL)):
        for j in range(len(dv_NH)):
            x_axis = np.arange(0,len(arch[2][0,0,:]),1)
            plt.plot(x_axis, arch[2][i, j, :], ls = linetype[j], marker = markers[j], c = color[i], label = '%i, %i'%(dv_HL[i], dv_NH[j]) )
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training MSE for training dataset')
    plt.yscale('log')
    plt.grid()
    plt.legend(title = "Epoch")

    ax = fig.add_subplot(gs[1,1])
    for i in range(len(dv_HL)):
        for j in range(len(dv_NH)):
            x_axis = np.arange(0,len(arch[2][0,0,:]),1)
            plt.plot(x_axis, arch[3][i, j, :], ls = linetype[j],marker = markers[j], c = color[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training MSE for validation dataset')
    plt.yscale('log')
    plt.grid()
    plt.legend(title = "Hidden layers // Neurons per layer")

    plt.savefig("./Results/Study_RegParams/RegNet_paramOpt_arch.png", dpi = 100)
    plt.show()

    ###########################################################
    fig = plt.figure(figsize=(12,12), constrained_layout=True)
    gs = fig.add_gridspec(2,2)

    fig.suptitle("Sweep parameter \n Regression Network", fontsize=16)
    
    ax = fig.add_subplot(gs[0,0])
    plt.plot(dv_EP, train[0][:,0], 'x-', c = color[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Training MSE for training dataset')
    plt.grid()

    ax = fig.add_subplot(gs[1,0])
    plt.plot(dv_EP, train[0][:,1], 'x-', c = color[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Training MSE for validation dataset')
    plt.grid()

    ax = fig.add_subplot(gs[0:2,1])
    plt.plot(dv_EP, train[0][:,2], 'x-', c = color[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Computation time (s)')
    plt.grid()

    # ax = fig.add_subplot(gs[0,0:2])
    # for i in range(len(dv_EP)):
    #     x_axis = np.arange(0, dv_EP[i], 1)
    #     plt.plot(x_axis, train[0][:,2], 'x-', c = color[i])
    # plt.xlabel('Training epochs')
    # plt.ylabel('Computation time (s)')
    # plt.grid()

    # plt.tight_layout()

    plt.savefig("./Results/Study_RegParams/RegNet_paramOpt_train.png", dpi = 100)
    plt.show()

if __name__ == "__main__":
    # Choose which ones to choose:
    base = "./databaseANN/3_DatabaseLast/deltakeplerian/"
 
    train_file_path = base +'Together.txt'

    dataset_np = DTS.LoadNumpy(train_file_path, save_file_path = base,\
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            labelType = 3, 
            plotDistribution=False, plotErrors=False,
            plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            # plotOutputDistr = True, plotEpvsEv = True,
            data_augmentation =  Dataset_conf.Dataset_config['dataAugmentation']['type'])

    # dataset_np = DTS.LoadNumpy(train_file_path)
    traindata, testdata = DTS.splitData_reg(dataset_np)
    perceptron = ANN_reg(dataset_np)
    perceptron.get_traintestdata(traindata, testdata)

    # hidden_layers, neuron_hidden, n_spilts, n_repeats
    dv_HL = [5, 8, 15, 25]
    dv_NH = [50, 80, 100, 250, 400, 600]
    dv_EP = [10, 30, 50, 100, 250, 500]

    # dv_HL = [1, 2]
    # dv_NH = [2, 3]
    # dv_EP = [10, 20]

    # ACCURACY
    # optArch(perceptron,  dv_HL, dv_NH)
    # optTra(perceptron, dv_EP)
    
    plot(dv_HL, dv_NH, dv_EP)


    # OPTIMIZATION
    dv_HL = [1, 15]
    dv_NH = [3, 500]
    dv_EP = [10, 500]
    dv_LR = [0.0001, 0.01]

    # bounds = [dv_HL, dv_NH, dv_EP, dv_LR]
    # nind = 10
    # optComplete(base, perceptron, bounds, nind )
    # plot_complete(base, bounds)


    