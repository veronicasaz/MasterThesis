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
import TrainingDataKeras as DTS

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
        for layer in range(dv[0]):
            model.add(keras.layers.Dense(
                dv[1],
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
                    epochs = dv[2] )
        
        return self.history.history['loss'][-1], self.history.history['val_loss'][-1]


def optComplete(base, perceptron, bounds):
    solution = np.zeros((nind, len(bounds) ))

    FileName1 = base+"Results/Study_RegParams/regparams.txt"

    def f(inputs):
        # Study architecture
        start_time = time.time()
        train_accuracy, val_accuracy = perceptron.training(inputs)
        t = (time.time() - start_time)

        fobj = train_accuracy + abs(train_accuracy - val_accuracy)/train_accuracy
        
        vector = np.column_stack((fobj, train_accuracy))
        vector = np.column_stack((vector, val_accuracy))
        vector = np.column_stack((vector, t))
        vector = np.column_stack((vector, inputs))

        with open(FileName1, "a") as f:
            for value in vector:
                    if value != vector[-1]:
                        myfile.write(str(value) +" ")
                    else:
                        myfile.write(str(value))
                myfile.write("\n")
        f.close()

        return fobj

    # Select which inputs are forced to be integers
    int_input = np.ones(len(bounds))
    int_input[-1] = 0

    f_min, Best = AL_OPT.EvolAlgorithm_integerinput(f, bounds , x_add = False, \
        ind = 10, 
        max_iter = 10,
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
    dv2 = 100

    test_accuracy_Matrix_arch = np.zeros((len(dv_HL), len(dv_NH)))

    # Study architecture
    for i, dv0 in enumerate(dv_HL):
        for j, dv1 in enumerate(dv_NH):
            print("OPT arch", dv0, dv1)
            dv = [dv0, dv1, dv2]
            test_accuracy = perceptron.training(dv)
            test_accuracy_Matrix_arch[i, j] = test_accuracy

    FileName1 = "./Results/Study_RegParams/arch_test.txt"

    mat = np.matrix(test_accuracy_Matrix_arch)

    with open(FileName1, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
   

def optTra(perceptron, dv_EP):
    test_accuracy_Matrix_train = np.zeros((len(dv_EP), 2 ))
    dv0 = 5
    dv1 = 40

    for k, dv2 in enumerate(dv_EP):
        dv = [dv0, dv1, dv2]
        t0 = time.time()
        test_accuracy = perceptron.training(dv)
        tf = (time.time() - t0) 
        print('Time network eval', tf)
        test_accuracy_Matrix_train[k, 0] = test_accuracy
        test_accuracy_Matrix_train[k, 1] = tf
                
    FileName2 =  "./Results/Study_RegParams/train_test.txt"
    mat = np.matrix(test_accuracy_Matrix_train) # cannot save 3d matrix

    with open(FileName2, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()

def loadData(): 
    
    test_acc_arch = np.loadtxt("./Results/Study_RegParams/arch_test.txt")
    test_acc_train = np.loadtxt("./Results/Study_RegParams/train_test.txt")

    return  test_acc_arch, test_acc_train

def plot(dv_HL, dv_NH, dv_EP):
    te1, te2= loadData()
    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']
    
    fig = plt.figure(figsize=(12,12), constrained_layout=True)
    gs = fig.add_gridspec(3,1)

    fig.suptitle("Sweep parameter \n Regression Network", fontsize=16)
    
    ax = fig.add_subplot(gs[0,:])
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, te1[i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training MAE')
    plt.grid()
    plt.legend(title = "Hidden layers")

    ax = fig.add_subplot(gs[1,:])
    plt.plot(dv_EP, te2[:,0], 'x-', c = color[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Training MAE')
    plt.grid()

    ax = fig.add_subplot(gs[2,:])
    plt.plot(dv_EP, te2[:,1], 'x-', c = color[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Computation time (s)')
    plt.grid()

    # plt.tight_layout()

    plt.savefig("./Results/Study_RegParams/RegNet_paramOpt.png", dpi = 100)
    plt.show()

if __name__ == "__main__":
    # Choose which ones to choose:
    base = "./databaseANN/DatabaseBestDeltaV/deltakeplerian/"
 
    train_file_path = base +'databaseSaved_fp100/Random_MBH_5000_3.txt'

    dataset_np = DTS.LoadNumpy(train_file_path, save_file_path = base, error='vector',\
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            labelType = 0, 
            plotDistribution=False, plotErrors=False,
            # plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            plotOutputDistr = True, plotEpvsEv = True,
            data_augmentation = 'False')

    # dataset_np = DTS.LoadNumpy(train_file_path)
    traindata, testdata = DTS.splitData_reg(dataset_np)
    perceptron = ANN_reg(dataset_np)
    perceptron.get_traintestdata(traindata, testdata)

    # hidden_layers, neuron_hidden, n_spilts, n_repeats
    dv_HL = [2, 5, 8, 15]
    dv_NH = [3, 5, 10, 20, 50, 80, 100, 250, 500]
    dv_EP = [10, 30, 50, 100, 250, 500]

    # ACCURACY
    # optArch(perceptron,  dv_HL, dv_NH)
    # optTra(perceptron, dv_EP)
    
    # plot(dv_HL, dv_NH, dv_EP)


    # OPTIMIZATION
    dv_HL = [1, 15]
    dv_NH = [3, 500]
    dv_EP = [10, 500]
    dv_LR = [0.0001, 0.01]

    bounds = [dv_HL, dv_NH, dv_EP]
    optComplete(base, perceptron, bounds )
    plot_complete(base, bounds)


    