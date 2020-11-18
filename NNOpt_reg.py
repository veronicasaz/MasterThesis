import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.model_selection import RepeatedKFold

import LoadConfigFiles as CONF
import TrainingDataKeras as DTS

ANN = CONF.ANN_reg()
ANN_train = ANN.ANN_train
ANN_archic = ANN.ANN_archic


class ANN_reg:
    def __init__(self, dataset):
        self.dataset_np = dataset

        self.n_classes = self.dataset_np.n_classes # Labels
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

    def create_model(self, dv):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()

        if ANN_train['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            for layer in range(dv[0]):
                model.add(keras.layers.Dense(
                    dv[1], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN_archic['regularizer_value']) ))
        else:
            for layer in range(dv[0]):
                model.add(keras.layers.Dense(
                    dv[1],
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer) )

        model.add(keras.layers.Dense(self.n_classes) ) # output layer
       
        # Compile
        model.compile(loss='mae', optimizer ='adam')

        return model
            
    def get_traintestdata(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def training(self, dv):
        # Create a callback that saves the model's weights
        X, y = self.traindata
        
        results = list()

        # define evaluation procedure
        cv = RepeatedKFold(n_splits=dv[2], 
                        n_repeats=dv[3], 
                        random_state=ANN_train['random_state'])

        # enumerate folds
        for train_ix, test_ix in cv.split(X):
            # prepare data
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            # define model
            self.model = self.create_model(dv)
            # fit model
            self.model.fit(X_train, y_train, verbose=2, epochs=dv[4]  )
            # evaluate model on test set: mean absolute error
            mae = self.model.evaluate(X_test, y_test, verbose=0)
            # store result
            results.append(mae)
        # return results

        print('MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))

        return np.mean(results)


def optArch(perceptron, dv_HL, dv_NH):
    # Fix training values to study architecture
    dv2 = 5
    dv3 = 2
    dv4 = 50

    test_accuracy_Matrix_arch = np.zeros((len(dv_HL), len(dv_NH)))

    # Study architecture
    for i, dv0 in enumerate(dv_HL):
        for j, dv1 in enumerate(dv_NH):
            print("OPT arch", dv0, dv1)
            dv = [dv0, dv1, dv2, dv3, dv4]
            test_accuracy = perceptron.training(dv)
            test_accuracy_Matrix_arch[i, j] = test_accuracy

    FileName1 = "./Results/TrainingPopulation/NNoptimization_reg_arch_test.txt"

    mat = np.matrix(test_accuracy_Matrix_arch)

    with open(FileName1, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
   

def optTra(perceptron, dv_SP, dv_RP, dv_EP):
    test_accuracy_Matrix_train = np.zeros((len(dv_SP), len(dv_RP)))
    test_accuracy_Matrix_train2 = np.zeros((len(dv_EP)))
    dv0 = 5
    dv1 = 40
    dv4 = 100
    for i, dv2 in enumerate(dv_SP):
        for j, dv3 in enumerate(dv_RP):
            print("OPT train", dv2, dv3)
            dv = [dv0, dv1, dv2, dv3, dv4]
            test_accuracy = perceptron.training(dv)
            test_accuracy_Matrix_train[i, j] = test_accuracy

    dv2 = 5
    dv3 = 2
    for k, dv4 in enumerate(dv_EP):
        dv = [dv0, dv1, dv2, dv3, dv4]
        test_accuracy = perceptron.training(dv)
        test_accuracy_Matrix_train2[k] = test_accuracy
                
    FileName2 =  "./Results/TrainingPopulation/NNoptimization_reg_train_test.txt"
    FileName3 =  "./Results/TrainingPopulation/NNoptimization_reg_train_test2.txt"
    mat = np.matrix(test_accuracy_Matrix_train) # cannot save 3d matrix
    mat2 = np.matrix(test_accuracy_Matrix_train2)

    with open(FileName2, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
    with open(FileName3, "wb") as f:
        for line in mat2:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()



def loadData(): 
    
    test_acc_arch = np.loadtxt("./Results/TrainingPopulation/OptRegression/NNoptimization_reg_arch_test.txt")
    test_acc_train = np.loadtxt("./Results/TrainingPopulation/OptRegression/NNoptimization_reg_train_test.txt")
    test_acc_train2 = np.loadtxt("./Results/TrainingPopulation/OptRegression/NNoptimization_reg_train_test2.txt")

    return  test_acc_arch, test_acc_train, test_acc_train2

def plot(dv_HL, dv_NH, dv_SP, dv_RP, dv_EP):
    te1, te2, te22 = loadData()
    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']
    
    fig = plt.figure(figsize=(12,12))
    fig.suptitle("Sweep parameter \n Regression Network", fontsize=16)
    
    ax = fig.add_subplot(3,1, 1)
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, te1[i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training MAE')
    plt.grid()
    plt.legend(title = "Hidden layers")

    ax = fig.add_subplot(3,1,2)
    for i in range(len(dv_SP)):
        plt.plot(dv_RP, te2[i, :], 'x-', c = color[i], label = dv_SP[i])
    plt.xlabel('Training repeats')
    plt.ylabel('Training MAE')
    plt.grid()
    plt.legend(title = "Training splits")

    ax = fig.add_subplot(3,1,3)
    plt.plot(dv_EP, te22, 'x-', c = color[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Training MAE')
    plt.grid()

    # plt.tight_layout()

    plt.savefig("./Results/TrainingPopulation/OptRegression/RegNet_paramOpt.png", dpi = 100)
    plt.show()

if __name__ == "__main__":
    train_file_path = "./Results/TrainingPopulation/OptRegression/trainingData_Feas_big2.txt"

    dataset_np = DTS.LoadNumpy(train_file_path, error= True,\
            equalize = True,
            plotDistribution=False, plotErrors=False)
    # dataset_np = DTS.LoadNumpy(train_file_path)
    traindata, testdata = DTS.splitData_reg(dataset_np)
    perceptron = ANN_reg(dataset_np)
    perceptron.get_traintestdata(traindata, testdata)

    # hidden_layers, neuron_hidden, n_spilts, n_repeats
    dv_HL = [2, 5, 8]
    dv_NH = [3, 5, 10, 20, 50, 80, 100]
    dv_SP = [2, 4, 5, 8, 15]
    dv_RP = [2, 3, 5, 8 ]
    dv_EP = [10, 20, 30, 50, 100]


    # ACCURACY
    optArch(perceptron,  dv_HL, dv_NH)
    optTra(perceptron, dv_SP, dv_RP, dv_EP)
    
    plot(dv_HL, dv_NH, dv_SP, dv_RP, dv_EP)


    