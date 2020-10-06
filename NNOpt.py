import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

import LoadConfigFiles as CONF
import TrainingDataKeras as DTS

train_file_path = "./databaseANN/trainingData_Feas.txt"

ANN = CONF.ANN()
ANN_train = ANN.ANN_train
ANN_archic = ANN.ANN_archic

class ANN:
    def __init__(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

        self.n_input = traindata[0].shape[1] #inputs
        self.n_classes = 2 # Labels
        self.n_examples = traindata[0].shape[0] # samples

    def create_model(self, dv):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()
        for layer in range(dv[0]):
            model.add(keras.layers.Dense(
                        dv[1], 
                        activation='relu', 
                        use_bias=True, bias_initializer='zeros',
                        kernel_initializer = initializer) )
        model.add(keras.layers.Dense(self.n_classes) ) # output layer
       
        # Compile
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model
            
    def training(self, dv):

        # Create model architecture
        self.model = self.create_model(dv)
        
        # Train
        self.history = self.model.fit(self.traindata[0], self.traindata[1], 
                    validation_split= ANN_train['validation_size'],
                    epochs = dv[2], 
                    batch_size = dv[3] )

    def retrieveAccuracy(self):
        train_loss = self.history.history['loss']
        train_accuracy = self.history.history['accuracy']

        val_loss = self.history.history['val_loss']
        val_accuracy = self.history.history['val_accuracy']

        test_loss, test_accuracy = self.model.evaluate(self.testdata[0], self.testdata[1], verbose=2)

        return train_accuracy, val_accuracy, test_accuracy

def optArch(traindata, testdata, dv_HL, dv_NH):

    perceptron = ANN(traindata, testdata)

    # Fix training values to study architecture
    dv2 = 50
    dv3 = 30

    train_accuracy_Matrix_arch = np.zeros((len(dv_HL), len(dv_NH)))
    val_accuracy_Matrix_arch = np.zeros((len(dv_HL), len(dv_NH)))
    test_accuracy_Matrix_arch = np.zeros((len(dv_HL), len(dv_NH)))

    # Study architecture
    for i, dv0 in enumerate(dv_HL):
        for j, dv1 in enumerate(dv_NH):
            print("OPT arch", dv0, dv1)
            dv = [dv0, dv1, dv2, dv3]
            perceptron.training(dv)
            train_accuracy, val_accuracy, test_accuracy = perceptron.retrieveAccuracy()
            train_accuracy_Matrix_arch[i, j] = train_accuracy[-1]
            val_accuracy_Matrix_arch[i, j] = val_accuracy[-1]
            test_accuracy_Matrix_arch[i, j] = test_accuracy

    FileName1 = "./Results/TrainingPopulation/NNoptimization_arch_train.txt"
    FileName2 = "./Results/TrainingPopulation/NNoptimization_arch_test.txt"
    FileName3 = "./Results/TrainingPopulation/NNoptimization_arch_val.txt"

    mat = np.matrix(train_accuracy_Matrix_arch)
    mat2 = np.matrix(test_accuracy_Matrix_arch)
    mat3 = np.matrix(val_accuracy_Matrix_arch)

    with open(FileName1, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
    with open(FileName2, "wb") as f:
        for line in mat2:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
    with open(FileName3, "wb") as f:
        for line in mat3:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()

def optTra(traindata, testdata, dv_TE, dv_BS):
    perceptron = ANN(traindata, testdata)

    train_accuracy_Matrix_train = np.zeros((len(dv_TE), len(dv_BS)))
    test_accuracy_Matrix_train = np.zeros((len(dv_TE), len(dv_BS)))
    val_accuracy_Matrix_train = np.zeros((len(dv_TE), len(dv_BS)))

    dv0 = 2
    dv1 = 50
    for i, dv2 in enumerate(dv_TE):
        for j, dv3 in enumerate(dv_BS):
            print("OPT train", dv2, dv3)
            dv = [dv0, dv1, dv2, dv3]
            perceptron.training(dv)
            train_accuracy, val_accuracy, test_accuracy = perceptron.retrieveAccuracy()
            train_accuracy_Matrix_train[i, j] = train_accuracy[-1]
            test_accuracy_Matrix_train[i, j] = test_accuracy
            val_accuracy_Matrix_train[i, j] = val_accuracy[-1]

    FileName4 = "./Results/TrainingPopulation/NNoptimization_train_train.txt"
    FileName5 = "./Results/TrainingPopulation/NNoptimization_train_test.txt"
    FileName6 = "./Results/TrainingPopulation/NNoptimization_train_val.txt"

    mat = np.matrix(train_accuracy_Matrix_train)
    mat2 = np.matrix(test_accuracy_Matrix_train)
    mat3 = np.matrix(val_accuracy_Matrix_train)

    with open(FileName4, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
    with open(FileName5, "wb") as f:
        for line in mat2:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
    with open(FileName6, "wb") as f:
        for line in mat3:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()

def loadData(): 
    train_acc_arch = np.loadtxt("./Results/TrainingPopulation/NNoptimization_arch_train.txt")
    test_acc_arch = np.loadtxt("./Results/TrainingPopulation/NNoptimization_arch_test.txt")
    val_acc_arch = np.loadtxt("./Results/TrainingPopulation/NNoptimization_arch_val.txt")

    train_acc_train = np.loadtxt("./Results/TrainingPopulation/NNoptimization_train_train.txt")
    test_acc_train = np.loadtxt("./Results/TrainingPopulation/NNoptimization_train_test.txt")
    val_acc_train = np.loadtxt("./Results/TrainingPopulation/NNoptimization_train_val.txt")

    return train_acc_arch, test_acc_arch, val_acc_arch, train_acc_train, test_acc_train, val_acc_train

def plot(dv_HL, dv_NH, dv_TE, dv_BS):
    ta1, te1, va1, ta2, te2, va2 = loadData()
    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']

    fig= plt.figure()

    ax = fig.add_subplot(2,3, 1)
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, ta1[i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training accuracy')
    plt.grid()
    plt.legend(title = "Hidden layers")

    ax = fig.add_subplot(2,3, 2)
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, te1[i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Testing accuracy')
    plt.grid()
    plt.legend(title = "Hidden layers")
    # plt.show()

    ax = fig.add_subplot(2,3, 3)
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, va1[i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Validation accuracy')
    plt.grid()
    plt.legend(title = "Hidden layers")
    # plt.show()


    ax = fig.add_subplot(2,3,4)
    for i in range(len(dv_BS)):
        plt.plot(dv_TE, ta2[:, i], 'x-', c = color[i], label = dv_BS[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Training accuracy')
    plt.grid()
    plt.legend(title = "Batch size")

    ax = fig.add_subplot(2,3,5)
    for i in range(len(dv_BS)):
        plt.plot(dv_TE, te2[:, i], 'x-', c = color[i], label = dv_BS[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Testing accuracy')
    plt.grid()
    plt.legend(title = "Batch size")

    ax = fig.add_subplot(2,3,6)
    for i in range(len(dv_BS)):
        plt.plot(dv_TE, va2[:, i], 'x-', c = color[i], label = dv_BS[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Validation accuracy')
    plt.grid()
    plt.legend(title = "Batch size")

    # plt.layout('tight')
    
    plt.show()


if __name__ == "__main__":

    dataset_np = DTS.LoadNumpy()
    traindata, testdata = DTS.splitData(dataset_np)

    dv_HL = [2, 5, 8]
    dv_NH = [3, 5, 10, 20, 50, 80, 100]
    dv_TE = [1, 5, 20, 50, 80, 150]
    dv_BS = [10, 30, 60, 100]

    # optArch(traindata, testdata, dv_HL, dv_NH)
    # optTra(traindata, testdata, dv_TE, dv_BS)
    # hidden_layers, neuron_hidden, training_epochs, batch_size

    plot(dv_HL, dv_NH, dv_TE, dv_BS)


    