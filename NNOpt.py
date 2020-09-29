# import tensorflow as tf
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

import LoadConfigFiles as CONF

import TrainingDataKeras as TDK

tf.disable_v2_behavior() # To run v1 of tf

###################################################################
# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
# https://www.tensorflow.org/tutorials/keras/classification
# https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b
###################################################################

# train_file_path = "./databaseANN/trainingData_Feas_shortTest.txt"
train_file_path = "./databaseANN/trainingData_Feas.txt"

ANN = CONF.ANN()
ANN_train = ANN.ANN_train
ANN_archic = ANN.ANN_archic

# def multilayer_perceptron(x, weights, biases, keep_prob): #with dropout
def multilayer_perceptron(x, weights, biases, dv_HL):
    """

    """
    layer_prev = tf.add(tf.matmul(x, weights[0]), biases[0])
    layer_prev = tf.nn.relu(layer_prev)

    for layer in range(dv_HL-1):
        layer_current = tf.add(tf.matmul(layer_prev, weights[layer+1]), biases[layer+1])
        layer_current = tf.nn.relu(layer_current)
        layer_prev = layer_current

    out_layer = tf.matmul(layer_current, weights[-1]) + biases[-1]
    out_layer = tf.sigmoid(out_layer)
    
    return out_layer

def ANN_setup(traindata, testdata, dv): 

    dv_HL, dv_NH, dv_TE, dv_BS = dv

    s = tf.InteractiveSession()

    n_input = traindata[0].shape[1] #inputs
    n_classes = 2 # Labels
    n_examples = traindata[0].shape[0]

    # x = tf.cast(traindata[0], dtype = tf.float32)
    # y = np.reshape( traindata[1], (traindata[1].size, 1) )

    x = tf.placeholder('float32', shape = (None, n_input), name= "x")
    y = tf.placeholder("float32", shape = (None, n_classes), name = "y")
    keep_prob = tf.placeholder(tf.float32)

    weights = [ tf.Variable(tf.random.normal([n_input, dv_NH]))] +\
            [ tf.Variable(tf.random.normal([dv_NH, dv_NH])) ] * (dv_HL -1) +\
            [  tf.Variable(tf.random.normal([dv_NH, n_classes])) ]

    biases = [tf.Variable(tf.random.normal([dv_NH]))] +\
             [tf.Variable(tf.random.normal([dv_NH]))] * (dv_HL -1) +\
             [tf.Variable(tf.random.normal([n_classes]))]

    # out_layer = multilayer_perceptron(x, weights, biases, keep_prob)
    out_layer = multilayer_perceptron(x, weights, biases, dv_HL) # without dropout

    # Cost function and optimizer

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = out_layer, labels = y)) 
            # + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))

    # optimizer = tf.train.AdamOptimizer(learning_rate = ANN_train['learning_rate']).minimize(cost) 
    # optimizer = tf.train.AdamOptimizer(learning_rate = ANN_train['learning_rate']).minimize(cost, \
    #     var_list = list(weights.values()) + list(biases.values())  ) # When weights and biases are dictionaries

    optimizer = tf.train.AdamOptimizer(learning_rate = ANN_train['learning_rate']).minimize(cost, \
        var_list = weights + biases  ) 

    # Metrics definition
    correct_prediction = tf.equal(tf.argmax(traindata[1], 1), \
                                  tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))


    training_accuracy = []
    training_loss = []
    testing_accuracy = []

    s.run(tf.global_variables_initializer())
    for epoch in range(dv_TE):    
        arr = np.arange(traindata[0].shape[0])
        np.random.shuffle(arr)
        for index in range(0, traindata[0].shape[0], dv_BS):
            s.run(optimizer, {x: traindata[0][arr[index:index + dv_BS]],
                             y: traindata[1][arr[index:index + dv_BS]],
                             keep_prob: ANN_train['dropout_prob']})

        training_accuracy.append(s.run(accuracy, feed_dict= {x: traindata[0], 
                                                            y: traindata[1], 
                                                            keep_prob: 1}))
        training_loss.append(s.run(cost, {x: traindata[0], 
                                        y: traindata[1], 
                                        keep_prob: 1}))
        
        ## Evaluation of model
        testing_accuracy.append(accuracy_score(testdata[1].argmax(1), 
                                s.run(out_layer, {x: testdata[0],keep_prob:1}).argmax(1)))

    print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
    print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1])) 

    return training_accuracy, testing_accuracy

def optArch(traindata, testdata, dv_HL, dv_NH):

    # Fix training values to study architecture
    dv2 = 50
    dv3 = 30

    train_accuracy_Matrix_arch = np.zeros((len(dv_HL), len(dv_NH)))
    test_accuracy_Matrix_arch = np.zeros((len(dv_HL), len(dv_NH)))

    # Study architecture
    for i, dv0 in enumerate(dv_HL):
        for j, dv1 in enumerate(dv_NH):
            print("OPT arch", dv0, dv1)
            dv = [dv0, dv1, dv2, dv3]
            train_accuracy, test_accuracy = ANN_setup(traindata, testdata, dv)
            train_accuracy_Matrix_arch[i, j] = train_accuracy[-1]
            test_accuracy_Matrix_arch[i, j] = test_accuracy[-1]

    FileName1 = "./Results/TrainingPopulation/NNoptimization_arch_train.txt"
    FileName2 = "./Results/TrainingPopulation/NNoptimization_arch_test.txt"

    mat = np.matrix(train_accuracy_Matrix_arch)
    mat2 = np.matrix(test_accuracy_Matrix_arch)

    with open(FileName1, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
    with open(FileName2, "wb") as f:
        for line in mat2:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()

def optTra(traindata, testdata, dv_TE, dv_BS):
    train_accuracy_Matrix_train = np.zeros((len(dv_TE), len(dv_BS)))
    test_accuracy_Matrix_train = np.zeros((len(dv_TE), len(dv_BS)))

    dv0 = 2
    dv1 = 50
    for i, dv2 in enumerate(dv_TE):
        for j, dv3 in enumerate(dv_BS):
            print("OPT train", dv2, dv3)
            dv = [dv0, dv1, dv2, dv3]
            train_accuracy, test_accuracy = ANN_setup(traindata, testdata, dv)
            train_accuracy_Matrix_train[i, j] = train_accuracy[-1]
            test_accuracy_Matrix_train[i, j] = test_accuracy[-1]

    FileName3 = "./Results/TrainingPopulation/NNoptimization_train_train.txt"
    FileName4 = "./Results/TrainingPopulation/NNoptimization_train_test.txt"

    mat = np.matrix(train_accuracy_Matrix_train)
    mat2 = np.matrix(test_accuracy_Matrix_train)

    with open(FileName3, "wb") as f:
        for line in mat:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()
    with open(FileName4, "wb") as f:
        for line in mat2:  
            np.savetxt(f, line, fmt='%.2f')  
    f.close()

def loadData(): 
    train_acc_arch = np.loadtxt("./Results/TrainingPopulation/NNoptimization_arch_train.txt")
    test_acc_arch = np.loadtxt("./Results/TrainingPopulation/NNoptimization_arch_test.txt")
    train_acc_train = np.loadtxt("./Results/TrainingPopulation/NNoptimization_train_train.txt")
    test_acc_train = np.loadtxt("./Results/TrainingPopulation/NNoptimization_train_test.txt")

    return train_acc_arch, test_acc_arch, train_acc_train, test_acc_train

def plot(dv_HL, dv_NH, dv_TE, dv_BS):
    ta1, te1, ta2, te2 = loadData()
    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']

    fig= plt.figure()

    ax = fig.add_subplot(2,2, 1)
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, ta1[i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Training accuracy')
    plt.grid()
    plt.legend(title = "Hidden layers")

    ax = fig.add_subplot(2,2, 2)
    for i in range(len(dv_HL)):
        plt.plot(dv_NH, te1[i, :], 'x-', c = color[i], label = dv_HL[i])
    plt.xlabel('Neurons per layer')
    plt.ylabel('Testing accuracy')
    plt.grid()
    plt.legend(title = "Hidden layers")
    # plt.show()


    ax = fig.add_subplot(2,2, 3)
    for i in range(len(dv_BS)):
        plt.plot(dv_TE, ta2[:, i], 'x-', c = color[i], label = dv_BS[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Training accuracy')
    plt.grid()
    plt.legend(title = "Batch size")

    ax = fig.add_subplot(2,2, 4)
    for i in range(len(dv_BS)):
        plt.plot(dv_TE, te2[:, i], 'x-', c = color[i], label = dv_BS[i])
    plt.xlabel('Training epochs')
    plt.ylabel('Testing accuracy')
    plt.grid()
    plt.legend(title = "Batch size")
    
    plt.show()


if __name__ == "__main__":

    feasible_txt = TDK.loadPandas()
    feasible_stnd, data_feature = TDK.adaptData(feasible_txt)
    traindata, testdata = TDK.splitData(feasible_stnd, data_feature)

    dv_HL = [2, 5, 8]
    dv_NH = [3, 5, 10, 20, 50, 80, 100]
    dv_TE = [1, 5, 20, 50, 80, 150]
    dv_BS = [10, 30, 60, 100]

    # optArch(traindata, testdata, dv_HL, dv_NH)
    # optTra(traindata, testdata, dv_TE, dv_BS)
    # hidden_layers, neuron_hidden, training_epochs, batch_size

    plot(dv_HL, dv_NH, dv_TE, dv_BS)


    