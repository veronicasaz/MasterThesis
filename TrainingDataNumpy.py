# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

import LoadConfigFiles as CONF

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

class Dataset:
    def __init__(self, file_path, dataset = False, shuffle = True):
        # Load with numpy
        if np.any(dataset) == False:
            dataset = np.loadtxt(file_path, skiprows = 1)
            # Load labels
            fh = open(file_path,'r')
            for i, line in enumerate(fh):
                if i == 1: 
                    break
                line = line[:-1] # remove the /n
                self.labels = line.split(" ")
            fh.close()

        else:
            self.dataset = dataset

        if shuffle == True:
            self.dataset = np.random.shuffle(dataset) # shuffle rows

        self.input_data = dataset[:,1:]
        self.output = dataset[:,0]

    def statisticsFeasible(self):
        self.nsamples = len(self.dataset[:,0])
        self.count_feasible = np.count_nonzero(self.output)
        print("Samples", self.nsamples, "Feasible", self.count_feasible)

    def plotDistributionOfFeasible(self):
        count_unfeasible = np.count_nonzero(self.output==0)
        count_feasible = len(self.output) - count_unfeasible

        fig, ax = plt.subplots() 
        plt.bar([0,1], [count_unfeasible, count_feasible])
        plt.xticks([0,1], ['Unfeasible', 'Feasible'])
        for i, v in enumerate([count_unfeasible, count_feasible]):
            ax.text(i, v+5, str(v), color='black', fontweight='bold')
        plt.show()

    def standardizationInputs(self):
        # Standarization of the inputs
        scaler = StandardScaler()
        scaler.fit(self.input_data)
        self.input_data_std = scaler.transform(self.input_data)

    def convertLabels(self): # Labels are [Unfeasible feasible]
        self.output_2d = np.zeros((len(self.output), 2))
        for i in range(len(self.output)):
            if self.output[i] == 0: # Non feasible
                self.output_2d[i,:] = np.array([1,0])
            else:
                self.output_2d[i,:] = np.array([0,1])


def plotInitialDataPandas(pairplot = False, corrplot = False):
    feasible_txt = pd.read_csv(train_file_path, sep=" ", header = 0)
    labels_feas = feasible_txt.columns.values

    if pairplot == True: # pairplot
        sns.pairplot(feasible_txt[labels_feas], hue = 'Label')
        plt.show()

    if corrplot == True: # correlations matrix 
        corr_mat = feasible_txt.corr()
        fig, ax = plt.subplots(figsize =(20,12))
        sns.heatmap(corr_mat, vmax = 1.0, square= True, ax=ax)
        plt.show()   

def LoadNumpy(plotDistribution = False):
    # Load with numpy to see plot
    dataset_np = Dataset(train_file_path, shuffle = True)

    # Plot distribution of feasible/unfeasible
    if plotDistribution == True:
        dataset_np.plotDistributionOfFeasible()
    # dataset_np.statisticsFeasible()
    # dataset_np.plotDistributionOfDataset()

    dataset_np.standardizationInputs()
    dataset_np.convertLabels()

    return dataset_np


def splitData( dataset_np):
    train_x, train_y = dataset_np.input_data_std, dataset_np.output_2d

    train_cnt = floor(train_x.shape[0] * ANN_train['train_size'])
    x_train = train_x[0:train_cnt]
    y_train = train_y[0:train_cnt]
    x_test = train_x[train_cnt:]  
    y_test = train_y[train_cnt:]

    return [x_train, y_train], [x_test, y_test]


class ANN:
    def __init__(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

        self.n_input = traindata[0].shape[1] #inputs
        self.n_classes = traindata[1].shape[1] # Labels
        self.n_examples = traindata[0].shape[0] # samples

        self.weights = [ tf.Variable(tf.random.normal([self.n_input, ANN_archic['neuron_hidden']]))] +\
            [ tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], ANN_archic['neuron_hidden']])) ] * (ANN_archic['hidden_layers'] -1) +\
            [  tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], self.n_classes])) ]

        self.biases = [tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']]))] +\
             [tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']]))] * (ANN_archic['hidden_layers'] -1) +\
             [tf.Variable(tf.random.normal([self.n_classes]))]

    def multilayer_perceptron(self, x, weights, biases):
        # layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_prev = tf.add(tf.matmul(x, weights[0]), biases[0])
        layer_prev = tf.nn.relu(layer_prev)
        # layer_1 = tf.nn.dropout(layer_1, keep_prob)

        for layer in range(ANN_archic['hidden_layers']-1):
            # layer_current = tf.add(tf.matmul(layers[layer], weights['h2']), biases['b2'])
            layer_current = tf.add(tf.matmul(layer_prev, weights[layer+1]), biases[layer+1])
            layer_current = tf.nn.relu(layer_current)
            layer_prev = layer_current
        # layer_2 = tf.nn.dropout(layer_2, keep_prob)

        out_layer = tf.matmul(layer_current, weights[-1]) + biases[-1]
        self.out_layer = tf.sigmoid(out_layer)


    def training(self):
        s = tf.InteractiveSession()

        x = tf.placeholder('float32', shape = (None, self.n_input), name= "x")
        y = tf.placeholder("float32", shape = (None, self.n_classes), name = "y")
        keep_prob = tf.placeholder(tf.float32)

        self.multilayer_perceptron(x, self.weights, self.biases) # without dropout
        # out_layer = multilayer_perceptron(x, weights, biases, keep_prob)

         # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits = self.out_layer, labels = y)) 
                # + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))

        optimizer = tf.train.AdamOptimizer(learning_rate = ANN_train['learning_rate']).minimize(cost, \
            var_list = self.weights + self.biases  ) 

         # Metrics definition
        correct_prediction = tf.equal(tf.argmax(self.traindata[1], 1), \
                                    tf.argmax(self.out_layer, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

        self.training_accuracy = []
        self.training_loss = []
        self.testing_accuracy = []

        s.run(tf.global_variables_initializer())
        for epoch in range(ANN_train['training_epochs']):    
            arr = np.arange(traindata[0].shape[0])
            for index in range(0, traindata[0].shape[0], ANN_train['batch_size']):
                s.run(optimizer, {x: traindata[0][arr[index:index + ANN_train['batch_size']]],
                                y: traindata[1][arr[index:index + ANN_train['batch_size']]],
                                keep_prob: ANN_train['dropout_prob']})

            self.training_accuracy.append(s.run(accuracy, feed_dict= {x: traindata[0], 
                                                                y: traindata[1], 
                                                                keep_prob: 1}))
            self.training_loss.append(s.run(cost, {x: traindata[0], 
                                            y: traindata[1], 
                                            keep_prob: 1}))
            
            ## Evaluation of model
            self.testing_accuracy.append(accuracy_score(testdata[1].argmax(1), 
                                    s.run(self.out_layer, {x: testdata[0],keep_prob:1}).argmax(1)))
            print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                            self.training_loss[epoch],
                                                                            self.training_accuracy[epoch],
                                                                        self.testing_accuracy[epoch]))
        

    def plotTraining(self):
        ## Plotting chart of training and testing accuracy as a function of iterations
        iterations = list(range(ANN_train['training_epochs']))
        plt.plot(iterations, self.training_accuracy, label='Train')
        plt.plot(iterations, self.testing_accuracy, label='Test')
        plt.ylabel('Accuracy')
        plt.xlabel('iterations')
        plt.legend()
        plt.show()
        print("Train Accuracy: {0:.2f}".format(self.training_accuracy[-1]))
        print("Test Accuracy:{0:.2f}".format(self.testing_accuracy[-1])) 


if __name__ == "__main__":

    # plotInitialDataPandas(pairplot= True, corrplot= True)
    # dataset_np = LoadNumpy(plotDistribution = True)
    dataset_np = LoadNumpy()
    traindata, testdata = splitData(dataset_np)
    
    perceptron = ANN(traindata, testdata)
    perceptron.training()
    perceptron.plotTraining()