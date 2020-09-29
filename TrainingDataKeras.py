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
    def __init__(self, file_path, labels_feas, dataset = False):
        # Load with numpy
        self.labels_feas =  labels_feas
        if np.any(dataset) == False:
            dataset = np.loadtxt(file_path, skiprows = 1)
            self.dataset = dataset[:,1:]
            self.output = dataset[:,0]
        else:
            self.dataset = dataset
  
    def statisticsFeasible(self):
        self.nsamples = len(self.dataset[:,0])
        self.count_feasible = np.count_nonzero(self.output)
        print("Samples", self.nsamples, "Feasible", self.count_feasible)

    def plotDistributionOfDataset(self):
        fig = plt.figure(figsize=(15,8))

        for i in range(len(self.dataset[0,:])):
            # Normalization
            x = np.ones(len(self.dataset[:,0]))*i
            y_norm = self.dataset[:,i]
            
            # if min(y) > 0:
            #     y_norm = y / max(y)
            # else:
            #     y_norm = np.zeros(len(y))
            #     pos = np.where(y >0)[0]
            #     neg = np.where(y<0)[0]
            #     y_norm[pos] = y[pos] / max(y)
            #     y_norm[neg] = y[neg] / -min(y)s

            # Plotting
            plt.scatter(x, y_norm)
        
        plt.xticks(np.arange(len(self.dataset[:,0])), self.labels_feas[1:])
        plt.tick_params(axis='both', which='major', labelsize=12)
        # ax.set_xticklabels(self.labels_feas[1:])
        plt.axis([-0.1, 7.1,-1.05,1.05])
        # plt.axis('equal')
        plt.grid(alpha = 0.5)

        dpi = 100
        layoutSave = 'tight'
        # plt.savefig('databaseFeasibility.png', dpi = dpi, bbox_inches = layoutSave)
        plt.title("Distribution of input parameters \n of the training population",\
            size = 25)
        plt.show()

def LoadNumpy():
    # Load with numpy to see plot
    dataset_np = Dataset(train_file_path, labels)
    dataset_np.statisticsFeasible()
    dataset_np.plotDistributionOfDataset()

# def plotDistributionStandardized(database):

def convertLabels(Labels):
    Labels2 = np.zeros((len(Labels), 2))
    for i in range(len(Labels)):
        if Labels[i] == 0: # Non feasible
            Labels2[i,:] = np.array([1,0])
        else:
            Labels2[i,:] = np.array([0,1])

    # Labels2 = Labels
    return Labels2

def loadPandas():
    # Load with pandas
    feasible_txt = pd.read_csv(train_file_path, sep=" ", header = 0)
    return feasible_txt

def adaptData(feasible_txt, plotDistribution = False, pairplot = False, corrplot = False):
    labels_feas = feasible_txt.columns.values
    input_data = feasible_txt.drop('Label', axis = 1)
    input_labels = convertLabels( feasible_txt['Label'] )
    # print(feasible_txt.dtypes)

    # Plot distribution of feasible/unfeasible
    if plotDistribution == True:
        feasible_txt.Label.value_counts().plot(kind = "bar", rot = 0)
        plt.show()

    # Standarization of the inputs
    scaler = StandardScaler()
    scaler.fit(input_data)
    standardized = scaler.transform(input_data)

    # Numpy append label column to standard data
    numpy_stnd = np.c_[feasible_txt.iloc[:,0].values, standardized]

    # Pandas dataframe from the previous data standardized
    feasible_stnd = pd.DataFrame(data= numpy_stnd, columns = labels_feas )  # Complete dataset standardized
    # print(type(standardized))

    # dataset_distribution = Dataset("None", labels_feas, dataset = standardized)
    # dataset_distribution.plotDistributionOfDataset()

    if pairplot == True: # pairplot
        sns.pairplot(feasible_stnd[labels_feas], hue = 'Label')
        plt.show()

    if corrplot == True: # correlations matrix 
        corr_mat = feasible_stnd.corr()
        fig, ax = plt.subplots(figsize =(20,12))
        sns.heatmap(corr_mat, vmax = 1.0, square= True, ax=ax)
        plt.show()

    return feasible_stnd, [standardized, input_labels]

# def encodeData(feasible_stnsd):
#     def enconde(series)

def splitData(feasible_stnd, data_feature ):
    train_x, train_y = data_feature 

    train_cnt = floor(train_x.shape[0] * ANN_train['train_size'])
    # x_train = train_x.iloc[0:train_cnt].values
    x_train = train_x[0:train_cnt] # it comes from numpy
    # y_train = train_y.iloc[0:train_cnt].values
    y_train = train_y[0:train_cnt]
    # x_test = train_x.iloc[train_cnt:].values
    x_test = train_x[train_cnt:] # it comes from numpy 
    # y_test = train_y.iloc[train_cnt:].values
    y_test = train_y[train_cnt:]

    return [x_train, y_train], [x_test, y_test]


# def multilayer_perceptron(x, weights, biases, keep_prob): #with dropout
def multilayer_perceptron(x, weights, biases):
    """

    """
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
    out_layer = tf.sigmoid(out_layer)
    
    return out_layer

def ANN_setup(traindata, testdata): 

    s = tf.InteractiveSession()

    n_input = traindata[0].shape[1] #inputs
    n_classes = 2 # Labels
    n_examples = traindata[0].shape[0]

    # x = tf.cast(traindata[0], dtype = tf.float32)
    # y = np.reshape( traindata[1], (traindata[1].size, 1) )

    x = tf.placeholder('float32', shape = (None, n_input), name= "x")
    y = tf.placeholder("float32", shape = (None, n_classes), name = "y")
    keep_prob = tf.placeholder(tf.float32)

    # weights = {
    #     'h1': tf.Variable(tf.random.normal([n_input, ANN_archic['neuron_hidden']])),
    #     'h2': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], ANN_archic['neuron_hidden']])),
    #     'out': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], n_classes]))
    # }

    # biases = {
    #     'b1': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']])),
    #     'b2': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']])),
    #     'out': tf.Variable(tf.random.normal([n_classes]))
    # }

    weights = [ tf.Variable(tf.random.normal([n_input, ANN_archic['neuron_hidden']]))] +\
            [ tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], ANN_archic['neuron_hidden']])) ] * (ANN_archic['hidden_layers'] -1) +\
            [  tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], n_classes])) ]

    biases = [tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']]))] +\
             [tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']]))] * (ANN_archic['hidden_layers'] -1) +\
             [tf.Variable(tf.random.normal([n_classes]))]

    # out_layer = multilayer_perceptron(x, weights, biases, keep_prob)
    out_layer = multilayer_perceptron(x, weights, biases) # without dropout

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
    for epoch in range(ANN_train['training_epochs']):    
        arr = np.arange(traindata[0].shape[0])
        np.random.shuffle(arr)
        for index in range(0, traindata[0].shape[0], ANN_train['batch_size']):
            s.run(optimizer, {x: traindata[0][arr[index:index + ANN_train['batch_size']]],
                             y: traindata[1][arr[index:index + ANN_train['batch_size']]],
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
        print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                        training_loss[epoch],
                                                                        training_accuracy[epoch],
                                                                    testing_accuracy[epoch]))


    ## Plotting chart of training and testing accuracy as a function of iterations
    iterations = list(range(ANN_train['training_epochs']))
    plt.plot(iterations, training_accuracy, label='Train')
    plt.plot(iterations, testing_accuracy, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()
    print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
    print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1])) 

if __name__ == "__main__":

    # LoadNumpy()
    feasible_txt = loadPandas()
    # feasible_stnd, data_feature = adaptData(feasible_txt, plotDistribution = True, pairplot= True, corrplot= True)
    feasible_stnd, data_feature = adaptData(feasible_txt)
    
    # encodeData(feasible_stnd) #needed?
    traindata, testdata = splitData(feasible_stnd, data_feature)
    ANN_setup(traindata, testdata)