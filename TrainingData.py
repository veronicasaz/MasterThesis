# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

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
    # Labels2 = np.zeros((len(Labels), 2))
    # for i in range(len(Labels)):
    #     if Labels[i] == 0: # Non feasible
    #         Labels2[i,:] = np.array([1,0])
    #     else:
    #         Labels2[i,:] = np.array([0,1])

    Labels2 = Labels
    return Labels2

def loadPandas(plotDistribution = False, pairplot = False, corrplot = False):
    # Load with pandas
    feasible_txt = pd.read_csv(train_file_path, sep=" ", header = 0)
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
    y_train = train_y.iloc[0:train_cnt].values
    # x_test = train_x.iloc[train_cnt:].values
    x_test = train_x[train_cnt:] # it comes from numpy 
    y_test = train_y.iloc[train_cnt:].values
    return [x_train, y_train], [x_test, y_test]

# def multilayer_perceptron(x, weights, biases, keep_prob): #with dropout
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.dropout(layer_1, keep_prob)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.dropout(layer_2, keep_prob)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    return out_layer

# def get_compiled_model():
#     # Dense(number of neurons per layer)
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1)
#   ])

def ANN_setup(traindata, testdata): 

    n_input = traindata[0].shape[1] #inputs
    n_classes = 2 # Labels
    n_examples = traindata[0].shape[0]

    x = tf.cast(traindata[0], dtype = tf.float32)
    y = np.reshape( traindata[1], (traindata[1].size, 1) )

    weights = {
        'h1': tf.Variable(tf.random.normal([n_input, ANN_archic['neuron_hidden']])),
        'h2': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], ANN_archic['neuron_hidden']])),
        'out': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden'], n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']])),
        'b2': tf.Variable(tf.random.normal([ANN_archic['neuron_hidden']])),
        'out': tf.Variable(tf.random.normal([n_classes]))
    }

    # keep_prob = tf.placeholder("float")
    # out_layer = multilayer_perceptron(x, weights, biases, keep_prob)
    out_layer = multilayer_perceptron(x, weights, biases) # without dropout

    # Cost function and optimizer

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits = out_layer, labels = y)) # TODO: change from sigmoid. Relu?

    # optimizer = tf.train.AdamOptimizer(learning_rate = ANN_train['learning_rate']).minimize(cost) 
    optimizer = tf.train.AdamOptimizer(learning_rate = ANN_train['learning_rate']).minimize(cost) 
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = ANN_train['learning_rate']).minimize(cost)


    ###############################################
    # Plot settings 
    ###############################################
    avg_set = [] 
    epoch_set = [] 

    # Initializing the variables 
    init = tf.global_variables_initializer() 

    # Launch the graph 
    with tf.Session() as sess: 
        sess.run(init) 
        
        # Training cycle
        for epoch in range(ANN_train['training_epochs']): 
            avg_cost = 0. 
            total_batch = int(n_examples / ANN_train['batch_size']) # number of batches
            
            # Loop over all batches 
            for i in range(total_batch): 
                batch_xs, batch_ys = mnist.train.next_batch(ANN_train['batch_size']) 
                # Fit training using batch data sess.run(optimizer, feed_dict = {
                    # x: batch_xs, y: batch_ys}) 
                # Compute average loss 
                avg_cost += sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch
            # Display logs per epoch step 
            if epoch % display_step == 0: 
                print( "Epoch:", '%04d' % (epoch + 1), \
                    "cost=", "{:.9f}".format(avg_cost) )
            avg_set.append(avg_cost) 
            epoch_set.append(epoch + 1)
        print ("Training phase finished") 
        
        plt.plot(epoch_set, avg_set, 'o', label = 'MLP Training phase') 
        plt.ylabel('cost') 
        plt.xlabel('epoch') 
        plt.legend() 
        plt.show() 
        
        # Test model 
        correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1)) 
        
        # Calculate accuracy 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
        print ("Model Accuracy:", accuracy.eval({x: testdata[0], y: testdata[1]}))
        





# # Convert objects into discrete numerical values
# for label in labels_feas:
#   feasible_txt[label] = pd.Categorical(feasible_txt[label])
#   feasible_txt[label] = getattr(feasible_txt, label).cat.codes

# # Create data.Dataset
# output = feasible_txt.pop('Label')
# dataset = tf.data.Dataset.from_tensor_slices((feasible_txt.values, output.values))

# # for feat, targ in dataset.take(5):
# #     print ('Features: {}, Target: {}'.format(feat, targ))

# # Shuffle and batch the dataset
# train_dataset = dataset.shuffle(len(feasible_txt)).batch(1)

# # Create and train a model
# def get_compiled_model():
#     # Dense(number of neurons per layer)
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1)
#   ])

#     model.compile(optimizer='adam',
#                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                     metrics=['accuracy']) # TODO: choose measure for accuracy for classification

#     return model

# model = get_compiled_model()
# model.fit(train_dataset, epochs=10)


if __name__ == "__main__":

    # LoadNumpy()
    # feasible_stnd, data_feature = loadPandas(plotDistribution = True, pairplot= True, corrplot= True)
    feasible_stnd, data_feature = loadPandas()
    
    # encodeData(feasible_stnd) #needed?
    traindata, testdata = splitData(feasible_stnd, data_feature)
    ANN_setup(traindata, testdata)