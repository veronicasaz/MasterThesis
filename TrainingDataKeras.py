import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

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

###################################################################
# https://deeplizard.com/learn/video/8krd5qKVw-Q
###################################################################

# train_file_path = "./databaseANN/trainingData_Feas_V2.txt"
train_file_path = "./databaseANN/trainingData_Feas_V2plusfake.txt"

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


def plotInitialDataPandas(pairplot = False, corrplot = False, inputsplotbar = False, inputsplotbarFeas = False):
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

    if inputsplotbar == True: # plot distribution of inputs
        fig= plt.figure()

        rows = floor( np.sqrt(len(labels_feas)-1) )
        cols = ceil(int((len(labels_feas)-1 ) /rows))

        for i in range(len(labels_feas)-1): # number of inputs
            ax = fig.add_subplot(rows, cols, i+1)
            ax = sns.histplot(feasible_txt[labels_feas[i+1]], kde = True)
            # ax.set_title(labels_feas[i+1])
            # ax.set(ylim=(0, None))

        plt.show()

    if inputsplotbarFeas == True: # plot distribution of inputs
        fig= plt.figure()

        rows = floor( np.sqrt(len(labels_feas)-1) )
        cols = ceil(int((len(labels_feas)-1 ) /rows))

        for i in range(len(labels_feas)-1): # number of inputs
            if i == 4 or i == 5:
                log_scale = True
            else:
                log_scale = False
            ax = fig.add_subplot(rows, cols, i+1)
            ax = sns.histplot(data = feasible_txt, x= labels_feas[i+1], hue = 'Label')
            # ax.set_title(labels_feas[i+1])
            # ax.set(ylim=(0, None))

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
    # dataset_np.convertLabels()

    return dataset_np


def splitData( dataset_np):
    train_x, train_y = dataset_np.input_data_std, dataset_np.output

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
        self.n_classes = 2 # Labels
        self.n_examples = traindata[0].shape[0] # samples

        self.checkpoint_path = "./trainedCNet_Class/training_1/cp.ckpt"

    def create_model(self, regularization = True):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()

        if regularization == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            for layer in range(ANN_archic['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN_archic['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN_archic['regularizer_value']) ))
        else:
            for layer in range(ANN_archic['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN_archic['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer) )

        model.add(keras.layers.Dense(self.n_classes) ) # output layer
       
        # Compile
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model
            
    def training(self):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        # Create model architecture
        self.model = self.create_model()
        
        # Train
        self.history = self.model.fit(self.traindata[0], self.traindata[1], 
                    validation_split= ANN_train['validation_size'],
                    epochs = ANN_train['training_epochs'], 
                    batch_size = ANN_train['batch_size'],
                    callbacks=[cp_callback] )

    def plotTraining(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def evaluateTest(self):
        test_loss, test_acc = self.model.evaluate(self.testdata[0], self.testdata[1], verbose=2)

        print('\nTest accuracy:', test_acc)

    def predict(self, fromFile = False):
        """
        INPUTS:
            fromFile: model is not trained in this run but loaded
        """
        if fromFile == True:
            model_fromFile = self.create_model()
            model_fromFile.load_weights(self.checkpoint_path)
            self.model = model_fromFile

        self.probability_model = tf.keras.Sequential([self.model, 
                                         tf.keras.layers.Softmax()])
        
        predictions = self.probability_model.predict(self.testdata[0])

        return predictions

    def plotPredictions(self, predictions):
        true_labels = self.testdata[1]
        choice_prediction = [np.argmax(pred) for pred in predictions]
        

        True_Positive = 0
        True_Negative = 0
        False_Positive = 0
        False_Negative = 0

        for i in range(len(choice_prediction)):
            if choice_prediction[i] == 1 and true_labels[i] == 1:
                True_Positive += 1
            elif choice_prediction[i] == 0 and true_labels[i] == 0:
                True_Negative += 1
            elif choice_prediction[i] == 1 and true_labels[i] == 0:
                False_Positive += 1
            elif choice_prediction[i] == 0 and true_labels[i] == 1:
                False_Negative += 1

        fig, ax = plt.subplots() 
        plt.xticks(range(4), labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative'])
        plt.yticks([])
        plot = plt.bar(range(4), [True_Positive, True_Negative, False_Positive, False_Negative] )
        for i, v in enumerate([True_Positive, True_Negative, False_Positive, False_Negative]):
            ax.text(i, v+5, str(v), color='black', fontweight='bold')
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def singlePrediction(self, input_case):
        print(input_case)
        input_batch = np.array([input_case])
        prediction = self.probability_model.predict(input_batch)
        print("Prediction", prediction, "predicted label", np.argmax(prediction))

    def printWeights(self):
        weights_h = list()
        bias_h = list()

        for layer in range(ANN_archic['hidden_layers']):
            weights_h.append( self.model.layers[layer].get_weights()[0] )
            bias_h.append( self.model.layers[layer].get_weights()[1] )

        weights_output = self.model.layers[-1].get_weights()[0]
        bias_output = self.model.layers[-1].get_weights()[1]
        print("WEIGHTS", weights_h)
        print("WEIGHTS", weights_output)

class ANN_fromFile:
    def __init__(self):
        self.n_classes = 2 # Labels
        self.checkpoint_path = "./trainedCNet_Class/training_1/cp.ckpt"

    def create_model(self, regularization = True):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()

        if regularization == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            for layer in range(ANN_archic['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN_archic['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN_archic['regularizer_value']) ))
        else:
            for layer in range(ANN_archic['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN_archic['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer) )

        model.add(keras.layers.Dense(self.n_classes) ) # output layer
       
        # Compile
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model
    
    def load_model_fromFile(self):
        model_fromFile = self.create_model()
        model_fromFile.load_weights(self.checkpoint_path)
        self.model = model_fromFile

        self.probability_model = tf.keras.Sequential([self.model, 
                                         tf.keras.layers.Softmax()])

    def predict(self, input_case):
        prediction = self.probability_model.predict(input_case)
        return [np.argmax(pred) for pred in prediction]

    def predict_single(self, input_case):
        input_batch = np.array([input_case])
        prediction = self.probability_model.predict(input_batch)
        return np.argmax(prediction)

if __name__ == "__main__":

    # plotInitialDataPandas(pairplot= False, corrplot= False, inputsplotbar = False, inputsplotbarFeas = True)
    # dataset_np = LoadNumpy(plotDistribution = True)
    dataset_np = LoadNumpy()
    traindata, testdata = splitData(dataset_np)
    
    perceptron = ANN(traindata, testdata)
    # perceptron.training()
    # perceptron.plotTraining()
    
    # print("EVALUATE")
    # perceptron.evaluateTest()
    predictions = perceptron.predict(fromFile=True)
    perceptron.plotPredictions(predictions)

    # Print weights of trained
    # perceptron.printWeights()

    # Simple prediction
    print("SINGLE PREDICTION")
    predictions = perceptron.singlePrediction(testdata[0][10, :])
    print("Correct label", testdata[1][10])