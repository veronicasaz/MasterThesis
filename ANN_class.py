import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

import LoadConfigFiles as CONF
import TrainingDataKeras as TD

###################################################################
# https://deeplizard.com/learn/video/8krd5qKVw-Q
###################################################################

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

    def predict(self, fromFile = False, testfile = False):
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
        
        if type(testfile) == bool:
            predictions = self.probability_model.predict(self.testdata[0])
        else:
            predictions = self.probability_model.predict(testfile)

        return predictions

    def plotPredictions(self, predictions, labels = False):
        if type(labels) == bool:
            true_labels = self.testdata[1]
        else:
            true_labels = labels
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
        # plt.tight_layout()
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

    ###############################################
    # LOAD TRAINING DATA
    ###############################################
    train_file_path = "./databaseANN/ErrorIncluded/trainingData_Feas_big.txt"
    # train_file_path = "./databaseANN/trainingData_Feas_V2plusfake.txt"

    # TD.plotInitialDataPandas(pairplot= False, corrplot= False, inputsplotbar = False, inputsplotbarFeas = True)
    dataset_np = TD.LoadNumpy(train_file_path, plotDistribution = True)
    # dataset_np = TD.LoadNumpy()
    traindata, testdata = TD.splitData(dataset_np)

    
    ###############################################
    # CREATE AND TRAIN CLASS NETWORK
    ###############################################
    perceptron = ANN(traindata, testdata)
    perceptron.training()
    perceptron.plotTraining()
    perceptron.evaluateTest()
    
    print("EVALUATE")

    predictions = perceptron.predict(fromFile=True)
    perceptron.plotPredictions(predictions)

    # Print weights of trained
    # perceptron.printWeights()

    # Simple prediction
    print("SINGLE PREDICTION")
    predictions = perceptron.singlePrediction(testdata[0][10, :])
    print("Correct label", testdata[1][10])


    # # Use test file 
    # testfile =  "./databaseANN/ErrorIncluded/trainingData_Feas_V2.txt"
    # dataset_Test = np.loadtxt(testfile, skiprows = 1)
    # # Load labels
    # fh = open(testfile,'r')
    # for i, line in enumerate(fh):
    #     if i == 1: 
    #         break
    #     line = line[:-1] # remove the /n
    #     labels = line.split(" ")
    # fh.close()
    # dataset_np = Dataset(" ", dataset = [dataset_Test, labels], shuffle = True)
    # dataset_np.standardizationInputs()

    # perceptron = ANN(traindata, testdata)
    
    # predictions = perceptron.predict(fromFile=True, testfile = dataset_np.input_data_std)
    # perceptron.plotPredictions(predictions, labels = dataset_np.output)

