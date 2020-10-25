import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold

import LoadConfigFiles as CONF
import TrainingDataKeras as TD

###################################################################
# https://stackabuse.com/tensorflow-2-0-solving-classification-and-regression-problems/
# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
###################################################################

ANN = CONF.ANN_reg()
ANN_train = ANN.ANN_train
ANN_archic = ANN.ANN_archic


class ANN_reg:
    def __init__(self, dataset):
        self.dataset_np = dataset

        self.n_classes = self.dataset_np.n_classes # Labels
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

        self.checkpoint_path = "./trainedCNet_Reg/training_1/cp.ckpt"

    def create_model(self, regularization = False):
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
        model.compile(loss='mae', optimizer ='adam')

        return model
            
    def get_traintestdata(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def training(self):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=0)

        X, y = self.traindata
        
        results = list()

        # define evaluation procedure
        cv = RepeatedKFold(n_splits=ANN_train['n_splits'], 
                        n_repeats=ANN_train['n_repeats'], 
                        random_state=ANN_train['random_state'])

        # enumerate folds
        self.history = []
        for train_ix, test_ix in cv.split(X):
            # prepare data
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            # define model
            self.model = self.create_model()
            # fit model
            self.history.append( self.model.fit(X_train, y_train, verbose=2, epochs=100,
                    callbacks=[cp_callback])    )
            # evaluate model on test set: mean absolute error
            mae = self.model.evaluate(X_test, y_test, verbose=0)
            # store result
            print(mae)
            results.append(mae)
        # return results

        print('MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))


    def plotTraining(self):
        # summarize history for loss
        colors = ['r-.','g-.','k-.','b-.','r-.','g-.','k-.','b-.','r-','g-','k-','b-','r--','g--','k.-','b.-']
        for i in range(len(self.history)):
            plt.plot(self.history[i].history['loss'], colors[i])
        # plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def load_model_fromFile(self):
        model_fromFile = self.create_model()
        model_fromFile.load_weights(self.checkpoint_path).expect_partial()
        self.model = model_fromFile
        
    def predict(self, fromFile = False, testfile = False, rescale = False):
        """
        INPUTS:
            fromFile: model is not trained in this run but loaded. True or False
            testfile: the predict function uses the test from the database or
                    an external dataset. To use the testdataset: False, otherwise 
                    testfile = dataset to be evaluated.
            rescale: rescale the output (True) or not (False)
        """
        if fromFile == True:
            self.load_model_fromFile()

        if type(testfile) == bool:
            pred_test = self.model.predict(self.testdata[0])

            if rescale == False: # No inverse standarization possible

                self.Error_pred = np.zeros((len(pred_test),2))
                for i in range(len(pred_test)):
                    print('i', i)
                    print("%e, %e"%(pred_test[i,0], pred_test[i,1]) )
                    print("%e, %e"%(self.testdata[1][i,0], self.testdata[1][i,1] ))
                    print("------------------------")
                    self.Error_pred[i,0] = abs( pred_test[i,0] - self.testdata[1][i,0] )
                    self.Error_pred[i,1] = abs( pred_test[i,1] - self.testdata[1][i,1] )
            else:
                predictions_unscaled = self.dataset_np.inverseStandardizationError(pred_test) #Obtain predictions in actual 
                true_value = self.dataset_np.inverseStandardizationError(self.testdata[1]) #Obtain predictions in actual

                self.Error_pred_unscale = np.zeros((len(pred_test),2)) 
                for i in range(len(predictions_unscaled)):
                    print('i', i)
                    print("%e, %e"%(predictions_unscaled[i,0], predictions_unscaled[i,1]) )
                    print("%e, %e"%(true_value[i,0], true_value[i,1] ))
                    print("------------------------")
                    self.Error_pred_unscale[i,0] = abs( predictions_unscaled[i,0] - true_value[i,0 ])
                    self.Error_pred_unscale[i,1] = abs( predictions_unscaled[i,1] - true_value[i,1 ])
        else:
            pred_test0 = self.model.predict(testfile)
            if rescale == True: # inverse standarization 
                pred_test = self.dataset_np.inverseStandardizationError(pred_test0) #Obtain predictions in actual
            else:
                pred_test = pred_test0
        return pred_test

    def plotPredictions(self, dataset):

        if type(dataset) == bool: # No inverse standarization possible
            fig, ax = plt.subplots() 
            plt.plot(np.arange(0, len(self.Error_pred)), self.Error_pred[:,0], 'r-x', label = 'Error in position')
            plt.plot(np.arange(0, len(self.Error_pred)), self.Error_pred[:,1], 'g-x', label = 'Error in velocity')

        else:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(np.arange(0, len(self.Error_pred_unscale)), self.Error_pred_unscale[:,0], 'r-x', label = 'Error in position')
            ax2.plot(np.arange(0, len(self.Error_pred_unscale)), self.Error_pred_unscale[:,1], 'g-x', label = 'Error in velocity')

        # plt.grid(False)
        plt.legend()
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

if __name__ == "__main__":

    ###############################################
    # LOAD TRAINING DATA
    ###############################################
    train_file_path = "./databaseANN/ErrorIncluded/trainingData_Feas_big2.txt"
    # train_file_path = "./databaseANN/trainingData_Feas_V2plusfake.txt"

    # TD.plotInitialDataPandas(train_file_path, pairplot= False, corrplot= False, inputsplotbar = False, inputsplotbarFeas = True)
    # dataset_np = TD.LoadNumpy(train_file_path, plotDistribution = True)
    dataset_np = TD.LoadNumpy(train_file_path)
    traindata, testdata = TD.splitData_reg(dataset_np)

    
    ###############################################
    # CREATE AND TRAIN CLASS NETWORK
    ###############################################
    perceptron = ANN_reg(dataset_np)
    perceptron.get_traintestdata(traindata, testdata)
    perceptron.training()
    perceptron.plotTraining()
    
    print("EVALUATE")
    # predictions = perceptron.predict(fromFile=True, rescale = False)
    print("Rescaled:")
    predictions = perceptron.predict(fromFile=True, rescale = True)

    # predictions_unscaled = dataset_np.inverseStandardizationError(predictions) #Obtain predictions in actual 
    # true_value = dataset_np.inverseStandardizationError(testdata[1]) #Obtain predictions in actual 
    # for i in range(len(predictions_unscaled)):
    #     print('i', i)
    #     print("%e, %e"%(predictions_unscaled[i,0], predictions_unscaled[i,1]) )
    #     print("%e, %e"%(true_value[i,0], true_value[i,1] ))
    #     print("------------------------")
    
    
    # perceptron.plotPredictions(predictions)

    # # Print weights of trained
    # # perceptron.printWeights()

    # # Simple prediction
    # print("SINGLE PREDICTION")
    # predictions = perceptron.singlePrediction(testdata[0][10, :])
    # print("Correct label", testdata[1][10])


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

