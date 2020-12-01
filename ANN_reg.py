import os
import sys
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

ANN_reg = CONF.ANN_reg()
ANN = ANN_reg.ANN_config

Dataset_conf = CONF.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']

class ANN_reg:
    def __init__(self, dataset):
        self.dataset_np = dataset

        self.n_classes = int(len(self.dataset_np.output_reg[0,:])) # Outputs
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

        self.checkpoint_path = "./trainedCNet_Reg/training_1/"

    def create_model(self):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()

        if ANN['Training']['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            for layer in range(ANN['Architecture']['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN['Architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN['Architecture']['regularizer_value']) ))
        else:
            for layer in range(ANN['Architecture']['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN['Architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer) )

        model.add(keras.layers.Dense(self.n_classes) ) # output layer
       
        # Compile
        self.loss = 'mse'
        model.compile(loss=self.loss, optimizer ='adam')

        return model
            
    def get_traintestdata(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def training(self):
        # Create a callback that saves the model's weights
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
        #                                                 save_weights_only=True,
        #                                                 verbose=0)
        self.model = self.create_model()

        self.history = self.model.fit(self.traindata[0], self.traindata[1], 
                    validation_split= ANN['Training']['validation_size'],
                    epochs = ANN['Training']['epochs'] )

        self.model.save_weights(self.checkpoint_path+"cp.ckpt")

        
        print('MSE: %.3f (%.3f)' %(self.history.history['loss'][-1], self.history.history['val_loss'][-1]))

        return self.history.history['loss'][-1], self.history.history['val_loss'][-1]

    def trainingKFoldCross(self):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path+"cp.ckpt",
                                                        save_weights_only=True,
                                                        verbose=0)

        X, y = self.traindata
        
        results = list()

        # define evaluation procedure
        cv = RepeatedKFold(n_splits=ANN['Training']['n_splits'], 
                        n_repeats=ANN['Training']['n_repeats'], 
                        random_state=ANN['Training']['random_state'])

        # enumerate folds
        self.history = []
        for train_ix, test_ix in cv.split(X):
            # prepare data
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            # define model
            self.model = self.create_model()
            # fit model
            self.history.append( self.model.fit(X_train, y_train, verbose=2, 
                    epochs=ANN['Training']['epochs'],
                    callbacks=[cp_callback])    )
            # evaluate model on test set: mean absolute error
            mae = self.model.evaluate(X_test, y_test, verbose=0)
            # store result
            print(mae)
            results.append(mae)
        # return results

        print('MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))

        return np.mean(results), np.std(results)


    def plotTraining(self):
        colors = ['r-.','g-.','k-.','b-.','r-.','g-.','k-.','b-.','r-','g-','k-','b-','r--','g--','k.-','b.-']
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel(self.loss)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.grid(alpha = 0.5)
        plt.tight_layout()
        plt.savefig(self.checkpoint_path+"trainingloss.png", dpi = 100)
        plt.show()

    def plotTrainingKFold(self):
        # summarize history for loss
        colors = ['r-.','g-.','k-.','b-.','r-.','g-.','k-.','b-.','r-','g-','k-','b-','r--','g--','k.-','b.-']
        for i in range(len(self.history)):
            plt.plot(self.history[i].history['loss'], colors[i%len(colors)])
        # plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.savefig(self.checkpoint_path+"trainingloss_kfold.png", dpi = 100)
        # plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def load_model_fromFile(self):
        model_fromFile = self.create_model()
        model_fromFile.load_weights(self.checkpoint_path+"cp.ckpt").expect_partial()
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

                self.Output_pred = np.zeros((len(pred_test),3))
                for i in range(len(pred_test)):
                    print('i', i)
                    print("%e, %e, %e"%(pred_test[i,0], pred_test[i,1], pred_test[i,2]) )
                    print("%e, %e, %e"%(self.testdata[1][i,0], self.testdata[1][i,1], self.testdata[1][i,2] ))
                    print("------------------------")
                    self.Output_pred[i,0] = abs( pred_test[i,0] - self.testdata[1][i,0] )
                    self.Output_pred[i,1] = abs( pred_test[i,1] - self.testdata[1][i,1] )
                    self.Output_pred[i,2] = abs( pred_test[i,2] - self.testdata[1][i,2] )

            else:
                if Scaling['type_stand'] == 0:
                    predictions_unscaled, inputs_unscaled = self.dataset_np.commonInverseStandardization(pred_test, self.testdata[0]) #Obtain predictions in actual 
                    true_value, inputs_unscaled = self.dataset_np.commonInverseStandardization(self.testdata[1], self.testdata[0]) #Obtain predictions in actual
                elif Scaling['type_stand'] == 1:
                    predictions_unscaled = self.dataset_np.inverseStandardization(pred_test, typeR='E') #Obtain predictions in actual 
                    true_value = self.dataset_np.inverseStandardization(self.testdata[1], typeR='E') #Obtain predictions in actual
                elif Scaling['type_stand'] == 2:
                    predictions_unscaled = np.zeros(np.shape(self.testdata[1]))
                    true_value = np.zeros(np.shape(self.testdata[1]))

                    #Mf
                    predictions_unscaled[:,0] = self.dataset_np.inverseStandardization(pred_test[:,0], typeR='Mf') #Obtain predictions in actual 
                    true_value[:,0] = self.dataset_np.inverseStandardization(self.testdata[1][:,0], typeR='Mf') #Obtain predictions in actual
                    #Ep
                    predictions_unscaled[:,1] = self.dataset_np.inverseStandardization(pred_test[:,1], typeR='Ep') #Obtain predictions in actual 
                    true_value[:,1] = self.dataset_np.inverseStandardization(self.testdata[1][:,1], typeR='Ep') #Obtain predictions in actual
                    #Ev
                    predictions_unscaled[:,2] = self.dataset_np.inverseStandardization(pred_test[:,2], typeR='Ev') #Obtain predictions in actual 
                    true_value[:,2] = self.dataset_np.inverseStandardization(self.testdata[1][:,2], typeR='Ev') #Obtain predictions in actual


                self.Output_pred_unscale = np.zeros((len(pred_test),3)) 
                for i in range(len(predictions_unscaled)):
                    print('i', i)
                    print("Predictions, %e, %e, %e"%(predictions_unscaled[i,0], predictions_unscaled[i,1], predictions_unscaled[i,2]) )
                    print("True value, %e, %e, %e"%(true_value[i,0], true_value[i,1], true_value[i,2] ))
                    print("------------------------")
                    self.Output_pred_unscale[i,0] = abs( predictions_unscaled[i,0] - true_value[i,0 ])
                    self.Output_pred_unscale[i,1] = abs( predictions_unscaled[i,1] - true_value[i,1 ])
                    self.Output_pred_unscale[i,2] = abs( predictions_unscaled[i,2] - true_value[i,2 ])
        else:
            pred_test0 = self.model.predict(testfile)
            if rescale == True: # inverse standarization 
                pred_test = self.dataset_np.inverseStandardizationError(pred_test0) #Obtain predictions in actual
            else:
                pred_test = pred_test0
        return pred_test

    def plotPredictions(self, std):

        if std == False: # No inverse standarization possible or not needed
            fig, ax = plt.subplots() 
            plt.plot(np.arange(0, len(self.Output_pred)), self.Output_pred[:,0], 'r-x', label = 'Difference in mass of fuel')
            plt.plot(np.arange(0, len(self.Output_pred)), self.Output_pred[:,1], 'g-x', label = 'Difference in position error')
            plt.plot(np.arange(0, len(self.Output_pred)), self.Output_pred[:,2], 'b-x', label = 'Difference in velocity error')

            plt.xlabel("Samples to predict")
            plt.grid(alpha = 0.5)
            plt.legend()
            
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.subplots_adjust(wspace=0.1, hspace=0.05)
            ax1.plot(np.arange(0, len(self.Output_pred_unscale)), self.Output_pred_unscale[:,0], 'r-x', label = 'Difference in mass of fuel')
            ax2.plot(np.arange(0, len(self.Output_pred_unscale)), self.Output_pred_unscale[:,1], 'g-x', label = 'Difference in position error')
            ax3.plot(np.arange(0, len(self.Output_pred_unscale)), self.Output_pred_unscale[:,2], 'b-x', label = 'Difference in velocity error')
            
            if Scaling['scaling'] == 0:
                ax2.set_yscale('log')
                ax3.set_yscale('log')
            elif Scaling['scaling'] == 1:
                ax2.set_yscale('symlog')
                ax3.set_yscale('symlog')

            for ax in [ax1, ax2, ax3]:
                ax.grid(alpha = 0.5)
                ax.legend()
                ax.set_xlabel("Samples to predict")

        # plt.tight_layout()
        plt.savefig(self.checkpoint_path+"TestPredictionDifference_std" + str(std) +  ".png", dpi = 100)
        plt.show()

    def singlePrediction(self, input_case):
        print(input_case)
        input_batch = np.array([input_case])
        prediction = self.probability_model.predict(input_batch)
        print("Prediction", prediction, "predicted label", np.argmax(prediction))

    def printWeights(self):
        weights_h = list()
        bias_h = list()

        for layer in range(ANN['Architecture']['hidden_layers']):
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
    # train_file_path = "./databaseANN/DeltaCartesian_ErrorIncluded/trainingData_Feas_Lambert_big.txt"

    # Choose which ones to choose:
    base = "./databaseANN/DatabaseOptimized/deltakeplerian/"
    
    # Join files together into 1
    train_file_path = base +'Random.txt'
    # TD.join_files(file_path, train_file_path)


    # TD.plotInitialDataPandasError(train_file_path, save_file_path,  pairplot= True, corrplot= True)
    dataset_np = TD.LoadNumpy(train_file_path, base, error= 'vector',\
            equalize = False, \
            standardizationType = Scaling['type_stand'], scaling = Scaling['scaling'], \
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            labelType = False,
            plotDistribution=False, plotErrors=False)
    
    traindata, testdata = TD.splitData_reg(dataset_np)

    # sys.exit(0)
    ###############################################
    # CREATE AND TRAIN CLASS NETWORK
    ###############################################
    perceptron = ANN_reg(dataset_np)
    perceptron.get_traintestdata(traindata, testdata)
    perceptron.training()
    perceptron.plotTraining()


    
    print("EVALUATE")
    predictions = perceptron.predict(fromFile=True, rescale = False)
    perceptron.plotPredictions(std = False)
    print("Rescaled:")
    predictions = perceptron.predict(fromFile=True, rescale = True)
    perceptron.plotPredictions(std = True)

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

