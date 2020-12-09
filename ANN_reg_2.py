import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold

import LoadConfigFiles as CONF
import TrainingData as TD

###################################################################
# https://stackabuse.com/tensorflow-2-0-solving-classification-and-regression-problems/
# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
###################################################################

ANN_reg = CONF.ANN_reg()
ANN = ANN_reg.ANN_config

Dataset_conf = CONF.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']

FIT_C = CONF.Fitness_config()
FIT = FIT_C.Fit_config

class ANN_reg:
    def __init__(self, dataset, save_path = False):
        self.dataset_np = dataset

        self.n_classes = int(dataset.n_outputs) # Outputs
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

        if save_path == False:
            self.checkpoint_path = "./trainedCNet_Reg/training/"+Dataset_conf.Dataset_config['Creation']['typeoutputs']+'/'+Dataset_conf.Dataset_config['Outputs']+'/'
        else:
            self.checkpoint_path = save_path +"./trainedCNet_Reg/"+Dataset_conf.Dataset_config['Outputs']+'/'

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
        opt = keras.optimizers.Adam(learning_rate=ANN['Training']['learning_rate'])
        model.compile(loss=self.loss, optimizer =opt)

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
        cv = RepeatedKFold(n_splits=ANN['TrainingKfold']['n_splits'], 
                        n_repeats=ANN['TrainingKfold']['n_repeats'], 
                        random_state=ANN['TrainingKfold']['random_state'])

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
                    epochs=ANN['TrainingKfold']['epochs'],
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
        
        f = plt.figure()
        ax = f.add_subplot(111)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])

        text = "Train loss = %e\n Validation loss = %e"%(self.history.history['loss'][-1], self.history.history['val_loss'][-1]) 
        plt.text(0.5, 0.5, text, horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes)

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
        else:
            pred_test = self.model.predict(testfile)

        if rescale == False and type(testfile) == bool: # No inverse standarization possible

            self.Output_pred = np.zeros((len(pred_test),self.n_classes))
            if self.n_classes > 1:
                for i in range(len(pred_test)):
                    print('i', i)
                    print(pred_test[i], self.testdata[1][i])
                    # print("%e, %e, %e"%(pred_test[i,0], pred_test[i,1], pred_test[i,2]) )
                    # print("%e, %e, %e"%(self.testdata[1][i,0], self.testdata[1][i,1], self.testdata[1][i,2] ))
                    print("------------------------")

                    for output_i in range(self.n_classes):
                        self.Output_pred[i,output_i] = abs( pred_test[i,output_i] - self.testdata[1][i,output_i] )
            else:
                for i in range(len(pred_test)):
                    print(pred_test[i], self.testdata[1][i])
                    self.Output_pred[i,0] = abs( pred_test[i] - self.testdata[1][i])

        else:
            predictions_unscaled, inputs_unscaled = self.dataset_np.commonInverseStandardization(pred_test, self.testdata[0]) #Obtain predictions in actual 
            true_value, inputs_unscaled = self.dataset_np.commonInverseStandardization(self.testdata[1], self.testdata[0]) #Obtain predictions in actual
            
            if type(testfile) == bool:
                self.Output_pred_unscale = np.zeros((len(pred_test),self.n_classes)) 
                for i in range(len(predictions_unscaled)):
                    print('i', i)
                    # print("Predictions, %e, %e, %e"%(predictions_unscaled[i,0], predictions_unscaled[i,1], predictions_unscaled[i,2]) )
                    # print("True value, %e, %e, %e"%(true_value[i,0], true_value[i,1], true_value[i,2] ))
                    print("------------------------")
                    for output_i in range(self.n_classes):
                        self.Output_pred_unscale[i,output_i] = abs( predictions_unscaled[i,output_i] - true_value[i,output_i ])
                        print(predictions_unscaled[i,output_i], true_value[i,output_i ])
    
        return pred_test

    def plotPredictions(self, std):

        labels = ['Difference in mass of fuel', 'Difference in position error', 'Difference in velocity error']
        symbols = ['r-x', 'g-x', 'b-x']
        if std == False: # No inverse standarization possible or not needed
            fig, ax = plt.subplots() 
            
            for i in range(self.n_classes):
                plt.plot(np.arange(0, len(self.Output_pred)), self.Output_pred[:,i], symbols[i], label = labels[i])

            plt.xlabel("Samples to predict")
            plt.grid(alpha = 0.5)
            plt.legend()
            
        else:
            if self.n_classes == 3:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax_i = [ax1, ax2, ax3]
            elif self.n_classes == 2:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax_i = [ax1, ax2]
            elif self.n_classes == 1:
                fig, (ax1) = plt.subplots(1, 1)
                ax_i = [ax1]
            
            fig.subplots_adjust(wspace=0.1, hspace=0.05)
            for i in range(self.n_classes):
                ax_i[i].plot(np.arange(0, len(self.Output_pred_unscale)), self.Output_pred_unscale[:,i], symbols[i], label = labels[i])

            if Dataset_conf.Dataset_config['Outputs'] == 'epev' or\
               Dataset_conf.Dataset_config['Outputs'] == 'ep' or\
               Dataset_conf.Dataset_config['Outputs'] == 'ev':
                for ax in ax_i:
                    if Scaling['scaling'] == 0:
            #         for i in range(self.n_classes -1 )
                        ax.set_yscale('log')
                    elif Scaling['scaling'] == 1:
                        ax.set_yscale('symlog')

            if Dataset_conf.Dataset_config['Outputs'] == 'epevmf':
                for ax in ax_i[1:]:
                    if Scaling['scaling'] == 0:
            #         for i in range(self.n_classes -1 )
                        ax.set_yscale('log')
                    elif Scaling['scaling'] == 1:
                        ax.set_yscale('symlog')


            #         ax3.set_yscale('log')
            #     elif Scaling['scaling'] == 1:

            # if self.n_classes == 2 or self.n_classes == 3:
            #     if Scaling['scaling'] == 0:
            #         for i in range(self.n_classes -1 )
            #         ax2.set_yscale('log')
            #         ax3.set_yscale('log')
            #     elif Scaling['scaling'] == 1:
            #         ax2.set_yscale('symlog')
            #         ax3.set_yscale('symlog')

            for ax in ax_i:
                ax.grid(alpha = 0.5)
                ax.legend()
                ax.set_xlabel("Samples to predict")

        # plt.tight_layout()
        plt.savefig(self.checkpoint_path+"TestPredictionDifference_std" + str(std) +  ".png", dpi = 100)
        plt.show()


        if std == False:
            dataset = pd.DataFrame(data = np.log10(self.Output_pred)) 
        else:
            dataset = pd.DataFrame(data = np.log10(self.Output_pred_unscale)) 
        sns.displot(dataset)
        # plt.legend(self.output_label, loc='upper left')
        plt.tight_layout()
        plt.savefig(self.checkpoint_path+"TestPredictionDifference_std" + str(std) +  "_pd.png", dpi = 100)
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

def Network(dataset_np, save_path):
    """
    Call the network to train and evaluate
    """
    traindata, testdata = TD.splitData_reg(dataset_np)

    perceptron = ANN_reg(dataset_np, save_path = save_path)
    perceptron.get_traintestdata(traindata, testdata)
    
    perceptron.training()
    perceptron.plotTraining()
    
    # perceptron.trainingKFoldCross()
    # perceptron.plotTrainingKFold()
    
    print("EVALUATE")
    predictions = perceptron.predict(fromFile=True, rescale = False)
    perceptron.plotPredictions(std = False)
    print("Rescaled:")
    predictions = perceptron.predict(fromFile=True, rescale = True)
    perceptron.plotPredictions(std = True)

def Fitness_network():
    base = "./databaseANN/DatabaseFitness/deltakeplerian/"

    # file_path = [base+ 'Random.txt', base+ 'Random_opt_2.txt', base+ 'Random_opt_5.txt',\
    #     base+ 'Lambert_opt.txt']
    file_path = base+ '500/Random_MBH_3.txt'
    # [
                # base+ '500/Random_MBH_eval.txt', 
                
                # base+ '1000/Random_MBH_eval.txt', base+ '1000/Random_MBH_3.txt',
                # base+ '5000/Random_MBH_eval.txt', base+ '5000/Random_MBH_3.txt',
                # base+'Lambert_eval.txt', base+'Lambert.txt'
                # ]


    # file_path = [base+ 'Random.txt',  base+ 'Random_opt_5.txt']
    
    train_file_path = file_path
    # train_file_path = base +'Random.txt'

    dataset_np = TD.LoadNumpy(train_file_path, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            plotDistribution=False, plotErrors=False,
            # plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            plotOutputDistr = True, plotEpvsEv = False,
            data_augmentation = Dataset_conf.Dataset_config['dataAugmentation']['type'])

    
    Network(dataset_np, base)

def Fitness_network_optDecV():
    base = "./databaseANN/DatabaseBestDeltaV/deltakeplerian/"

    # file_path = [base+ 'Random.txt', base+ 'Random_opt_2.txt', base+ 'Random_opt_5.txt',\
    #     base+ 'Lambert_opt.txt']
    file_path = [base + 'Random1000_MBH_eval.txt',
                base + 'Random1000_MBH.txt',
                base + 'Random_1000.txt',
                base + 'Random_500.txt'
                # base + 'OthersFromFitness/Random_MBH5000_3.txt',
                # base + 'OthersFromFitness/Random_MBH1000_3.txt',
                
                # base + 'OthersFromFitness/Random_MBH500_3.txt', 
                # base + 'OthersFromFitness/Random_MBH500_5.txt', 
                # base + 'OthersFromFitness/Lambert_3.txt'
                ]



    # file_path = [base+ 'Random.txt',  base+ 'Random_opt_5.txt']
    
    train_file_path = file_path
    # train_file_path = base +'Random.txt'

    dataset_np = TD.LoadNumpy(train_file_path, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            plotDistribution=False, plotErrors=False,
            # plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            plotOutputDistr = True, plotEpvsEv = True,
            data_augmentation = Dataset_conf.Dataset_config['dataAugmentation']['type'])

    
    Network(dataset_np, base)

def Opt_network():
    base = "./databaseANN/DatabaseOptimized/deltakeplerian/"

    train_file_path = base +'Random.txt'


    dataset_np = TD.LoadNumpy(train_file_path, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': [1], 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            plotDistribution=True, plotErrors=True,
            plotOutputDistr = True, plotEpvsEv = True,
            # data_augmentation = Dataset_conf.Dataset_config['dataAugmentation'])
                data_augmentation= Dataset_conf.Dataset_config['dataAugmentation']['type'])
    
    Network(dataset_np, base)


if __name__ == "__main__":
    # Fitness_network()

    Fitness_network_optDecV()
    # Opt_network()