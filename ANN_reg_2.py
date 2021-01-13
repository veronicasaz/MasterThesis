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

from FitnessFunction_normalized import Fitness
import LoadConfigFiles as CONF
import TrainingData as TD

import time

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

SF = CONF.SimsFlan_config() # Load Sims-Flanagan config variables 
Fitness = Fitness(Nimp = SF.Nimp)

class ANN_reg:
    def __init__(self, save_path = False, n_classes = 2, n_input = 8, output_label = ['ep', 'ev']):

        self.n_classes = n_classes # Outputs
        self.n_input = n_input #inputs
        self.output_label = output_label #labels for the outputs

        if save_path == False:
            self.checkpoint_path = "./trainedCNet_Reg/training/"+Dataset_conf.Dataset_config['Creation']['typeoutputs']+'/'+Dataset_conf.Dataset_config['Outputs']+'/'
        else:
            self.checkpoint_path = save_path +"trainedCNet_Reg/"+Dataset_conf.Dataset_config['Outputs']+'/'

    def create_model(self):
        # Create architecture
        if ANN['Training']['initializer'] == 1:
            initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        else:
            def my_init(shape, dtype=None):            
                return np.random.randn(shape[0], shape[1]) * np.sqrt(2/(shape[1]))

            initializer = my_init

        model = keras.Sequential()

        if ANN['Training']['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            for layer in range(ANN['Architecture']['hidden_layers']):
                
                model.add(keras.layers.Dense(
                    ANN['Architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN['Training']['regularizer_value']) ))
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

        if ANN['Training']['learning_rate'] == 'variable':
            # exponential
            # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            #         initial_learning_rate=ANN['Training']['lr_i'],
            #         decay_steps=ANN['Training']['lr_decaysteps'],
            #         decay_rate=ANN['Training']['lr_decayrate'])

            # polinomial
            # lr_schedule = keras.optimizers.schedules.PolynomialDecay(
            #         initial_learning_rate=ANN['Training']['lr_i'],
            #         decay_steps=ANN['Training']['lr_decaysteps'],
            #         end_learning_rate=ANN['Training']['lr_f'],
            #         power=1.0)

            # time based: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/#:~:text=Keras%20has%20a%20time%2Dbased%20learning%20rate%20schedule%20built%20in.&text=When%20the%20decay%20argument%20is,effect%20on%20the%20learning%20rate.&text=When%20the%20decay%20argument%20is%20specified%2C%20it%20will%20decrease%20the,by%20the%20given%20fixed%20amount
            lr_schedule = ANN['Training']['lr_i']
            decay_rate = ANN['Training']['lr_decayrate']

        else:
            lr_schedule = ANN['Training']['learning_rate']
            decay_rate = 0

        # opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, 
                                    decay = decay_rate )
                                    # clipnorm=ANN['Training']['clipnorm'])

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


        self.model.save(self.checkpoint_path + "model.h5")

        self.model.summary()        
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
        plt.yscale('log')
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

        self.model = keras.models.load_model(self.checkpoint_path+"model.h5")
        
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

        if type(testfile) != bool:
            pred_test = self.model.predict(testfile)
        else:
            pred_test = self.model.predict(self.testdata[0])


        # Rescale
        if rescale == False and type(testfile) == bool: # No inverse standarization possible

            self.Output_pred = np.zeros((len(pred_test),self.n_classes))
            if self.n_classes > 1:
                for i in range(len(pred_test)):
                    print('i', i)
                    print(pred_test[i, :], self.testdata[1][i, :])
                    print("------------------------")

                    for output_i in range(self.n_classes):
                        self.Output_pred[i,output_i] = abs( pred_test[i,output_i] - self.testdata[1][i,output_i] )
            else:
                for i in range(len(pred_test)):
                    print(pred_test[i], self.testdata[1][i])
                    self.Output_pred[i,0] = abs( pred_test[i] - self.testdata[1][i])
            
            return pred_test

        elif rescale == True and type(testfile) == bool:
            predictions_unscaled, inputs_unscaled = TD.commonInverseStandardization(pred_test, self.testdata[0], self.checkpoint_path) #Obtain predictions in actual 
            true_value, inputs_unscaled = TD.commonInverseStandardization(self.testdata[1], self.testdata[0], self.checkpoint_path) #Obtain predictions in actual
            
            self.Output_pred_unscale = np.zeros((len(pred_test),self.n_classes)) 
            self.Output_pred_unscale_ptg = np.zeros((len(pred_test),self.n_classes)) 
            
            for i in range(len(predictions_unscaled)):
                print('i', i)
                # print("Predictions, %e, %e, %e"%(predictions_unscaled[i,0], predictions_unscaled[i,1], predictions_unscaled[i,2]) )
                # print("True value, %e, %e, %e"%(true_value[i,0], true_value[i,1], true_value[i,2] ))
                print("------------------------")
                for output_i in range(self.n_classes):
                    self.Output_pred_unscale[i,output_i] = abs( predictions_unscaled[i,output_i] - true_value[i,output_i ])
                    self.Output_pred_unscale_ptg[i,output_i] = abs( predictions_unscaled[i,output_i] - true_value[i,output_i] ) /true_value[i,output_i]
                    print(predictions_unscaled[i,output_i], true_value[i,output_i ])

            return predictions_unscaled

        elif rescale == True and type(testfile) != bool:
            predictions_unscaled, inputs_unscaled = TD.commonInverseStandardization(pred_test, testfile,self.checkpoint_path) #Obtain predictions in actual 
            return  predictions_unscaled

        else:
            return pred_test

    def plotPredictions(self, std):

        labels = 'Difference in predicted' 
        symbols = ['r-x', 'g-x', 'b-x']
        if std == False: # No inverse standarization possible or not needed
            fig, ax = plt.subplots() 
            
            for i in range(self.n_classes):
                plt.plot(np.arange(0, len(self.Output_pred)), self.Output_pred[:,i], symbols[i], label = labels+ self.output_label[i])

            plt.xlabel("Samples to predict")
            plt.grid(alpha = 0.5)
            plt.legend()
            
        else:
        #     if self.n_classes == 3:
        #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #         ax_i = [ax1, ax2, ax3]
        #     elif self.n_classes == 2:
        #         fig, (ax1, ax2) = plt.subplots(1, 2)
        #         ax_i = [ax1, ax2]
        #     elif self.n_classes == 1:
        #         fig, (ax1) = plt.subplots(1, 1)
        #         ax_i = [ax1]
            
        #     fig.subplots_adjust(wspace=0.1, hspace=0.05)
        #     for i in range(self.n_classes):
        #         ax_i[i].scatter(np.arange(0, len(self.Output_pred_unscale)), self.Output_pred_unscale[:,i], marker =  'x', label = labels+ self.dataset_np.output_label[i])

        #     if Dataset_conf.Dataset_config['Outputs'] == 'epev' or\
        #        Dataset_conf.Dataset_config['Outputs'] == 'ep' or\
        #        Dataset_conf.Dataset_config['Outputs'] == 'ev':
        #         for ax in ax_i:
        #             if Scaling['scaling'] == 0:
        #     #         for i in range(self.n_classes -1 )
        #                 ax.set_yscale('log')
        #             elif Scaling['scaling'] == 1:
        #                 ax.set_yscale('symlog')

        #     if Dataset_conf.Dataset_config['Outputs'] == 'epevmf':
        #         for ax in ax_i[1:]:
        #             if Scaling['scaling'] == 0:
        #     #         for i in range(self.n_classes -1 )
        #                 ax.set_yscale('log')
        #             elif Scaling['scaling'] == 1:
        #                 ax.set_yscale('symlog')


        #     #         ax3.set_yscale('log')
        #     #     elif Scaling['scaling'] == 1:

        #     # if self.n_classes == 2 or self.n_classes == 3:
        #     #     if Scaling['scaling'] == 0:
        #     #         for i in range(self.n_classes -1 )
        #     #         ax2.set_yscale('log')
        #     #         ax3.set_yscale('log')
        #     #     elif Scaling['scaling'] == 1:
        #     #         ax2.set_yscale('symlog')
        #     #         ax3.set_yscale('symlog')

        #     for ax in ax_i:
        #         ax.grid(alpha = 0.5)
        #         ax.legend()
        #         ax.set_xlabel("Samples to predict")

        # # plt.tight_layout()
        # plt.savefig(self.checkpoint_path+"TestPredictionDifference_std" + str(std) +  ".png", dpi = 100)
        # plt.show()

            # Ep vs Ev
            fig, (ax1) = plt.subplots(1, 1)
            ax_i = ax1
            ax_i.scatter(self.Output_pred_unscale[:,0], self.Output_pred_unscale[:,1], marker =  'x')
            
            ax_i.set_yscale('log')
            ax_i.set_xscale('log')

            ax_i.grid(alpha = 0.5)
            ax_i.set_xlabel("Samples to predict")

            plt.tight_layout()
            plt.savefig(self.checkpoint_path+"TestPredictionDifference_std" + str(std) +  ".png", dpi = 100)
            plt.show()

        # As a pde distribution
        if std == False:
            dataset = pd.DataFrame(data = np.log10(self.Output_pred)) 
        else:
            dataset = pd.DataFrame(data = np.log10(self.Output_pred_unscale_ptg)) 
        sns.displot(dataset)
        # plt.legend(self.output_label, loc='upper left')
        plt.tight_layout()
        plt.savefig(self.checkpoint_path+"TestPredictionDifference_std" + str(std) +  "_pd.png", dpi = 100)
        plt.show()

    def singlePrediction(self, input_case, fromFile = False):
        if fromFile == True:
            self.load_model_fromFile()
        # input_batch = np.array([input_case])
        input_batch = input_case.reshape(1, -1)
        prediction = self.model.predict(input_batch)
        return prediction[0,:]

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



def Network(dataset_np, perceptron, save_path):
    """
    Call the network to train and evaluate
    """
    
    perceptron.training()
    perceptron.plotTraining()
    
    # perceptron.trainingKFoldCross()
    # perceptron.plotTrainingKFold()
    
    print("EVALUATE")
    # rescale = False
    # predictions = perceptron.predict(fromFile=True, rescale = rescale)
    # perceptron.plotPredictions(std = rescale)
    print("Rescaled:")
    rescale = True
    predictions = perceptron.predict(fromFile=True, rescale = rescale)
    perceptron.plotPredictions(std = rescale)

def Fitness_network(train = False):
    base = "./databaseANN/3_DatabaseLast/deltakeplerian/"
    # file_path = [base+ 'Random.txt', base+ 'Random_opt_2.txt', base+ 'Random_opt_5.txt',\
    #     base+ 'Lambert_opt.txt']
    file_path = base+ 'Together.txt'

    dataset_np = TD.LoadNumpy(file_path, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],
            output_type = Dataset_conf.Dataset_config['Outputs'],
            labelType = 3, 
            decV = False,
            plotDistribution=False, plotErrors=False,
            plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            # plotOutputDistr = True, plotEpvsEv = True,
            data_augmentation = Dataset_conf.Dataset_config['dataAugmentation']['type'])

    traindata, testdata = TD.splitData_reg(dataset_np)

    perceptron = ANN_reg(dataset_np, save_path =base)
    perceptron.get_traintestdata(traindata, testdata)

    if train == True:
        Network(dataset_np, perceptron, base)
    else:
        evaluatePredictionsNewData(dataset_np, perceptron)

def Fitness_network_join():
    base = "./databaseANN/3_DatabaseLast/deltakeplerian/"

    file_path = [
                base+ 'databaseSaved_fp100/Random_MBH_1000_eval.txt',
                base+ 'databaseSaved_fp10/Random_MBH_5000_3.txt',
                base+ 'databaseSaved_fp100/Random_MBH_5000.txt'
                ]

    
    file_path_together = base +'Together.txt'
    TD.join_files(file_path, file_path_together)
    
    train_file_path = file_path_together
    # train_file_path = base+ 'databaseSaved/Random_MBH_5000_3.txt'

    dataset_np = TD.LoadNumpy(train_file_path, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            labelType = len(file_path), 
            plotDistribution=False, plotErrors=False,
            plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            # plotOutputDistr = True, plotEpvsEv = True,
            data_augmentation = Dataset_conf.Dataset_config['dataAugmentation']['type'])

    
    Network(dataset_np, base)


    # PREDICT WITHOUT GAUSSIAN NOISE

def evaluatePredictionsNewData(dataset_np, ANN):
    ind = 20
    pop_0 = np.zeros([ind, len(SF.bnds)])
    for i in range(len(SF.bnds)):
        pop_0[:,i] = np.random.rand(ind) * (SF.bnds[i][1]-SF.bnds[i][0]) + SF.bnds[i][0]
    
    Fitness function
    feas1 = np.zeros((ind, 2))
    t0_fit = time.time()
    for i in range(ind):
        DecV = pop_0[i,:]
        fitness = Fitness.calculateFitness(DecV)
        feas1[i, 0] = Fitness.Epnorm
        feas1[i, 1] = Fitness.Evnorm
    tf_1 = (time.time() - t0_fit) 
    

    # ANN batch
    feas2 = np.zeros((ind, 2))
    t0_class = time.time()
    input_Vector = np.zeros((ind,8))
    for i in range(ind):
        DecV = pop_0[i,:]

        # Transform inputs
        input_Vector_i = Fitness.DecV2inputV('deltakeplerian', newDecV = DecV)
        input_Vector[i,:] = dataset_np.standardize_withoutFitting(input_Vector_i, 'I')

    t0_class_2 = time.time()


    # Feasibility
    feas2_unscaled = ANN.predict(fromFile = True, testfile = input_Vector, rescale = False)
    feas2 = ANN.predict(fromFile = True, testfile = input_Vector, rescale = True)
    tf_3 = (time.time() - t0_class) 
    tf_3_mid = ( t0_class_2 - t0_class) 

    difference = np.zeros((ind, 2))
    print(feas2[0:10, :])
    print(feas1[0:10, :])
    difference = np.absolute(feas2 - feas1) 

    print(difference[0:10, :])
    # PLOT DIFFERENCES
    labels = 'Difference in predicted' 
    symbols = ['r-x', 'g-x', 'b-x']

    fig, (ax1) = plt.subplots(1, 1)
    ax_i = [ax1]
    
    fig.subplots_adjust(wspace=0.1, hspace=0.05)
    for i in range(1):
        ax_i[i].scatter(difference[:,0], difference[:,1], marker =  'x', label = labels+ dataset_np.output_label[i])
        
        ax_i[i].set_yscale('log')
        ax_i[i].set_xscale('log')

        ax_i[i].grid(alpha = 0.5)
        ax_i[i].legend()
        ax_i[i].set_xlabel("Samples to predict")

    # plt.tight_layout()
    plt.savefig(ANN.checkpoint_path+"TestPredictionDifference_std_newdata.png", dpi = 100)
    plt.show()

     # As a pde distribution

    # dataset = pd.DataFrame(data = np.log10(difference)) 
    # sns.displot(dataset)
    # # plt.legend(self.output_label, loc='upper left')
    # plt.tight_layout()
    # plt.savefig(ANN.checkpoint_path+"TestPredictionDifference_std_pd_newdata_pd.png", dpi = 100)
    # plt.show()


def checkDatabase():
    """
    Check that database provides same fitness as evaluation
    """
    base = "./databaseANN/3_DatabaseLast/deltakeplerian/"
    

    perceptron = ANN_reg(save_path =base)
    perceptron.load_model_fromFile()

    input_Vector = traindata[0][0:10,:]
    # Feasibility
    feas2_unscaled = perceptron.predict(testfile = input_Vector, rescale = False)
    # feas2 = perceptron.predict(testfile = input_Vector, rescale = True)

    print(feas2_unscaled[0:10, :])

    result = dataset_np.commonInverseStandardization(traindata[1][0:10,:], traindata[0][0:10,:])
    print(traindata[1][0:10,:])
    # print(result[0])
    #  traindata[1][0:10,:])

    # ind = 2
    # pop = traindata[0][0:ind, :]
    # popdecv = dataset_np.decV[0:ind, :]

    # pop_out, non= dataset_np.commonInverseStandardization(traindata[1][0:ind,:], pop)

    # # FITNESS
    # feas1 = np.zeros((ind, 2))
    # t0_fit = time.time()
    # for i in range(ind):
    #     DecV = popdecv[i,:]
    #     fitness = Fitness.calculateFitness(DecV)
    #     feas1[i, 0] = Fitness.Epnorm
    #     feas1[i, 1] = Fitness.Evnorm

    #     print("=========== ============")
    #     print("Fitness", feas1[i,:])
    #     print("Dataset", pop_out[i])

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
    # Fitness_network(train = True)
    # Fitness_network(train = False)

    checkDatabase()

    # Fitness_network_join()
    # Opt_network()