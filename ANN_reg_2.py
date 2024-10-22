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
import GenerateTrainingDataFromOpt as GTD
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
            self.save_path = save_path
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
            inputs = testfile
        else:
            inputs = self.testdata[0]

        pred_test = self.model.predict(inputs)

        pred_test = np.abs(pred_test)
        # Apply absolute value so there are no negative values


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
            predictions_unscaled, inputs_unscaled = TD.commonInverseStandardization(pred_test, self.testdata[0], self.save_path) #Obtain predictions in actual 
            true_value, inputs_unscaled = TD.commonInverseStandardization(self.testdata[1], self.testdata[0], self.save_path) #Obtain predictions in actual
            
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
            predictions_unscaled, inputs_unscaled = TD.commonInverseStandardization(pred_test, testfile, self.save_path) #Obtain predictions in actual 
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




def loadDatabase(base):
    # file_path = [base+ 'Random.txt', base+ 'Random_opt_2.txt', base+ 'Random_opt_5.txt',\
    #     base+ 'Lambert_opt.txt']
    file_path = base+ 'Together.txt'
    # file_path = base+ 'Random_1000_eval.txt'

    dataset_np = TD.LoadNumpy(file_path, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], 
            Log = Dataset_conf.Dataset_config['Log'],
            output_type = Dataset_conf.Dataset_config['Outputs'],
            labelType = 3, 
            decV = True,
            plotDistribution=False, plotErrors=False,
            plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            # plotOutputDistr = True, plotEpvsEv = True,
            data_augmentation = Dataset_conf.Dataset_config['dataAugmentation']['type'])

    traindata, testdata = TD.splitData_reg(dataset_np, path = base)

    return dataset_np.dataset

def Network(perceptron, save_path):
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

def Fitness_network(base, train = False):
    
    # Load so that it is always the same (easier for comparisons)

    traindata_x = np.load(base+"1_CurrentLoadSave/traindata_x.npy")
    traindata_y = np.load(base+"1_CurrentLoadSave/traindata_y.npy")
    testdata_x = np.load(base+"1_CurrentLoadSave/testdata_x.npy")
    testdata_y = np.load(base+"1_CurrentLoadSave/testdata_y.npy")
    traindata = [traindata_x, traindata_y]
    testdata = [testdata_x, testdata_y]

    perceptron = ANN_reg(save_path =base)
    perceptron.get_traintestdata(traindata, testdata)

    if train == True:
        Network(perceptron, base)
    else:
        evaluatePredictionsNewData(base, perceptron)

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

def evaluatePredictionsNewData(base, ANN):
    ##################################################
    ####   DIFFERENCE IN NEW DATA ####################
    ##################################################
    ind = 500
    pop_0 = np.zeros([ind, len(SF.bnds)])
    for i in range(len(SF.bnds)):
        pop_0[:,i] = np.random.rand(ind) * (SF.bnds[i][1]-SF.bnds[i][0]) + SF.bnds[i][0]
    
    # Fitness function
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
    output_Vector = np.zeros((ind,2))
    for i in range(ind):
        DecV = pop_0[i,:]
        fitness = Fitness.calculateFitness(DecV)

        # Transform inputs
        input_Vector_i = Fitness.DecV2inputV('deltakeplerian', newDecV = DecV)
        print(input_Vector_i)
        input_Vector[i,:] = TD.standardize_withoutFitting(input_Vector_i, "I", base)

        output_Vector_i = np.array([Fitness.Epnorm_norm, Fitness.Evnorm_norm])
        output_Vector[i,:] = TD.standardize_withoutFitting(output_Vector_i, "O", base)

    t0_class_2 = time.time()

    # Feasibility
    feas2_unscaled = ANN.predict(fromFile = True, testfile = input_Vector, rescale = False)
    feas2 = ANN.predict(fromFile = True, testfile = input_Vector, rescale = True)
    tf_3 = (time.time() - t0_class) 
    tf_3_mid = ( t0_class_2 - t0_class) 

    difference1 = np.zeros((ind, 2))
    difference1 = np.abs(feas2 - feas1) 
    difference1_unscaled = np.abs(feas2_unscaled, output_Vector)


    ##################################################
    ####   DIFFERENCE IN TRAIN/TEST DATA #############
    ##################################################
    #Test predictions for database data
    traindata = ANN.traindata
    testdata = ANN.testdata

    feas_train1 = ANN.predict(fromFile = True, testfile = traindata[0], rescale = True)
    feas_train1_unscaled = ANN.predict(fromFile = True, testfile = traindata[0], rescale = False)
    feas_train2 = TD.commonInverseStandardization(traindata[1],
                                            traindata[0], base)[0] 

    difference2 = np.zeros((ind, 2))
    difference2 = np.abs(feas_train2 - feas_train1)
    difference2_unscaled = np.abs(traindata[1], feas_train1_unscaled)


    feas_test1 = ANN.predict(fromFile = True, testfile = testdata[0], rescale = True)
    feas_test1_unscaled = ANN.predict(fromFile = True, testfile = testdata[0], rescale = False)
    feas_test2 = TD.commonInverseStandardization(testdata[1],
                                            testdata[0], base)[0]

    difference3 = np.zeros((ind, 2))
    difference3 = np.abs(feas_test2 - feas_test1) 
    difference3_unscaled = np.abs(testdata[1], feas_test1_unscaled)

    displaynumber = 0
    for i in range(displaynumber):
        print("Predicted", feas2[i,:], "Fitness",feas1[i, :])
    
    for i in range(displaynumber):
        print("Predicted", feas_train1, "Database", feas_train2)

    for i in range(displaynumber):
        print("Predicted", feas_test1, "Database", feas_test2)


    # PLOT DIFFERENCES
    labels = 'Difference in predicted' 
    symbols = ['r-x', 'g-x', 'b-x']

    fig, (ax1) = plt.subplots(1, 1)
    ax_i = [ax1]
    
    fig.subplots_adjust(wspace=0.1, hspace=0.05)
    for i in range(1):

        ax_i[i].scatter(difference2_unscaled[:,0], difference2_unscaled[:,1], color = 'green', marker =  'x', label = 'traindata')
        ax_i[i].scatter(difference3_unscaled[:,0], difference3_unscaled[:,1], color = 'orange', marker =  'x', label = 'testdata')
        ax_i[i].scatter(difference1_unscaled[:,0], difference1_unscaled[:,1], color = 'black', marker =  'x', label = 'newdata')

        ax_i[i].set_yscale('log')
        ax_i[i].set_xscale('log')

        ax_i[i].set_xlabel("Difference in prediction position")
        ax_i[i].set_ylabel("Difference in prediction velocity")
        ax_i[i].grid(alpha = 0.5)
        ax_i[i].legend()
        

    # plt.tight_layout()
    plt.savefig(ANN.checkpoint_path+"TestPredictionDifference_std_newdata.png", dpi = 100)
    plt.show()

    fig.subplots_adjust(wspace=0.1, hspace=0.05)
    for i in range(1):
        ax_i[i].scatter(difference2[:,0], difference2[:,1], color = 'green', marker =  'x', label = 'traindata')
        ax_i[i].scatter(difference3[:,0], difference3[:,1], color = 'orange', marker =  'x', label = 'testdata')
        ax_i[i].scatter(difference1[:,0], difference1[:,1], color = 'black', marker =  'x', label = 'newdata')

        ax_i[i].set_yscale('log')
        ax_i[i].set_xscale('log')

        ax_i[i].set_xlabel("Difference in prediction position")
        ax_i[i].set_ylabel("Difference in prediction velocity")
        ax_i[i].grid(alpha = 0.5)
        ax_i[i].legend()
        

    # plt.tight_layout()
    plt.savefig(ANN.checkpoint_path+"TestPredictionDifference_newdata.png", dpi = 100)
    plt.show()


    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax_i = [ax1, ax2]
    
    fig.subplots_adjust(wspace=0.1, hspace=0.05)
    for i in range(2):
        ax_i[i].scatter(feas_train2[:,i], feas_train1[:,i], color = 'green', marker =  'x', label = 'traindata')
        ax_i[i].scatter(feas_test2[:,i], feas_test1[:,i], color = 'orange', marker =  'x', label = 'testdata')
        ax_i[i].scatter(feas1[:,i], feas2[:,i], color = 'black', marker =  'x', label = 'newdata')

        ax_i[i].plot([0,0], [max(feas_train2[:,i]), max(feas_train2[:,i])])
        ax_i[i].set_yscale('log')
        ax_i[i].set_xscale('log')

        ax_i[i].grid(alpha = 0.5)
        ax_i[i].legend()
        ax_i[i].set_xlabel("Real value")
        ax_i[i].set_ylabel("Predicted value")



    # plt.tight_layout()
    plt.savefig(ANN.checkpoint_path+"TestPredictionDifference_std_newdata_real.png", dpi = 100)
    plt.show()




    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax_i = [ax1, ax2]
    
    fig.subplots_adjust(wspace=0.1, hspace=0.05)
    for i in range(2):
        ax_i[i].scatter(feas_train2[:,i], difference2[:,i], color = 'green', marker =  'x', label = 'traindata')
        ax_i[i].scatter(feas_test2[:,i], difference3[:,i], color = 'orange', marker =  'x', label = 'testdata')
        ax_i[i].scatter(feas1[:,i], difference1[:,i], color = 'black', marker =  'x', label = 'newdata')

        ax_i[i].set_yscale('log')
        ax_i[i].set_xscale('log')

        ax_i[i].grid(alpha = 0.5)
        ax_i[i].legend()
        ax_i[i].set_xlabel("Real value")
        ax_i[i].set_ylabel("Difference in unscaled prediction")



    # plt.tight_layout()
    plt.savefig(ANN.checkpoint_path+"TestPredictionDifference_std_newdata_2.png", dpi = 100)
    plt.show()

     # As a pde distribution

    # dataset = pd.DataFrame(data = np.log10(difference)) 
    # sns.displot(dataset)
    # # plt.legend(self.output_label, loc='upper left')
    # plt.tight_layout()
    # plt.savefig(ANN.checkpoint_path+"TestPredictionDifference_std_pd_newdata_pd.png", dpi = 100)
    # plt.show()


def checkDatabase(database):
    """
    Check that database provides same fitness as evaluation
    """
    base =  "./databaseANN/4_DatabaseTest_repit/deltakeplerian/"

    # Test 1
    ind = 500
    dataset1 = GTD.latinhypercube(len(SF.bnds), len(SF.bnds), 500)
    
    pop_0 = np.zeros([ind, len(SF.bnds)])
    for i in range(len(SF.bnds)):
        pop_0[:,i] = np.random.rand(ind) * (SF.bnds[i][1]-SF.bnds[i][0]) + SF.bnds[i][0]
    
        # plt.scatter(np.arange(0,ind,1),pop_0[:,i], color = 'red')
        # plt.scatter(np.arange(0,ind,1), dataset1[:,i], color = 'blue')
        # plt.show()
    # for i in range(len(SF.bnds)):
        # print(i, max(dataset1[:,i]), max(pop_0[:,i]))

    # plots equally distributed
    # same ranges

    # # Test2 
    # for i in range(1):
    #     fit1 = Fitness.calculateFitness(dataset1[i,:])
    #     v1 = Fitness.DecV2inputV('deltakeplerian', )
    #     v2 = Fitness.DecV2inputV('deltakeplerian', newDecV=dataset1[i,:]) 
    #     print(v1)
    #     print(v2)

    # # It works




    traindata_x = np.load(base+"1_CurrentLoadSave/traindata_x.npy")
    traindata_y = np.load(base+"1_CurrentLoadSave/traindata_y.npy")

    decV_data = database[:,16:-1]

    # # Test3
    # for i in range(1):
    #     fit1 = Fitness.calculateFitness(dataset1[i,:])
    #     v1 = Fitness.DecV2inputV('deltakeplerian', )
    #     v2 = Fitness.DecV2inputV('deltakeplerian', newDecV=dataset1[i,:]) 
    #     print(decV_data[i,:])
    #     print(v2)

    # Test4
    for i in range(6):
        DecV = decV_data[i,:]
        input_Vector_i = Fitness.DecV2inputV('deltakeplerian', newDecV = DecV)
        print(input_Vector_i)
        input_Vector_j = TD.standardize_withoutFitting(input_Vector_i, "I", base)

        print("Traindata", traindata_x[i,:])
        print("New _data", input_Vector_j)
        print('===========================')


        Fitness.calculateFitness(DecV)
        Ep = Fitness.Epnorm_norm
        Ev = Fitness.Evnorm_norm
        E = np.array([Ep, Ev])
        Er = TD.standardize_withoutFitting(E, "O", base)
        print(Er)
        print(traindata_y[i])
        print('===========================')

    # perceptron = ANN_reg(save_path =base)
    # perceptron.load_model_fromFile()

    # input_Vector = traindata[0][0:10,:]
    # # Feasibility
    # feas2_unscaled = perceptron.predict(testfile = input_Vector, rescale = False)
    # # feas2 = perceptron.predict(testfile = input_Vector, rescale = True)

    # print(feas2_unscaled[0:10, :])

    # result = dataset_np.commonInverseStandardization(traindata[1][0:10,:], traindata[0][0:10,:])
    # print(traindata[1][0:10,:])
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
    base = "./databaseANN/4_DatabaseTest_repit/deltakeplerian/"
    # dataset= loadDatabase(base)
    # Fitness_network(base, train = True)
    Fitness_network(base, train = False)

    # checkDatabase(dataset)

    # Fitness_network_join()
    # Opt_network()