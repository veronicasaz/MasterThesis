import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil

import AstroLibraries.AstroLib_Basic as AL_BF 
import LoadConfigFiles as CONF

ANN = CONF.ANN_reg()
ANN_train = ANN.ANN_train
ANN_datab = ANN.ANN_datab

FIT = CONF.Fitness_config()
FIT_FEAS = FIT.FEAS

###################################################################
# https://deeplizard.com/learn/video/8krd5qKVw-Q
###################################################################

class Dataset:
    def __init__(self, file_path, dataset_preloaded = False, shuffle = True):
        # Load with numpy
        if type(dataset_preloaded) == bool:
            dataset = np.loadtxt(file_path, skiprows = 1)
            # self.dataset = dataset
            # Load labels
            fh = open(file_path,'r')
            for i, line in enumerate(fh):
                if i == 1: 
                    break
                line = line[:-1] # remove the /n
                self.labels = line.split(" ")
            fh.close()

        else:
            dataset, labels = dataset_preloaded

        if shuffle == True:
            np.random.shuffle(dataset ) # shuffle rows
            self.dataset = dataset
        else: 
            self.dataset = dataset

        self.nsamples = len(self.dataset[:,0])

        self.input_data = dataset[:,7:]
        self.output = dataset[:,0]
        error_p = [np.linalg.norm(dataset[i, 1:4]) for i in range(self.nsamples)]
        error_v = [np.linalg.norm(dataset[i, 4:7]) for i in range(self.nsamples)]
        self.error = np.column_stack((error_p, error_v)) # error in position and velocity

        self.n_input = self.input_data.shape[1]
        self.n_classes = self.error.shape[1]

    def statisticsFeasible(self):
        self.count_feasible = np.count_nonzero(self.output)
        print("Samples", self.nsamples, "Feasible", self.count_feasible)
    
    def statisticsError(self):
        plt.scatter(self.error[:,0], self.error[:,1])
        
        # Plot limit lines for feasibility
        x = [min(self.error[:,0]) , max(self.error[:,0]) ]
        y = [min(self.error[:,1]) , max(self.error[:,1]) ]
        plt.plot(FIT_FEAS['feas_ep'] / AL_BF.AU*np.ones(len(y)), y)
        plt.plot(x, FIT_FEAS['feas_ev'] / AL_BF.AU * AL_BF.year2sec(1) *np.ones(len(x)))
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    def plotDistributionOfFeasible(self):
        count_unfeasible = np.count_nonzero(self.output==0)
        count_feasible = len(self.output) - count_unfeasible

        fig, ax = plt.subplots() 
        plt.bar([0,1], [count_unfeasible, count_feasible])
        plt.xticks([0,1], ['Unfeasible', 'Feasible'])
        for i, v in enumerate([count_unfeasible, count_feasible]):
            ax.text(i, v+5, str(v), color='black', fontweight='bold')
        plt.show()

    def plotDistributionOfErrors(self):
        # fig = plt.figure(figsize = (30,30))
        
        # Error in position
        fig = plt.figure(figsize = (15,15))
        for i in range(self.n_input):
            ax = fig.add_subplot(self.n_input//2, 2, i+1)
            ax.plot(self.input_data_std[:,i] , self.error_std[:, 0], 'ko', markersize = 5)
            ax.set_xlabel(self.labels[i+7], labelpad = -2)
            ax.set_ylabel("Error in position")
            plt.yscale("log")

        plt.tight_layout()
        plt.savefig("./databaseANN/ErrorIncluded/Inputs_ErrorPosition_std.png", dpi = 100)
        plt.show()

        # Error in velocity
        fig = plt.figure(figsize = (15,15))
        for i in range(self.n_input):
            ax = fig.add_subplot(self.n_input//2, 2, i+1)
            ax.plot(self.input_data_std[:,i] , self.error_std[:, 1], 'ko', markersize = 5)
            ax.set_xlabel(self.labels[i+7], labelpad = -2)
            ax.set_ylabel("Error in velocity")
            plt.yscale("log")

        plt.tight_layout()
        plt.savefig("./databaseANN/ErrorIncluded/Inputs_ErrorVelocity_std.png", dpi = 100)
        plt.show()


    def standardizationInputs(self):
        # Standarization of the inputs
        scaler = StandardScaler()
        scaler.fit(self.input_data)
        self.input_data_std = scaler.transform(self.input_data)
    

    def standardizationError(self):
        # Normalization of errors TODO: eliminate and inlcude in database already
        self.error[:,0] /= AL_BF.AU # Normalize with AU
        self.error[:,1] = self.error[:,1] / AL_BF.AU * AL_BF.year2sec(1)

        # Standarization of the error
        if ANN_datab['scaling'] == 0:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.error)
            self.error_std = self.scaler.transform(self.error) 

        elif ANN_datab['scaling'] == 1:
            self.scaler = StandardScaler()
            self.scaler.fit(self.error)
            self.error_std = self.scaler.transform(self.error) 
        
        elif ANN_datab['scaling'] == 2:
            self.scaler = Normalizer()
            transformer = Normalizer(0).fit
        

    def inverseStandardizationError(self, x):
        x2 = self.scaler.inverse_transform(x)
        x2[:,0] *= AL_BF.AU # Normalize with AU
        x2[:,1] = x2[:,1] * AL_BF.AU / AL_BF.year2sec(1)
        return x2
        

    def convertLabels(self): # Labels are [Unfeasible feasible]
        self.output_2d = np.zeros((len(self.output), 2))
        for i in range(len(self.output)):
            if self.output[i] == 0: # Non feasible
                self.output_2d[i,:] = np.array([1,0])
            else:
                self.output_2d[i,:] = np.array([0,1])

    def equalizeclasses(self, en):
        indexes = np.where(self.output == 1)[0]
        indexes_un = np.arange(0, self.nsamples,1)
        indexes_un = np.delete(indexes_un, indexes)
        np.random.shuffle(indexes_un)

        self.input_data_std_e = np.zeros((2*len(indexes) + en, len(self.input_data[0,:])))
        self.output_e = np.zeros(2*len(indexes) + en)
        self.error_std_e = np.zeros((2*len(indexes) + en, len(self.error[0,:])))
        for i in range(len(indexes)):
            self.input_data_std_e[2*i,:] = self.input_data_std[indexes[i],:]
            self.output_e[2*i] = self.output[indexes[i]]
            self.error_std_e[2*i,:] = self.error_std[indexes[i],:]
            
            self.input_data_std_e[2*i+1,:] = self.input_data_std[indexes_un[i],:]
            self.output_e[2*i+1] = self.output[indexes_un[i]]
            self.error_std_e[2*i+1,:] = self.error_std[indexes_un[i],:]

        # fill with unfeasible data
        self.input_data_std_e[2*len(indexes)+2:,:] = self.input_data_std[indexes_un[len(indexes)+2:len(indexes)+en],:]
        self.output_e[2*len(indexes)+2:] = self.output[indexes_un[len(indexes)+2:len(indexes)+en]]
        self.error_std_e[2*len(indexes)+2:,:] = self.error_std[indexes_un[len(indexes)+2:len(indexes)+en],:]

        self.n_examples = 2*len(indexes)

def plotInitialDataPandas(train_file_path, pairplot = False, corrplot = False, \
                            inputsplotbar = False, inputsplotbarFeas = False):
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
            print(i)
            if i == 4 or i == 5:
                log_scale = True
            else:
                log_scale = False
            ax = fig.add_subplot(rows, cols, i+1)
            ax = sns.histplot(data = feasible_txt, x= labels_feas[i+1], hue = 'Label',\
                legend=False,  kde=False)
            # ax.set_title(labels_feas[i+1])
            # ax.set(ylim=(0, None))

        plt.show()
        print("Here2")

def LoadNumpy(train_file_path, plotDistribution = False, plotErrors = False,\
    equalize = False):
    # Load with numpy to see plot
    dataset_np = Dataset(train_file_path, shuffle = True)

    # Plot distribution of feasible/unfeasible
    if plotDistribution == True:
        dataset_np.plotDistributionOfFeasible()

    # dataset_np.statisticsFeasible()
    # dataset_np.plotDistributionOfDataset()

    if plotErrors == True:
        dataset_np.plotDistributionOfErrors()

    dataset_np.standardizationInputs()
    dataset_np.standardizationError()
    # dataset_np.convertLabels()

    if equalize == True:
        dataset_np.equalizeclasses(50)



    return dataset_np


def splitData_class( dataset_np, equalize = False):
    if equalize == True:
        train_x, train_y = dataset_np.input_data_std_e, dataset_np.output_e
    else:
        train_x, train_y = dataset_np.input_data_std, dataset_np.output

    train_cnt = floor(train_x.shape[0] * ANN_train['train_size'])
    x_train = train_x[0:train_cnt]
    y_train = train_y[0:train_cnt]
    x_test = train_x[train_cnt:]  
    y_test = train_y[train_cnt:]

    return [x_train, y_train], [x_test, y_test]

def splitData_reg(dataset_np, equalize = False):
    if equalize == True:
        train_x, train_y = dataset_np.input_data_std_e, dataset_np.error_std_e
    else:
        train_x, train_y = dataset_np.input_data_std, dataset_np.error_std

    train_cnt = floor(train_x.shape[0] * ANN_train['train_size'])
    x_train = train_x[0:train_cnt]
    y_train = train_y[0:train_cnt]
    x_test = train_x[train_cnt:]  
    y_test = train_y[train_cnt:]

    return [x_train, y_train], [x_test, y_test]


if __name__ == "__main__":

    train_file_path = "./databaseANN/ErrorIncluded/trainingData_Feas_big.txt"
    # train_file_path = "./databaseANN/trainingData_Feas_V2plusfake.txt"

    # plotInitialDataPandas(pairplot= False, corrplot= False, inputsplotbar = False, inputsplotbarFeas = True)
    # dataset_np = LoadNumpy(train_file_path, plotDistribution = True)
    equalize = True
    dataset_np = LoadNumpy(train_file_path, equalize = equalize)
    dataset_np.statisticsError()
    traindata, testdata = splitData_class(dataset_np, equalize = equalize)
    
