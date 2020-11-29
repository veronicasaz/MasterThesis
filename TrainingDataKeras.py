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

ANN_reg = CONF.ANN_reg()
ANN = ANN_reg.ANN_config

Dataset_conf = CONF.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']

FIT_C = CONF.Fitness_config()
FIT = FIT_C.Fit_config

SF = CONF.SimsFlan_config() # Load Sims-Flanagan config variables  

###################################################################
# https://deeplizard.com/learn/video/8krd5qKVw-Q
###################################################################
def join_files(file_path, filename):
    dataset = np.loadtxt(file_path[0], skiprows = 1)
    label = np.ones(len(dataset[:,0])) * 0
    dataset = np.column_stack((dataset, label))

    print("Type %i, Number samples: %i"%(0, len(dataset[:,0])))

    for i, file_i in enumerate(file_path[1:]):
        dataset_i = np.loadtxt(file_i, skiprows = 1)
        label = np.ones(len(dataset_i[:,0]))* (i+1)
        dataset_i = np.column_stack((dataset_i, label))

        print("Type %i, Number samples: %i"%(i+1, len(dataset_i[:,0])))

        dataset = np.vstack((dataset, dataset_i))

    # Load labels
    fh = open(file_path[1],'r')
    for i, line in enumerate(fh):
        if i == 1: 
            break
        line = line[:-1] # remove the /n
        labels = line.split(" ")
    fh.close()

    labels.append( 'Datatype')

    # Save labels
    with open(filename, "w") as myfile:
        for i in labels:
            if i != labels[-1]:
                myfile.write(i +" ")
            else:
                myfile.write(i+"\n")
        np.savetxt(myfile, dataset)
    myfile.close()

def save_standard(dataset, save_file_path):
    """
        0: feasibility label
        1: norm ep
        2: norm ev
        3-end: inputs
        load as:
            labelType = False
            errorType = 'norm'
    """
    filename = save_file_path + str(Scaling['type_stand']) + '_' + str(Scaling['scaling']) + '.txt' 

    # Outputs:
    dataset_2 = np.column_stack(( dataset.output , dataset.error_std ))
    dataset_2 = np.column_stack(( dataset_2, dataset.input_data_std ))

    # Heading
    if dataset.errorType == 'vector': # eliminate cartesian names
        labels2 = [dataset.labels[0]]
        labels2.extend( ['Ep', 'Ev'] )
        if dataset.labelType != False: # delete last label
            labels2.extend( dataset.labels[7:-1] )
        else: 
            labels2.extend( dataset.labels[7:] )
    else:
        labels2 = dataset.labels

    # Save to file
    with open(filename, "w") as myfile:
        for i in labels2:
            if i != labels2[-1]:
                myfile.write(i +" ")
            else:
                myfile.write(i+"\n")
        np.savetxt(myfile, dataset_2)
    myfile.close()


# def equalizeToInputs(dataset, limits, divisions):
#     def count(limit, division):

#     dataset2 = np.zeros((len(limits), divisions**len(limits)))

    

class Dataset:
    def __init__(self, file_path, dataset_preloaded = False, shuffle = True, \
        error = 'vector', equalize = False, labelType = False):
        """
        error: False-database doesnt contain error, vector: cartesian  components, norm: only norm given
        labelType: Number: the last column of the database is an integer indicating from which file it comes.  
                    False: not included 
        """
        self.labelType = labelType
        self.errorType = error
        self.equalize = equalize

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
            dataset, self.labels = dataset_preloaded

        if shuffle == True:
            np.random.shuffle(dataset ) # shuffle rows
            self.dataset = dataset
        else: 
            self.dataset = dataset

        self.nsamples = len(self.dataset[:,0])

        
        self.output = self.dataset[:,0]

        if self.errorType == 'vector':
            startinput = 7
            error_p = [np.linalg.norm(self.dataset[i, 1:4]) for i in range(self.nsamples)]
            error_v = [np.linalg.norm(self.dataset[i, 4:7]) for i in range(self.nsamples)]
            self.error = np.column_stack((error_p, error_v)) # error in position and velocity
        elif self.errorType == 'norm':
            startinput = 3
            error_p = self.dataset[:, 1]
            error_v = self.dataset[:, 2]
            self.error = np.column_stack((error_p, error_v)) # error in position and velocity
        else:
            startinput = 1

        if labelType == False:
            self.input_data = self.dataset[:,startinput:]
        else:
            self.input_data = self.dataset[:,startinput:-1]
            

        self.n_input = self.input_data.shape[1]
        self.n_classes = 2

        if self.equalize == True:
            self.equalize_fun(self.error[:,0])

    def equalize_fun(self, base_vector):
        """
        equalize_fun: eliminate most common samples based on a criterion 
        INPUTS:
            x: database
            column: column to equalize
        """
        # By orders of magnitude
        indexes = [ int(np.log10( base_vector[i] )) for i in range(self.nsamples)  ]

        counter = np.zeros(12)
        for i in np.arange(0,12,1):
            counter[i] = indexes.count(i)

        mean = int( np.mean(counter[np.nonzero(counter)]) )
        print("equalize", counter)

        indexes_delete = list()
        for j in range(len(counter)):
            if counter[j] > mean: # more samples than it should
                ii = np.where((np.array(indexes) == j))[0]
                np.random.shuffle(ii)
                a = (ii[0:int(counter[j])-mean]).tolist()
                indexes_delete.extend( a )

        self.input_data = np.delete(self.input_data, indexes_delete, 0)
        self.output = np.delete(self.output, indexes_delete, 0)
        self.error = np.delete(self.error, indexes_delete, 0)
        self.dataset = np.delete(self.dataset, indexes_delete, 0)
        self.nsamples = len(self.output)

    def statisticsFeasible(self):
        self.count_feasible = np.count_nonzero(self.output)
        print("Samples", self.nsamples, "Feasible", self.count_feasible)
    
    def statisticsError(self, save_file_path):
        plt.scatter(self.error[:,0], self.error[:,1], color = 'black')
        
        # Plot limit lines for feasibility
        x = [min(self.error[:,0]) , max(self.error[:,0]) ]
        y = [min(self.error[:,1]) , max(self.error[:,1]) ]
        # plt.plot(FIT['FEASIB']['feas_ep'] / AL_BF.AU*np.ones(len(y)), y)
        # plt.plot(x, FIT['FEASIB']['feas_ev'] / AL_BF.AU * AL_BF.year2sec(1) *np.ones(len(x)))

        # Before standardizing
        plt.plot(FIT['FEASIB']['feas_ep'] *np.ones(len(y)), y, color = 'orange')
        plt.plot(x, FIT['FEASIB']['feas_ev']  *np.ones(len(x)), color = 'red')

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Error in position (m)")
        plt.ylabel("Error in velocity (m/s)")
        plt.title("Distribution of errors")
        
        plt.tight_layout()
        plt.savefig(save_file_path+"EpvsEv.png", dpi = 100)
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

    def plotDistributionOfErrors(self, save_file_path, std = True):
        # fig = plt.figure(figsize = (30,30))
        colors = ['black', 'red', 'green', 'blue', 'orange']

        std = True # use standard values
        if std == True: 
            x = self.input_data_std
            y = self.error_std
            stdlabel = "_std_"+str(self.scaling)

            ylabel_p = " Standardized"
            ylabel_v = " Standardized"
                
            if self.Log == True:
                limits_p = (min(y[:,0]), max(y[:,0]))
                limits_v =  (min(y[:,1]), max(y[:,1]))

                ylabel_p = ylabel_p +"log"
                ylabel_v = ylabel_v + "log"
            else:
                if self.scaling == 0:
                    limits_p = (1.5e-7, 1.01e0)
                    limits_v = (1.5e-7, 1.01e0)
                elif self.scaling == 1:
                    limits_p = (-1.01e0, 1.01e0)
                    limits_v = (-1.01e0, 1.01e0)
            
            
        else:
            x = self.input_data
            y = self.error
            stdlabel = ""

            if self.dataUnits == "AU":
                limits_p = (1e-6,1e1)
                limits_v = (1e-2,1e2)

                ylabel_p = " (AU)"
                ylabel_v = " (AU/year)"

            elif self.dataUnits == "SI":
                limits_p = (1e4,1e12)
                limits_v = (1.5e1, 1e5)

                ylabel_p = " (m)"
                ylabel_v = " (m/s)"
            
            if self.Log == True:
                limits_p = (np.log10(limits_p[0]), np.log10(limits_p[1]))
                limits_v = (np.log10(limits_v[0]), np.log10(limits_v[1]))

                ylabel_p = " log" + ylabel_p
                ylabel_v = " log" + ylabel_v
            

        # Error in position adn velocity
        save_file_path_epev = [save_file_path+"Inputs_ErrorPosition" +stdlabel+".png",\
                            save_file_path+"Inputs_ErrorVelocity" +stdlabel+".png"]

        ylabel_epev = ["Error in position" + ylabel_p,\
                        "Error in velocity" + ylabel_v]

        limits_epev = [limits_p, limits_v]

        for plot in range(2): # one for ep, one for ev
            fig = plt.figure(figsize = (15,15))
            for i in range(self.n_input):
                ax = fig.add_subplot(self.n_input//2, 2, i+1)

                if self.labelType != False:
                    for j in range(self.labelType): # how many files are there
                        indexes = np.where(self.dataset[:,-1] == j)[0] # find which values correspond to a certain creation method

                        ax.scatter(x[indexes,i] , y[indexes, 0],\
                            color = colors[j%len(colors)], marker = 'o', alpha = 0.5, label = j)
                else:
                    ax.plot(x[:,i] , y[:, 0], 'ko', markersize = 5)
                
                # print(np.log(min(self.error_std[:, 0])), np.log(max(self.error_std[:, 0])))
                
                ax.set_xlabel(self.labels[i+7], labelpad = -2)

                ax.set_ylabel(ylabel_epev[plot])
                if self.Log == False:
                    if std == True and self.scaling ==0:
                        ax.set_yscale('log')
                    elif std == True and self.scaling == 1:
                        ax.set_yscale('symlog') # To display negative values
                
                ax.set_ylim(limits_epev[plot])

                plt.legend()

            plt.tight_layout()
            plt.savefig(save_file_path_epev[plot], dpi = 100)
            plt.show()


    def commonStandardization(self, scaling, dataUnits, Log):
        """
        standardize inputs and errors together
        """
        self.scaling = scaling
        self.dataUnits = dataUnits
        self.Log = Log
        if scaling == 0:
            self.scaler = MinMaxScaler()
        elif scaling == 1:
            self.scaler = StandardScaler()

        
        if dataUnits == 'AU':
            self.error[:,0] /= AL_BF.AU # Normalize with AU
            self.error[:,1] = self.error[:,1] / AL_BF.AU * AL_BF.year2sec(1)
        if Log == True: # Apply logarithm 

            self.error[:,0] = [np.log10(self.error[i,0]) for i in range(len(self.error[:,0]))]
            self.error[:,1] = [np.log10(self.error[i,1]) for i in range(len(self.error[:,1]))]            

        database = np.column_stack((self.error, self.input_data))
        self.scaler.fit(database)

        database2 = self.scaler.transform(database)
        self.error_std = database2[:,0:2]
        self.input_data_std = database2[:,2:]

        print('here')


    def commonInverseStandardization(self, y, x):
        database = np.column_stack((y,x))
        print(np.shape(x), np.shape(y))
        
        x2 = self.scaler.inverse_transform(database)
        E = x2[:, 0:2]
        I = x2[:, 2:]

        if self.Log == True:
            E[:,0] = np.array([10**(E[i,0]) for i in range(len(E[:,0]))])
            E[:,1] = np.array([10**(E[i,1]) for i in range(len(E[:,1]))])
        if self.dataUnits == "AU":
            E[:,0] *= AL_BF.AU # Normalize with AU
            E[:,1] = E[:,1] * AL_BF.AU / AL_BF.year2sec(1)
        
        return E, I

    def standardizationInputs(self, scaling):
        self.scaling = scaling
        if scaling == 0:
            self.scaler_I = MinMaxScaler()
        elif scaling == 1:
            self.scaler_I = StandardScaler()
        # Standarization of the inputs
        # scaler = StandardScaler()
        self.scaler_I.fit(self.input_data)
        self.input_data_std = self.scaler_I.transform(self.input_data)

    def standardizationError(self, scaling, dataUnits, Log, sep = False):
        # Normalization of errors TODO: eliminate and inlcude in database already
        self.scaling = scaling
        self.dataUnits = dataUnits
        self.Log = Log
        if dataUnits == 'AU':
            self.error[:,0] /= AL_BF.AU # Normalize with AU
            self.error[:,1] = self.error[:,1] / AL_BF.AU * AL_BF.year2sec(1)
        if Log == True: # Apply logarithm 
            self.error[:,0] = [np.log10(self.error[i,0]) for i in range(len(self.error[:,0]))]
            self.error[:,1] = [np.log10(self.error[i,1]) for i in range(len(self.error[:,1]))]
            
        # Standarization of the error
        if scaling == 0:
            self.scaler = MinMaxScaler()
            self.scalerEp = MinMaxScaler()
            self.scalerEv = MinMaxScaler()

        elif scaling == 1:
            self.scaler = StandardScaler()
            self.scalerEp = StandardScaler()
            self.scalerEv = StandardScaler()
        
        if sep == False:
            self.scaler.fit(self.error)
            self.error_std = self.scaler.transform(self.error)
        else:
            self.error_std = np.zeros(np.shape(self.error))

            self.scalerEp.fit(self.error[:,0].reshape(-1, 1) )
            self.error_std[:,0] = self.scalerEp.transform(self.error[:,0].reshape(-1, 1)).flatten()
            self.scalerEv.fit(self.error[:,1].reshape(-1, 1))
            self.error_std[:,1] = self.scalerEv.transform(self.error[:,1].reshape(-1, 1)).flatten()
    
        # elif Scaling['scaling'] == 2:
        #     self.scaler = Normalizer()
        #     transformer = Normalizer(0).fit
        

    def inverseStandardization(self, x, typeR='E'):
        """
            typeR: E, rescale errors together
                  Ep, rescale error pos
                  Ev, rescale error vel
                  I, rescale input
        """
        if typeR == 'E':
            x2 = self.scaler.inverse_transform(x)

            if self.Log == True:
                x2[:,0] = [10**(x2[i,0]) for i in range(len(x2[:,0]))]
                x2[:,1] = [10**(x2[i,1]) for i in range(len(x2[:,1]))]
                
            if self.dataUnits == "AU":
                x2[:,0] *= AL_BF.AU # Normalize with AU
                x2[:,1] = x2[:,1] * AL_BF.AU / AL_BF.year2sec(1)
            

        elif typeR == 'Ep':
            x2 = self.scalerEp.inverse_transform(x.reshape(-1,1)).flatten()
            if self.Log == True:
                x2 = np.array([10**(x2[i]) for i in range(len(x2))])
            if self.dataUnits == "AU":
                x2 *= AL_BF.AU # Normalize with AU
            

        elif typeR == 'Ev':
            x2 = self.scalerEv.inverse_transform(x.reshape(-1,1)).flatten()

            if self.Log == True:
                x2 = np.array([10**(x2[i]) for i in range(len(x2))])
            if self.dataUnits == "AU":
                x2 = x2 * AL_BF.AU / AL_BF.year2sec(1)
            

        elif typeR == 'I':
            x2 = self.scalerI.inverse_transform(x)
        
        return x2
        

    def convertLabels(self): # Labels are [Unfeasible feasible]
        self.output_2d = np.zeros((len(self.output), 2))
        for i in range(len(self.output)):
            if self.output[i] == 0: # Non feasible
                self.output_2d[i,:] = np.array([1,0])
            else:
                self.output_2d[i,:] = np.array([0,1])

    def equalizeclasses(self, en, error = False):
        indexes = np.where(self.output == 1)[0]
        indexes_un = np.arange(0, self.nsamples,1)
        indexes_un = np.delete(indexes_un, indexes)
        np.random.shuffle(indexes_un)

        self.input_data_std_e = np.zeros((2*len(indexes) + en, len(self.input_data[0,:])))
        self.output_e = np.zeros(2*len(indexes) + en)
        
        if error == True:
            self.error_std_e = np.zeros((2*len(indexes) + en, len(self.error[0,:])))
        for i in range(len(indexes)):
            self.input_data_std_e[2*i,:] = self.input_data_std[indexes[i],:]
            self.output_e[2*i] = self.output[indexes[i]]
            
            if error == True:
                self.error_std_e[2*i,:] = self.error_std[indexes[i],:]
                self.error_std_e[2*i+1,:] = self.error_std[indexes_un[i],:]
            
            self.input_data_std_e[2*i+1,:] = self.input_data_std[indexes_un[i],:]
            self.output_e[2*i+1] = self.output[indexes_un[i]]
            

        # fill with unfeasible data
        self.input_data_std_e[2*len(indexes)+2:,:] = self.input_data_std[indexes_un[len(indexes)+2:len(indexes)+en],:]
        self.output_e[2*len(indexes)+2:] = self.output[indexes_un[len(indexes)+2:len(indexes)+en]]
        if error == True:
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

def plotInitialDataPandasError(train_file_path, save_file_path, pairplot = False, corrplot = False):
    feasible_txt = pd.read_csv(train_file_path, sep=" ", header = 0)
    labels_feas = feasible_txt.columns.values

    database = np.loadtxt(train_file_path, skiprows = 1)
    Ep = [np.linalg.norm(database[i,1:4]) for i in range(len(database[:,0]))]
    Ev = [np.linalg.norm(database[i,4:7]) for i in range(len(database[:,0]))]

    database_2 = np.column_stack((Ep, Ev))
    database_2 = np.column_stack((database_2, database[:,7:]))

    labels =['Ep', 'Ev']
    labels.extend(labels_feas[7:])

    df = pd.DataFrame(data=database_2, columns =  labels)

    if pairplot == True: # pairplot
        # g = sns.pairplot(df, hue = 'Datatype') # TODO: change if not using file "Together"
        g = sns.pairplot(df)
        # g.set(yscale = 'log', xscale= 'log')
        plt.tight_layout()
        plt.savefig(save_file_path+"/Pairplot.png", dpi = 100)
        plt.show()

    if corrplot == True: # correlations matrix
        sns.set_theme(style="white") 
        corr_mat = df.corr()
        fig, ax = plt.subplots(figsize =(20,12))
        cmap = sns.color_palette("mako", as_cmap=True)
        sns.heatmap(corr_mat, vmax = 1.0, square= True, ax=ax, \
            annot=True, cmap=cmap)
        plt.tight_layout()
        plt.savefig(save_file_path+"Corrplot.png", dpi = 100)
        plt.show()   


def LoadNumpy(train_file_path, save_file_path, \
            plotDistribution = False, plotErrors = False,\
            equalize = False, error = False, \
            standardizationType =  0, scaling = 0,\
            dataUnits = "AU", Log = False,\
            labelType = False):

    # Load with numpy to see plot
    dataset_np = Dataset(train_file_path, shuffle = True, error = error, 
        equalize = equalize, labelType = labelType)

    # Plot distribution of feasible/unfeasible
    if plotDistribution == True:
        dataset_np.plotDistributionOfFeasible()

    # if plotErrors == True:
    #     dataset_np.statisticsError(save_file_path)
    
    if standardizationType == 0: # common
        dataset_np.commonStandardization(scaling, dataUnits, Log)
    elif standardizationType == 1: #'input_output'
        dataset_np.standardizationInputs(scaling)
        if error != False:
            dataset_np.standardizationError(scaling, dataUnits, Log, sep=False)
    elif standardizationType == 2: #'input_sepoutput'
        dataset_np.standardizationInputs(scaling)
        dataset_np.standardizationError(scaling, dataUnits, Log, sep=True)
    else:
        setattr(dataset_np, "Log", Log)
        setattr(dataset_np, "DataUnits", dataUnits)
        
    
            
    if plotErrors == True:
        dataset_np.plotDistributionOfErrors(save_file_path)

    
    # dataset_np.convertLabels()

    return dataset_np


def splitData_class( dataset_np):
    if equalize == True:
        train_x, train_y = dataset_np.input_data_std_e, dataset_np.output_e
    else:
        train_x, train_y = dataset_np.input_data_std, dataset_np.output

    train_cnt = floor(train_x.shape[0] * ANN['Training']['train_size'])
    x_train = train_x[0:train_cnt]
    y_train = train_y[0:train_cnt]
    x_test = train_x[train_cnt:]  
    y_test = train_y[train_cnt:]

    return [x_train, y_train], [x_test, y_test]

def splitData_reg(dataset_np, samples = False):
    """
        equalize: decreases the number of inputs with larger repetition
        samples: takes a certain number of samples instead of the complete file
    """
    if dataset_np.equalize == True:
        train_x, train_y = dataset_np.input_data_std_e, dataset_np.error_std_e
    else:
        train_x, train_y = dataset_np.input_data_std, dataset_np.error_std

    if samples != False:
        train_x = train_x[0:samples,:]
        train_y = train_y[0:samples,:]

    train_cnt = floor(train_x.shape[0] * ANN['Training']['train_size'])
    x_train = train_x[0:train_cnt]
    y_train = train_y[0:train_cnt]
    x_test = train_x[train_cnt:]  
    y_test = train_y[train_cnt:]

    return [x_train, y_train], [x_test, y_test]


if __name__ == "__main__":

    # Choose which ones to choose:
    base = "./databaseANN/Organized/deltakeplerian/"
    file_path = [base + 'Random.txt', base +'Random_opt_5.txt', \
                base +'Random_opt_2.txt',\
                base +'Lambert.txt', base +'Lambert_opt.txt']
    # file_path = [base + 'Random.txt', base +'Random_opt.txt']
    
    # Join files together into 1
    file_path_together = base +'Together.txt'
    join_files(file_path, file_path_together)


    # See inputs
    # plotInitialDataPandasError(file_path_together, base,  pairplot= True, corrplot= False)
    dataset_np = LoadNumpy(file_path_together, base, error= 'vector',\
            equalize =  False, \
            standardizationType = Scaling['type_stand'], scaling = Scaling['scaling'],\
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            plotDistribution=False, plotErrors=True, labelType = len(file_path))

    # save_standard(dataset_np, base + 'Together_')