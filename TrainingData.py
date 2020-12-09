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
from AstroLibraries import AstroLib_2BP as AL_2BP
import LoadConfigFiles as CONF
import FitnessFunction_normalized as FIT_f

ANN_reg = CONF.ANN_reg()
ANN = ANN_reg.ANN_config

Dataset_conf = CONF.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']

FIT_C = CONF.Fitness_config()
FIT = FIT_C.Fit_config

SF = CONF.SimsFlan_config() # Load Sims-Flanagan config variables  

fitness = FIT_f.Fitness()

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
        if dataset.labelType != 0: # delete last label
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



class Dataset:
    def __init__(self, file_path, dataset_preloaded = False, \
        inputs = {'labelType': 0},\
        outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': 1, 'add': 'vector'}, \
        actions = {'shuffle': True }
        ):
        """
        labelType: Number: the last column of the database is an integer indicating from which file it comes.  
                    0: not included 
        outputs: objfunc gives one output with the result of the objective function.
                 epevmf: gives three outputs, mass of fuel, error in position, error in velocity. 
                 epev: gives two outputs, the error in position and the error in velocity
                 ep: gives the error in position
                 ev: gives the error in velocity
                 mf: gives the mass of fuel

                 add: to include additional information:
                    vector implies that the error is in cartesian coordinates, then outputs_err is the range
                    error_norm: indicates that the error is in its norm, then outputs_err is the range!!!
                    
        """
        self.inputs_conf = inputs
        self.outputs = outputs

        # Load with numpy
        if type(dataset_preloaded) == bool: # Not preloaded

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
            dataset, self.labels = dataset_preloaded

        if actions['shuffle'] == True:
            np.random.shuffle(dataset ) # shuffle rows

        self.dataset = dataset

        self.nsamples = len(self.dataset[:,0])
        if self.inputs_conf['labelType'] != 0:
            self.labelType = self.dataset[:,-1]

        self.output_adapt(self.outputs['outputs_err']) # Adapt error 
    
    def output_adapt(self, output_err):   
        if self.outputs['add'] == 'vector':
            cuterror = int( (output_err[1] - output_err[0]) /2 + output_err[0] )       
            error_p = [np.linalg.norm(self.dataset[i, output_err[0]:cuterror]) for i in range(self.nsamples)]
            error_v = [np.linalg.norm(self.dataset[i, cuterror:output_err[1]]) for i in range(self.nsamples)]
            self.error = np.column_stack((error_p, error_v)) # error in position and velocity

        elif self.outputs['add'] == 'norm':
            error_p = self.dataset[:, output_err[0]]
            error_v = self.dataset[:, output_err[1]-1]
            self.error = np.column_stack((error_p, error_v)) # error in position and velocity


    def select_output(self, typeOutput):
        """
        INPUTS:
            typeinput: 
                objfunc
                epevmf
                epev
                ep
                ev
                mf
        """
        self.typeOutput = typeOutput

        # Choose outputs
        self.output_class = self.dataset[:,self.outputs['outputs_class'][0]:self.outputs['outputs_class'][1]]
        self.n_classes = 2
                
        if self.typeOutput == 'objfunc':
            self.n_outputs = 1
            self.output_label = [self.typeOutput]
            self.output_reg = np.zeros( (self.nsamples, self.n_outputs))
            for i in range(self.nsamples):
                self.output_reg[i] = fitness.objFunction(self.error[i,:], m_fuel = self.dataset[i,self.outputs['outputs_mf']])

        elif self.typeOutput == 'epevmf':
            self.n_outputs = 3
            self.output_label = ['ep', 'ev', 'mf']
            self.output_reg = np.zeros( (self.nsamples, self.n_outputs))
            self.output_reg = np.column_stack((self.dataset[:,self.outputs['outputs_mf']], self.error))

        elif self.typeOutput == 'epev':
            self.n_outputs = 2
            self.output_label = ['ep', 'ev']
            self.output_reg = self.error.copy()

        elif self.typeOutput == 'ep':
            self.n_outputs = 1
            self.output_label = ['ep']
            self.output_reg = self.error[:,0].copy()
        
        elif self.typeOutput == 'ev':
            self.n_outputs = 1
            self.output_label = ['ev']
            self.output_reg = self.error[:,1].copy()
        
        elif self.typeOutput == 'mf':
            self.n_outputs = 1
            self.output_label = ['mf']
            self.output_reg = self.dataset[:,self.outputs['outputs_mf']].copy()

        else:
            print("Error: choose valid type of output")

        startinput = self.outputs['outputs_err'][1]

        if self.inputs_conf['labelType'] == 0:
            self.input_data = self.dataset[:,startinput:]
        else:
            self.input_data = self.dataset[:,startinput:-1]
            
        self.n_input = self.input_data.shape[1]


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
        self.output_class = np.delete(self.output_class, indexes_delete, 0)
        self.output_reg = np.delete(self.output_reg, indexes_delete, 0)
        self.error = np.delete(self.error, indexes_delete, 0)
        self.dataset = np.delete(self.dataset, indexes_delete, 0)
        self.nsamples = len(self.output_class)

    def statisticsFeasible(self):
        self.count_feasible = np.count_nonzero(self.output_class)
        print("Samples", self.nsamples, "Feasible", self.count_feasible)
    
    def statisticsError(self, save_file_path):
        """
        plot Ep vs Ev with limits for feasibility
        """
        colors = ['black', 'red', 'green', 'blue', 'orange', 'yellow', 'purple',\
            'pink']
        
        # Plot limit lines for feasibility
        x = [min(self.error[:,0]) , max(self.error[:,0]) ]
        y = [min(self.error[:,1]) , max(self.error[:,1]) ]
        # plt.plot(FIT['FEASIB']['feas_ep'] / AL_BF.AU*np.ones(len(y)), y)
        # plt.plot(x, FIT['FEASIB']['feas_ev'] / AL_BF.AU * AL_BF.year2sec(1) *np.ones(len(x)))

        # Before standardizing
        plt.plot(FIT['FEASIB']['feas_ep'] *np.ones(len(y)), y, color = 'orange')
        plt.plot(x, FIT['FEASIB']['feas_ev']  *np.ones(len(x)), color = 'red')

        if self.inputs_conf['labelType'] != 0:
            for j in range(self.inputs_conf['labelType']): # how many files are there
                indexes = np.where(self.labelType == j)[0] # find which values correspond to a certain creation method

                plt.scatter(self.error[indexes,0], self.error[indexes,1], color = colors[j], label = j)
            plt.legend()

        else:
            plt.scatter(self.error[:,0], self.error[:,1], color = colors[0])
        
        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Error in position (m)")
        plt.ylabel("Error in velocity (m/s)")
        plt.title("Distribution of errors")
        
        plt.tight_layout()
        plt.savefig(save_file_path+"EpvsEv.png", dpi = 100)
        plt.show()

    def plotDistributionOfFeasible(self):

        count_unfeasible = np.count_nonzero(self.output_class==0)
        count_feasible = len(self.output_class) - count_unfeasible

        fig, ax = plt.subplots() 
        plt.bar([0,1], [count_unfeasible, count_feasible])
        plt.xticks([0,1], ['Unfeasible', 'Feasible'])
        for i, v in enumerate([count_unfeasible, count_feasible]):
            ax.text(i, v+5, str(v), color='black', fontweight='bold')
        plt.show()

    def plotDistributionOfErrors(self, save_file_path, std = True):
        colors = ['black', 'red', 'green', 'blue', 'orange']

        # Standardization
        if std == True: 
            x = self.input_data_std
            y = self.error_std
            stdlabel = "_std_"+str(self.scaling)

            ylabel_p = " Standardized"
            ylabel_v = " Standardized"
            
        else:
            x = self.input_data
            y = self.error
            stdlabel = ""
            ylabel_p = ""
            ylabel_v = ""

        limitFactor = 1.1
        limits_p = (min(y[:,0])*limitFactor, max(y[:,0])*limitFactor)
        limits_v =  (min(y[:,1])*limitFactor, max(y[:,1])*limitFactor)
        
        # Log
        if self.Log == True:
            ylabel_p = ylabel_p +" log"
            ylabel_v = ylabel_v + " log"
        else:
            if self.scaling == 0:
                limits_p = (1.5e-7, 1.01e0)
                limits_v = (1.5e-7, 1.01e0)
            elif self.scaling == 1:
                limits_p = (-1.01e0, 1.01e0)
                limits_v = (-1.01e0, 1.01e0)

        # Units
        if self.dataUnits == "AU":
            # limits_p = (1e-6,1e1)
            # limits_v = (1e-2,1e1)

            ylabel_p = ylabel_p +" (AU)"
            ylabel_v = ylabel_v +" (AU/year)"

        elif self.dataUnits == "SI":
            # limits_p = (1e4,1e12)
            # limits_v = (1.5e1, 1e5)

            ylabel_p = ylabel_p +" (m)"
            ylabel_v = ylabel_v +" (m/s)"
            

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

                if self.inputs_conf['labelType'] != 0:
                    for j in range(self.inputs_conf['labelType']): # how many files are there
                        indexes = np.where(self.labelType == j)[0] # find which values correspond to a certain creation method
                        ax.scatter(x[indexes,i] , y[indexes, plot],\
                            color = colors[j%len(colors)], marker = 'o', alpha = 0.5, label = j)
                else:
                    ax.plot(x[:,i] , y[:, plot], 'ko', markersize = 5, label = "Optimized")
                
                # print(np.log(min(self.error_std[:, 0])), np.log(max(self.error_std[:, 0])))
                
                ax.set_xlabel(self.labels[i+7], labelpad = -2)

                if self.Log == False:
                    if self.scaling ==0:
                        ax.set_yscale('log')
                    elif self.scaling == 1:
                        ax.set_yscale('symlog') # To display negative values
                
                ax.set_ylim(limits_epev[plot])

                plt.legend()

            plt.suptitle(ylabel_epev[plot], y=1.01)

            plt.tight_layout()
            plt.savefig(save_file_path_epev[plot], dpi = 100)
            plt.show()


#TODO: not used:
    def plotOutputsWRTInputs(self, save_file_path, std = True):
        colors = ['black', 'red', 'green', 'blue', 'orange']

        # Standardization
        if std == True: 
            x = self.input_data_std
            y = self.error_std
            stdlabel = "_std_"+str(self.scaling)

            ylabel_p = " Standardized"
            ylabel_v = " Standardized"
            
        else:
            x = self.input_data
            y = self.error
            stdlabel = ""
            ylabel_p = ""
            ylabel_v = ""

        limitFactor = 1.1
        limits_p = (min(y[:,0])*limitFactor, max(y[:,0])*limitFactor)
        limits_v =  (min(y[:,1])*limitFactor, max(y[:,1])*limitFactor)
        
        # Log
        if self.Log == True:
            ylabel_p = ylabel_p +" log"
            ylabel_v = ylabel_v + " log"
        else:
            if self.scaling == 0:
                limits_p = (1.5e-7, 1.01e0)
                limits_v = (1.5e-7, 1.01e0)
            elif self.scaling == 1:
                limits_p = (-1.01e0, 1.01e0)
                limits_v = (-1.01e0, 1.01e0)

        # Units
        if self.dataUnits == "AU":
            # limits_p = (1e-6,1e1)
            # limits_v = (1e-2,1e1)

            ylabel_p = ylabel_p +" (AU)"
            ylabel_v = ylabel_v +" (AU/year)"

        elif self.dataUnits == "SI":
            # limits_p = (1e4,1e12)
            # limits_v = (1.5e1, 1e5)

            ylabel_p = ylabel_p +" (m)"
            ylabel_v = ylabel_v +" (m/s)"
            

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

                if self.inputs_conf['labelType'] != 0:
                    for j in range(self.inputs_conf['labelType']): # how many files are there
                        indexes = np.where(self.labelType == j)[0] # find which values correspond to a certain creation method
                        ax.scatter(x[indexes,i] , y[indexes, plot],\
                            color = colors[j%len(colors)], marker = 'o', alpha = 0.5, label = j)
                else:
                    ax.plot(x[:,i] , y[:, plot], 'ko', markersize = 5, label = "Optimized")
                
                # print(np.log(min(self.error_std[:, 0])), np.log(max(self.error_std[:, 0])))
                
                ax.set_xlabel(self.labels[i+7], labelpad = -2)

                if self.Log == False:
                    if self.scaling ==0:
                        ax.set_yscale('log')
                    elif self.scaling == 1:
                        ax.set_yscale('symlog') # To display negative values
                
                ax.set_ylim(limits_epev[plot])

                plt.legend()

            plt.suptitle(ylabel_epev[plot], y=1.01)

            plt.tight_layout()
            plt.savefig(save_file_path_epev[plot], dpi = 100)
            plt.show()

    def plotDistributionErrorsPD(self, save_file_path):
        save_file_path_epev = save_file_path+"Distribution_output_" +str(self.typeOutput)+".png"
        dataset = pd.DataFrame(data = np.log10(self.output_reg), columns = self.output_label )
        sns.displot(dataset)
        # plt.legend(self.output_label, loc='upper left')
        plt.tight_layout()
        plt.savefig(save_file_path_epev, dpi = 100)
        plt.show()


    def commonStandardization(self, scaling, dataUnits, Log, typeOutput):
        """
        standardize inputs and errors together
        """
        self.scaling = scaling
        self.dataUnits = dataUnits
        self.Log = Log

        # Normalization
        if dataUnits == 'AU':
            self.error[:,0] /= AL_BF.AU # Normalize with AU
            self.error[:,1] = self.error[:,1] / AL_BF.AU * AL_BF.year2sec(1)
        
        self.Spacecraft = AL_2BP.Spacecraft( )

        if Log == True: # Apply logarithm 
            self.error[:,0] = [np.log10(self.error[i,0]) for i in range(len(self.error[:,0]))]
            self.error[:,1] = [np.log10(self.error[i,1]) for i in range(len(self.error[:,1]))]  

        if typeOutput == 'epevmf' or typeOutput == 'mf':
            self.dataset[:,self.outputs['outputs_mf']] /= self.Spacecraft.m_dry 

        self.select_output(typeOutput)

        # Scaling
        if scaling == 0:
            self.scaler = MinMaxScaler()
        elif scaling == 1:
            self.scaler = StandardScaler()

        database = np.column_stack((self.output_reg, self.input_data))
        self.scaler.fit(database)

        database2 = self.scaler.transform(database)
        if self.typeOutput == 'epevmf':
            self.error_std = database2[:,1:3]
            self.output_reg_std = database2[:, 0:3]
            self.input_data_std = database2[:,3:]
        
        elif self.typeOutput == 'epev':
            self.error_std = database2[:,0:2]
            self.output_reg_std = database2[:, 0:2]
            self.input_data_std = database2[:,2:]

        elif self.typeOutput == 'objfunc' or self.typeOutput == 'ep' or self.typeOutput == 'ev' or self.typeOutput == 'mf':
            self.error_std = np.zeros(np.shape(self.error)) # does not apply
            self.output_reg_std = database2[:, 0]
            self.input_data_std = database2[:,1:]



    def commonInverseStandardization(self, y, x):
        database = np.column_stack((y,x))
        
        x2 = self.scaler.inverse_transform(database)
        
        if self.typeOutput == 'epevmf':
            startinput = 3
            starterror = 1
        elif self.typeOutput == 'epev':
            startinput = 2
            starterror = 0
        elif self.typeOutput == 'objfunc' or self.typeOutput == 'ep' or self.typeOutput == 'ev' or self.typeOutput == 'mf':
            startinput = 1
            starterror = 0


        E = x2[:, 0:startinput]
        I = x2[:, startinput:]

        print(database[0:10, :], startinput)

        if self.typeOutput == 'ep' or self.typeOutput == 'epev' or self.typeOutput == 'epevmf':
            if self.Log == True:
                E[:,starterror] = np.array([10**(E[i,starterror]) for i in range(len(E[:,starterror]))])
            if self.dataUnits == "AU":
                E[:,starterror] *= AL_BF.AU # Normalize with AU
        
        if self.typeOutput == 'epev':
            if self.Log == True:
                E[:,starterror+1] = np.array([10**(E[i,starterror+1]) for i in range(len(E[:,starterror+1]))])
            if self.dataUnits == "AU":
                E[:,starterror+1] = E[:,starterror+1] * AL_BF.AU / AL_BF.year2sec(1)
        if self.typeOutput == 'ev':
            if self.Log == True:
                E[:,starterror] = np.array([10**(E[i,starterror]) for i in range(len(E[:,starterror]))])
            if self.dataUnits == "AU":
                E[:,starterror] = E[:,starterror] * AL_BF.AU / AL_BF.year2sec(1)
       
        if self.typeOutput == 'epevmf' or self.typeOutput == 'mf':
            E[:,0] *= self.Spacecraft.m_dry

        return E, I

    def noise_gauss(self, mean, std):
        if self.inputs_conf['labelType'] == 0:
            self.labelType == np.zeros((len(self.nsamples))).flatten()
            self.inputs_conf['labelType'] = 1

        # Create copies
        output_2 = self.output_reg_std.copy()
        input_2 = self.input_data_std.copy()
        labelType_2 = self.labelType.copy()

        # Add noise
        output_2 += np.random.normal(mean, std, np.shape(self.output_reg_std)) 
        input_2 += np.random.normal(mean, std, np.shape(self.input_data_std)) 
        labelType_2 += self.inputs_conf['labelType']

        self.inputs_conf['labelType'] *= 2

        # Append to current list
        if self.n_outputs == 1:
            self.output_reg_std = np.concatenate((self.output_reg_std, output_2))
        else:
            self.output_reg_std = np.vstack((self.output_reg_std, output_2))
        self.input_data_std = np.vstack((self.input_data_std, input_2))
        self.labelType = np.concatenate((self.labelType, labelType_2))

        
        # Find error again
        if self.typeOutput == 'epevmf':
            self.error_std = self.output_reg_std[:,1:3] 
        elif self.typeOutput == 'epev':
            self.error_std = self.output_reg_std[:,0:2]
        elif self.typeOutput == 'objfunc' or self.typeOutput == 'ep' or self.typeOutput == 'ev' or self.typeOutput == 'mf':
            self.error_std = np.zeros(np.shape(self.error)) 
    # def plotDataPandas(self):
    #     df = 
    #     g = sns.pairplot(df)
    #     # g.set(yscale = 'log', xscale= 'log')
    #     plt.tight_layout()
    #     plt.savefig(save_file_path+"/Pairplot.png", dpi = 100)
    #     plt.show()


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
    Ep = [np.linalg.norm(database[i,2:5]) for i in range(len(database[:,0]))]
    Ev = [np.linalg.norm(database[i,5:8]) for i in range(len(database[:,0]))]
    
    Err = np.column_stack((Ep, Ev))
    database_2 = np.column_stack((database[:,1], Err))
    database_2 = np.column_stack((database_2, database[:,8:]))

    labels =['Mf','Ep', 'Ev']
    labels.extend(labels_feas[8:])

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


def LoadNumpy(train_file_path, save_file_path = None, \
            scaling = 0,\
            dataUnits = "AU", Log = False, \
            outputs =  {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': 1, 'add': 'vector'},
            output_type = 'epev',
            labelType = 0,
            plotDistribution = False, plotErrors = False, 
            plotOutputDistr = False, plotEpvsEv = False,
            data_augmentation = 'False'):

    if data_augmentation != 'False' and data_augmentation['type'] == 'multiplication': # Augmentation = true
        train_file_path2 = train_file_path.copy()
        for rep in range(data_augmentation['times']-1):
            train_file_path2.extend(train_file_path)

        train_file_path = train_file_path2

    if type(train_file_path) == list: # files have to be joined
        labelType = len(train_file_path)
        train_file_path2 = save_file_path +'Together.txt'
        join_files( train_file_path, train_file_path2)
        train_file_path = train_file_path2
        
    # Load with numpy to see plot
    dataset_np = Dataset(train_file_path, \
        inputs = {'labelType': labelType},\
        outputs =outputs, \
        actions = {'shuffle': True })

    if plotEpvsEv == True and save_file_path != None:
        dataset_np.statisticsError(save_file_path)

    if scaling == 0 or scaling == 1: # common
        dataset_np.commonStandardization(scaling, dataUnits, Log, output_type)
    else:
        dataset_np.select_output(output_type)
    # else:
    #     setattr(dataset_np, "Log", Log)
    #     setattr(dataset_np, "DataUnits", dataUnits)


    if plotDistribution == True:
        dataset_np.plotDistributionOfFeasible()

    if plotOutputDistr == True:
        dataset_np.plotDistributionErrorsPD(save_file_path)

    if data_augmentation != 'False':
        save_file_path = save_file_path +"_aumented_"+data_augmentation['type']+"_"

    if data_augmentation != 'False' and data_augmentation['type'] == 'noise_gauss':
        dataset_np.noise_gauss(data_augmentation['mean'], data_augmentation['std'])
        std_Error = True
    else:
        std_Error = False # TODO: Can be modified to obtain other plots

    
    if plotErrors == True:
        dataset_np.plotDistributionOfErrors(save_file_path, std_Error)
        # dataset_np.plotDistributionOfErrors(save_file_path)

    return dataset_np



def splitData_class( dataset_np):
    if equalize == True:
        train_x, train_y = dataset_np.input_data_std_e, dataset_np.output_class_e
    else:
        train_x, train_y = dataset_np.input_data_std, dataset_np.output_class

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
    # if dataset_np.equalize == True:
    #     train_x, train_y = dataset_np.input_data_std_e, dataset_np.output_reg_std_e
    # else:
    train_x, train_y = dataset_np.input_data_std, dataset_np.output_reg_std

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
    base = "./databaseANN/DatabaseFitness/deltakeplerian/"
    # file_path = [base + 'Random.txt']
    # file_path = [base +'Random_MBH_eval.txt', base +'Random_MBH.txt']
                # base +'Random_MBH_eval.txt', base +'Random_MBH.txt']
    file_path = [base+ 'fp1/Random_MBH.txt',
                base+ 'fp2/Random_MBH.txt',
                base+ 'fp5/Random_MBH.txt',
                base+ '500/Random_MBH.txt',
                base+ 'fp50/Random_MBH.txt',
                base+ 'fp100/Random_MBH.txt']
    
    # file_path_together = base + 'Random.txt'
    # # Join files together into 1
    file_path_together = base +'ComparisonObjFunction.txt'
    join_files(file_path, file_path_together)


    dataset_np = LoadNumpy(file_path_together, save_file_path = base, 
            scaling = Scaling['scaling'], 
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            outputs = {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': False, 'add': 'vector'},
            output_type = Dataset_conf.Dataset_config['Outputs'],
            labelType = len(file_path),
            plotDistribution=False, plotErrors=False,
            # plotOutputDistr = False, plotEpvsEv = False,
            # plotDistribution=True, plotErrors=True,
            plotOutputDistr = True, plotEpvsEv = True,
            data_augmentation = Dataset_conf.Dataset_config['dataAugmentation'])

    # See inputs
    # plotInitialDataPandasError(file_path_together, base,  pairplot= True, corrplot= True)
    # dataset_np = LoadNumpy(file_path_together, base, error= 'vector',\
    #         equalize =  False, \
    #         standardizationType = Scaling['type_stand'], scaling = Scaling['scaling'],\
    #         dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
    #         Outputs= Dataset_conf.Dataset_config['Outputs'],
    #         plotDistribution=False, plotErrors=True, labelType = 2)

    # save_standard(dataset_np, base + 'Together_')