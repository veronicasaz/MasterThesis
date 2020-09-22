import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler 

###################################################################
# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
# https://www.tensorflow.org/tutorials/keras/classification
###################################################################

train_file_path = "./databaseANN/trainingData_Feas_shortTest.txt"

class Dataset:
    def __init__(self, file_path, labels_feas, dataset = False):
        # Load with numpy
        self.labels_feas =  labels_feas
        if np.any(dataset) == False:
            dataset = np.loadtxt(file_path, skiprows = 1)
            self.dataset = dataset[:,1:]
            self.output = dataset[:,0]
        else:
            self.dataset = dataset
  
    def statisticsFeasible(self):
        self.nsamples = len(self.dataset[:,0])
        self.count_feasible = np.count_nonzero(self.output)
        print("Samples", self.nsamples, "Feasible", self.count_feasible)

    def plotDistributionOfDataset(self):
        fig = plt.figure(figsize=(15,8))

        for i in range(len(self.dataset[0,:])):
            # Normalization
            x = np.ones(len(self.dataset[:,0]))*i
            y_norm = self.dataset[:,i]
            
            # if min(y) > 0:
            #     y_norm = y / max(y)
            # else:
            #     y_norm = np.zeros(len(y))
            #     pos = np.where(y >0)[0]
            #     neg = np.where(y<0)[0]
            #     y_norm[pos] = y[pos] / max(y)
            #     y_norm[neg] = y[neg] / -min(y)s

            # Plotting
            plt.scatter(x, y_norm)
        
        plt.xticks(np.arange(len(self.dataset[:,0])), self.labels_feas[1:])
        plt.tick_params(axis='both', which='major', labelsize=12)
        # ax.set_xticklabels(self.labels_feas[1:])
        plt.axis([-0.1, 7.1,-1.05,1.05])
        # plt.axis('equal')
        plt.grid(alpha = 0.5)

        dpi = 100
        layoutSave = 'tight'
        # plt.savefig('databaseFeasibility.png', dpi = dpi, bbox_inches = layoutSave)
        plt.title("Distribution of input parameters \n of the training population",\
            size = 25)
        plt.show()

def LoadNumpy():
    # Load with numpy to see plot
    dataset_np = Dataset(train_file_path, labels)
    dataset_np.statisticsFeasible()
    dataset_np.plotDistributionOfDataset()

# def plotDistributionStandardized(database):


def loadPandas(plotDistribution = False, pairplot = False, corrplot = False):
    # Load with pandas
    feasible_txt = pd.read_csv(train_file_path, sep=" ", header = 0)
    labels_feas = feasible_txt.columns.values
    input_data = feasible_txt.drop('Label', axis = 1)
    # print(feasible_txt.dtypes)

    # Plot distribution of feasible/unfeasible
    if plotDistribution == True:
        feasible_txt.Label.value_counts().plot(kind = "bar", rot = 0)
        plt.show()

    # Standarization of the inputs
    scaler = StandardScaler()
    scaler.fit(input_data)
    standardized = scaler.transform(input_data)

    # Numpy append label column to standard data
    numpy_stnd = np.c_[feasible_txt.iloc[:,0].values, standardized]

    # Pandas dataframe from the previous data standardized
    feasible_stnd = pd.DataFrame(data= numpy_stnd, columns = labels_feas )  # Complete dataset standardized
    # print(type(standardized))

    # dataset_distribution = Dataset("None", labels_feas, dataset = standardized)
    # dataset_distribution.plotDistributionOfDataset()

    if pairplot == True: # pairplot
        sns.pairplot(feasible_stnd[labels_feas], hue = 'Label')
        plt.show()

    if corrplot == True: # correlations matrix 
        corr_mat = feasible_stnd.corr()
        fig, ax = plt.subplots(figsize =(20,12))
        sns.heatmap(corr_mat, vmax = 1.0, square= True, ax=ax)
        plt.show()

    return feasible_stnd

def encodeData(feasible_stnd):

# # Convert objects into discrete numerical values
# for label in labels_feas:
#   feasible_txt[label] = pd.Categorical(feasible_txt[label])
#   feasible_txt[label] = getattr(feasible_txt, label).cat.codes

# # Create data.Dataset
# output = feasible_txt.pop('Label')
# dataset = tf.data.Dataset.from_tensor_slices((feasible_txt.values, output.values))

# # for feat, targ in dataset.take(5):
# #     print ('Features: {}, Target: {}'.format(feat, targ))

# # Shuffle and batch the dataset
# train_dataset = dataset.shuffle(len(feasible_txt)).batch(1)

# # Create and train a model
# def get_compiled_model():
#     # Dense(number of neurons per layer)
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1)
#   ])

#     model.compile(optimizer='adam',
#                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                     metrics=['accuracy']) # TODO: choose measure for accuracy for classification

#     return model

# model = get_compiled_model()
# model.fit(train_dataset, epochs=10)


if __name__ == "__main__":
    # LoadNumpy()
    loadPandas(plotDistribution = False)
    encodeData(feasible_stnd)