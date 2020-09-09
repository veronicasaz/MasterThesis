import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_file_path = "trainingData_Feas_shortTest.txt"

class Dataset:
    def __init__(self, file_path):
        # Load with numpy
        self.labels_feas =  ["Label", "t_t", "m_0", "|Delta_a|", \
              "|Delta_e|", "cos(Delta_i)", "Delta_Omega",\
              "Delta_omega", "Delta_theta"]
        self.dataset = np.loadtxt(file_path, skiprows = 1)
  
    def statisticsFeasible(self):
        self.nsamples = len(self.dataset[:,0])
        self.count_feasible = np.count_nonzero(self.dataset[:, 0])
        print("Samples", self.nsamples, "Feasible", self.count_feasible)

    def plotDistributionOfDataset(self, train_file_path):
        fig = plt.figure(figsize=(15,8))

        for i in range(len(self.dataset[0,1:])):
            # Normalization
            x = np.ones(len(self.dataset[:,0]))*i
            y = self.dataset[:,i+1]
            
            if min(y) > 0:
                y_norm = y / max(y)
            else:
                y_norm = np.zeros(len(y))
                pos = np.where(y >0)[0]
                neg = np.where(y<0)[0]
                y_norm[pos] = y[pos] / max(y)
                y_norm[neg] = y[neg] / -min(y)

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

dataset_np = Dataset(train_file_path)
dataset_np.statisticsFeasible()
dataset_np.plotDistributionOfDataset(train_file_path)

# Load with pandas
feasible_txt = pd.read_csv(train_file_path, sep=" ", header = 0)
labels_feas = feasible_txt.columns.values
# print(feasible_txt.dtypes)

# # Convert objects into discrete numerical values
# for label in labels_feas:
#   feasible_txt[label] = pd.Categorical(feasible_txt[label])
#   feasible_txt[label] = getattr(feasible_txt, label).cat.codes

# Create data.Dataset
output = feasible_txt.pop('Label')
dataset = tf.data.Dataset.from_tensor_slices((feasible_txt.values, output.values))


for feat, targ in dataset.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))

# Shuffle and batch the dataset
train_dataset = dataset.shuffle(len(feasible_txt)).batch(1)

# Create and train a model

