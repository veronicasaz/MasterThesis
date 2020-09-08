import tensorflow as tf
import numpy as np
import pandas as pd

train_file_path = "trainingData_Feas_shortTest.txt"

class Dataset:
    def __init__(self, train_file_path):
        # Load with numpy
        self.labels_feas =  ["Label", "t_t", "m_0", "|Delta_a|", \
              "|Delta_e|", "cos(Delta_i)", "Delta_Omega",\
              "Delta_omega", "Delta_theta"]
        self.dataset = np.loadtxt(train_file_path, skiprows = 1)
  
    def statisticsFeasible(self):
        self.nsamples = len(self.dataset[:,0])
        self.count_feasible = np.count_nonzero(self.dataset[:, 0])
        print("Samples", self.nsamples, "Feasible", self.count_feasible)

  # def plotDistributionOfDataset(train_file_path):

dataset_np = Dataset(train_file_path)
dataset_np.statisticsFeasible()

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


for feat, targ in train_dataset.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))

# Shuffle and batch the dataset
train_dataset = dataset.shuffle(len(feasible_txt)).batch(1)

# Create and train a model

