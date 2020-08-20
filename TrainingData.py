import tensorflow as tf
import numpy as np

train_file_path = "trainingData_Feas_shortTest.txt"

class Dataset:
    def __init__(self, dataset):
      self.nsamples = len(dataset[:,0])
      self.count_feasible = np.count_nonzero(dataset[:, 0])


# Load with numpy
dataset_feas = np.loadtxt(train_file_path, skiprows = 1)
dataset = Dataset(dataset_feas)

print(dataset.nsamples, dataset.count_feasible)


# Check how much percentage of feasible have been found


# # Name of column
# # Label, params, params,
# def get_dataset(file_path, **kwargs):
#     dataset = tf.data.experimental.make_csv_dataset(
#       file_path,
#       batch_size=5, # Artificially small to make examples easier to show.
#       label_name="Label   ",
#       na_value="?",
#       num_epochs=1,
#       ignore_errors=True, 
#       **kwargs)
#     return datasetLabel

# def show_batch(dataset):
#   for batch, label in dataset.take(1):
#     for key, value in batch.items():
#       print("{:20s}: {}".format(key,value.numpy()))


# raw_train_data = get_dataset(train_file_path)
# # raw_test_data = get_dataset(test_file_path)

# show_batch(raw_train_data)
