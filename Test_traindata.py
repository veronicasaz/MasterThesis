import numpy as np
import TrainingData as TD

import LoadConfigFiles as CONF

Dataset_conf = CONF.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']


# Path to dataset
base = "./databaseANN/3_DatabaseLast/deltakeplerian/"
file_path = base+ 'Together.txt'


scaling = Scaling['scaling']
dataUnits = Dataset_conf.Dataset_config['DataUnits']
Log = Dataset_conf.Dataset_config['Log']
output_type = Dataset_conf.Dataset_config['Outputs']

dataset = TD.Dataset(file_path, 
        inputs = {'labelType': 3, 'decV': False},\
        outputs =  {'outputs_class': [0,1], 'outputs_err': [2, 8], 'outputs_mf': 1, 'add': 'vector'}, \
        actions = {'shuffle': False })



dataset.select_output(output_type)
print(dataset.output_reg[0:2,:])
print(dataset.input_data[0:2,:])

dataset.normalize_outputs(scaling, dataUnits, Log, output_type)
dataset.select_output(output_type)

print("Standardized")
dataset.commonStandardization(base) # Try without saving
print(dataset.output_reg_std[0:2,:])
print(dataset.input_data_std[0:2,:])

print("Unstandardize")
E, I = TD.commonInverseStandardization(dataset.output_reg_std, 
                                    dataset.input_data_std, base)

print(E[0:2,:])
print(I[0:2,:])

print("Standardize without fitting")
I = TD.standardize_withoutFitting(dataset.input_data, "I", base)
print(I[0:2,:])


# VERIFIED !!

traindata, testdata = TD.splitData_reg(dataset, samples = 10)

print("Train and test data")
print(traindata)

print("======")
print(testdata)

# VERIFIED !!
