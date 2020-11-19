import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

import seaborn as sns
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from keras.models import model_from_json

import LoadConfigFiles as CONF
import TrainingDataKeras as TD

###################################################################
# https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://machinelearningmastery.com/keras-functional-api-deep-learning/
# ###################################################################

ANN_C = CONF.ANN_GAN()
ANN = ANN_C.ANN_config

class GAN_training:
    def __init__(self, dataset):
        self.dataset_np = dataset

        self.n_classes = self.dataset_np.n_classes # Labels
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

    def get_traintestdata(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def Classification_model(self):
        # input vector
        in_vector = keras.layers.Input(shape= (self.n_input,))

        for layer in range(ANN['Discriminator']['architecture']['hidden_layers']):
            merge = keras.layers.Dense(ANN['Discriminator']['architecture']['neuron_hidden'], \
                activation = 'relu')(in_vector)

        out_layer = keras.layers.Dense(1, activation = 'sigmoid')(merge)

        # define model
        model = keras.models.Model(in_vector, out_layer)

        # compile model
        opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss = 'binary_crossentropy', optimizer = opt, \
            metrics = ['accuracy'])

        return model

    def Discriminator_model(self):
        # label input
        in_label = keras.layers.Input(shape=(1,))
        # embedding for categorical input
        # li = keras.layers.Embedding(self.n_classes, self.n_input)(in_label)
        # input vector
        in_vector = keras.layers.Input(shape= (self.n_input,))
        #concat label as a channel
        merge = keras.layers.Concatenate()([in_vector, in_label])

        for layer in range(ANN['Discriminator']['architecture']['hidden_layers']):
            merge = keras.layers.Dense(ANN['Discriminator']['architecture']['neuron_hidden'], \
                activation = 'relu')(merge)

        out_layer = keras.layers.Dense(1, activation = 'sigmoid')(merge)

        # define model
        model = keras.models.Model([in_vector, in_label], out_layer)

        # compile model
        opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss = 'binary_crossentropy', optimizer = opt, \
            metrics = ['accuracy'])

        return model
            
    def Generator_model(self, latent_dim):
        
        # label input 
        in_label = keras.layers.Input(shape=(1,))
        #embedding for categorical input 
        # li = keras.layers.Embedding(self.n_classes, 50)()

        # image generator input
        in_lat = keras.layers.Input(shape=(latent_dim,))

        # merge label and input
        merge = keras.layers.Concatenate()([in_lat, in_label])

        for layer in range(ANN['Generator']['architecture']['hidden_layers']):
            merge = keras.layers.Dense(ANN['Generator']['architecture']['neuron_hidden'], \
                activation = 'relu')(merge)

        out_layer = keras.layers.Dense(self.n_input, activation = 'linear')(merge)

        # define model
        model = keras.models.Model([in_lat, in_label], out_layer)

        return model

    def define_GAN(self, Generator, Discriminator):
        # make weights in the discriminator not trainable
        Discriminator.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = Generator.input

        # get output from the generator model
        gen_output = Generator.output

        # connect output and label input from generator as inputs to discriminator
        gan_output = Discriminator([gen_output, gen_label])

        # define gan model as taking noise and label and outputting a classification
        model = keras.models.Model([gen_noise, gen_label], gan_output)
        
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def select_supervised_samples(self ):
        X, y = self.traindata
        X_list, y_list = list(), list()
        n_per_class = np.count_nonzero(y == 1)
        
        for i in range(self.n_classes):
            # get all samples of this class
            X_with_class = X[y == i]
            # choose random instances
            ix = np.random.randint(0, len(X_with_class), n_per_class)
            # add to list
            [X_list.append(X_with_class[j]) for j in ix]
            [y_list.append(i) for j in ix]
        return np.array(X_list), np.array(y_list)
        

    def generate_real_samples(self, dataset, n):
        input_d, labels = dataset
        # choose random instances
        ix = np.random.randint(0, len(input_d), n)
        x, labels = input_d[ix,:], labels[ix]
        # generate class labels
        y = np.ones((n,1)) # Label 1 indicates they are real
        return [x,labels], y

    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        # Generate labels
        labels = np.random.randint(0, self.n_classes, n_samples)
        return [x_input, labels]

    def generate_fake_samples(self, Generator, latent_dim, n_samples):
        x_input, labels_input = self.generate_latent_points(latent_dim, n_samples)

        # Predict output
        X = Generator.predict([x_input, labels_input])

        # Create class labels
        y = np.zeros((n_samples,1))

        return [X, labels_input], y

    def summarize_performance(self, epoch, Generator, Discriminator_sup,\
                    latent_dim, n_samples = 100 ):

        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(Generator, latent_dim, n_samples)

        # Real samples
        x_real, y_real = self.traindata
        # evaluate discriminator on real examples
        _, acc_real = Discriminator_sup.evaluate(x_real, y_real, verbose=0)
        print('Classifier Accuracy: %.3f%%' % (acc_real * 100))
        
        # # evaluate discriminator on fake examples
        # _, acc_fake = Discriminator_uns.evaluate(x_fake, y_fake, verbose=0)
        # # summarize discriminator performance
        # print("Discriminator performance", epoch, acc_real, acc_fake)


    def train(self, Generator, Discriminator, GAN, latent_dim):
    
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(self.traindata[1])/ ANN['Training']['n_batch'])
        # calculate the number of training iterations
        n_steps = bat_per_epo * ANN['Training']['n_epochs']
        # calculate the size of half a batch of samples
        half_batch = int(ANN['Training']['n_batch'] / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % 
                (ANN['Training']['n_epochs'], 
                ANN['Training']['n_batch'], 
                half_batch, 
                bat_per_epo, 
                n_steps))

        # manually enumerate epochs
        for i in range(ANN['Training']['n_epochs']):
            #enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = self.generate_real_samples(self.traindata, half_batch)
                # update discriminator model weights
                d_loss1, _ = Discriminator.train_on_batch([X_real, labels_real], y_real)

                # generate fake samples
                [X_fake, labels], y_fake = self.generate_fake_samples(Generator, latent_dim, half_batch)
                # update discriminator model weights
                d_loss2, _ = Discriminator.train_on_batch([X_fake, labels], y_fake)

                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(latent_dim, ANN['Training']['n_batch'])
                # create inverted labels for the fake samples 
                y_gan = np.ones((ANN['Training']['n_batch'], 1))
                # update the generator via de discriminator's error
                g_loss = GAN.train_on_batch([z_input, labels_input], y_gan)

                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            

            # serialize weights to HDF5
            Generator.save_weights("./trainedCNet_GAN_cond/Generator.h5")
            Discriminator.save_weights("./trainedCNet_GAN_cond/Discriminator.h5")
            print("Saved model to disk")

    def start(self):
        latent_dim = ANN['Training']['latent_dim']

        # Create discriminator models
        Discriminator = self.Discriminator_model()
        # Create generator models
        Generator = self.Generator_model(latent_dim)
        # Create the GAN
        GAN = self.define_GAN(Generator, Discriminator)

        # Load data
        dataset = self.traindata

        # Train model
        self.train(Generator, Discriminator, GAN, latent_dim)

    def generate_samples(self, n_samples, latent_dim, namefile):
        # Load trained generator
        loaded_model = self.Generator_model(latent_dim)
        # load weights into new model
        loaded_model.load_weights("./trainedCNet_GAN_cond/Generator.h5")
        print("Loaded model from disk")

        # Create samples
        [x, labels], y  = self.generate_fake_samples(loaded_model, latent_dim, n_samples)
        save_input = np.column_stack((labels,x))

        # Add real samples
        save_real_input = np.column_stack((np.ones(self.n_examples), self.dataset_np.input_data_std))

        # Save to file
        with open(namefile, "w") as myfile:
            myfile.write("Label t_t m_0 |Delta_a| |Delta_e| cos(Delta_i) Delta_Omega Delta_omega Delta_theta\n")
            for i, line in enumerate(save_real_input):
                for value in line:
                    if value != save_real_input[i, -1]:
                        myfile.write(str(value) + " ")
                    else:
                        myfile.write(str(value) + "\n")

        with open(namefile, "a") as myfile:
            for i, line in enumerate(save_input):
                for value in line:
                    if value != save_input[i, -1]:
                        myfile.write(str(value) + " ")
                    else:
                        myfile.write(str(value) + "\n")

        myfile.close()

    def see_samples(self, namefile):
        feasible_txt = pd.read_csv(namefile, sep=" ", header = 0)
        labels_feas = feasible_txt.columns.values

        sns.pairplot(feasible_txt[labels_feas], hue = 'Label')
        plt.show()

    def evaluate_discriminator(self):
        loaded_model = self.Classification_model()
        ref_model = self.Discriminator_model()
        # load weights into new model
        ref_model.load_weights("./trainedCNet_GAN_cond/Discriminator.h5")

        # Transfer weights from discriminator to classificator
        weights = ref_model.get_weights()
        print(weights)
        weights = np.delete(weights, [1,2], 0)
        loaded_model.set_weights(weights)

        # Predict outcome with discriminator
        predictions = loaded_model.predict(self.testdata[0])
        choice_prediction = [np.argmax(pred) for pred in predictions]
        true_labels = self.testdata[1]

        True_Positive = 0
        True_Negative = 0
        False_Positive = 0
        False_Negative = 0

        for i in range(len(choice_prediction)):
            if choice_prediction[i] == 1 and true_labels[i] == 1:
                True_Positive += 1
            elif choice_prediction[i] == 0 and true_labels[i] == 0:
                True_Negative += 1
            elif choice_prediction[i] == 1 and true_labels[i] == 0:
                False_Positive += 1
            elif choice_prediction[i] == 0 and true_labels[i] == 1:
                False_Negative += 1

        print('Accuracy', (True_Positive+True_Negative)/len(predictions))

        fig, ax = plt.subplots() 
        plt.xticks(range(4), labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative'])
        plt.yticks([])
        plot = plt.bar(range(4), [True_Positive, True_Negative, False_Positive, False_Negative] )
        for i, v in enumerate([True_Positive, True_Negative, False_Positive, False_Negative]):
            ax.text(i, v+5, str(v), color='black', fontweight='bold')
        plt.grid(False)
        # plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    ###############################################
    # LOAD TRAINING DATA
    ###############################################
    train_file_path = "./databaseANN/ErrorIncluded/trainingData_Feas_big.txt"
    # train_file_path = "./databaseANN/trainingData_Feas_V2plusfake.txt"

    # TD.plotInitialDataPandas(pairplot= False, corrplot= False, inputsplotbar = False, inputsplotbarFeas = True)
    # dataset_np = TD.LoadNumpy(train_file_path, plotDistribution = True)
    dataset_np = TD.LoadNumpy(train_file_path, error = True)
    traindata, testdata = TD.splitData_class(dataset_np)

    
    ###############################################
    # CREATE AND TRAIN CLASS NETWORK
    ###############################################
    perceptron = GAN_training(dataset_np)
    perceptron.get_traintestdata( traindata, testdata)
    perceptron.start() # Train GAN

    nameFile = "./databaseANN/GAN/RealvsFakeData/fakesamples.txt"
    perceptron.generate_samples(60, ANN['Training']['latent_dim'], nameFile) # Datbase with real and fake data. 
                               # Label indicates if it is real (1) or fake (0)
    perceptron.see_samples(nameFile)

    # perceptron.evaluate_discriminator()