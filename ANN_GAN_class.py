import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

import LoadConfigFiles as CONF
import TrainingDataKeras as TD

###################################################################
# https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/
###################################################################

ANN = CONF.ANN_GAN()

class GAN_training:
    def __init__(self, dataset):
        self.dataset_np = dataset

        self.n_classes = self.dataset_np.n_classes # Labels
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

        self.checkpoint_path = "./trainedCNet_Class/training_1/cp.ckpt"

    def get_traintestdata(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def Discriminator_model(self, regularization = True):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        # model_sup = keras.Sequential()
        model_uns = keras.Sequential()

        # for model in [model_sup, model_uns]:
        if ANN.Discriminator['architecture']['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            for layer in range(ANN.Discriminator['architecture']['hidden_layers']):
                model_uns.add(keras.layers.Dense(
                    ANN.Discriminator['architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN.Discriminator['architecture']['regularizer_value']) ))
        else:
            for layer in range(ANN.Discriminator['architecture']['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN.Discriminator['architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer) )

        # model_sup.add(keras.layers.Dense(self.n_classes, activation ='softmax' )) # output layer
        model_uns.add(keras.layers.Dense(self.n_classes, activation ='sigmoid' ))
        
        # Compile
        # model_sup.compile(optimizer='adam',
        #       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #       metrics=['accuracy'])
        model_uns.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

        return model_uns
            
    def Generator_model(self, latent_dim):
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()

        if ANN.Generator['architecture']['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            for layer in range(ANN.Generator['architecture']['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN.Generator['architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN.Generator['architecture']['regularizer_value']) ))
        else:
            for layer in range(ANN.Generator['architecture']['hidden_layers']):
                model.add(keras.layers.Dense(
                    ANN.Generator['architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer) )

        model.add(keras.layers.Dense(self.n_input, activation = 'linear')) # output layer

        return model

    def define_GAN(self, Generator, Discriminator_uns):
        # make weights in the discriminator not trainable
        Discriminator_uns.trainable = False
        # connect output from generator as input to discriminator
        model = keras.Sequential()
        model.add(Generator)
        
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model


    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input
  

    def generate_fake_samples(self, Generator, latent_dim, n_samples):
        x_input = self.generate_latent_points(latent_dim, n_samples)

        # Predict output
        X = Generator.predict(x_input)

        # Create class labels
        y = np.zeros((n_samples,1))

        return X, y

    def summarize_performance(self, epoch, Generator, Discriminator_uns,\
                    latent_dim, dataset, n_samples = 100 ):

        # Real samples
        x_real, y_real = self.traindata
        # evaluate discriminator on real examples
        _, acc_real = Discriminator_uns.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(Generator, latent_dim, n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = Discriminator_uns.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print("Discriminator performance", epoch, acc_real, acc_fake)



    def train(self, Generator, Discriminator_uns, GAN, latent_dim, 
                n_epochs=20, n_batch=100, n_eval = 50):
    
        # calculate the number of batches per training epoch
        bat_per_epo = int(self.n_examples / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
        
        # manually enumerate epochs
        for i in range(n_epochs):
            # real data
            x_real = self.traindata[0][half_batch, :]
            y_real = self.traindata[1][half_batch]
            # fake data
            x_fake, y_fake = self.generate_fake_samples(Generator, latent_dim, half_batch)
            
            # update unsupervised discriminator (d)
            Discriminator_uns.train_on_batch(x_real, y_real)
            Discriminator_uns.train_on_batch(x_fake, y_fake)
            
            # update generator (g)
            x_gan = self.generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch,1))
            GAN.train_on_batch(x_gan, y_gan)
            # summarize loss on this batch
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
            # evaluate the model performance every so often
            if (i+1) % n_eval == 0:
                self.summarize_performance(i, g_model, d_model, latent_dim)

    def start(self):
        latent_dim = 5

        # Create discriminator models
        Discriminator_uns= self.Discriminator_model()
        # Create generator models
        Generator = self.Generator_model(latent_dim)
        # Create the GAN
        GAN = self.define_GAN(Generator, Discriminator_uns)

        # Load data
        dataset = self.traindata

        # Train model
        self.train(Generator, Discriminator_uns, GAN, latent_dim)

if __name__ == "__main__":

    ###############################################
    # LOAD TRAINING DATA
    ###############################################
    train_file_path = "./databaseANN/ErrorIncluded/trainingData_Feas_big.txt"
    # train_file_path = "./databaseANN/trainingData_Feas_V2plusfake.txt"

    # TD.plotInitialDataPandas(pairplot= False, corrplot= False, inputsplotbar = False, inputsplotbarFeas = True)
    # dataset_np = TD.LoadNumpy(train_file_path, plotDistribution = True)
    dataset_np = TD.LoadNumpy(train_file_path)
    traindata, testdata = TD.splitData_class(dataset_np)

    
    ###############################################
    # CREATE AND TRAIN CLASS NETWORK
    ###############################################
    perceptron = GAN_training(dataset_np)
    perceptron.get_traintestdata( traindata, testdata)
    perceptron.start()
