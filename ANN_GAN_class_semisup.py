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
###################################################################

ANN = CONF.ANN_GAN()

class GAN_training:
    def __init__(self, dataset):
        self.dataset_np = dataset

        self.n_classes = self.dataset_np.n_classes # Labels
        self.n_examples = self.dataset_np.nsamples # samples
        self.n_input = self.dataset_np.n_input #inputs

    def get_traintestdata(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def Discriminator_model(self, regularization = True):
        # Create architecture
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        # model_sup = keras.Sequential()
        model_uns = keras.Sequential()
        model_sup = keras.Sequential()

        for model in [model_sup, model_uns]:
            if ANN.Discriminator['architecture']['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
                model.add(keras.layers.Dense(
                        ANN.Discriminator['architecture']['neuron_hidden'], 
                        input_dim = self.n_input,
                        activation='relu', 
                        use_bias=True, bias_initializer='zeros',
                        kernel_initializer = initializer,
                        kernel_regularizer= keras.regularizers.l2(ANN.Discriminator['architecture']['regularizer_value']) ))

                for layer in range(ANN.Discriminator['architecture']['hidden_layers']-1):
                    model.add(keras.layers.Dense(
                        ANN.Discriminator['architecture']['neuron_hidden'], 
                        activation='relu', 
                        use_bias=True, bias_initializer='zeros',
                        kernel_initializer = initializer,
                        kernel_regularizer= keras.regularizers.l2(ANN.Discriminator['architecture']['regularizer_value']) ))
            else:
                model.add(keras.layers.Dense(
                        ANN.Discriminator['architecture']['neuron_hidden'], 
                        input_dim = self.n_input,
                        activation='relu', 
                        use_bias=True, bias_initializer='zeros',
                        kernel_initializer = initializer))

                for layer in range(ANN.Discriminator['architecture']['hidden_layers'] -1):
                    model.add(keras.layers.Dense(
                        ANN.Discriminator['architecture']['neuron_hidden'], 
                        activation='relu', 
                        use_bias=True, bias_initializer='zeros',
                        kernel_initializer = initializer) )

        model_sup.add(keras.layers.Dense(self.n_classes, activation ='softmax' )) # output layer
        model_uns.add(keras.layers.Dense(1, activation ='sigmoid' ))
        
        # Compile
        model_sup.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        model_uns.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

        return model_uns, model_sup
            
    def Generator_model(self, latent_dim):
        initializer = tf.keras.initializers.GlorotNormal() # Glorot uniform by defaut
        
        model = keras.Sequential()

        if ANN.Generator['architecture']['regularization'] == True:  # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
            model.add(keras.layers.Dense(
                    ANN.Generator['architecture']['neuron_hidden'], 
                    activation='relu', 
                    input_dim = latent_dim,
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN.Generator['architecture']['regularizer_value']) ))
                    
            for layer in range(ANN.Generator['architecture']['hidden_layers']-1):
                model.add(keras.layers.Dense(
                    ANN.Generator['architecture']['neuron_hidden'], 
                    activation='relu', 
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer,
                    kernel_regularizer= keras.regularizers.l2(ANN.Generator['architecture']['regularizer_value']) ))
        else:
            model.add(keras.layers.Dense(
                    ANN.Generator['architecture']['neuron_hidden'], 
                    activation='relu', 
                    input_dim = latent_dim,
                    use_bias=True, bias_initializer='zeros',
                    kernel_initializer = initializer) )
            for layer in range(ANN.Generator['architecture']['hidden_layers']-1):
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
        model.add(Discriminator_uns)
        
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def select_supervised_samples(self ):
        X, y = self.traindata
        X_list, y_list = list(), list()
        n_per_class = np.count_nonzero(y == 1)

        # TODO: remove. Extra number of unfeasible
        
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
        return x_input

    def generate_fake_samples(self, Generator, latent_dim, n_samples):
        x_input = self.generate_latent_points(latent_dim, n_samples)

        # Predict output
        X = Generator.predict(x_input)

        # Create class labels
        y = np.zeros((n_samples,1))

        return X, y

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


    def train(self, Generator, Discriminator_uns, Discriminator_sup, GAN, latent_dim):
    
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(self.traindata[1])/ ANN.Training['n_batch'])
        # calculate the number of training iterations
        n_steps = bat_per_epo * ANN.Training['n_epochs']
        # calculate the size of half a batch of samples
        half_batch = int(ANN.Training['n_batch'] / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % 
                (ANN.Training['n_epochs'], 
                ANN.Training['n_batch'], 
                half_batch, 
                bat_per_epo, 
                n_steps))

        # select supervised dataset
        X_sup, y_sup = self.select_supervised_samples()
        print("Size of supervised samples", X_sup.shape, y_sup.shape)

        # manually enumerate epochs
        for i in range(ANN.Training['n_epochs']):
            # update supervised discriminator (c)
            [Xsup_real, ysup_real], _ = self.generate_real_samples([X_sup, y_sup], half_batch)
            c_loss, c_acc = Discriminator_sup.train_on_batch(Xsup_real, ysup_real)

            # update unsupervised discriminator (d)
            [X_real, _], y_real = self.generate_real_samples(self.traindata, half_batch)
            d_loss1 = Discriminator_uns.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(Generator, latent_dim, half_batch)
            d_loss2 = Discriminator_uns.train_on_batch(X_fake, y_fake)

            # update generator
            X_gan, y_gan = self.generate_latent_points(latent_dim, ANN.Training['n_batch']), \
                                                        np.ones((ANN.Training['n_batch'], 1))
            g_loss = GAN.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
            # evaluate the model performance every so often
            if (i+1) % (bat_per_epo * 1) == 0:
                self.summarize_performance(i, Generator, Discriminator_sup, latent_dim)

        # serialize weights to HDF5
        Generator.save_weights("./trainedCNet_GAN_real/Generator.h5")
        Discriminator_sup.save_weights("./trainedCNet_GAN_real/Discriminator_sup.h5")
        print("Saved model to disk")

    def start(self):
        latent_dim = ANN.Training['latent_dim']

        # Create discriminator models
        Discriminator_uns, Discriminator_sup = self.Discriminator_model()
        # Create generator models
        Generator = self.Generator_model(latent_dim)
        # Create the GAN
        GAN = self.define_GAN(Generator, Discriminator_uns)

        # Load data
        dataset = self.traindata

        # Train model
        self.train(Generator, Discriminator_uns, Discriminator_sup, GAN, latent_dim)

    def generate_samples(self, n_samples, latent_dim, namefile):
        # Load trained generator
        loaded_model = self.Generator_model(latent_dim)
        # load weights into new model
        loaded_model.load_weights("./trainedCNet_GAN_real/Generator.h5")
        print("Loaded model from disk")

        # Create samples
        x, y  = self.generate_fake_samples(loaded_model, latent_dim, n_samples)
        save_input = np.column_stack((y,x))

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
        loaded_model_un, loaded_model = self.Discriminator_model()
        # load weights into new model
        loaded_model.load_weights("./trainedCNet_GAN_real/Discriminator_sup.h5")

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
    perceptron.generate_samples(60, ANN.Training['latent_dim'], nameFile) # Datbase with real and fake data. 
     #                           # Label indicates if it is real (1) or fake (0)
    perceptron.see_samples(nameFile)

    perceptron.evaluate_discriminator()