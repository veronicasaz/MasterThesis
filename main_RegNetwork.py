import numpy as np
import scipy.optimize as spy

from FitnessFunction_normalized import Fitness
from ANN_reg import ANN_reg
import TrainingDataKeras as TD

import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_OPT as AL_OPT
import GenerateTrainingDataFromOpt as GTD
import ANN_reg as AR

import LoadConfigFiles as CONFIG

import time

########################
# Initial settings
########################
SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables 
FIT_C = CONFIG.Fitness_config()
FIT = FIT_C.Fit_config

Fitness = Fitness(Nimp = SF.Nimp)

Dataset_conf = CONFIG.Dataset()
Scaling = Dataset_conf.Dataset_config['Scaling']

opt_config = CONFIG.OPT_config()
EA = opt_config.EA
MBH = opt_config.MBH
MBH_batch = opt_config.MBH_batch


# Database for inverse standardization
train_file_path = "./databaseANN/DatabaseOptimized/deltakeplerian/500_AU/Random.txt"
save_file_path = "./databaseANN/DatabaseOptimized/deltakeplerian/500_AU/"

dataset_np = TD.LoadNumpy(train_file_path, save_file_path, error= 'vector',\
            standardizationType = Scaling['type_stand'], scaling = Scaling['scaling'], \
            dataUnits = Dataset_conf.Dataset_config['DataUnits'], Log = Dataset_conf.Dataset_config['Log'],\
            plotDistribution=False, plotErrors=False) # It is necessary to fit the data for the rescaling
    
traindata, testdata = TD.splitData_reg(dataset_np)
ANN = AR.ANN_reg(dataset_np)
ANN.get_traintestdata(traindata, testdata)


# optimization


########################
# Calculate fitness
########################
def f(DecV):
    # Error
    t0_reg = time.time()
    # Transform inputs
    ind = len(DecV[:,0])
    input_Vector = np.zeros((ind,8))
    for i in range(ind):
        input_Vector[i,:] = Fitness.DecV2inputV( 'deltakeplerian', newDecV = DecV[i,:])
    
    # Standardize input vector
    input_Vector_std = dataset_np.standardize_withoutFitting(input_Vector, 'I')
    # Feasibility 
    feas = ANN.predict(fromFile = True, 
                        testfile = input_Vector_std,
                        rescale = True)
    tf_reg = (time.time() - t0_reg) 
    print('Time network eval', tf_reg)

    # Fitness Function
    value = np.zeros((ind,1))
    for i in range(ind):
        value[i] = Fitness.objFunction(feas[i,1:], feas[i,0])
    return value 

def f_notANN(DecV):
    """
    pass 1 dec v, not an array
    """

    # value = np.zeros((ind,1))
    # for i in range(ind):
    value = Fitness.calculateFitness(DecV)

    # ind = len(DecV[:,0])
    # value = np.zeros((ind,1))
    # for i in range(ind):
    #     value[i] = Fitness.calculateFitness(DecV[i])
    return value 

def EA(): # Evolutionary Algorithm
    # Optimize outer loop
    start_time = time.time()
    f_min, Best = AL_OPT.EvolAlgorithm(f, SF.bnds , x_add = False, \
        ind = EA['ind'], 
        max_iter = EA['iterat'],
        max_iter_success = EA['itersuccess'],
        elitism = EA['elitism'], 
        mutation = EA['mutat'], 
        immig = EA['immig'],
        bulk_fitness = True )
    t = (time.time() - start_time) 

    print("Min", f_min,'time',t)    
    # AL_BF.writeData(f_min, 'w', 'SolutionEA.txt')

    # SHOW RESULT
    Fitness.calculateFitness(f_min, plot = True)
    Fitness.printResult()

def MBH_self():
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    DecV = np.zeros(len(SF.bnds))
    for i in range(len(SF.bnds)):
        DecV[i] = ( SF.bnds[i][0] + SF.bnds[i][1] ) /2

    cons = {'type': 'eq', 'fun': f_class}

    start_time = time.time()
    fmin_4, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep,\
                niter = MBH['niter_total'], 
                niter_local = MBH['niter_local'], \
                niter_sucess = MBH['niter_success'], 
                bnds = SF.bnds, \
                jumpMagnitude = MBH['jumpMagnitude'], 
                tolLocal = MBH['tolLocal'],\
                tolGlobal = MBH['tolGlobal'], 
                cons = cons)
    
    t = (time.time() - start_time) 
    print("Min4", fmin_4, 'time', t)
    AL_BF.writeData(fmin_4, 'w', 'SolutionMBH_self.txt')
    Fitness.calculateFitness(fmin_4, plot = True)
    Fitness.printResult()

def MBH_batch_f():
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    DecV = np.zeros(len(SF.bnds))
    DecV = GTD.latinhypercube(len(SF.bnds), MBH_batch['nind']) #initialize with latin hypercube


    start_time = time.time()
    fmin_4, Best = AL_OPT.MonotonicBasinHopping_batch(f_notANN, DecV, mytakestep,\
                f_opt = f, 
                nind = MBH_batch['nind'], 
                niter = MBH_batch['niter_total'], 
                niter_sucess = MBH_batch['niter_success'], \
                bnds = SF.bnds, \
                jumpMagnitude = MBH_batch['jumpMagnitude'],\
                tolGlobal = MBH_batch['tolGlobal'], )
    
    t = (time.time() - start_time) 
    print("Min4", min(Best), 'time', t)
    best_input = fmin_4[np.where(Best == min(Best))[0] ]
    AL_BF.writeData(best_input, 'w', 'SolutionMBH_batch.txt')

    # Locally optimize best to obtain actual inputs of that one
    solutionLocal = spy.minimize(f_notANN, best_input, method = 'SLSQP', \
                tol = 0.01, bounds = SF.bnds)
    Fitness.calculateFitness(solutionLocal.x, plot = True)
    Fitness.printResult()

def evaluateFeasibility():
    ind = 1000
    pop_0 = np.zeros([ind, len(SF.bnds)])
    for i in range(len(SF.bnds)):
        pop_0[:,i] = np.random.rand(ind) * (SF.bnds[i][1]-SF.bnds[i][0]) + SF.bnds[i][0]
    
    #Fitness function
    feas = np.zeros(ind)
    t0_fit = time.time()
    for i in range(ind):
        DecV = pop_0[i,:]
        Fitness.calculateFeasibility(DecV)
        feas[i] = Fitness.studyFeasibility()

    tf_fit = (time.time() - t0_fit) 
    print('Time fitness eval', tf_fit)
    print('Number feasible', np.count_nonzero(feas==1))

    # ANN individual
    feas2 = np.zeros(ind)
    t0_class = time.time()
    for i in range(ind):
        DecV = pop_0[i,:]
        # Transform inputs
        input_Vector = Fitness.DecV2inputV(newDecV = DecV)
        # Feasibility
        feas2[i] = ANN.predict_single(input_Vector)

    tf_class = (time.time() - t0_class) 
    print('Time network eval', tf_class)
    print('Number feasible', np.count_nonzero(feas2==1))

    # ANN batch
    feas2 = np.zeros(ind)
    t0_class = time.time()
    input_Vector = np.zeros((ind,8))
    for i in range(ind):
        DecV = pop_0[i,:]
        # Transform inputs
        input_Vector[i,:] = Fitness.DecV2inputV(newDecV = DecV)
    # Feasibility
    feas2 = ANN.predict(input_Vector)
    feas2 = abs(np.array(feas2)-1)

    tf_class = (time.time() - t0_class) 
    print('Time network eval', tf_class)
    print('Number feasible', np.count_nonzero(feas2==0))

def propagateOne():
    DecV_I = np.loadtxt("SolutionMBH_self.txt")

    # f = Fitness.calculateFeasibility(DecV_I)
    Fitness.calculateFitness(DecV_I, plot = True)

if __name__ == "__main__":
    # EA()
    # MBH_self()
    MBH_batch_f()
    # propagateOne()
    # evaluateFeasibility() # Compare speed of 3 evaluations

