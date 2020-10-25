import numpy as np

from FitnessFunction_normalized import Fitness
from ANN_reg import ANN_reg
import TrainingDataKeras as TD

import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_OPT as AL_OPT

import LoadConfigFiles as CONFIG

import time

########################
# Initial settings
########################
# Sims Flanagan
SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables 
FIT = CONFIG.Fitness_config()
Fitness = Fitness(Nimp = SF.Nimp)

# Database for inverse standardization
train_file_path = "./databaseANN/ErrorIncluded/trainingData_Feas_big.txt"
dataset_np = TD.LoadNumpy(train_file_path)

# ANN
ANN = ANN_reg(dataset_np)
ANN.load_model_fromFile()

# optimization
opt_config = CONFIG.OPT_config()

########################
# Calculate fitness
########################
def f(DecV):
    # Mass
    mass = np.zeros((opt_config.EA['ind']))
    for i in range(opt_config.EA['ind']):
        mass[i] = Fitness.calculateMass(DecV[i,:])

    # Error
    t0_reg = time.time()
    # Transform inputs
    ind = len(DecV[:,0])
    input_Vector = np.zeros((ind,8))
    for i in range(ind):
        input_Vector[i,:] = Fitness.DecV2inputV(newDecV = DecV[i,:])
    
    # Feasibility 
    feas = ANN.predict(fromFile = True, 
                        testfile = input_Vector,
                        rescale = True)
    tf_reg = (time.time() - t0_reg) 
    print('Time network eval', tf_reg)

    # Fitness Function
    fc1 = feas[:,0] / AL_BF.AU # Normalize with AU
    fc2 = feas[:,1] / AL_BF.AU * AL_BF.year2sec(1)

    print(feas[0:5,:])
    print(fc1[0:5], fc2[0:5], mass[0:5])

    value = fc1 * FIT.FEAS['factor_pos'] + \
            fc2 * FIT.FEAS['factor_vel'] + \
            mass * FIT.FEAS['factor_mass']
    return value 

def EA(): # Evolutionary Algorithm
    # Optimize outer loop
    EA = opt_config.EA
    start_time = time.time()
    f_min, Best = AL_OPT.EvolAlgorithm(f, SF.bnds , x_add = False, \
        ind = EA['ind'], max_iter = EA['iterat'], max_iter_success = EA['itersuccess'],
        elitism = EA['elitism'], mutation = EA['mutat'], immig = EA['immig'],
        bulk_fitness = True )
    t = (time.time() - start_time) 

    print("Min", f_min,'time',t)    
    # AL_BF.writeData(f_min, 'w', 'SolutionEA.txt')

    # SHOW RESULT
    Fitness.calculateFitness(f_min, plot = True)
    Fitness.printResult()

def MBH_self():
    MBH = opt_config.MBH
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    DecV = np.zeros(len(SF.bnds))
    for i in range(len(SF.bnds)):
        DecV[i] = ( SF.bnds[i][0] + SF.bnds[i][1] ) /2

    cons = {'type': 'eq', 'fun': f_class}

    start_time = time.time()
    fmin_4, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep,\
                niter = MBH['niter_total'], niter_local = MBH['niter_local'], \
                niter_sucess = MBH['niter_success'], bnds = SF.bnds, \
                jumpMagnitude = MBH['jumpMagnitude'], tolLocal = MBH['tolLocal'],\
                tolGlobal = MBH['tolGlobal'], cons = cons)
    
    t = (time.time() - start_time) 
    print("Min4", fmin_4, 'time', t)
    AL_BF.writeData(fmin_4, 'w', 'SolutionMBH_self.txt')
    Fitness.calculateFitness(fmin_4, plot = True)
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
    EA()
    # MBH_self()
    # propagateOne()
    # evaluateFeasibility() # Compare speed of 3 evaluations

