import numpy as np
import scipy.optimize as spy

from FitnessFunction_normalized import Fitness
from ANN_reg_2 import ANN_reg
import TrainingData as TD
from GenerateTrainingDataFromOpt import latinhypercube

import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_OPT as AL_OPT
from AstroLibraries import AstroLib_Ephem as AL_Eph

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


# Load database
base = "./databaseANN/3_DatabaseLast/deltakeplerian/"
ANN = ANN_reg(save_path = base, 
              n_classes = 2, n_input = 8, 
              output_label= ['ep', 'ev']) # from dataset_np to not load it
ANN.load_model_fromFile() # To speed it up later

########################
# Calculate fitness
########################
def f_ANN(DecV):
    # Error
    t0_reg = time.time()
    # Transform inputs
    ind = len(DecV[:,0])
    input_Vector = np.zeros((ind,8))
    for i in range(ind):
        input_Vector[i,:] = Fitness.DecV2inputV( 'deltakeplerian', newDecV = DecV[i,:])
    
    # Standardize input vector
    input_Vector_std = TD.standardize_withoutFitting(input_Vector, 'I', base)
    
    # Feasibility 
    feas = ANN.predict(fromFile = False, 
                        testfile = input_Vector_std,
                        rescale = True) # Already loaded above
    tf_reg = (time.time() - t0_reg) 
    print('Time eval eval', tf_reg)

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

def f_notANN_test(DecV):
    """
    pass 1 dec v, not an array
    """

    v0 = np.sqrt(11.5) * 1000
    v1 = np.sqrt(18.6) * 1000
    t0 = AL_Eph.DateConv([16, 6, 2022], 'calendar')
    t0 = t0.JD_0
    tf = AL_Eph.DateConv([5, 2, 2025], 'calendar')
    t_t = AL_BF.days2sec( tf.JD_0 - t0)

    fixed = [v0, v1, t0, t_t]
    fix = [0, 3, 6, 7]

    # value = np.zeros((ind,1))
    # for i in range(ind):
    DecV_2 = np.zeros(len(DecV)+len(fix))

    index = 0
    for i in range(len(DecV)):
        if i in fix:
            DecV_2[i] = fixed[fix.index(i)] 
        else:
            DecV_2[i] = DecV[index]
            index += 1
   
    value = Fitness.calculateFitness(DecV_2)

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


    start_time = time.time()
    fmin_4, Best = AL_OPT.MonotonicBasinHopping(f_notANN, DecV, mytakestep,\
                niter = MBH['niter_total'], 
                niter_local = MBH['niter_local'], \
                niter_sucess = MBH['niter_success'], 
                bnds = SF.bnds, \
                jumpMagnitude = MBH['jumpMagnitude'], 
                tolLocal = MBH['tolLocal'],\
                tolGlobal = MBH['tolGlobal'])
    
    t = (time.time() - start_time) 
    print("Min4", fmin_4, 'time', t)
    AL_BF.writeData(fmin_4, 'w', 'SolutionMBH_self.txt')
    Fitness.calculateFitness(fmin_4, plot = True)
    Fitness.printResult()

def MBH_batch_f(ML = False):
    if ML == True:
        f_opt = f_ANN
    else:
        f_opt = None
        
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    DecV = np.zeros(len(SF.bnds))
    DecV = latinhypercube(len(SF.bnds), len(SF.bnds), MBH_batch['nind']) #initialize with latin hypercube

    start_time = time.time()
    fmin_4, Best = AL_OPT.MonotonicBasinHopping_batch(f_notANN, DecV, mytakestep,\
                f_opt = f_opt, 
                nind = MBH_batch['nind'], 
                niter = MBH_batch['niter_total'], 
                niter_success = MBH_batch['niter_success'], \
                bnds = SF.bnds, \
                jumpMagnitude = MBH_batch['jumpMagnitude'],\
                tolGlobal = MBH_batch['tolGlobal'],
                tolLocal = MBH_batch['tolLocal'] )
    
    t = (time.time() - start_time) 
    print("Min4", min(Best), 'time', t)
    best_input = fmin_4[np.where(Best == min(Best))[0] ]
    AL_BF.writeData(best_input, 'w', './OptSol/SolutionMBH_batch.txt')

    # Locally optimize best to obtain actual inputs of that one
    solutionLocal = spy.minimize(f_notANN, best_input, method = 'SLSQP', \
                tol = 0.01, bounds = SF.bnds)
    Fitness.calculateFitness(solutionLocal.x, plot = True)
    Fitness.printResult()

def MBH_batch_test(ML = False):
    """
    test with reference problem to see the optimization results. The velocities are 
    fixed in magnitude to overcome the fact that in the paper mars is just a GA
    """
    if ML == True:
        f_opt = f_ANN
    else:
        f_opt = None
        
    # exclude limits of v magn in the dec vector
    lims = list(SF.bnds)
    fix = [0, 3, 6, 7] # choose indixes of values of decv to fix
    for i in reversed(fix):
        del lims[i]

    lims = tuple(lims)

    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, lims)

    DecV = np.zeros(len(lims))
    DecV = latinhypercube(len(lims), len(lims), MBH_batch['nind'], lims = lims) #initialize with latin hypercube

    start_time = time.time()
    fmin_4, Best = AL_OPT.MonotonicBasinHopping_batch(f_notANN_test, DecV, mytakestep,\
                f_opt = f_opt, 
                nind = MBH_batch['nind'], 
                niter = MBH_batch['niter_total'], 
                niter_success = MBH_batch['niter_success'], \
                bnds = lims, \
                jumpMagnitude = MBH_batch['jumpMagnitude'],\
                tolGlobal = MBH_batch['tolGlobal'],
                tolLocal = MBH_batch['tolLocal'] )
    
    t = (time.time() - start_time) 
    print("Min4", min(Best), 'time', t)
    best_input = fmin_4[np.where(Best == min(Best))[0] ]
    AL_BF.writeData(best_input, 'w', './OptSol/SolutionMBH_batch.txt')

def propagate_test():
    DecV = np.genfromtxt("./OptSol/SolutionMBH_batch.txt", delimiter = ' ', dtype = float)

    v0 = np.sqrt(11.5) * 1000
    v1 = np.sqrt(18.6) * 1000
    t0 = AL_Eph.DateConv([16, 6, 2022], 'calendar')
    t0 = t0.JD_0
    tf = AL_Eph.DateConv([5, 2, 2025], 'calendar')
    t_t = AL_BF.days2sec( tf.JD_0 - t0)

    fixed = [v0, v1, t0, t_t]
    fix = [0, 3, 6, 7]

    # value = np.zeros((ind,1))
    # for i in range(ind):
    DecV_2 = np.zeros(len(DecV)+len(fix))

    index = 0
    for i in range(len(DecV)):
        if i in fix:
            DecV_2[i] = fixed[fix.index(i)] 
        else:
            DecV_2[i] = DecV[index]
            index += 1
   
    value = Fitness.calculateFitness(DecV_2, plot = True)
    Fitness.printResult()
    # ind = len(DecV[:,0])
    # value = np.zeros((ind,1))
    # for i in range(ind):
    #     value[i] = Fitness.calculateFitness(DecV[i])
    return value 

    Fitness.printResult()

def evaluateFeasibility():
    """
    compare speeds of ML and numerical calculation of the fitness
    """
    ind = 10
    pop_0 = np.zeros([ind, len(SF.bnds)])
    for i in range(len(SF.bnds)):
        pop_0[:,i] = np.random.rand(ind) * (SF.bnds[i][1]-SF.bnds[i][0]) + SF.bnds[i][0]
    
    #Fitness function
    feas = np.zeros((ind, 2))
    t0_fit = time.time()
    for i in range(ind):
        DecV = pop_0[i,:]
        fitness = Fitness.calculateFitness(DecV)
        feas[i, 0] = Fitness.Epnorm
        feas[i, 1] = Fitness.Evnorm
    tf_1 = (time.time() - t0_fit) 
    

    # ANN individual
    feas2 = np.zeros((ind, 2))
    t0_class = time.time()
    for i in range(ind):
        DecV = pop_0[i,:]
        # Transform inputs
        input_Vector = Fitness.DecV2inputV( 'deltakeplerian', newDecV = DecV)
        input_Vector_std = dataset_np.standardize_withoutFitting(input_Vector, 'I')
    
        # Feasibility
        feas2[i, :] = ANN.singlePrediction(input_Vector_std)

    tf_2 = (time.time() - t0_class) 


    # ANN batch
    feas3 = np.zeros((ind, 2))
    t0_class = time.time()
    input_Vector = np.zeros((ind,8))
    for i in range(ind):
        DecV = pop_0[i,:]
        # Transform inputs
        input_Vector_i = Fitness.DecV2inputV('deltakeplerian', newDecV = DecV)
        input_Vector[i,:] = dataset_np.standardize_withoutFitting(input_Vector_i, 'I')
    t0_class_2 = time.time()
    # Feasibility
    feas3 = ANN.predict(testfile = input_Vector, rescale = True)
    tf_3 = (time.time() - t0_class) 
    tf_3_mid = ( t0_class_2 - t0_class) 


    print('Time fitness eval 1', tf_1)
    print('Time network eval 2', tf_2)
    print('Time network eval 3', tf_3, "Conversion of vector:", tf_3_mid)


    for i in range(ind):
        print(feas[i,:],'\n', feas3[i,:], '\n',feas2[i,:])
        print("===============================")

def propagateOne():
    DecV_I = np.genfromtxt("./OptSol/SolutionMBH_batch.txt", delimiter = ' ', dtype = float)

    # Random generation
    DecV_R = np.zeros(len(SF.bnds))
    for decv in range(len(SF.bnds)): # Add impulses that won't be used for Lambert
        DecV_R[decv] = np.random.uniform(low = SF.bnds[decv][0], \
            high = SF.bnds[decv][1], size = 1)

    # print(DecV_R)
    # f = Fitness.calculateFeasibility(DecV_I)
    Fitness.calculateFitness(DecV_I, plot = False, plot3D = True)
    # Fitness.plot_tvsT()

if __name__ == "__main__":
    # EA()
    # MBH_self()
    # MBH_batch_f(ML = False) # Without ML
    # MBH_batch_f(ML = True) # With ML

    # TEST WITH PAPER
    # MBH_batch_test(ML = False) # Without ML
    # propagate_test()

    # propagateOne()
    evaluateFeasibility() # Compare speed of 3 evaluations with and without ML

    # Without ML