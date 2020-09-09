import numpy as np
import pykep as pk
import scipy.optimize as spy

from FitnessFunction_normalized import Fitness
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT

import LoadConfigFiles as CONFIG

import time

########################
# Initial settings
########################
SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables 
Fitness = Fitness(Nimp = SF.Nimp)

########################
# Decision Vector Outer loop 
########################
transfertime = 250

########################
# Calculate fitness
########################
def f(DecV):
    return Fitness.calculateFeasibility(DecV)
    
def optimize(): # Evolutionary Algorithm
    # Optimize outer loop
    EA = opt_config.EA
    start_time = time.time()
    f_min, Best = AL_OPT.EvolAlgorithm(f, SF.bnds , x_add = False, \
        ind = EA['ind'], max_iter = EA['iterat'], max_iter_success = EA['itersuccess'],
        elitism = EA['elitism'], mutation = EA['mutat'] )
    t = (time.time() - start_time) 

    print("Min", f_min,'time',t)    
    AL_BF.writeData(f_min, 'w', 'SolutionEA.txt')
    Fitness.calculateFitness(f_min, plot = True)
    Fitness.printResult()

def coordS():
    CS = opt_config.CS
    # Coordinate search opt
    f_min = np.loadtxt("SolutionEA.txt")

    f_min2, Best2 = AL_OPT.coordinateSearch(f, f_min, SF.bnds, CS['stepsize'], \
        x_add = False, max_iter = CS['maxiter'])
    print("Min2")
    print(f_min2)
    AL_BF.writeData(f_min2, 'w', 'SolutionCoord.txt')

def MBH():
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    DecV = np.zeros(len(SF.bnds))
    for i in range(len(SF.bnds)):
        DecV[i] = ( SF.bnds[i][0] + SF.bnds[i][1] ) /2

    cons = []
    for factor in range(len(DecV)):
        lower, upper = SF.bnds[factor]
        l = {'type': 'ineq',
            'fun': lambda x, a=lower, i=factor: x[i] - a}
        u = {'type': 'ineq',
            'fun': lambda x, b=upper, i=factor: b - x[i]}
        cons.append(l)
        cons.append(u)
        
    minimizer_kwargs = dict(method="COBYLA", constraints=(cons),options={'disp': False,  'maxiter': 100})#T

    start_time = time.time()
    fmin_3 = spy.basinhopping(f, DecV, niter = 30, minimizer_kwargs=minimizer_kwargs, niter_success = 10, take_step= mytakestep,callback=AL_OPT.print_fun)
    # DecV_optimized2 = spy.basinhopping(main, DecV, niter=20, minimizer_kwargs=minimizer_kwargs,niter_success = 5,callback=print_fun)
    # DecV_optimized2 = basinhopping(Problem, DecV, niter=2, minimizer_kwags=minimizer_kwargs,take_step=mytakestep,callback=print_fun)
    t = (time.time() - start_time) 

    print("Min3")
    AL_BF.writeData(fmin_3.x, 'w', 'SolutionMBH.txt')

def MBH_self():
    MBH = opt_config.MBH
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    DecV = np.zeros(len(SF.bnds))
    for i in range(len(SF.bnds)):
        DecV[i] = ( SF.bnds[i][0] + SF.bnds[i][1] ) /2

    cons = []
    for factor in range(len(DecV)):
        lower, upper = SF.bnds[factor]
        l = {'type': 'ineq',
            'fun': lambda x, a=lower, i=factor: x[i] - a}
        u = {'type': 'ineq',
            'fun': lambda x, b=upper, i=factor: b - x[i]}
        cons.append(l)
        cons.append(u)

    start_time = time.time()
    fmin_4, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep,\
                niter = MBH['niter_total'], niter_local = MBH['niter_local'], \
                niter_success = MBH['niter_success'], bnds = SF.bnds, \
                jumpMagnitude = MBH['jumpMagnitude'], tolLocal = MBH['tolLocal'],\
                tolGlobal = MBH['tolGlobal'])
    
    t = (time.time() - start_time) 
    print("Min4", fmin_4, 'time', t)
    AL_BF.writeData(fmin_4, 'w', 'SolutionMBH_self.txt')
    Fitness.calculateFitness(fmin_4, plot = True)
    Fitness.printResult()

def propagateSol():
    print("#######################################")
    print("Evolutionary Algorithm")
    print("#######################################") 
    DecV_I = np.loadtxt("SolutionEA.txt")

    f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
    Fitness.printResult()
    print(f) 

    print("#######################################")
    print("EA + coord search")
    print("#######################################") 
    DecV_I = np.loadtxt("SolutionCoord.txt")

    f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
    Fitness.printResult()
    print(f) 

    print("#######################################")
    print("MBH scipy")
    print("#######################################") 
    DecV_I = np.loadtxt("SolutionMBH.txt")

    f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
    Fitness.printResult()
    print(f) 
    feasibility = True
    for j in range(len(DecV_I)):
        if ( DecV_I[j] < SF.bnds[j][0] ) or ( DecV_I[j] > SF.bnds[j][1] ):
            # print(j, "Within bounds?", "min", bnds[j][0], "value",DecV_I[j], "max",bnds[j][1])
            feasibility = False
    print("Constraints:",feasibility)

    print("#######################################")
    print("MBH self implemented")
    print("#######################################")    
    DecV_I = np.loadtxt("SolutionMBH_self.txt")

    f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
    Fitness.printResult()
    print(f) 
    for j in range(len(DecV_I)):
        if ( DecV_I[j] < SF.bnds[j][0] ) or ( DecV_I[j] > SF.bnds[j][1] ):
            print(j, "Within bounds?", "min", SF.bnds[j][0], "value", DecV_I[j], "max", SF.bnds[j][1])

def propagateOne():
    DecV_I = np.loadtxt("SolutionMBH_self.txt")

    # f = Fitness.calculateFeasibility(DecV_I)
    Fitness.calculateFitness(DecV_I, plot = True)

if __name__ == "__main__":
    # optimize()
    # coordS()
    # MBH()
    # MBH_self()
    # propagateSol()
    propagateOne()

