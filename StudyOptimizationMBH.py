import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from FitnessFunction_normalized import Fitness
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT
import LoadConfigFiles as CONFIG

import time

SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables 
opt_config = CONFIG.OPT_config()

def runOpt(stepMagn, iterGlobal):
    ########################
    # Initial settings
    ########################
    FitnessF = Fitness(Nimp = SF.Nimp)

    ########################
    # Calculate fitness
    ########################
    def f(DecV):
        return FitnessF.calculateFeasibility(DecV)
        
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    DecV = np.zeros(len(SF.bnds))
    for i in range(len(SF.bnds)):
        DecV[i] = ( SF.bnds[i][0] + SF.bnds[i][1] ) /2

    ########################
    # Parameter sweep
    ########################
    Time = np.zeros( (len(iterGlobal), len(stepMagn) ))
    MIN = np.zeros( (len(iterGlobal), len(stepMagn) ))
    MASS = np.zeros( (len(iterGlobal), len(stepMagn) ))
    ERROR = np.zeros( (len(iterGlobal), len(stepMagn), 6 ))

    def deleteFile(name):
        file = open(name,"r+")
        file.truncate(0)
        file.close()

    deleteFile("./Results/StudyMBHSettings/1/Time.txt")
    deleteFile("./Results/StudyMBHSettings/1/Min.txt")
    deleteFile("./Results/StudyMBHSettings/1/Mass.txt")
    deleteFile("./Results/StudyMBHSettings/1/Error.txt")

    for i in range(len(iterGlobal)):
        for j in range(len(stepMagn)):
    # for i in range(1):
    #     for j in range(1):
            print("##################################################")
            print("IterGlobal",iterGlobal[i], "IterLocal", stepMagn[j])
            print("##################################################")
            start_time = time.time()
            fmin, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep, \
                niter = iterGlobal[i], niter_local = opt_config['MBH']['niter_local'], \
                niter_success = opt_config['MBH']['niter_success'], bnds = SF.bnds, jumpMagnitude = stepMagn[j], \
                tolLocal = opt_config['MBH']['tolLocal'], tolGlobal = opt_config['MBH']['tolGlobal'])
            t = (time.time() - start_time) 
            fit = FitnessF.calculateFitness(fmin, optMode = True, plot = False)
            
            # print(Best)
            Time[i,j] = t
            MIN[i,j] = Best
            MASS[i,j] = FitnessF.m_fuel
            ERROR[i,j,:] = FitnessF.Error
        
            with open('./Results/StudyMBHSettings/1/Time.txt', "a") as myfile:
                myfile.write(str(Time[i,j]) +' ' )
            with open('./Results/StudyMBHSettings/1/Min.txt', "a") as myfile:
                myfile.write(str(MIN[i,j]) +' ' )
            with open('./Results/StudyMBHSettings/1/Mass.txt', "a") as myfile:
                myfile.write(str(MASS[i,j]) +' ' )
            with open('./Results/StudyMBHSettings/1/Error.txt', "a") as myfile:
                myfile.write(str(ERROR[i,j]) +' ' )

        with open('./Results/StudyMBHSettings/1/Time.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyMBHSettings/1/Min.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyMBHSettings/1/Mass.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyMBHSettings/1/Error.txt', "a") as myfile:
            myfile.write('\n')

    # AL_BF.writeData(Time, 'w', 'Time.txt')
    # AL_BF.writeData(MIN, 'w', 'Min.txt')
    # AL_BF.writeData(MASS, 'w', 'Mass.txt')
    # AL_BF.writeData(ERROR, 'w', 'Error.txt')



    # print("Min")
    # print(f_min)
    # AL_BF.writeData(f_min, 'w', 'Solution.txt')

    # return IND, ITER

def loadData():
    time_1 = np.loadtxt("./Results/StudyMBHSettings/1/Time.txt")
    minVal_1 = np.loadtxt("./Results/StudyMBHSettings/1/Min.txt")
    mass_1 = np.loadtxt("./Results/StudyMBHSettings/1/Mass.txt")

    return [time_1], [minVal_1]

def plotConvergence(stepMagn, iterGlobal):
    time, minVal = loadData()

    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']

    fig= plt.figure()

    ax = fig.add_subplot(2, 3, 1)
    for i in range(len(stepMagn)):
        # plt.plot(ITER[:], time[i,:], 'x-', c = color[i], label = IND[i])
        plt.plot(iterGlobal[:], time[0][:,i] /60, 'x-', c = color[i], label = stepMagn[i])
    plt.xlabel('Iterations Global')
    plt.ylabel('Time of computation 1 (min)')
    plt.grid()
    plt.legend(title = "Individuals")
    # plt.show()

    ax = fig.add_subplot(2, 3, 2)
    for i in range(len(stepMagn)):
        # plt.plot(ITER[:], time[i,:], 'x-', c = color[i], label = IND[i])
        plt.plot(iterGlobal[:], minVal[0][:,i], 'x-', c = color[i], label = stepMagn[i])
    plt.xlabel('Iterations Global')
    plt.ylabel('Minimum value 1')
    plt.grid()
    ax.set_yscale('log')
    plt.legend(title = "Individuals")

    # ax = fig.add_subplot(2, 3, 3)
    # for i in range(len(IND)):
    #     # plt.plot(ITER[:], time[i,:], 'x-', c = color[i], label = IND[i])
    #     plt.plot(ITER[:], time[2][i,:] /60, 'x-', c = color[i], label = IND[i])
    # plt.xlabel('Iterations')
    # plt.ylabel('Time of computation 2 (min)')
    # plt.grid()
    # plt.ylim(-100/60, 3000/60)
    # plt.legend(title = "Individuals")
    # # plt.show()

    # ax = fig.add_subplot(2, 3, 4)
    # for i in range(len(IND)):
    #     plt.plot(ITER[:], minVal[0][i,:], 'x-', c = color[i], label = IND[i])
    # plt.xlabel('Iterations')
    # plt.ylabel('Minimum value 1')
    # plt.grid()
    # ax.set_yscale('log')
    # plt.legend(title = "Individuals")
 


    # ax = fig.add_subplot(2, 3, 5)
    # for i in range(len(IND)):
    #     plt.plot(ITER[:], minVal[1][i,:], 'x-', c = color[i], label = IND[i])
    # plt.xlabel('Iterations')
    # plt.ylabel('Minimum value 2')
    # plt.grid()
    # ax.set_yscale('log')
    # plt.legend(title = "Individuals")

    # ax = fig.add_subplot(2, 3, 6)
    # for i in range(len(IND)):
    #     plt.plot(ITER[:], minVal[2][i,:], 'x-', c = color[i], label = IND[i])
    # plt.xlabel('Iterations')
    # plt.ylabel('Minimum value 2')
    # plt.grid()
    # ax.set_yscale('log')
    # plt.legend(title = "Individuals")

    plt.show()

def studyNiterlocal(nsamples, niter):
    ########################
    # Initial settings
    ########################
    FitnessF = Fitness(Nimp = SF.Nimp)

    ########################
    # Calculate fitness
    ########################
    def f(DecV):
        return FitnessF.calculateFeasibility(DecV)
        
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    ########################
    # Parameter sweep
    ########################
    Time = np.zeros( (nsamples, len(niter) ))
    MIN = np.zeros( (nsamples, len(niter) ))
    ERROR = np.zeros( (nsamples, len(niter), 6 ))

    def deleteFile(name):
        file = open(name,"r+")
        file.truncate(0)
        file.close()

    deleteFile("./Results/StudyMBHSettings/2/Time.txt")
    deleteFile("./Results/StudyMBHSettings/2/Min.txt")
    deleteFile("./Results/StudyMBHSettings/2/Error.txt")

    samples = np.zeros((nsamples, len(SF.bnds)))
    for decv in range(len(SF.bnds)): 
        samples[:, decv] = np.random.uniform(low = SF.bnds[decv][0], \
            high = SF.bnds[decv][1], size = nsamples)


    for i in range(nsamples):
        DecV = samples[i,:]
        for j in range(len(niter)):
            print("##################################################")
            print("samples",i, "IterLocal", j)
            print("##################################################")
            niter_success = niter[j]
            
            start_time = time.time()

            fmin, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep, \
                niter = niter[j], niter_local = 100, \
                niter_success = niter_success,bnds = SF.bnds, jumpMagnitude = MBH['jumpMagnitude'], \
                tolLocal = MBH['tolLocal'], tolGlobal = MBH['tolGlobal'])
            t = (time.time() - start_time) 
            fit = FitnessF.calculateFitness(fmin, optMode = True, plot = False)
            
            DecV = FitnessF.DecV # Start next number of iterations from this one
            Time[i,j] = t
            MIN[i,j] = Best
            ERROR[i,j,:] = FitnessF.Error
        
            with open('./Results/StudyMBHSettings/2/Time.txt', "a") as myfile:
                myfile.write(str(Time[i,j]) +' ' )
            with open('./Results/StudyMBHSettings/2/Min.txt', "a") as myfile:
                myfile.write(str(MIN[i,j]) +' ' )
            with open('./Results/StudyMBHSettings/2/Error.txt', "a") as myfile:
                myfile.write(str(ERROR[i,j]) +' ' )

        with open('./Results/StudyMBHSettings/2/Time.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyMBHSettings/2/Min.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyMBHSettings/2/Error.txt', "a") as myfile:
            myfile.write('\n')

def loadDataLocal():
    time_1 = np.loadtxt("./Results/StudyMBHSettings/2/Time.txt")
    minVal_1 = np.loadtxt("./Results/StudyMBHSettings/2/Min.txt")

    return time_1, minVal_1

def plotConvergenceLocal(nsamples, niter):
    time, minVal = loadDataLocal()

    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']

    fig = plt.figure()

    niter_add = np.cumsum(niter)
    ax = fig.add_subplot(1, 2, 1)
    print(time[0,:])
    for i in range(nsamples):
        plt.plot(niter_add, time[i, :] /60, 'x-', c = color[i%len(color)])
    plt.xlabel('Iterations Global')
    plt.ylabel('Time of computation 1 (min)')
    plt.grid()
    # plt.legend(title = "Individuals")
 
    ax = fig.add_subplot(1, 2, 2)
    for i in range(nsamples):
        plt.plot(niter_add, minVal[i,:], 'x-', c = color[i%len(color)])
    plt.xlabel('Iterations')
    plt.ylabel('Minimum value 2')
    plt.grid()
    ax.set_yscale('log')

    plt.show()



if __name__ == "__main__":

    # Study for Convergece step global and step mag
    iterGlobal = np.array([10, 20, 50, 100, 200])
    stepMagn = np.array([0.01,0.02,0.05, 0.2])
    # runOpt(stepMagn, iterGlobal)
    # plotConvergence(stepMagn, iterGlobal)



    # Study for local jump convergence
    niter = np.array([10, 10, 10, 20]) # Adds to previous niter
    nsamples = 10

    # studyNiterlocal(nsamples, niter)
    plotConvergenceLocal(nsamples, niter)