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
EA = opt_config.EA

def runOpt(IND, ITER):
    ########################
    # Initial settings
    ########################
    FitnessF = Fitness(Nimp = SF.Nimp)

    ########################
    # Calculate fitness
    ########################
    def f(DecV):
        return FitnessF.calculateFeasibility(DecV)
        

    ########################
    # Parameter sweep
    ########################

    Time = np.zeros( (len(IND), len(ITER) ))
    MIN = np.zeros( (len(IND), len(ITER) ))
    MASS = np.zeros( (len(IND), len(ITER) ))
    ERROR = np.zeros( (len(IND), len(ITER), 6 ))

    def deleteFile(name):
        file = open(name,"r+")
        file.truncate(0)
        file.close()

    deleteFile("./Results/StudyEASettings/AfterModifyingDynamics_1/Time.txt")
    deleteFile("./Results/StudyEASettings/AfterModifyingDynamics_1/Min.txt")
    deleteFile("./Results/StudyEASettings/AfterModifyingDynamics_1/Mass.txt")
    deleteFile("./Results/StudyEASettings/AfterModifyingDynamics_1/Error.txt")

    for i in range(len(IND)):
        for j in range(len(ITER)):
    # for i in range(1):
    #     for j in range(1):
            print("##################################################")
            print("IND",IND[i], "ITER", ITER[j])
            print("##################################################")
            start_time = time.time()
            f_min, Best = AL_OPT.EvolAlgorithm(f, SF.bnds , x_add = False, \
                ind = IND[i], max_iter = ITER[j], max_iter_success = EA['itersuccess'] )
            t = (time.time() - start_time) 
            fit = FitnessF.calculateFitness(f_min, optMode = True, plot = False)
            
            # print(Best)
            Time[i,j] = t
            MIN[i,j] = Best
            MASS[i,j] = FitnessF.m_fuel
            ERROR[i,j,:] = FitnessF.Error
        
            with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Time.txt', "a") as myfile:
                myfile.write(str(Time[i,j]) +' ' )
            with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Min.txt', "a") as myfile:
                myfile.write(str(MIN[i,j]) +' ' )
            with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Mass.txt', "a") as myfile:
                myfile.write(str(MASS[i,j]) +' ' )
            with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Error.txt', "a") as myfile:
                myfile.write(str(ERROR[i,j]) +' ' )

        with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Time.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Min.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Mass.txt', "a") as myfile:
            myfile.write('\n')
        with open('./Results/StudyEASettings/AfterModifyingDynamics_1/Error.txt', "a") as myfile:
            myfile.write('\n')

    # AL_BF.writeData(Time, 'w', 'Time.txt')
    # AL_BF.writeData(MIN, 'w', 'Min.txt')
    # AL_BF.writeData(MASS, 'w', 'Mass.txt')
    # AL_BF.writeData(ERROR, 'w', 'Error.txt')



    # print("Min")
    # print(f_min)
    # AL_BF.writeData(f_min, 'w', 'Solution.txt')

    return IND, ITER

def loadData():
    # time_1 = np.loadtxt("./Results/StudyEASettings/1/Time.txt")
    # minVal_1 = np.loadtxt("./Results/StudyEASettings/1/Min.txt")
    # mass_1 = np.loadtxt("./Results/StudyEASettings/1/Mass.txt")
    # error = np.loadtxt("./Results/StudyOptSettings/1/Error.txt")
    time_1 = np.loadtxt("./Results/StudyEASettings/AfterModifyingDynamics_1/Time.txt")
    minVal_1 = np.loadtxt("./Results/StudyEASettings/AfterModifyingDynamics_1/Min.txt")
    mass_1 = np.loadtxt("./Results/StudyEASettings/AfterModifyingDynamics_1/Mass.txt")

    # time_2 = np.loadtxt("./Results/StudyEASettings/2/Time.txt")
    # minVal_2 = np.loadtxt("./Results/StudyEASettings/2/Min.txt")

    # time_3 = np.loadtxt("./Results/StudyEASettings/3/Time.txt")
    # minVal_3 = np.loadtxt("./Results/StudyEASettings/3/Min.txt")

    return [time_1], [minVal_1]
    # return [time_1, time_2, time_3], [minVal_1, minVal_2, minVal_3]

def plotConvergence(IND, ITER):
    time, minVal = loadData()

    color = ['red', 'green', 'blue', 'black', 'orange', 'yellow']

    fig= plt.figure()

    ax = fig.add_subplot(2, 3, 1)
    for i in range(len(IND)):
        # plt.plot(ITER[:], time[i,:], 'x-', c = color[i], label = IND[i])
        plt.plot(ITER[:], time[0][i,:] /60, 'x-', c = color[i], label = IND[i])
    plt.xlabel('Iterations')
    plt.ylabel('Time of computation 1 (min)')
    plt.grid()
    plt.legend(title = "Individuals")
    # plt.show()

    # ax = fig.add_subplot(2, 3, 2)
    # for i in range(len(IND)):
    #     # plt.plot(ITER[:], time[i,:], 'x-', c = color[i], label = IND[i])
    #     plt.plot(ITER[:], time[1][i,:] /60, 'x-', c = color[i], label = IND[i])
    # plt.xlabel('Iterations')
    # plt.ylabel('Time of computation 2 (min)')
    # plt.grid()
    # plt.ylim(-100/60, 3000/60)
    # plt.legend(title = "Individuals")
    # # plt.show()

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

    ax = fig.add_subplot(2, 3, 4)
    for i in range(len(IND)):
        plt.plot(ITER[:], minVal[0][i,:], 'x-', c = color[i], label = IND[i])
    plt.xlabel('Iterations')
    plt.ylabel('Minimum value 1')
    plt.grid()
    # ax.set_yscale('log')
    plt.legend(title = "Individuals")
 


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

    # fig = plt.figure()
    # for i in range(len(IND)):
    #     plt.plot(ITER[:], mass[i,:], 'x-', c = color[i], label = IND[i])
    # plt.xlabel('Iterations')
    # plt.ylabel('Mass of fuel')
    # plt.grid()
    # plt.legend(title = "Individuals")
    # plt.show()

    # errorX = np.zeros((len(IND), len(ITER)))
    # errorV = np.zeros((len(IND), len(ITER)))
    # for i in range(len(IND)):
    #     for j in range(len(ITER)):
    #         errorX[i,j] = np.linalg.norm( error[i,j,0:3] )
    #         errorV[i,j] = np.linalg.norm( error[i,j,3:] )

    # fig, ax = plt.figure()
    # for i in range(IND):
    #     ax.plot(ITER[:], errorX[i,:], 'x-', c = 'red', label = IND[i])
    #     ax.plot(ITER[:], errorV[i,:], 'x-', c = 'red', label = IND[i])
    # plt.legend()
    

if __name__ == "__main__":
    IND = np.array([10, 20, 50, 100, 200])
    ITER = np.array([10, 20, 50, 100])
    
    # runOpt(IND, ITER)
    plotConvergence(IND, ITER)