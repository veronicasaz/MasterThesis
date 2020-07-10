import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from FitnessFunction_1loop import Fitness
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT

import time

def runOpt(stepMagn, iterGlobal):
    ########################
    # Initial settings
    ########################
    Nimp = 25
    FitnessF = Fitness(Nimp = Nimp)

    ########################
    # Decision Vector Outer loop 
    ########################
    date0 = np.array([15,6,2019,0])
    t0 = AL_Eph.DateConv(date0,'calendar') #To JD
    # m0 = 747
    transfertime = 250
    iterLocal = 50
    
    # Bounds of the outer loop
    # bnd_v0 = (vp_Hohm * 0.2, vp_Hohm *0.6) 
    bnd_v0 = (0, 4e3) # Relative to the planet
    bnd_v0_angle = (0., 2*np.pi)
    bnd_vf = ( 0.0, 5e3) # Relative to the planet
    # bnd_vf = ( v_escape *0.9, v_escape *1.1)
    bnd_vf_angle = (0., 2*np.pi)
    bnd_t0 = (t0.JD_0, t0.JD_0+1000) # Launch date
    # bnd_m0 = (0, 200) # Mass should never be 0 as you add dry mass
    bnd_t_t = (AL_BF.days2sec(200), AL_BF.days2sec(1200) )
    bnd_deltavmag = (0., 1.) # magnitude
    bnd_deltavang = (-np.pi, np.pi) # angle

    bnds = (bnd_v0, bnd_v0_angle, bnd_v0_angle, \
            bnd_vf, bnd_vf_angle, bnd_vf_angle, \
            bnd_t0, bnd_t_t)

    for i in range(Nimp): # 3 times because impulses are 3d vectors
        bnds += (bnd_deltavmag, bnd_deltavang, bnd_deltavang)

    ########################
    # Calculate fitness
    ########################
    def f(DecV):
        return FitnessF.calculateFeasibility(DecV)
        
    mytakestep = AL_OPT.MyTakeStep(Nimp, bnds)

    DecV = np.zeros(len(bnds))
    for i in range(len(bnds)):
        DecV[i] = ( bnds[i][0] + bnds[i][1] ) /2

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

    niter_success = 15

    for i in range(len(iterGlobal)):
        for j in range(len(stepMagn)):
    # for i in range(1):
    #     for j in range(1):
            print("##################################################")
            print("IterGlobal",iterGlobal[i], "IterLocal", stepMagn[j])
            print("##################################################")
            start_time = time.time()
            fmin, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep, \
                niter = iterGlobal[i], niter_local = iterLocal, \
                niter_success = niter_success,bnds = bnds, jumpMagnitude = stepMagn[j], \
                tolLocal = 1e3, tolGlobal = 1e3)
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

    # error = np.loadtxt("./Results/StudyOptSettings/1/Error.txt")

    # time_2 = np.loadtxt("./Results/StudyMBHSettings/2/Time.txt")
    # minVal_2 = np.loadtxt("./Results/StudyMBHSettings/2/Min.txt")

    # time_3 = np.loadtxt("./Results/StudyMBHSettings/3/Time.txt")
    # minVal_3 = np.loadtxt("./Results/StudyMBHSettings/3/Min.txt")

    return [time_1], [minVal_1]
    # return [time_1, time_2, time_3], [minVal_1, minVal_2, minVal_3]

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

if __name__ == "__main__":
    iterGlobal = np.array([10, 20, 50, 100, 200])
    stepMagn = np.array([0.01,0.02,0.05, 0.2])
    
    # runOpt(stepMagn, iterGlobal)
    plotConvergence(stepMagn, iterGlobal)