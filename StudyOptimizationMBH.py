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

def runOpt(iterLocal, iterGlobal):
    ########################
    # Initial settings
    ########################
    Nimp = 55
    FitnessF = Fitness(Nimp = Nimp)

    ########################
    # Decision Vector Outer loop 
    ########################
    date0 = np.array([15,6,2019,0])
    t0 = AL_Eph.DateConv(date0,'calendar') #To JD
    # m0 = 747
    transfertime = 250

    # # # TODO: change. For now use the initial and final velocities to rendez 
    # # # vous Mars starting from the Earth with Lambert terminal velocity vector
    # earthephem = pk.planet.jpl_lp('earth')
    # marsephem = pk.planet.jpl_lp('mars')
    # r_E, v_E = earthephem.eph(t0.JD_0)
    # r_M, v_M = marsephem.eph(t0.JD_0 + transfertime)

    # lambert = AL_TR.Lambert(np.array(r_E), np.array(r_M), AL_BF.days2sec(transfertime), AL_BF.mu_S)
    # v1, v2 = lambert.TerminalVelVect()
    # v1 = np.array(v_E)
    # # DecV_O = [v1, np.array(v_M), date0, m0]
    # print(v1, v2)


    # Initial approx of velocity for the launch at the Earth
    vp_Hohm = np.sqrt( 2*AL_BF.mu_S * (1/AL_BF.AU - 1/(227940e6 + AL_BF.AU) ) )
    va_Hohm = np.sqrt( 2*AL_BF.mu_S * (1/227940e6 - 1/(227940e6 + AL_BF.AU) ) )
    v_escape = np.sqrt(2*AL_BF.mu_S / 227940e6) # velocity of a parabola at mars

    # Bounds of the outer loop
    # bnd_v0 = (vp_Hohm * 0.2, vp_Hohm *0.6) 
    bnd_v0 = (0, 4e3) # Relative to the planet
    bnd_v0_angle = (0., 2*np.pi)
    bnd_vf = ( 0.0, 5e3) # Relative to the planet
    # bnd_vf = ( v_escape *0.9, v_escape *1.1)
    bnd_vf_angle = (0., 2*np.pi)
    bnd_t0 = (t0.JD_0, t0.JD_0+1000) # Launch date
    bnd_m0 = (0, 200) # Mass should never be 0 as you add dry mass
    bnd_t_t = (AL_BF.days2sec(200), AL_BF.days2sec(600) )
    bnd_deltavmag = (0., 1.) # magnitude
    bnd_deltavang = (-np.pi, np.pi) # angle

    bnds = (bnd_v0, bnd_v0_angle, bnd_v0_angle, \
            bnd_vf, bnd_vf_angle, bnd_vf_angle, \
            bnd_t0, bnd_m0, bnd_t_t)

    for i in range(Nimp): # 3 times because impulses are 3d vectors
        bnds += (bnd_deltavmag, bnd_deltavang, bnd_deltavang)


    ########################
    # Calculate fitness
    ########################
    def f(DecV):
        return FitnessF.calculateFitness(DecV, plot = False)
        
    mytakestep = AL_OPT.MyTakeStep(Nimp, bnds)

    DecV = np.zeros(len(bnds))
    for i in range(len(bnds)):
        DecV[i] = ( bnds[i][0] + bnds[i][1] ) /2

    ########################
    # Parameter sweep
    ########################
    Time = np.zeros( (len(iterGlobal), len(iterLocal) ))
    MIN = np.zeros( (len(iterGlobal), len(iterLocal) ))
    MASS = np.zeros( (len(iterGlobal), len(iterLocal) ))
    ERROR = np.zeros( (len(iterGlobal), len(iterLocal), 6 ))

    def deleteFile(name):
        file = open(name,"r+")
        file.truncate(0)
        file.close()

    deleteFile("Time.txt")
    deleteFile("Min.txt")
    deleteFile("Mass.txt")
    deleteFile("Error.txt")

    niter_success = 15

    for i in range(len(iterGlobal)):
        for j in range(len(iterLocal)):
    # for i in range(1):
    #     for j in range(1):
            print("##################################################")
            print("IterGlobal",iterGlobal[i], "IterLocal", iterLocal[j])
            print("##################################################")
            start_time = time.time()
            fmin, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep, \
                niter = iterGlobal[i], niter_local = iterLocal[j], \
                niter_success = niter_success,bnds = bnds, jumpMagnitude = 0.01, \
                tolLocal = 1e2, tolGlobal = 1e3)
            t = (time.time() - start_time) 
            fit = FitnessF.calculateFitness(fmin, optMode = True, plot = False)
            
            # print(Best)
            Time[i,j] = t
            MIN[i,j] = Best
            MASS[i,j] = FitnessF.m_fuel
            ERROR[i,j,:] = FitnessF.Error
        
            with open('Time.txt', "a") as myfile:
                myfile.write(str(Time[i,j]) +' ' )
            with open('Min.txt', "a") as myfile:
                myfile.write(str(MIN[i,j]) +' ' )
            with open('Mass.txt', "a") as myfile:
                myfile.write(str(MASS[i,j]) +' ' )
            with open('Error.txt', "a") as myfile:
                myfile.write(str(ERROR[i,j]) +' ' )

        with open('Time.txt', "a") as myfile:
            myfile.write('\n')
        with open('Min.txt', "a") as myfile:
            myfile.write('\n')
        with open('Mass.txt', "a") as myfile:
            myfile.write('\n')
        with open('Error.txt', "a") as myfile:
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
    time_1 = np.loadtxt("./Results/StudyOptSettings/1/Time.txt")
    minVal_1 = np.loadtxt("./Results/StudyOptSettings/1/Min.txt")
    mass_1 = np.loadtxt("./Results/StudyOptSettings/1/Mass.txt")
    # error = np.loadtxt("./Results/StudyOptSettings/1/Error.txt")

    time_2 = np.loadtxt("./Results/StudyOptSettings/2/Time.txt")
    minVal_2 = np.loadtxt("./Results/StudyOptSettings/2/Min.txt")

    time_3 = np.loadtxt("./Results/StudyOptSettings/3/Time.txt")
    minVal_3 = np.loadtxt("./Results/StudyOptSettings/3/Min.txt")

    return [time_1, time_2, time_3], [minVal_1, minVal_2, minVal_3]

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

    ax = fig.add_subplot(2, 3, 2)
    for i in range(len(IND)):
        # plt.plot(ITER[:], time[i,:], 'x-', c = color[i], label = IND[i])
        plt.plot(ITER[:], time[1][i,:] /60, 'x-', c = color[i], label = IND[i])
    plt.xlabel('Iterations')
    plt.ylabel('Time of computation 2 (min)')
    plt.grid()
    plt.ylim(-100/60, 3000/60)
    plt.legend(title = "Individuals")
    # plt.show()

    ax = fig.add_subplot(2, 3, 3)
    for i in range(len(IND)):
        # plt.plot(ITER[:], time[i,:], 'x-', c = color[i], label = IND[i])
        plt.plot(ITER[:], time[2][i,:] /60, 'x-', c = color[i], label = IND[i])
    plt.xlabel('Iterations')
    plt.ylabel('Time of computation 2 (min)')
    plt.grid()
    plt.ylim(-100/60, 3000/60)
    plt.legend(title = "Individuals")
    # plt.show()

    ax = fig.add_subplot(2, 3, 4)
    for i in range(len(IND)):
        plt.plot(ITER[:], minVal[0][i,:], 'x-', c = color[i], label = IND[i])
    plt.xlabel('Iterations')
    plt.ylabel('Minimum value 1')
    plt.grid()
    ax.set_yscale('log')
    plt.legend(title = "Individuals")
 


    ax = fig.add_subplot(2, 3, 5)
    for i in range(len(IND)):
        plt.plot(ITER[:], minVal[1][i,:], 'x-', c = color[i], label = IND[i])
    plt.xlabel('Iterations')
    plt.ylabel('Minimum value 2')
    plt.grid()
    ax.set_yscale('log')
    plt.legend(title = "Individuals")

    ax = fig.add_subplot(2, 3, 6)
    for i in range(len(IND)):
        plt.plot(ITER[:], minVal[2][i,:], 'x-', c = color[i], label = IND[i])
    plt.xlabel('Iterations')
    plt.ylabel('Minimum value 2')
    plt.grid()
    ax.set_yscale('log')
    plt.legend(title = "Individuals")

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
    iterGlobal = np.array([10, 20, 50, 100, 200])
    iterLocal = np.array([10, 20, 50, 100, 200, 500])
    
    runOpt(iterGlobal, iterLocal)
    # plotConvergence(iterGlobal, iterLocal)