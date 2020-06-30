import numpy as np
import pykep as pk
import scipy.optimize as spy

from FitnessFunction_1loop import Fitness
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT

import time

########################
# Initial settings
########################
Nimp = 55
Fitness = Fitness(Nimp = Nimp)

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
    return Fitness.calculateFitness(DecV, plot = False)
    

def optimize():
    # Optimize outer loop
    ind = 1000
    iterat = 100
    start_time = time.time()
    f_min, Best = AL_OPT.EvolAlgorithm(f, bnds , x_add = False, ind = ind, max_iter = iterat, max_iter_success = 10 )
    t = (time.time() - start_time) 

    print("Time", t)
    print("Min", f_min)    
    AL_BF.writeData(f_min, 'w', 'SolutionEA.txt')

def coordS():
    # Coordinate search opt
    f_min = np.loadtxt("SolutionEA.txt")
    stepsize = np.zeros(len(f_min)) + 0.1
    stepsize[0:6] = 5e3 # Velocities
    stepsize[6] = 15 # initial time
    stepsize[7] = 10 # mass
    stepsize[8] = AL_BF.days2sec(10) # transfer time

    f_min2, Best2 = AL_OPT.coordinateSearch(f, f_min, bnds, stepsize, x_add = False,max_iter = 10)
    print("Min2")
    print(f_min2)
    AL_BF.writeData(f_min2, 'w', 'SolutionCoord.txt')

def MBH():
    mytakestep = AL_OPT.MyTakeStep(Nimp, bnds)

    DecV = np.zeros(len(bnds))
    for i in range(len(bnds)):
        DecV[i] = ( bnds[i][0] + bnds[i][1] ) /2

    cons = []
    for factor in range(len(DecV)):
        lower, upper = bnds[factor]
        l = {'type': 'ineq',
            'fun': lambda x, a=lower, i=factor: x[i] - a}
        u = {'type': 'ineq',
            'fun': lambda x, b=upper, i=factor: b - x[i]}
        cons.append(l)
        cons.append(u)
        
    minimizer_kwargs = dict(method="COBYLA", constraints=(cons),options={'disp': False,  'maxiter': 100})#T

    start_time = time.time()
    fmin_3 = spy.basinhopping(f, DecV, niter = 10, minimizer_kwargs=minimizer_kwargs,niter_success = 50,take_step= mytakestep,callback=AL_OPT.print_fun)
    # DecV_optimized2 = spy.basinhopping(main, DecV, niter=20, minimizer_kwargs=minimizer_kwargs,niter_success = 5,callback=print_fun)
    # DecV_optimized2 = basinhopping(Problem, DecV, niter=2, minimizer_kwags=minimizer_kwargs,take_step=mytakestep,callback=print_fun)
    t = (time.time() - start_time) 

    print("Min3")
    AL_BF.writeData(fmin_3.x, 'w', 'SolutionMBH.txt')

def MBH_self():
    mytakestep = AL_OPT.MyTakeStep(Nimp, bnds)

    DecV = np.zeros(len(bnds))
    for i in range(len(bnds)):
        DecV[i] = ( bnds[i][0] + bnds[i][1] ) /2

    cons = []
    for factor in range(len(DecV)):
        lower, upper = bnds[factor]
        l = {'type': 'ineq',
            'fun': lambda x, a=lower, i=factor: x[i] - a}
        u = {'type': 'ineq',
            'fun': lambda x, b=upper, i=factor: b - x[i]}
        cons.append(l)
        cons.append(u)

    fmin_4, Best = AL_OPT.MonotonicBasinHopping(f, DecV, mytakestep, niter = 100, niter_local = 20, bnds = bnds, cons = cons)
    print("Min4", fmin_4)
    AL_BF.writeData(fmin_4, 'w', 'SolutionMBH_self.txt')


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
        if ( DecV_I[j] < bnds[j][0] ) or ( DecV_I[j] > bnds[j][1] ):
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
        if ( DecV_I[j] < bnds[j][0] ) or ( DecV_I[j] > bnds[j][1] ):
            print(j, "Within bounds?", "min", bnds[j][0], "value",DecV_I[j], "max",bnds[j][1])

if __name__ == "__main__":
    # optimize()
    # coordS()
    # MBH()
    # MBH_self()
    propagateSol()
