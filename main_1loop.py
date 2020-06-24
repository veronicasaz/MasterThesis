import numpy as np
import pykep as pk

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

# Bounds of the outer loop

# Initial approx of velocity for the launch at the Earth
v_Hohm = np.sqrt( 2*AL_BF.mu_S * (1/AL_BF.AU - 1/(2*227940e6)) )

bnd_v0 = (0., 1.)
bnd_v0_angle = (0., 2*np.pi)
bnd_vf_angle = (0., 2*np.pi)
bnd_t0 = (t0.JD_0, t0.JD_0+1000) # Launch date
bnd_m0 = (0, 200) # Mass should never be 0 as you add dry mass
bnd_t_t = (AL_BF.days2sec(200), AL_BF.days2sec(600) )
bnd_deltavmag = (-1., 1.) # magnitude

bnds = (bnd_v, bnd_v, bnd_v, bnd_v, bnd_v, bnd_v, bnd_t0, bnd_m0, bnd_t_t)

for i in range(Nimp): # 3 times because impulses are 3d vectors
    bnds += (bnd_deltavmag, bnd_deltavmag, bnd_deltavmag)


########################
# Calculate fitness
########################
def f(DecV):
    return Fitness.calculateFitness(DecV, plot = False)
    

def optimize():
    # Optimize outer loop
    # ind = 100
    # iterat = 100
    ind = 100
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

# def MBH():
#     mytakestep = AL_OPT.MyTakeStep(0.5)

#     cons = []
#     for factor in range(len(DecV)):
#         lower, upper = bnds[factor]
#         l = {'type': 'ineq',
#             'fun': lambda x, a=lower, i=factor: x[i] - a}
#         u = {'type': 'ineq',
#             'fun': lambda x, b=upper, i=factor: b - x[i]}
#         cons.append(l)
#         cons.append(u)
        
#     minimizer_kwargs = dict(args = [Nimp,deltav_magn], method="COBYLA",constraints=(cons),options={'disp': False,  'maxiter': 100})#T

#     start_time = time.time()
#     DecV_optimized2 = spy.basinhopping(EOF.mainOpt_Simple, DecV, niter = 50, minimizer_kwargs=minimizer_kwargs,niter_success = 50,take_step=mytakestep,callback=print_fun)
#     # DecV_optimized2 = spy.basinhopping(main, DecV, niter=20, minimizer_kwargs=minimizer_kwargs,niter_success = 5,callback=print_fun)
#     # DecV_optimized2 = basinhopping(Problem, DecV, niter=2, minimizer_kwags=minimizer_kwargs,take_step=mytakestep,callback=print_fun)
#     t[4] = (time.time() - start_time) 
#     x[4,1:] = DecV_optimized2.x



def propagateSol():
    DecV_I = np.loadtxt("SolutionEA.txt")

    f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
    Fitness.printResult()
    print(f) 

    # Test: Not optimize. Or load solution from file
    DecV_I = np.loadtxt("SolutionCoord.txt")

    f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
    Fitness.printResult()
    print(f) 

if __name__ == "__main__":
    # optimize()
    coordS()
    propagateSol()
