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
mag = 30e3
bnd_v = (0., 1.)
bnd_t0 = (t0.JD_0, t0.JD_0+1000) # Launch date
bnd_m0 = (0, 200) # Mass should never be 0 as you add dry mass
bnd_t_t = (AL_BF.days2sec(200), AL_BF.days2sec(600) )
bnd_deltavmag = (0., 1.) # magnitude

bnds = (bnd_v, bnd_v, bnd_v, bnd_v, bnd_v, bnd_v, bnd_t0, bnd_m0, bnd_t_t)

for i in range(Nimp): # 3 times because impulses are 3d vectors
    bnds += (bnd_deltavmag, bnd_deltavmag, bnd_deltavmag)

# DeltaV_list = np.zeros((Nimp, 3))
# DecV_I = [AL_BF.days2sec(transfertime), DeltaV_list]


########################
# Calculate fitness
########################
def f(DecV):
    return Fitness.calculateFitness(DecV, plot = False)
    

# Optimize outer loop
start_time = time.time()
f_min, Best = AL_OPT.EvolAlgorithm(f, bnds , x_add = False, ind = 1000, max_iter = 10, max_iter_success = 4 )
t = (time.time() - start_time) 

print("Min")
print(f_min)
Fitness.calculateFitness(f_min, plot = False)
AL_BF.writeData(f_min, 'w', 'Solution.txt')


# Test: Not optimize. Or load solution from file
DecV_I = np.loadtxt("Solution.txt")
f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
Fitness.printResult()
print(f) 

