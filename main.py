import numpy as np
import pykep as pk

from FitnessFunction import Fitness
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT

import time

########################
# Initial settings
########################
Nimp = 35
Fitness = Fitness(Nimp = Nimp)

########################
# Decision Vector Outer loop 
########################
date0 = np.array([15,6,2019,0])
t_0 = AL_Eph.DateConv(date0,'calendar') #To JD
m0 = 747
transfertime = 250

# # TODO: change. For now use the initial and final velocities to rendez 
# # vous Mars starting from the Earth with Lambert terminal velocity vector
earthephem = pk.planet.jpl_lp('earth')
marsephem = pk.planet.jpl_lp('mars')
r_E, v_E = earthephem.eph(t_0.JD_0)
r_M, v_M = marsephem.eph(t_0.JD_0 + transfertime)

# lambert = AL_TR.Lambert(np.array(r_E), np.array(r_M), AL_BF.days2sec(transfertime), AL_BF.mu_S)
# v1, v2 = lambert.TerminalVelVect()

v1 = np.array(v_E)
DecV_O = [v1, np.array(v_M), date0, m0]

########################
# Decision vector inner loop
########################
# Bounds of the inner loop
bnd_t_t = (AL_BF.days2sec(300), AL_BF.days2sec(1200) )
bnd_deltavmag = (0., 1.) # magnitude

bnds = (bnd_t_t,)
for i in range(Nimp): # 3 times because impulses are 3d vectors
    bnds += (bnd_deltavmag, bnd_deltavmag, bnd_deltavmag)

# DeltaV_list = np.zeros((Nimp, 3))
# DecV_I = [AL_BF.days2sec(transfertime), DeltaV_list]


########################
# Calculate fitness
########################
def f_O(DecV_O):
    Fitness.DecV_O(DecV_O)

    start_time = time.time()
    f_min, Best = AL_OPT.EvolAlgorithm(f, bnds , x_add = False, ind = 50, max_iter = 20, max_iter_success = 5 )
    t = (time.time() - start_time) 

    print("Min")
    print(f_min)
    Fitness.calculateFitness(f_min, plot = True)
    AL_BF.writeData(f_min, 'w', 'Solution.txt')

def f(DecV_I):
    return Fitness.calculateFitness(DecV_I, plot = False)
    



# Optimize outer loop
# start_time = time.time()
# f_min, Best = AL_OPT.EvolAlgorithm(f_0, bnds_O , x_add = Nimp, ind = ind[i], max_iter = max_iter[j], max_iter_success = 15 )
# t = (time.time() - start_time) 

# Optimize only inner loop and outer loop fixed
f_O(DecV_O)


# Test: Not optimize. Or load solution from file
DecV_I = np.loadtxt("Solution.txt")

Fitness.DecV_O(DecV_O)
f = Fitness.calculateFitness(DecV_I, optMode = True, plot = True)
print(f) 

