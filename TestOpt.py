import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

import pykep as pk

import time

import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_2BP as AL_2BP
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_Plots as AL_Plot
from AstroLibraries import AstroLib_Trajectories as AL_TR
from AstroLibraries import AstroLib_OPT as AL_OPT


# # In[]
# ##############################################
# ############# VALIDATION #####################
# ##############################################
Cts = AL_BF.ConstantsBook()

def testfunction1(r):
    # f(1,3) = 0. bounds (-10,10)
    x, y = r
    z = (x+2*y-7)**2+(2*x+y-5)**2
    return z

def testfunction2(r):
    # f(0,0) = 0.  bounds (-10,10)
    x, y = r
    z = 0.26*(x**2+y**2)-0.48*x*y
    return z

class MyTakeStep(object):
    def __init__(self, bnds):
        self.bnds = bnds
    def __call__(self, x):
        self.call(x)
        return x
    def call(self, x, magnitude):
        x2 = np.zeros(len(x))
        for j in range(len(x)):
            x2[j] = x[j] + magnitude* np.random.uniform(self.bnds[j][0] - x[j], \
                                    self.bnds[j][1] - x[j], 1)[0]
        return x2

def MonotonicBasinHopping():
    niter = 200
    niterlocal = 10
    niter_success = 5

    bnds = ((-10,10),(-10,10))
    mytakestep = MyTakeStep(bnds)

    DecV = np.array([4, 5])

    cons = []
    for factor in range(len(DecV)):
        lower, upper = bnds[factor]
        l = {'type': 'ineq',
            'fun': lambda x, a=lower, i=factor: x[i] - a}
        u = {'type': 'ineq',
            'fun': lambda x, b=upper, i=factor: b - x[i]}
        cons.append(l)
        cons.append(u)

    start_time = time.time()
    fmin, Best = AL_OPT.MonotonicBasinHopping(testfunction1, DecV, mytakestep, jummpMagnitude = 0.3,  niter = niter, niter_local = niterlocal, niter_success = niter_success, bnds = bnds)
    t = (time.time() - start_time) 
    print("MBH: min 1", fmin, Best, 'time', t)

    start_time = time.time()
    fmin, Best = AL_OPT.MonotonicBasinHopping(testfunction2, DecV, mytakestep, jummpMagnitude = 0.5,  niter = niter, niter_local = niterlocal, niter_success = niter_success, bnds = bnds)
    t = (time.time() - start_time) 
    print("MBH: min 2", fmin, Best, 'time', t)

def EvolutionaryAlgorithm():
    bnds = [[-10,10], [-10, 10]]

    x_minVal, lastMin = AL_OPT.EvolAlgorithm(testfunction1, bnds, ind = 200, max_iter = 10 )
    print("EA: min 1", x_minVal, lastMin)

    x_minVal, lastMin = AL_OPT.EvolAlgorithm(testfunction2, bnds, ind = 500, max_iter = 100 )
    print("EA: min 2", x_minVal, lastMin)

if __name__ == "__main__":
    MonotonicBasinHopping()
    EvolutionaryAlgorithm()