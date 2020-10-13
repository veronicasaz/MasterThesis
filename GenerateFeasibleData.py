import numpy as np
import pykep as pk
import time

from FitnessFunction_normalized import Fitness, Propagate
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT
from AstroLibraries import AstroLib_2BP as AL_2BP

import LoadConfigFiles as CONFIG

Cts = AL_BF.ConstantsBook()
SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables
Fit = Propagate(Nimp = SF.Nimp) # Load fitness class

def f(DecV):
    return Fit.calculateFeasibility(DecV)

def virtualBodyData():
    ####################
    # FILE CREATION
    ####################
    feasibilityFileName = "trainingData_Feas_fake.txt"
    massFileName = "trainingData_Opt_fake.txt"
    Heading = [ "Label", "t_t", "m_0", "|Delta_a|", \
        "|Delta_e|", "cos(Delta_i)", "Delta_Omega",\
        "Delta_omega", "Delta_theta"]
    for fileName in [feasibilityFileName, massFileName]:
        with open(fileName, "w") as myfile:
            for i in Heading:
                if i != Heading[-1]:
                    myfile.write(i +" ")
                else:
                    myfile.write(i)
            myfile.write("\n")
        myfile.close()

    ####################
    # CREATION OF RANDOM POPULATION
    ####################
    nsamples = 100 # number of training samples. 
    sample_input = np.zeros((nsamples, len(SF.bnds)))

    # for decv in range(6,8):
    for decv in range(len(SF.bnds)): # Add impulses 
        sample_input[:, decv] = np.random.uniform(low = SF.bnds[decv][0], \
            high = SF.bnds[decv][1], size = nsamples)

    for i in range(nsamples):
        Fit = Propagate(Nimp = SF.Nimp)
        Fit.prop(sample_input[i,:])
        Fit.savetoFile()
    

if __name__ == "__main__":
    virtualBodyData()
    
 

        