import numpy as np
import pykep as pk
import time

from FitnessFunction_normalized import Fitness
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT
from AstroLibraries import AstroLib_2BP as AL_2BP

import LoadConfigFiles as CONFIG


if __name__ == "__main__":
    ########################
    # Initial settings
    ########################
    SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables   
    
    Fitness = Fitness(Nimp = SF.Nimp) # Load fitness class
    def f(DecV):
        return Fitness.calculateFeasibility(DecV)

    ####################
    # FILE CREATION
    ####################
    feasibilityFileName = "trainingData_Feas.txt"
    massFileName = "trainingData_Opt.txt"
    Heading = [ "Label", "t_t", "m_0", "|Delta_a|", \
        "|Delta_e|", "cos(Delta_i)", "Delta_Omega",\
        "Delta_omega", "Delta_theta"]
    with open(feasibilityFileName, "w") as myfile:
        for i in Heading:
            myfile.write(i +"   ")
        myfile.write("\n")
    myfile.close()
    with open(massFileName, "w") as myfile:
        for i in Heading:
            myfile.write(i +"   ")
        myfile.write("\n")
    myfile.close()

    ####################
    # CREATION OF RANDOM POPULATION
    ####################
    nsamples = 5000 # number of training samples. TODO: increase
    samples_Lambert = np.zeros((nsamples, len(SF.bnds)))

    ####################
    # CHOICE OF RANDOM POPULATION WITH LAMBERT
    ####################
    start_time = time.time()

    for decv in range(6,8): 
        samples_Lambert[:, decv] = np.random.uniform(low = SF.bnds[decv][0], \
            high = SF.bnds[decv][1], size = nsamples)
    
    # Lambert for calculation of the velocity vectors 
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')

    notvalid = list()
    for i in range(nsamples):
        t_0 = samples_Lambert[i, 6]
        t_t = samples_Lambert[i, 7]

        r_0, vE = earthephem.eph(t_0)
        r_1, vM = marsephem.eph(t_0 + AL_BF.sec2days(t_t))

        nrevs = 2
        l = pk.lambert_problem(r1 = r_0, r2 = r_1, tof = t_t, cw = False, mu = pk.MU_SUN, max_revs=nrevs)
        v1 = np.array(l.get_v1())
        v2 = np.array(l.get_v2())

        # Check if any of the solutions for the revolutions has velocities
        # within the bounds
        nrevs_valid = -1
        for rev in range(len(v1)):
            v_i = np.linalg.norm(v1[rev] - np.array(vE)) # Relative velocities for the bounds 
            v_i2 = np.linalg.norm(v2[rev] - np.array(vM))
            # Change to polar for the bounds
            if v_i >= SF.bnds[0][0] and  v_i <= SF.bnds[0][1] and \
            v_i2 >= SF.bnds[3][0] and  v_i2 <= SF.bnds[3][1]:
                
                samples_Lambert[i, 0:3] = AL_BF.convert3dvector(v1[rev]-vE, "cartesian")
                samples_Lambert[i, 3:6] = AL_BF.convert3dvector(v2[rev]-vM, "cartesian")

                break
            elif rev == len(v1)-1:
                notvalid.append(i)
                # sample_inputs[i,:] = np.zeros(len(SF.bnds))

        ####################
        # Evaluate similarity between lambert and propagated trajectory
        ####################
                
    # Delete not valid rows:
    sample_inputs = np.delete(samples_Lambert, notvalid, axis = 0)

    t = (time.time() - start_time) 
    print("Samples", nsamples, "Non valid", len(notvalid))
    print("Time for Lambert", t)


    ####################
    # EVALUATION OF EACH SAMPLE
    # The idea is to use each sample and optimize it so that it is easier to 
    # find feasible trajectories
    # Only the initial and the optimized trajectory will be saved
    ####################
    opt_config = CONFIG.OPT_config()
    MBH = opt_config.MBH
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)
    
    for i_sample in range(len(sample_inputs)):
        print("-------------------------------")
        print("Sample %i"%i_sample)
        print("-------------------------------")
        sample = sample_inputs[i_sample, :]
        fvalue = f(sample)
        Fitness.savetoFile() # saves the current values
        
        Fitness.printResult()
        
        # optimize starting from sample
        xmin, Best = AL_OPT.MonotonicBasinHopping(f, sample, mytakestep,\
                niter = MBH['niter_total'], niter_local = MBH['niter_local'], \
                niter_success = MBH['niter_success'], bnds = SF.bnds, \
                jumpMagnitude = MBH['jumpMagnitude'], tolLocal = MBH['tolLocal'],\
                tolGlobal = MBH['tolGlobal'])
            
        fvalue = f(xmin)
        Fitness.savetoFile()
        Fitness.printResult()
        