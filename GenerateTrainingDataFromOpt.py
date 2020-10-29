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
    Cts = AL_BF.ConstantsBook()
    SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables   
    
    Fit = Fitness(Nimp = SF.Nimp) # Load fitness class
    def f(DecV):
        return Fit.calculateFeasibility(DecV)

    ####################
    # FILE CREATION
    ####################
    feasibilityFileName = "./databaseANN/ErrorIncluded/trainingData_Feas.txt"
    massFileName = "./databaseANN/ErrorIncluded/trainingData_Opt.txt"
    Heading = [ "Label", "Ep_x", "Ep_y", "Ep_z", "Ev_x", "Ev_y", "Ev_z","t_t", "m_0", "|Delta_a|", \
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
    nsamples = 5000 # number of training samples. 
    samples_Lambert = np.zeros((nsamples, len(SF.bnds)))

    ####################
    # CHOICE OF RANDOM POPULATION WITH LAMBERT
    ####################
    start_time = time.time()

    # for decv in range(6,8):
    for decv in range(6,len(SF.bnds)): # Add impulses that won't be used for Lambertt
        samples_Lambert[:, decv] = np.random.uniform(low = SF.bnds[decv][0], \
            high = SF.bnds[decv][1], size = nsamples)
    
    Lambert = True # Use lambert or not
    if Lambert == True:
        # Lambert for calculation of the velocity vectors 
        earthephem = pk.planet.jpl_lp('earth')
        marsephem = pk.planet.jpl_lp('mars')

        notvalid = list()
        for i in range(nsamples):
            t_0 = samples_Lambert[i, 6]
            t_t = samples_Lambert[i, 7]

            r_0, vE = earthephem.eph(t_0)
            r_1, vM = marsephem.eph(t_0 + AL_BF.sec2days(t_t))

            nrevs = 4
            l = pk.lambert_problem(r1 = r_0, r2 = r_1, tof = t_t, cw = False, \
                mu = Cts.mu_S_m, max_revs=nrevs)
            v1 = np.array(l.get_v1())
            v2 = np.array(l.get_v2())

            # Check if any of the solutions for the revolutions has velocities
            # within the bounds

            # Correction for the continuous thrust:
            Spacecraft = AL_2BP.Spacecraft( )
            # Apply correction of half DeltaV_max for approximation
            correction = 2
            DeltaV_max = Spacecraft.T / Spacecraft.m_dry * t_t / correction
            
            v1 -= v1/np.linalg.norm(v1) * DeltaV_max # applied in the same direction
            v2 += v2/np.linalg.norm(v2) * DeltaV_max

            v_i_prev = 1e12 # Excessive random value
            for rev in range(len(v1)):
                v_i = np.linalg.norm(v1[rev] - np.array(vE))  # Relative velocities for the bounds 
                v_i2 = np.linalg.norm(v2[rev] - np.array(vM))
                
                # Change to polar for the bounds
                if v_i >= (SF.bnds[0][0] ) and  v_i <= (SF.bnds[0][1] ) and \
                v_i2 >= (SF.bnds[3][0] ) and  v_i2 <= (SF.bnds[3][1] ):
                    if abs(v_i2 - v_i) < v_i_prev or rev == 0:
                        samples_Lambert[i, 0:3] = AL_BF.convert3dvector(v1[rev]-vE, "cartesian")
                        samples_Lambert[i, 3:6] = AL_BF.convert3dvector(v2[rev]-vM, "cartesian")
                        v_i_prev = abs(v_i2 - v_i)

                    # Choose the revolutions with the lowest velocity at the earth
                    
                elif rev == len(v1)-1 and v_i_prev == 1e12:
                    notvalid.append(i)
                    # sample_inputs[i,:] = np.zeros(len(SF.bnds))

                
        # Delete not valid rows:
        sample_inputs = np.delete(samples_Lambert, notvalid, axis = 0)

        t = (time.time() - start_time) 
        print("Samples", nsamples, "Non valid", len(notvalid))
        print("Time for Lambert", t)

    else: 
        sample_inputs = samples_Lambert 

    for i in range(len(sample_inputs)): # Correct angles to be between 0 and 2pi
        sample_inputs[i,1] = AL_BF.convertRange(sample_inputs[i,1], 'rad', 0, 2*np.pi)
        sample_inputs[i,2] = AL_BF.convertRange(sample_inputs[i,2], 'rad', 0, 2*np.pi)
        sample_inputs[i,4] = AL_BF.convertRange(sample_inputs[i,4], 'rad', 0, 2*np.pi)
        sample_inputs[i,5] = AL_BF.convertRange(sample_inputs[i,5], 'rad', 0, 2*np.pi)



    # print(sample_inputs)
    # sample_inputs = samples_Lambert

    ####################
    # EVALUATION OF EACH SAMPLE
    # The idea is to use each sample and optimize it so that it is easier to 
    # find feasible trajectories
    # Only the initial and the optimized trajectokry will be saved
    ####################
    opt_config = CONFIG.OPT_config()
    MBH = opt_config.MBH_generateDatabase
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)
    
    for i_sample in range(len(sample_inputs)):
        print("-------------------------------")
        print("Sample %i"%i_sample)
        print("-------------------------------")
        sample = sample_inputs[i_sample, :]
        fvalue = Fit.calculateFeasibility(sample, printValue = True)
        Fit.savetoFile() # saves the current values
        
        Fit.printResult()
        
        # optimize starting from sample
        xmin, Best = AL_OPT.MonotonicBasinHopping(f, sample, mytakestep,\
                niter = MBH['niter_total'], niter_local = MBH['niter_local'], \
                niter_success = MBH['niter_success'], bnds = SF.bnds, \
                jumpMagnitude = MBH['jumpMagnitude'], tolLocal = MBH['tolLocal'],\
                tolGlobal = MBH['tolGlobal'])
        
        fvalue = f(xmin)
        print(xmin)
        Fit.savetoFile()
        Fit.printResult()
        