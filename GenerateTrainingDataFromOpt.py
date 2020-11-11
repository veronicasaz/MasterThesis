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

def to_helioc(r, v, gamma):
    v_body = np.array([v*np.sin(gamma), \
                        v*np.cos(gamma),\
                        0])

    # Convert to heliocentric
    angle = np.arctan2(r[1], r[0])
    v_h = AL_BF.rot_matrix(v_body, angle, 'z')

    return v_h

def exposin_opt(r_1, r_2, v_E, v_M, t_t, mu):
    r_1_norm = np.linalg.norm( r_1 )
    r_2_norm = np.linalg.norm( r_2 )

    dot = np.dot(r_1[0:2], r_2[0:2])      # dot product between [x1, y1] and [x2, y2]
    det = r_1[0]*r_2[1] - r_2[0]*r_1[1]     # determinant
    psi = np.arctan2(det, dot) 
    psi = AL_BF.convertRange(psi, 'rad', 0 ,2*np.pi)
    
    k2 = [1/24, 1/12, 1/6, 1/4, 1/3, 1/2]
    N = np.arange(0,4,1)

    eSin = AL_TR.shapingMethod(mu / Cts.AU_m**3)
    v1_opt = 1e8 # random high value
    for k2_i in k2:
        gammaOptim_v = eSin.start(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi, t_t, k2_i)
        
        for j in range(len(N)):
            if gammaOptim_v[j] != 0:
                # eSin.plot_sphere(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi, gammaOptim_v[Ni], Ni)
                v1, v2 = eSin.calculateVel(N[j], gammaOptim_v[N[j]], r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi)

                v_1 = to_helioc(r_1, v1[0]*Cts.AU_m, v1[1])
                v_2 = to_helioc(r_2, v2[0]*Cts.AU_m, v2[1]) 
                # With respect to the body
                v1_E = v_1-v_E
                v2_M = v_2-v_M

                if np.linalg.norm(v1) < v1_opt:
                    v1_opt = np.linalg.norm(v1_E)
                    v2_opt = np.linalg.norm(v2_M)
                    k2_opt = k2_i
                    gammaOptim_opt = gammaOptim_v[j]

    return v1_opt, v2_opt

if __name__ == "__main__":
    ########################
    # Initial settings
    ########################
    Cts = AL_BF.ConstantsBook()
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
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
    nsamples = 10 # number of training samples. 
    samples_Lambert = np.zeros((nsamples, len(SF.bnds)))

    ####################
    # CHOICE OF RANDOM POPULATION WITH LAMBERT
    ####################
    start_time = time.time()

    # for decv in range(6,8):
    for decv in range(6,len(SF.bnds)): # Add impulses that won't be used for Lambertt
        samples_Lambert[:, decv] = np.random.uniform(low = SF.bnds[decv][0], \
            high = SF.bnds[decv][1], size = nsamples)
    

    # EXPOSIN
    Exposin = True # use exposin or not
    if Exposin == True:
        earthephem = pk.planet.jpl_lp('earth')
        marsephem = pk.planet.jpl_lp('mars')

        notvalid = list()
        for i in range(nsamples):
            t_0 = samples_Lambert[i, 6]
            t_t = samples_Lambert[i, 7]

            r_0, vE = earthephem.eph(t_0)
            r_1, vM = marsephem.eph(t_0 + AL_BF.sec2days(t_t))

            v1_opt, v2_opt = exposin_opt(r_0, r_1, vE, vM, AL_BF.sec2days(t_t), sun.mu)

            vi1 = np.linalg.norm(v1_opt)
            vi2 = np.linalg.norm(v2_opt)

            if vi1 >= (SF.bnds[0][0] ) and  vi1 <= (SF.bnds[0][1] ) and \
                vi2 >= (SF.bnds[3][0] ) and  vi2 <= (SF.bnds[3][1] ):
                    samples_Lambert[i, 0:3] = AL_BF.convert3dvector(v_1_opt, "cartesian")
                    samples_Lambert[i, 3:6] = AL_BF.convert3dvector(v_2_opt, "cartesian")
            else:
                notvalid.append(i)

        # Delete not valid rows:
        sample_inputs = np.delete(samples_Lambert, notvalid, axis = 0)

        t = (time.time() - start_time) 
        print("Samples", nsamples, "Non valid", len(notvalid))
        print("Time for Exposin", t)

    # LAMBERT
    Lambert = False # Use lambert or not
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
        