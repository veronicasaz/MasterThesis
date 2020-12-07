import numpy as np
import pykep as pk
import time
from pyDOE import *
import scipy.optimize as spy

from FitnessFunction_normalized import Fitness
from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT
from AstroLibraries import AstroLib_2BP as AL_2BP
from AstroLibraries import AstroLib_ShapingMethod as AL_Sh

import LoadConfigFiles as CONFIG


SF = CONFIG.SimsFlan_config() # Load Sims-Flanagan config variables   

opt_config = CONFIG.OPT_config()
MBH = opt_config.MBH_generateDatabase

Dataset_c = CONFIG.Dataset()
Dataset_conf = Dataset_c.Dataset_config

def createFile(typeinputs, creationMethod, appendToFile, evaluate):
    fileName = "./databaseANN/DatabaseOptimized/" + typeinputs + "/" + creationMethod + '_eval.txt'
    fileName_opt = "./databaseANN/DatabaseOptimized/" + typeinputs + "/" + creationMethod +'.txt'
    matrix_file = "./databaseANN/DatabaseOptimized/" + typeinputs + "/" 

    if typeinputs == 'deltakeplerian' or 'deltakeplerian_planet':
        Heading = [ "Label", "M_f", "Ep_x", "Ep_y", "Ep_z", "Ev_x", "Ev_y", "Ev_z","t_t", "m_0", "|Delta_a|", \
            "|Delta_e|", "cos(Delta_i)", "Delta_Omega",\
            "Delta_omega", "Delta_theta"]
    elif typeinputs == "cartesian":
        Heading = [ "Label", "M_f", "Ep_x", "Ep_y", "Ep_z", "Ev_x", "Ev_y", "Ev_z","t_t", "m_0", "Delta_x", \
            "Delta_y", "Delta_z", "Delta_vx",\
            "Delta_vy", "Delta_vz"]

    if appendToFile == False:
        with open(fileName, "w") as myfile:
            for i in Heading:
                if i != Heading[-1]:
                    myfile.write(i +" ")
                else:
                    myfile.write(i)
            myfile.write("\n")
        myfile.close()

    if appendToFile == False:
        with open(fileName_opt, "w") as myfile:
            for i in Heading:
                if i != Heading[-1]:
                    myfile.write(i +" ")
                else:
                    myfile.write(i)
            myfile.write("\n")
        myfile.close()

    return fileName, fileName_opt, matrix_file

def to_helioc(r, vx, vy):
    v_body = np.array([vx, vy, 0])

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
    
    k2 = [1/12, 1/6, 1/4, 1/2]
    N = np.arange(0,2,1)

    eSin = AL_Sh.shapingMethod(mu / Cts.AU_m**3)
    v1_opt = 1e8 # random high value
    for k2_i in k2:
        gammaOptim_v = eSin.calculategamma1(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi, \
        t_t, k2_i, plot = False)
        
        for j in range(len(N)):
            if gammaOptim_v[j] != 0:
                eSin.calculateExposin(j, gammaOptim_v[j],r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi )
                # eSin.plot_sphere(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi, gammaOptim_v[Ni], Ni)
                v1, v2 = eSin.terminalVel(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi)
                t, a_T = eSin.calculateThrustProfile(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi)

                v_1 = to_helioc(r_1, v1[0]*Cts.AU_m, v1[1]*Cts.AU_m)
                v_2 = to_helioc(r_2, v2[0]*Cts.AU_m, v2[1]*Cts.AU_m)
                
                # With respect to the body
                v1_E = v_1-v_E
                v2_M = v_2-v_M

                if np.linalg.norm(v1) < v1_opt:
                    v1_opt = np.linalg.norm(v1_E)
                    v2_opt = np.linalg.norm(v2_M)
                    v1_opt_v = v1_E
                    v2_opt_v = v2_M
                    k2_opt = k2_i
                    gammaOptim_opt = gammaOptim_v[j]

    # Calculate acceleration vector
    acc_vector = np.zeros((SF.Nimp*3))
    for i in range(SF.Nimp):
        # Acceleration on segment is average of extreme accelerations
        t_i = AL_BF.days2sec(t_t) / (SF.Nimp+1) *i
        t_i1 = AL_BF.days2sec(t_t) / (SF.Nimp+1) *(i+1)

        #find acceleration at a certain time
        a_i = eSin.accelerationAtTime(t_i)
        a_i1 = eSin.accelerationAtTime(t_i1)
        
        a = (a_i+a_i1)/2 # find the acceleration at a certain time
        deltav_i = AL_BF.days2sec(t_t) / (SF.Nimp+1) * a*Cts.AU_m
        acc_vector[3*i]  = abs(deltav_i) /100
        acc_vector[3*i+1]= 0
        acc_vector[3*i+2] = 0


    return v1_opt_v, v2_opt_v, acc_vector

def latinhypercube(ninputs, samples):
    lhd = lhs(ninputs, samples=samples)
    # lhd = norm(loc=0, scale=1).ppf(lhd)  # this applies to both factors here

    dataset = np.zeros(np.shape(lhd))
    for item in range(len(SF.bnds)):
        f = SF.bnds[item][1] - SF.bnds[item][0]
        a = np.ones((samples)) * SF.bnds[item][0]
        dataset[:,item] = lhd[:, item] * f + a

    return dataset


if __name__ == "__main__":
    ########################
    # Initial settings
    ########################
    Cts = AL_BF.ConstantsBook()
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    
    Fit = Fitness(Nimp = SF.Nimp) # Load fitness class
    def f(DecV):
        return Fit.calculateFitness(DecV)

    ######################################
    # CHOICE OF GENERATION OF THE DATABASE
    ######################################
    # TO MODIFY
    typeinputs = Dataset_conf['Creation']['typeinputs'] # cartesian or deltakeplerian deltakeplerian_planet
    creationMethod = Dataset_conf['Creation']['creationMethod'] # 'Exposin', 'Lambert', 'Random
    lhypercube = Dataset_conf['Creation']['lhypercube'] # Use latin hypercube for initial distribution of samples. 
                        #  only if creation method is Random or optimized 
    evaluate = Dataset_conf['Creation']['evaluate']
    samples_rand = Dataset_conf['Creation']['samples_rand'] # samples with random mor hypercube initialization
    samples_L = Dataset_conf['Creation']['samples_L'] # samples for Lambert and Exposin
    appendToFile = Dataset_conf['Creation']['appendToFile'] # append instead of creating a new file. To increase the number of values


    # sys.exit(0) # to make sure I don't do it accidentaly and have to create files over again
    
    ####################
    # FILE CREATION
    ####################
    feasibilityFileName, feasibilityFileName_opt, matrix_file = \
            createFile(typeinputs, creationMethod, appendToFile, evaluate)
    
    ####################
    # DATABASE CREATION
    ####################    
    if creationMethod == 'Lambert' or creationMethod == 'Exposin': # leave the 6 first inputs empty to calculate
        samples_initial = np.zeros((samples_L, len(SF.bnds)))
        for decv in range(6,len(SF.bnds)): # Add impulses that won't be used for Lambert
            samples_initial[:, decv] = np.random.uniform(low = SF.bnds[decv][0], \
                high = SF.bnds[decv][1], size = samples_L)
    else:
        if lhypercube == True:
            samples_initial = latinhypercube(len(SF.bnds), samples_rand)
        else:
            samples_initial = np.zeros((samples_L, len(SF.bnds)))
            for decv in range(len(SF.bnds)): 
                samples_initial[:, decv] = np.random.uniform(low = SF.bnds[decv][0], \
                    high = SF.bnds[decv][1], size = samples_L)
        
        np.save(matrix_file + "initial_hypercube", samples_initial)

    ####################
    # CHOICE OF RANDOM POPULATION WITH LAMBERT
    ####################
    start_time = time.time()

    # EXPOSIN
    if creationMethod == 'Exposin':
        earthephem = pk.planet.jpl_lp('earth')
        marsephem = pk.planet.jpl_lp('mars')

        notvalid = list()
        for i in range(samples_L):
            t_0 = samples_initial[i, 6]
            t_t = samples_initial[i, 7]

            r_0, vE = earthephem.eph(t_0)
            r_1, vM = marsephem.eph(t_0 + AL_BF.sec2days(t_t))

            v1_opt, v2_opt, acc_vector = exposin_opt(r_0, r_1, vE, vM, AL_BF.sec2days(t_t), sun.mu)
            print("Here", v1_opt, v2_opt)
            vi1 = np.linalg.norm(v1_opt)
            vi2 = np.linalg.norm(v2_opt)

            if vi1 >= (SF.bnds[0][0] ) and  vi1 <= (SF.bnds[0][1] ) and \
                vi2 >= (SF.bnds[3][0] ) and  vi2 <= (SF.bnds[3][1] ):
                    samples_initial[i, 0:3] = AL_BF.convert3dvector(v1_opt, "cartesian")
                    samples_initial[i, 3:6] = AL_BF.convert3dvector(v2_opt, "cartesian")
                    samples_initial[i, 8:] = acc_vector
            else:
                notvalid.append(i)

    # LAMBERT
    elif creationMethod == 'Lambert':
        # Lambert for calculation of the velocity vectors 
        earthephem = pk.planet.jpl_lp('earth')
        marsephem = pk.planet.jpl_lp('mars')

        notvalid = list()
        for i in range(samples_L):
            t_0 = samples_initial[i, 6]
            t_t = samples_initial[i, 7]

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
                        samples_initial[i, 0:3] = AL_BF.convert3dvector(v1[rev]-vE, "cartesian")
                        samples_initial[i, 3:6] = AL_BF.convert3dvector(v2[rev]-vM, "cartesian")
                        v_i_prev = abs(v_i2 - v_i)

                    # Choose the revolutions with the lowest velocity at the earth
                    
                elif rev == len(v1)-1 and v_i_prev == 1e12:
                    notvalid.append(i)
                    # sample_inputs[i,:] = np.zeros(len(SF.bnds))
    
    else:
        notvalid = [] # Not eliminate anything

    # Delete not valid rows:
    sample_inputs = np.delete(samples_initial, notvalid, axis = 0)

    t = (time.time() - start_time) 
    print( "Non valid", len(notvalid))
    print("Time for initial discrimination", t)


    # Correct angles to be between 0 and 2pi 
    for i in range(len(sample_inputs)): 
        sample_inputs[i,1] = AL_BF.convertRange(sample_inputs[i,1], 'rad', 0, 2*np.pi)
        sample_inputs[i,2] = AL_BF.convertRange(sample_inputs[i,2], 'rad', 0, 2*np.pi)
        sample_inputs[i,4] = AL_BF.convertRange(sample_inputs[i,4], 'rad', 0, 2*np.pi)
        sample_inputs[i,5] = AL_BF.convertRange(sample_inputs[i,5], 'rad', 0, 2*np.pi)

    
    ####################
    # EVALUATE
    if evaluate == True:
        for i_sample in range(len(sample_inputs)):
            print("-------------------------------")
            print("Sample %i"%i_sample)
            print("-------------------------------")
            sample = sample_inputs[i_sample, :]
            fvalue = Fit.calculateFitness(sample, printValue = False)
            Fit.savetoFile(typeinputs, feasibilityFileName) # saves the current values
            


    ####################
    # OPTIMIZE
    # The idea is to use each sample and optimize it so that it is easier to 
    # find feasible trajectories
    # Only the initial and the optimized trajectokry will be saved
    ####################
    mytakestep = AL_OPT.MyTakeStep(SF.Nimp, SF.bnds)

    for i_sample in range(len(sample_inputs) ):
        print("-------------------------------")
        print("Sample %i"%i_sample)
        print("-------------------------------")
        sample = sample_inputs[i_sample, :]
        fvalue = Fit.calculateFitness(sample)
        # Fit.savetoFile(typeinputs, feasibilityFileName, massFileName) # saves the current values
        # Not needed as saved before        
        Fit.printResult()
        
        # optimize starting from sample
        if creationMethod == 'Random_MBH':
            solutionLocal, Best = AL_OPT.MonotonicBasinHopping(f, sample, mytakestep,\
                niter = MBH['niter_total'], niter_local = MBH['niter_local'], \
                niter_success = MBH['niter_success'], bnds = SF.bnds, \
                jumpMagnitude = MBH['jumpMagnitude'], tolLocal = MBH['tolLocal'],\
                tolGlobal = MBH['tolGlobal'])
            fvalue = f(solutionLocal)
            Fit.savetoFile(typeinputs, feasibilityFileName_opt)
            Fit.printResult()
            
        else:
            solutionLocal = spy.minimize(f, sample, method = 'SLSQP', \
                tol = MBH['tolLocal'], bounds = SF.bnds, options = {'maxiter': MBH['niter_local']} )

        
            fvalue = f(solutionLocal.x)
            feasible = AL_OPT.check_feasibility(solutionLocal.x, SF.bnds)
            if feasible == True: 
                Fit.savetoFile(typeinputs, feasibilityFileName_opt)
                Fit.printResult()
            else:
                print("Out of bounds")
        print(fvalue)
        
    