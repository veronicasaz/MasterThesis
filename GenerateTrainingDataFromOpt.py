import numpy as np
import pykep as pk
import time

from AstroLibraries import AstroLib_Trajectories as AL_TR
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_OPT as AL_OPT
from AstroLibraries import AstroLib_2BP as AL_2BP


class Fitness:
    def __init__(self, *args, **kwargs):

        #Constants used
        Cts = AL_BF.ConstantsBook() #Constants for planets

        # Define bodies involved
        self.Spacecraft = AL_2BP.Spacecraft( )
        self.earthephem = pk.planet.jpl_lp('earth')
        self.marsephem = pk.planet.jpl_lp('mars')
        
        self.sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
        self.earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
        self.mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

        # Settings
        self.color = ['blue','green','black','red','yellow','orange']    

        ## Sims-Flanagan settings

        # Choose odd number to compare easily in middle point
        self.Nimp = kwargs.get('Nimp', 35) 

    def adaptDecisionVector(self, DecV, optMode):
        """ 
        adaptDecisionVector: modify decision vector to input in the problem
        """
        v0 = np.array(DecV[0:3]) # vector, [magnitude, angle, angle]
        vf = np.array(DecV[3:6]) # vector, [magnitude, angle, angle]
        self.t0, self.t_t = DecV[6:8]

        # Delta V
        if optMode == True:
            DeltaV_list = np.array(DecV[8:]).reshape(-1,3) # make a Nimp x 3 matrix
        else:
            DeltaV_list = DecV[8:][0]

        ########################################################################
        # INITIAL CALCULATION OF VARIABLES
        ########################################################################
        # Modify from magnitude angle angle to cartesian
        self.DeltaV_list = np.zeros(np.shape(DeltaV_list))
        DeltaV_sum = np.zeros(len(self.DeltaV_list)) # Magnitude of the impulses
        for i in range(len(self.DeltaV_list)):
            # mag = DeltaV_list[i,0]
            # angle1 = DeltaV_list[i,1]
            # angle2 = DeltaV_list[i,2]
            self.DeltaV_list[i, :] = AL_BF.convert3dvector(DeltaV_list[i,:], "polar")
            # self.DeltaV_list[i, 0] = mag * np.cos(angle1)*np.cos(angle2) 
            # self.DeltaV_list[i, 1] = mag * np.sin(angle1)*np.cos(angle2)
            # self.DeltaV_list[i, 2] = mag * np.sin(angle2)
            DeltaV_sum[i] = np.linalg.norm( self.DeltaV_list[i] )

        # Write velocity as x,y,z vector
        v0_cart = AL_BF.convert3dvector(v0, "polar")
        vf_cart = AL_BF.convert3dvector(vf, "polar")
        # v0_cart = v0[0] *np.array([ np.cos(v0[1])*np.cos(v0[2]) , \
        #                         np.sin(v0[1])*np.cos(v0[2]),
        #                         np.sin(v0[2]) ])
        # vf_cart = vf[0] *np.array([ np.cos(vf[1])*np.cos(vf[2]) , \
        #                         np.sin(vf[1])*np.cos(vf[2]),
        #                         np.sin(vf[2]) ])

        # Sum of all the Delta V = DeltaV max * sum Delta V_list
        # Assumption: the DeltaV_max is calculated as if mass is constant and 
        # equal to the dry mass as it is the largest contribution. This means 
        # that the DelaV_max is actually smaller than it will be obtained in this
        # problem
        # = Thrust for segment
        self.DeltaV_max = self.Spacecraft.T / self.Spacecraft.m_dry * \
            self.t_t / (self.Nimp + 1) 

        # Total DeltaV 
        DeltaV_total = sum(DeltaV_sum) * self.DeltaV_max

        #Calculate total mass of fuel for the given impulses
        self.m0 = \
            self.Spacecraft.MassChangeInverse(self.Spacecraft.m_dry, DeltaV_total)
        self.m_fuel = self.m0 - self.Spacecraft.m_dry

        return v0_cart, vf_cart


    def calculateFeasibility(self, DecV, optMode = True):
        """
        """
        ########################################################################
        # DecV
        ########################################################################
        v0_cart, vf_cart = self.adaptDecisionVector(DecV, optMode)

        ########################################################################
        # Propagation
        ########################################################################
        # Times and ephemeris
        # t_0 = AL_Eph.DateConv(self.date0,'calendar') #To JD
        t_1 = AL_Eph.DateConv(self.t0 + AL_BF.sec2days(self.t_t), 'JD_0' )

        r_p0, v_p0 = self.earthephem.eph(self.t0)
        r_p1, v_p1 = self.marsephem.eph(t_1.JD_0)
        
        # Change from relative to heliocentric velocity
        self.v0 = v0_cart + v_p0 
        self.vf = vf_cart + v_p1 

        # Create state vector for initial and final point
        self.SV_0 = np.append(r_p0, self.v0)
        self.SV_f = np.append(r_p1, self.vf) # - to propagate backwards
        self.SV_f_corrected = np.append(r_p1, -self.vf) # - to propagate backwards

        # Sims-Flanagan
        SV_list_forw = self.__SimsFlanagan(self.SV_0, saveState=True)
        SV_list_back = self.__SimsFlanagan(self.SV_f_corrected, backwards = True, saveState=True)

        # convert back propagation so that the signs of velocity match
        SV_list_back_corrected = np.copy(SV_list_back)
        SV_list_back_corrected[:,3:] *= -1 # change sign of velocity

        ########################################################################
        # Compare State in middle point
        ########################################################################
        # print("Error middle point", SV_list_back[-1, :],SV_list_forw[-1, :])
        self.Error = SV_list_back_corrected[-1, :] - SV_list_forw[-1, :]


        fc1 = np.linalg.norm(self.Error[0:3] / AL_BF.AU) # Normalize with AU
        fc2 = np.linalg.norm(self.Error[3:] / AL_BF.AU * AL_BF.year2sec(1))
        # print(fc1, fc2)

        return fc1* 1e2 + fc2 # *1000 so that in tol they are in same order of mag

    def printResult(self):
        print("Error", self.Error)

    def savetoFile(self):
        """
        savetoFile:

        Append to file
        save in different columns. 
            1: label: 0 unfeasible, 1 feasible
            2: t_t: transfer time in s
            3: m0: initial mass of the spacecraft
            4: difference in semi-major axis of the origin and goal
            5: difference in eccentricity of the origin and goal
            6: cosine of the difference in inclination
            7: difference in RAANs
            8: difference in omega
            9: difference in true anomaly

            max m0 should be added
        """
        feasibilityFileName = "trainingData_Feas.txt"
        massFileName = "trainingData_Opt.txt"
        # Inputs 
        inputs = np.zeros(8)
        inputs[0] = self.t_t
        inputs[1] = self.m0

        Orbit_0 = AL_2BP.BodyOrbit(self.SV_0, "Cartesian", self.sun)
        Orbit_f = AL_2BP.BodyOrbit(self.SV_f, "Cartesian", self.sun)
        K_0 = Orbit_0.KeplerElem
        K_0[-1] = Orbit_0.theta # change mean anomaly with true anomaly
        K_f = Orbit_f.KeplerElem
        K_f[-1] = Orbit_f.theta

        inputs[2:] = K_f - K_0
        inputs[2] = abs(inputs[2]) # absolute value
        inputs[3] = abs(inputs[3]) # absolute value
        inputs[4] = np.cos(inputs[4])# cosine

        # Feasibility
        # if np.linalg.norm(self.Error[0:3]) <= 100e3 and \
        #     np.linalg.norm(self.Error[3:]) <= 100: # Feasible trajectory
        if np.linalg.norm(self.Error[0:3]) <= 5e7 and \
            np.linalg.norm(self.Error[3:]) <= 5e3: # TODO: change. Too much
            feasible = 1
        else:
            feasible = 0

        # Write to file
        vectorFeasibility = np.append(feasible, inputs)
        with open(feasibilityFileName, "a") as myfile:
            for value in vectorFeasibility:
                if value != vectorFeasibility[-1]:
                    myfile.write(str(value) +" ")
                else:
                    myfile.write(str(value))
            myfile.write("\n")
        myfile.close()

        vectorMass = np.append(self.m_fuel, inputs)
        with open(massFileName, "a") as myfile:
            for value in vectorMass:
                if value != vectorMass[-1]:
                    myfile.write(str(value) +" ")
                else:
                    myfile.write(str(value))
            myfile.write("\n")
        myfile.close()

    def __SimsFlanagan(self, SV_i, backwards = False, saveState = False):
        """
        __SimsFlanagan:
            No impulse is not applied in the state 0 or final.
            Only in intermediate steps
        """
        t_i = self.t_t / (self.Nimp + 1) # Divide time in 6 segments

        if saveState == True: # save each state 
            SV_list = np.zeros(((self.Nimp + 1)//2 + 1, 6))
            SV_list[0, :] = SV_i

        # Propagate only until middle point to save computation time
        # ! Problems if Nimp is even
        for imp in range((self.Nimp + 1)//2): 
            # Create trajectory 
            trajectory = AL_2BP.BodyOrbit(SV_i, 'Cartesian', self.sun)
            
            # Propagate the given time 
            # Coordinates after propagation
            SV_i = trajectory.Propagation(t_i, 'Cartesian') 
            
            if saveState == True:
                SV_list[imp + 1, :] = SV_i

            # Add impulse if not in last point
            if imp != self.Nimp:
                if backwards == True:
                    SV_i[3:] -= self.DeltaV_list[imp, :] * self.DeltaV_max # Reduce the impulses
                else:
                    SV_i[3:] += self.DeltaV_list[imp, :] * self.DeltaV_max
        
        if saveState == True:
            return SV_list # All states
        else:
            return SV_i # Last state



if __name__ == "__main__":
    ########################
    # Initial settings
    ########################
    Nimp = 10
    Fitness = Fitness(Nimp = Nimp)
    date0 = np.array([15,6,2019,0])
    t0 = AL_Eph.DateConv(date0,'calendar') #To JD

    bnd_v0 = (0, 5e3) # Relative to the planet
    bnd_v0_angle = (0., 2*np.pi)
    bnd_vf = ( 0.0, 9e3) # Relative to the planet
    # bnd_vf = ( v_escape *0.9, v_escape *1.1)
    bnd_vf_angle = (0., 2*np.pi)
    bnd_t0 = (t0.JD_0, t0.JD_0+1000) # Launch date
    # bnd_m0 = (0, 200) # Mass should never be 0 as you add dry mass
    bnd_t_t = (AL_BF.days2sec(200), AL_BF.days2sec(900) )
    bnd_deltavmag = (0., 1.) # magnitude
    bnd_deltavang = (-np.pi, np.pi) # angle

    bnds = (bnd_v0, bnd_v0_angle, bnd_v0_angle, \
            bnd_vf, bnd_vf_angle, bnd_vf_angle, \
            bnd_t0, bnd_t_t)

    for i in range(Nimp): # 3 times because impulses are 3d vectors
        bnds += (bnd_deltavmag, bnd_deltavang, bnd_deltavang)


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
    nsamples = 500 # number of training samples. TODO: increase
    samples_Lambert = np.zeros((nsamples, len(bnds)))
    # for decv in range(len(bnds)): 
    #     sample_inputs[:, decv] = np.random.uniform(low = bnds[decv][0], \
    #         high = bnds[decv][1], size = nsamples)

    ####################
    # CREATION OF RANDOM POPULATION WITH LAMBERT
    ####################
    start_time = time.time()

    for decv in range(6,8): 
        samples_Lambert[:, decv] = np.random.uniform(low = bnds[decv][0], \
            high = bnds[decv][1], size = nsamples)
    
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
            if v_i >= bnds[0][0] and  v_i <= bnds[0][1] and \
            v_i2 >= bnds[3][0] and  v_i2 <= bnds[3][1]:
                
                samples_Lambert[i, 0:3] = AL_BF.convert3dvector(v1[rev]-vE, "cartesian")
                samples_Lambert[i, 3:6] = AL_BF.convert3dvector(v2[rev]-vM, "cartesian")

                break
            elif rev == len(v1)-1:
                notvalid.append(i)
                # sample_inputs[i,:] = np.zeros(len(bnds))

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
    niter = 10
    niterlocal = 100
    niter_success = 10
    mytakestep = AL_OPT.MyTakeStep(Nimp, bnds)
    
    for i_sample in range(len(sample_inputs)):
        print("-------------------------------")
        print("Sample %i"%i_sample)
        print("-------------------------------")
        sample = sample_inputs[i_sample, :]
        fvalue = f(sample)
        Fitness.savetoFile() # saves the current values
        
        Fitness.printResult()
        
        # optimize starting from sample
        xmin, Best = AL_OPT.MonotonicBasinHopping(f, sample, mytakestep, niter = niter, \
                  niter_local = niterlocal, niter_success = niter_success, bnds = bnds, \
                  jumpMagnitude = 0.005, tolLocal = 1e-2, tolGlobal = 1e-5)
            
        fvalue = f(xmin)
        Fitness.savetoFile()
        Fitness.printResult()
        