import numpy as np
import pykep as pk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import AstroLibraries.AstroLib_Basic as AL_BF # Imported in AstroLib_2BP
from AstroLibraries import AstroLib_2BP as AL_2BP
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_Plots as AL_Plot

import LoadConfigFiles as CONFIG
CONF_C = CONFIG.Fitness_config()
CONF = CONF_C.Fit_config

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
        self.Nimp = kwargs.get('Nimp', 11) 

    def adaptDecisionVector(self, DecV, optMode = True):
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

    def objFunction(self, Error, m_fuel = False):
        
        fc1 = np.linalg.norm(Error[0:3] / AL_BF.AU) # Normalize with AU
        fc2 = np.linalg.norm(Error[3:] / AL_BF.AU * AL_BF.year2sec(1))
        # fc1 = np.linalg.norm(Error[0:3])
        # fc2 = np.linalg.norm(Error[3:])

        self.Epnorm = fc1
        self.Evnorm = fc2
        # print('------------------------')
        # print(m_fuel, Error)
        # print(fc1, fc2)
        if type(m_fuel) == bool:
            value = fc1 * CONF['FEASIB']['factor_pos'] + \
                    fc2 * CONF['FEASIB']['factor_vel']
        else:
            fc0 = m_fuel / self.Spacecraft.m_dry

            # Standarize between 0 and 1:
            std = False
            if std == True:
                if fc1 == 0: # Avoid overflows
                    fc1 = 0.00000001
                if fc2 == 0:
                    fc2 = 0.00000001
                fc1 = np.log10(fc1)
                fc2 = np.log10(fc2)
                # array = ['range_mass', 'range_pos', 'range_vel']
                # array2 = [fc0, fc1, fc2]
                # print(array2) 
            #     res = np.zeros(len(array))
            #     for i in range(len(array)):
            #         res[i] = ( array2[i] - CONF['FEASIB'][array[i]][0]) /\
            #                 (CONF['FEASIB'][array[i]][1] - CONF['FEASIB'][array[i]][0]) *\
            #                 (1 - 0)

            # else: 
            res = [fc0, fc1, fc2]
                # print(res)

            # print(res)
            value = res[0]* CONF['FEASIB']['factor_mass'] +\
                    res[1]* CONF['FEASIB']['factor_pos'] + \
                    res[2]* CONF['FEASIB']['factor_vel']
        
        # Penalization functions
        # print("obje", f0, fc1, fc2)
        # print("mass",m_fuel, "Error", Error)
        return value


    def calculateFitness(self, DecV, optMode = True, thrust = 'free', 
        printValue = False, plot = False, plot3D = False, 
        massInFunct = True):
        """
        calculateFitness: obtain the value of the fitness function
        INPUTS:
            DecV_I: decision vector of the inner loop
                t_t: transfer time for the leg. In seconds
                DeltaV_list: list of size = Number of impulses 
                            with the 3D impulse between 0 and 1 
                            (will be multiplied by the max DeltaV)
            thrust = free: direction determined by two angles in heliocentric frame.
                    tangential: thrust applied in the direction of motion
            massInFunct: if true, objective function with mass. Otherwise just errors
        """
        
        ########################################################################
        # DecV
        ########################################################################
        self.DecV = DecV
        v0_cart, vf_cart = self.adaptDecisionVector(DecV, optMode=optMode)

        ########################################################################
        # Propagation
        ########################################################################
        # Times and ephemeris
        # t_0 = AL_Eph.DateConv(self.date0,'calendar') #To JD
        t_1 = AL_Eph.DateConv(self.t0 + AL_BF.sec2days(self.t_t), 'JD_0' )

        self.r_p0, self.v_p0 = self.earthephem.eph(self.t0)
        self.r_p1, self.v_p1 = self.marsephem.eph(t_1.JD_0)
        
        # Change from relative to heliocentric velocity
        self.v0 = v0_cart + self.v_p0 
        self.vf = vf_cart + self.v_p1 

        # Create state vector for initial and final point
        self.SV_0 = np.append(self.r_p0, self.v0)
        self.SV_f = np.append(self.r_p1, self.vf) # - to propagate backwards
        self.SV_f_corrected = np.append(self.r_p1, -self.vf) # - to propagate backwards

        # Sims-Flanagan
        SV_list_forw = self.__SimsFlanagan(self.SV_0, saveState=True, thrust = thrust)
        SV_list_back = self.__SimsFlanagan(self.SV_f_corrected, backwards = True,\
             saveState=True, thrust = thrust)

        # convert back propagation so that the signs of velocity match
        SV_list_back_corrected = np.copy(SV_list_back)
        SV_list_back_corrected[:,3:] *= -1 # change sign of velocity

        ########################################################################
        # Compare State in middle point
        ########################################################################
        # print("Error middle point", SV_list_back[-1, :],SV_list_forw[-1, :])
        self.Error = SV_list_back_corrected[-1, :] - SV_list_forw[-1, :]

        if massInFunct == True:
            self.f = self.objFunction(self.Error, m_fuel = self.m_fuel)
        else:
            self.f = self.objFunction(self.Error)

        if printValue == True:
            print("Value: ", self.f)

        if plot == True:
            # print(np.flipud(SV_list_back))
            self.plot2D(SV_list_forw, SV_list_back, [self.sun, self.earth, self.mars])
        if plot3D == True:
            # print(np.flipud(SV_list_back))
            self.plot3D(SV_list_forw, SV_list_back, [self.sun, self.earth, self.mars])

        return self.f # *1000 so that in 
                                                            # tol they are in same order of mag


    def calculateMass(self, DecV, optMode = True):
        ########################################################################
        # DecV
        ########################################################################
        v0_cart, vf_cart = self.adaptDecisionVector(DecV, optMode=optMode)

        return self.m_fuel

    def __propagateMass(self):
        """
        propagateMass: propagate mass forward to avoid having to use
                another variable for the final mass
        """
        m_current = self.m0
        for imp in range(self.Nimp):
            dv_current = np.linalg.norm( self.DeltaV_list[imp,:]) * self.DeltaV_max 
            m_current = self.Spacecraft.MassChange(m_current, dv_current)

        return m_current

    def __SimsFlanagan(self, SV_i, thrust = 'free',  backwards = False, saveState = False):
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

            # Add impulse if not in last point
            if imp != (self.Nimp + 1)//2:
                if thrust == 'free':
                    if backwards == True:
                        SV_i[3:] -= self.DeltaV_list[imp, :] * self.DeltaV_max # Reduce the impulses
                    else:
                        SV_i[3:] += self.DeltaV_list[imp, :] * self.DeltaV_max
                elif thrust == 'tangential':
                    # If magnitude is backwards:
                    sign = np.sign( np.dot(SV_i[3:], self.DeltaV_list[imp, :]) )
                    if backwards == True:
                        SV_i[3:] -= SV_i[3:]/np.linalg.norm(SV_i[3:])* sign* np.linalg.norm(self.DeltaV_list[imp, :]) * self.DeltaV_max # Reduce the impulses
                    else:
                        SV_i[3:] += SV_i[3:]/np.linalg.norm(SV_i[3:])* sign* np.linalg.norm(self.DeltaV_list[imp, :]) * self.DeltaV_max

            else: # To compare, only add impulse on the one forward to compare delta v
                 if backwards != True:
                        SV_i[3:] += self.DeltaV_list[imp, :] * self.DeltaV_max # Reduce the impulses

            if saveState == True:
                SV_list[imp + 1, :] = SV_i

        if saveState == True:
            return SV_list # All states
        else:
            return SV_i # Last state
    
    def printResult(self):
        print("Mass of fuel", self.m_fuel, "Error", self.Error)

    def adaptDecisionVector_simplified(self, DecV):
        v0_cart, vf_cart = self.adaptDecisionVector(DecV)

        ########################################################################
        # Propagation
        ########################################################################
        # Times and ephemeris
        # t_0 = AL_Eph.DateConv(self.date0,'calendar') #To JD
        t_1 = AL_Eph.DateConv(self.t0 + AL_BF.sec2days(self.t_t), 'JD_0' )

        self.r_p0, self.v_p0 = self.earthephem.eph(self.t0)
        self.r_p1, self.v_p1 = self.marsephem.eph(t_1.JD_0)
        
        # Change from relative to heliocentric velocity
        self.v0 = v0_cart + self.v_p0 
        self.vf = vf_cart + self.v_p1 

        # Create state vector for initial and final point
        self.SV_0 = np.append(self.r_p0, self.v0)
        self.SV_f = np.append(self.r_p1, self.vf)

    def DecV2inputV(self, typeinputs, newDecV = 0):

        if type(newDecV) != int:
            self.adaptDecisionVector_simplified(newDecV)

        inputs = np.zeros(8)
        inputs[0] = self.t_t
        inputs[1] = self.m0

        if typeinputs == "deltakeplerian":
            t_1 = self.t0 + AL_BF.sec2days(self.t_t)

            # Elements of the planets
            # elem_0 = self.earthephem.osculating_elements(pk.epoch(self.t0, 'mjd2000') )
            # elem_f = self.marsephem.osculating_elements(pk.epoch(t_1, 'mjd2000'))
            
            # Elements of the spacecraft
            earth_elem = AL_2BP.BodyOrbit(self.SV_0, "Cartesian", self.sun)
            elem_0 = earth_elem.KeplerElem
            mars_elem = AL_2BP.BodyOrbit(self.SV_f, "Cartesian", self.sun)
            elem_f = mars_elem.KeplerElem

            K_0 = np.array(elem_0)
            K_f = np.array(elem_f)

            # Mean anomaly to true anomaly
            K_0[-1] = AL_2BP.Kepler(K_0[-1], K_0[1], 'Mean')[0]
            K_f[-1] = AL_2BP.Kepler(K_f[-1], K_f[1], 'Mean')[0]

            inputs[2:] = K_f - K_0
            inputs[2] = abs(inputs[2]) # absolute value
            inputs[3] = abs(inputs[3]) # absolute value
            inputs[4] = np.cos(inputs[4])# cosine

        elif typeinputs == "deltakeplerian_planet":
            t_1 = self.t0 + AL_BF.sec2days(self.t_t)

            # Elements of the planets
            elem_0 = self.earthephem.osculating_elements(pk.epoch(self.t0, 'mjd2000') )
            elem_f = self.marsephem.osculating_elements(pk.epoch(t_1, 'mjd2000'))
            
            # Elements of the spacecraft
            # earth_elem = AL_2BP.BodyOrbit(self.SV_0, "Cartesian", self.sun)
            # elem_0 = earth_elem.KeplerElem
            # mars_elem = AL_2BP.BodyOrbit(self.SV_f, "Cartesian", self.sun)
            # elem_f = mars_elem.KeplerElem

            K_0 = np.array(elem_0)
            K_f = np.array(elem_f)

            # Mean anomaly to true anomaly
            K_0[-1] = AL_2BP.Kepler(K_0[-1], K_0[1], 'Mean')[0]
            K_f[-1] = AL_2BP.Kepler(K_f[-1], K_f[1], 'Mean')[0]

            inputs[2:] = K_f - K_0
            inputs[2] = abs(inputs[2]) # absolute value
            inputs[3] = abs(inputs[3]) # absolute value
            inputs[4] = np.cos(inputs[4])# cosine
        
        elif typeinputs == "cartesian":
            delta_r = np.array(self.SV_f[0:3]) - np.array(self.SV_0[0:3])
            delta_v = np.array(self.SV_f[3:]) - np.array(self.SV_0[3:])  
            inputs[2:5] = delta_r
            inputs[5:] = delta_v

        return inputs
    
    def studyFeasibility(self):
        
        if np.linalg.norm(self.Error[0:3]) <= CONF['FEASIB']['feas_ep'] and \
            np.linalg.norm(self.Error[3:]) <= CONF['FEASIB']['feas_ev']: 
            feasible = 1
        else:
            feasible = 0
        return feasible

    def savetoFile(self, typeinputs, filepath_feas, inputs = False):
        """
        savetoFile: save input parameters for neural network and the fitness and
        feasibility

        Append to file

            max m0 should be added
        INPUTS: 
            typeinputs: "deltakeplerian" or "cartesian"
                        save in different columns. 
                deltakeplerian:
                1: label: 0 unfeasible, 1 feasible
                2: t_t: transfer time in s
                3: m0: initial mass of the spacecraft
                4: difference in semi-major axis of the origin and goal
                5: difference in eccentricity of the origin and goal
                6: cosine of the difference in inclination
                7: difference in RAANs
                8: difference in omega
                9: difference in true anomaly
                cartesian:
                1: label: 0 unfeasible, 1 feasible
                2: t_t: transfer time in s
                3: m0: initial mass of the spacecraft
                4: delta x
                5: delta y
                6: delta z
                7: delta vx including the velocity of the spacecraft              
                8: delta vy                
                9: delta vz 
        """
        feasibilityFileName = filepath_feas
        # massFileName = filepath_m
        
        # Inputs 
        if type(inputs) == bool:
            inputs = self.DecV2inputV(typeinputs)
        else:
            inputs = self.DecV2inputV(typeinputs, newDecV = inputs)
        
        # Feasibility
        feasible = self.studyFeasibility()
        feasible = np.append(feasible, self.m_fuel)

        # Write to file
        vectorFeasibility = np.append(feasible, self.Error)
        vectorFeasibility = np.append(vectorFeasibility, inputs)
        with open(feasibilityFileName, "a") as myfile:
            for value in vectorFeasibility:
                if value != vectorFeasibility[-1]:
                    myfile.write(str(value) +" ")
                else:
                    myfile.write(str(value))
            myfile.write("\n")
        myfile.close()

        # vectorMass = np.append(self.m_fuel, inputs)
        # with open(massFileName, "a") as myfile:
        #     for value in vectorMass:
        #         if value != vectorMass[-1]:
        #             myfile.write(str(value) +" ")
        #         else:
        #             myfile.write(str(value))
        #     myfile.write("\n")
        # myfile.close()


    def plot3D(self, SV_f, SV_b, bodies, *args, **kwargs):

        """

        """
        # Create more points for display
        points = 10
        state_f = np.zeros(( (points+1)*(self.Nimp +1)//2 +1, 6 ))
        state_b = np.zeros(( (points+1)*(self.Nimp +1)//2 +1, 6 ))
        t_i = self.t_t / (self.Nimp +1) / (points+1)


        # Change velocity backwards to propagate backwards
        for j in range((self.Nimp+1)//2):
            state_f[(points+1)*j,:] = SV_f[j,:]
            state_b[(points+1)*j,:] = SV_b[j,:]
            for i in range(points):
                trajectory = AL_2BP.BodyOrbit(state_f[(points+1)*j + i], 'Cartesian', self.sun)
                state_f[(points+1)*j + i+1,:] = trajectory.Propagation(t_i, 'Cartesian')

                trajectory_b = AL_2BP.BodyOrbit(state_b[(points+1)*j + i,:], 'Cartesian', self.sun)
                state_b[(points+1)*j + i+1,:] = trajectory_b.Propagation(t_i, 'Cartesian') 

                # trajectory_b = AL_2BP.BodyOrbit(SV_b[j,:], 'Cartesian', self.sun)
                # state_b[(points+1)*j + i+1,:] = trajectory_b.Propagation(t_i*(i+1), 'Cartesian') 

        state_f[-1,:] = SV_f[-1,:]
        state_b[-1,:] = SV_b[-1,:]

        # Plot
        fig = plt.figure()
        ax = Axes3D(fig)
        # ax.view_init(azim=0, elev=10)
        # plot planets
        ax.scatter(0, 0, 0,  color = bodies[0].color, marker = 'o', s = 180, alpha = 0.5)
        ax.scatter(SV_f[0,0], SV_f[0,1], SV_f[0,2], c = bodies[1].color, marker = 'o', s = 150, alpha = 0.5)
        ax.scatter(SV_b[0,0], SV_b[0,1], SV_b[0,2], c = bodies[2].color, marker = 'o', s = 150, alpha = 0.5)

        # plot points for Sims-Flanagan
        x_f = SV_f[:,0]
        y_f = SV_f[:,1]
        z_f = SV_f[:,2]

        x_b = SV_b[:,0]
        y_b = SV_b[:,1]
        z_b = SV_b[:,2]

        ax.scatter(x_f, y_f, z_f, '^-', c = bodies[1].color)
        ax.scatter(x_b, y_b, z_b, '^-', c = bodies[2].color)

        # ax.scatter(x_f[-1],y_f[-1],z_f[-1], '^',c = bodies[1].color)
        # ax.scatter(x_b[-1],y_b[-1],z_b[-1], '^',c = bodies[2].color)

        # ax.plot(state_f[:,0],state_f[:,1], state_f[:,2], 'x-',color = bodies[1].color)
        # ax.plot(state_b[:,0],state_b[:,1], state_b[:,2], 'x-',color = bodies[2].color)
        ax.plot(state_f[:,0],state_f[:,1], state_f[:,2], color = bodies[1].color)
        ax.plot(state_b[:,0],state_b[:,1], state_b[:,2], color = bodies[2].color)


        # Plot settings
        AL_Plot.set_axes_equal(ax)

        dpi = kwargs.get('dpi', 200) 
        layoutSave = kwargs.get('layout', 'tight')
        plt.savefig('OptSol/resultopt3D.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()

    def plot2D(self, SV_f, SV_b, bodies, *args, **kwargs):
        # fig = plt.figure()
        fig, ax = plt.subplots()
        
        # plot planets
        ax.scatter(0, 0,  color = bodies[0].color, marker = 'o', s = 180, alpha = 0.5)
        ax.scatter(SV_f[0,0], SV_f[0,1], c = bodies[1].color, marker = 'o', s = 150, alpha = 0.5)
        ax.scatter(SV_b[0,0], SV_b[0,1], c = bodies[2].color, marker = 'o', s = 150, alpha = 0.5)

        x_f = SV_f[:,0]
        y_f = SV_f[:,1]
        z_f = SV_f[:,2]

        x_b = SV_b[:,0]
        y_b = SV_b[:,1]
        z_b = SV_b[:,2]

        dv1 = self.DeltaV_list[0:len(x_f), :]*1e11
        dv2 = self.DeltaV_list[1:len(x_b), :]*1e11

        for i in range(len(dv1)):
            plt.arrow(x_f[i], y_f[i], dv1[i, 0], dv1[i, 1], length_includes_head=True,
          head_width=5e9, head_length=5e9, fc='k', ec='k')
        for i in range(len(dv2)):
            plt.arrow(x_b[i], y_b[i], dv2[i, 0], dv2[i, 1], length_includes_head=True,
          head_width=5e9, head_length=5e9, fc='k', ec='k')

        ax.plot(x_f, y_f, '^-', c = bodies[1].color)
        ax.plot(x_b, y_b, '^-', c = bodies[2].color)

        ax.scatter(x_f[-1], y_f[-1], marker ='^', c = bodies[1].color)
        ax.scatter(x_b[-1], y_b[-1], marker ='^', c = bodies[2].color)

        plt.axis('equal')
        plt.grid(alpha = 0.5)
        # AL_Plot.set_axes_equal(ax)

        dpi = kwargs.get('dpi', 200) 
        layoutSave = kwargs.get('layout', 'tight')
        plt.savefig('OptSol/resultopt.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()

    def plot_tvsT(self):

        t = np.linspace(self.t0, self.t0 +AL_BF.sec2days(self.t_t), num = self.Nimp+2)
        deltaV_i = [np.linalg.norm(self.DeltaV_list[i,:]) *self.DeltaV_max for i in range(len(self.DeltaV_list[:,0])) ]
        deltaV = np.zeros(self.Nimp +2)
        deltaV[1:-1] = deltaV_i 

        fig, ax = plt.subplots()
        plt.plot(t, deltaV, 'o-', color = 'k')

        plt.title("Epoch vs Delta V")
        plt.ylabel("Delta V (m/s)")
        plt.xlabel("JD0 (days)")
        plt.grid(alpha = 0.5)
        # AL_Plot.set_axes_equal(ax)

        dpi = 200
        layoutSave = 'tight'
        plt.savefig('OptSol/tvsT.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()

class Propagate:
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
        self.Nimp = kwargs.get('Nimp', 11) 

    def adaptDecisionVector(self, DecV, optMode = True):
        """ 
        adaptDecisionVector: modify decision vector to input in the problem
        """
        v0 = np.array(DecV[0:3]) # vector, [magnitude, angle, angle]
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
            self.DeltaV_list[i, :] = AL_BF.convert3dvector(DeltaV_list[i,:], "polar")
            DeltaV_sum[i] = np.linalg.norm( self.DeltaV_list[i] )

        # Write velocity as x,y,z vector
        v0_cart = AL_BF.convert3dvector(v0, "polar")

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

        return v0_cart

    def prop(self, DecV, optMode = True, printValue = False, plot = False):
        self.DecV = DecV
        v0_cart = self.adaptDecisionVector(DecV, optMode=optMode)

        t_1 = AL_Eph.DateConv(self.t0 + AL_BF.sec2days(self.t_t), 'JD_0' )

        self.r_p0, self.v_p0 = self.earthephem.eph(self.t0)
        self.r_p1, self.v_p1 = self.marsephem.eph(t_1.JD_0)
    
        # Change from relative to heliocentric velocity
        self.v0 = v0_cart + self.v_p0 

        # Create state vector for initial and final point
        self.SV_0 = np.append(self.r_p0, self.v0)

        # Sims-Flanagan
        SV_list_forw = self.__SimsFlanagan(self.SV_0, saveState=True)

        self.rv_final = SV_list_forw[-1,:]

        if plot == True:
            self.plot2D(SV_list_forw, [self.sun, self.earth, self.mars])


    def __SimsFlanagan(self, SV_i, backwards = False, saveState = False):
        """
        __SimsFlanagan:
            No impulse is not applied in the state 0 or final.
            Only in intermediate steps
        """
        t_i = self.t_t / (self.Nimp + 1) # Divide time in 6 segments

        if saveState == True: # save each state 
            SV_list = np.zeros(((self.Nimp + 2), 6))
            SV_list[0, :] = SV_i

        # Propagate only until middle point to save computation time
        # ! Problems if Nimp is even
        for imp in range(self.Nimp + 1): 
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
    
    def DecV2inputV(self, newDecV = 0):

        if type(newDecV) != int:
            self.adaptDecisionVector_simplified(newDecV)

        inputs = np.zeros(8)
        inputs[0] = self.t_t
        inputs[1] = self.m0

        # Initial elements
        elem_0 = self.earthephem.osculating_elements(pk.epoch(self.t0, 'mjd2000') )
        K_0 = np.array(elem_0)
        # Mean anomaly to true anomaly
        K_0[-1] = AL_2BP.Kepler(K_0[-1], K_0[1], 'Mean')[0]

        # Final elements
        virtualBody_Orbit = AL_2BP.BodyOrbit(self.rv_final, 'Cartesian', self.sun)
        elem_f = virtualBody_Orbit.KeplerElem 
        K_f = np.array(elem_f)
        K_f[-1] = AL_2BP.Kepler(K_f[-1], K_f[1], 'Mean')[0]

        inputs[2:] = K_f - K_0
        inputs[2] = abs(inputs[2]) # absolute value
        inputs[3] = abs(inputs[3]) # absolute value
        inputs[4] = np.cos(inputs[4])# cosine

        return inputs

    def savetoFile(self):
        """
        savetoFile: save input parameters for neural network and the fitness and
        feasibility

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
        feasibilityFileName = "trainingData_Feas_fake.txt"
        massFileName = "trainingData_Opt_fake.txt"
        
        # Inputs 
        inputs = self.DecV2inputV()

        # Feasibility
        feasible = 1

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

    def plot2D(self, SV_f, bodies, *args, **kwargs):
        # fig = plt.figure()
        fig, ax = plt.subplots()
        
        # plot planets
        ax.scatter(0, 0,  color = bodies[0].color, marker = 'o', s = 180, alpha = 0.5)
        ax.scatter(SV_f[0,0], SV_f[0,1], c = bodies[1].color, marker = 'o', s = 150, alpha = 0.5)
        ax.scatter(self.r_p1[0], self.r_p1[1], c = bodies[2].color, marker = 'o', s = 150, alpha = 0.5)

        x_f = SV_f[:,0]
        y_f = SV_f[:,1]
        z_f = SV_f[:,2]

        dv1 = self.DeltaV_list[0:len(x_f), :]*1e11

        for i in range(len(dv1)):
            plt.arrow(x_f[i], y_f[i], dv1[i, 0], dv1[i, 1], length_includes_head=True,
          head_width=5e9, head_length=5e9, fc='k', ec='k')

        ax.plot(x_f, y_f, '^-', c = bodies[1].color)

        ax.scatter(x_f[-1], y_f[-1], marker ='^', c = bodies[1].color)

        plt.axis('equal')
        plt.grid(alpha = 0.5)
        # AL_Plot.set_axes_equal(ax)

        dpi = kwargs.get('dpi', 200) 
        layoutSave = kwargs.get('layout', 'tight')
        plt.savefig('./OptSol/resultopt.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()