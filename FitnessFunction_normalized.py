import numpy as np
import pykep as pk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import AstroLibraries.AstroLib_Basic as AL_BF # Imported in AstroLib_2BP
from AstroLibraries import AstroLib_2BP as AL_2BP
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_Plots as AL_Plot


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
            mag = DeltaV_list[i,0]
            angle1 = DeltaV_list[i,1]
            angle2 = DeltaV_list[i,2]
            self.DeltaV_list[i, 0] = mag * np.cos(angle1)*np.cos(angle2) 
            self.DeltaV_list[i, 1] = mag * np.sin(angle1)*np.cos(angle2)
            self.DeltaV_list[i, 2] = mag * np.sin(angle2)
            DeltaV_sum[i] = np.linalg.norm( self.DeltaV_list[i] )

        # Write velocity as x,y,z vector
        v0_cart = v0[0] *np.array([ np.cos(v0[1])*np.cos(v0[2]) , \
                                np.sin(v0[1])*np.cos(v0[2]),
                                np.sin(v0[2]) ])
        vf_cart = vf[0] *np.array([ np.cos(vf[1])*np.cos(vf[2]) , \
                                np.sin(vf[1])*np.cos(vf[2]),
                                np.sin(vf[2]) ])

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

    def objFunction(self, m_fuel, Error):
        f0 = m_fuel 
        fc1 = (np.linalg.norm(Error[0:3]) - 1e6)/1e6
        # fc2 = (np.linalg.norm(Error[3:])- 1e2)/1e2
        fc2 = np.linalg.norm(Error[3:])
        
        # Penalization functions
        # print("obje", f0, fc1, fc2)
        f = f0 + fc1 + fc2
        # print("mass",m_fuel, "Error", Error)
        return f

    def calculateFitness(self, DecV, optMode = True, plot = False):
        """
        calculateFitness: obtain the value of the fitness function
        INPUTS:
            DecV_I: decision vector of the inner loop
                t_t: transfer time for the leg. In seconds
                DeltaV_list: list of size = Number of impulses 
                            with the 3D impulse between 0 and 1 
                            (will be multiplied by the max DeltaV)
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
        SV_0 = np.append(r_p0, self.v0)
        SV_1 = np.append(r_p1, -self.vf) # - to propagate backwards

        # Sims-Flanagan
        SV_list_forw = self.__SimsFlanagan(SV_0, saveState=True)
        SV_list_back = self.__SimsFlanagan(SV_1, backwards = True, saveState=True)

        # convert back propagation so that the signs of velocity match
        SV_list_back_corrected = np.copy(SV_list_back)
        SV_list_back_corrected[:,3:] *= -1 # change sign of velocity

        ########################################################################
        # Compare State in middle point
        ########################################################################
        # print("Error middle point", SV_list_back[-1, :],SV_list_forw[-1, :])
        self.Error = SV_list_back_corrected[-1, :] - SV_list_forw[-1, :]

        if plot == True:
            # print(np.flipud(SV_list_back))
            self.plot2D(SV_list_forw, SV_list_back, [self.sun, self.earth, self.mars])

        ########################################################################
        # Return fitness 
        ########################################################################
        self.printResult()

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
        SV_0 = np.append(r_p0, self.v0)
        SV_1 = np.append(r_p1, -self.vf) # - to propagate backwards

        # Sims-Flanagan
        SV_list_forw = self.__SimsFlanagan(SV_0, saveState=True)
        SV_list_back = self.__SimsFlanagan(SV_1, backwards = True, saveState=True)

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


    def calculateMass(self, DecV, optMode = True):
        ########################################################################
        # DecV
        ########################################################################
        v0_cart, vf_cart = self.adaptDecisionVector(DecV, optMode)

        return self.m_fuel

    def printResult(self):
        print("Mass of fuel", self.m_fuel, "Error", self.Error)

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

    def plot(self, SV_f, SV_b, bodies, *args, **kwargs):
        fig = plt.figure()
        ax = Axes3D(fig)
        
        # plot planets
        ax.scatter(0, 0, 0,  color = bodies[0].color, marker = 'o', s = 180, alpha = 0.5)
        ax.scatter(SV_f[0,0], SV_f[0,1], SV_f[0,2], c = bodies[1].color, marker = 'o', s = 150, alpha = 0.5)
        ax.scatter(SV_b[0,0], SV_b[0,1], SV_b[0,2], c = bodies[2].color, marker = 'o', s = 150, alpha = 0.5)

        x_f = SV_f[:,0]
        y_f = SV_f[:,1]
        z_f = SV_f[:,2]

        x_b = SV_b[:,0]
        y_b = SV_b[:,1]
        z_b = SV_b[:,2]

        ax.plot(x_f, y_f, z_f, '^-', c = bodies[1].color)
        ax.plot(x_b, y_b, z_b, '^-', c = bodies[2].color)

        ax.scatter(x_f[-1],y_f[-1],z_f[-1], '^',c = bodies[1].color)
        ax.scatter(x_b[-1],y_b[-1],z_b[-1], '^',c = bodies[2].color)

        AL_Plot.set_axes_equal(ax)

        dpi = kwargs.get('dpi', 200) 
        layoutSave = kwargs.get('layout', 'tight')
        plt.savefig('resultopt.png', dpi = dpi, bbox_inches = layoutSave)

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
        plt.savefig('resultopt.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()