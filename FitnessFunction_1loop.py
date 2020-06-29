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
        # DecV 
        v0 = np.array(DecV[0:3]) # vector, [magnitude, angle, angle]
        vf = np.array(DecV[3:6]) # vector, [magnitude, angle, angle]
        self.t0, mf, self.t_t = DecV[6:9]
        self.m0  = mf + self.Spacecraft.m_dry
        
        if optMode == True:
            DeltaV_list = np.array(DecV[9:]).reshape(-1,3) # make a Nimp x 3 matrix
        else:
            DeltaV_list = DecV[9:][0]

        self.DeltaV_list = np.zeros(np.shape(DeltaV_list))

        # Modify from magnitude angle angle to cartesian
        for i in range(len(self.DeltaV_list)):
            mag = DeltaV_list[i,0]
            angle1 = DeltaV_list[i,1]
            angle2 = DeltaV_list[i,2]
            self.DeltaV_list[i, 0] = mag * np.cos(angle1)*np.cos(angle2) 
            self.DeltaV_list[i, 1] = mag * np.sin(angle1)*np.cos(angle2)
            self.DeltaV_list[i, 2] = mag * np.sin(angle2)

        # = Thrust for segment
        self.DeltaV_max = self.Spacecraft.T / self.m0 * self.t_t / (self.Nimp + 1) 
 
        # Write velocity as x,y,z vector
        v0_cart = v0[0] *np.array([ np.cos(v0[1])*np.cos(v0[2]) , \
                                np.sin(v0[1])*np.cos(v0[2]),
                                np.sin(v0[2]) ])
        vf_cart = vf[0] *np.array([ np.cos(vf[1])*np.cos(vf[2]) , \
                                np.sin(vf[1])*np.cos(vf[2]),
                                np.sin(vf[2]) ])

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

        # Compare state at middle point
        # print("Error middle point", SV_list_back[-1, :],SV_list_forw[-1, :])
        self.Error = SV_list_back_corrected[-1, :] - SV_list_forw[-1, :]

        # Calculate mass used
        self.m_fuel = self.m0 - self.__propagateMass()

        if plot == True:
            # print(np.flipud(SV_list_back))
            self.plot2D(SV_list_forw, SV_list_back, [self.sun, self.earth, self.mars])

        # Return fitness function
        self.f = self.objFunction(self.m_fuel, self.Error)
        return self.f

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