import matplotlib.pyplot as plt
import numpy as np
import scipy as spy

import time

from . import AstroLib_Basic as AL_BF
from . import AstroLib_2BP as AL_2BP
from . import AstroLib_Ephem as AL_Eph

class SystemBodies:
    """ 
    SystemBodies: calculate the general parameters for a Hohmann orbit as an 
    approximation using the information of the orbital parameters. 
    """
    def __init__(self, Planet, *args, **kwargs):
        """
        Constructor
        INPUTS: 
            Planet: matrix with the different bodies. In order from lower 
            orbit to higher
        OUTPUTS:
            T_syn: synodic time
            a_t: semi-major axis
            timeflight: time to arrive from one planet to the other. 
        """
        self.centerBody = Planet[0].orb.CentralBody #Valid if same central body
        self.Planets = Planet
#         self.PlanetsOrbit = PlanetOrbit

    def SynodicPeriod(self, units = 'sec'):
        """
        SynodicPeriod: calculate the synodic period of two planets
        INPUTS:
            units: units of time. 'sec', 'day'
        OUTPUTS:
            T_syn: synodic period
        """
        self.T_syn = 2*np.pi / abs(self.Planets[0].orb.n - self.Planets[1].orb.n)
        if units == 'day':
            self.T_syn = AL_BF.sec2days(self.T_syn)
        return self.T_syn

    def findDateWith180deg(self, date0, transfertime, planetEphem = None, \
        usingCircular = False):
        """
        findDateWith180deg: find the date at which the planets are 180 deg
        INPUTS:
            date0: initial date to start looking
            transfertime: time of transfer between the planets
            planetEphem: list with planet ephemeris
            usingCircular: using circular orbits
        """
        if usingCircular == False:
            continueLoop = False
            loop = 0
            t_0 = AL_Eph.DateConv(date0,'calendar') #To JD
            while continueLoop == False:
                loop += 1
                t = t_0.JD_0 + 15*loop

                r_0, v_0 = planetEphem[0].eph(t)
                r_1, v_1 = planetEphem[1].eph(t + transfertime)

                orbit_0 = AL_2BP.BodyOrbit(np.append(r_0, v_0), 'Cartesian', \
                    self.centerBody)
                self.Planets[0].addOrbit(orbit_0)

                orbit_1 = AL_2BP.BodyOrbit(np.append(r_1, v_1), 'Cartesian', \
                    self.centerBody)
                self.Planets[1].addOrbit(orbit_1)

                # Hohman transfer between Earth and Mars
                delta = difTAnom([r_0, r_1])
                if AL_BF.rad2deg(delta) > 177 and AL_BF.rad2deg(delta) <183:
                    continueLoop = True
            return [t, r_0, v_0, r_1, v_1]

        else: # Assuming circular orbits
            t_0 = AL_Eph.DateConv(date0,'calendar') #To JD
            self.r_0, self.v_0 = planetEphem[0].eph(t_0.JD_0)
            self.r_1, self.v_1 = planetEphem[1].eph(t_0.JD_0 + transfertime)
            delta = difTAnom([self.r_0, self.r_1])
            if AL_BF.rad2deg(delta) > 179 and AL_BF.rad2deg(delta) <181:
                return t_0
            t = (np.pi/2 - delta)/(self.Planets[1].orb.n + self.Planets[0].orb.n)
            return t_0.JD + AL_BF.sec2days(t)

    def HohmannTransfer(self, *args, **kwargs ):
        
        a_0 = kwargs.get('r0', self.Planets[0].orb.a) 
        a_1 = kwargs.get('r1', self.Planets[1].orb.a)            

        ########################
        # Semi-major axis of the transfer orbit between the planets
        self.a_H = (a_0 + a_1) / 2
        # self.a_H = (self.Planets[0].orb.a + self.Planets[1].orb.a) / 2
    
        # Time of the trajectory between the Earth and Mars
        # Half of the period of an elliptical orbit
        # Hohmann time:
        self.timeflight_H = np.pi * self.a_H**(3/2) / np.sqrt(self.centerBody.mu)  
    
        #Mean motion, period of the orbit
        self.n_H = np.sqrt(self.centerBody.mu / self.a_H**3) 
        self.T_H = 2*np.pi * np.sqrt(self.a_H**3 / self.centerBody.mu)
        
        #Semi-latus rectum, 
        self.rp_H = a_0
        self.ra_H = a_1
        self.vp_H = np.sqrt( 2*self.centerBody.mu * (1/self.rp_H - 1/(2*self.a_H)) )
        self.va_H = np.sqrt( 2*self.centerBody.mu * (1/self.ra_H - 1/(2*self.a_H)) )
        self.e_H = 1- self.rp_H / self.a_H
        self.p_H = self.a_H * (1-self.e_H**2)
        self.b_H = self.a_H * np.sqrt(1-self.e_H**2)

    
def difTAnom(state):
    """
    difTrueAnomaly: get the difference in true anomaly of the planets to see 
    if both have the right angle for the problem.
    INPUTS:
        state: position and velocity of the planets 
    OUTPUTS:
        deltanu: difference in true anomaly            
    """
    r_1 = state[0]
    r_2 = state[1]
    
    if r_1[1] >= 0 and r_1[0] <= 0: # Second quadrant
        deltanu_1 = np.arctan(r_1[1]/r_1[0])+np.pi
    elif r_1[1] < 0 and r_1[0] < 0: # Third quadrant
        deltanu_1 = np.arctan(r_1[1]/r_1[0])+np.pi
    else:
        deltanu_1 = np.arctan(r_1[1]/r_1[0])

    if r_2[1] >= 0 and r_2[0] <= 0: # Second quadrant
        deltanu_2 = np.arctan(r_2[1]/r_2[0])+np.pi
    elif r_2[1] < 0 and r_2[0] < 0: # Third quadrant
        deltanu_2 = np.arctan(r_2[1] / r_2[0]) + np.pi
    else:
        deltanu_2 = np.arctan(r_2[1] / r_2[0])

    deltanu = abs(deltanu_2 - deltanu_1)
    return deltanu
    

class Lambert:
    """
    Lambert: Lambert transfer between two points
    """
    def __init__(self, r_0, r_1, transfertime, centerbody_mu):
        """
        Constructor
        INPUTS:
            r_0: initial position
            r_1: final position
            transfertime: time of transfer between positions
            centerbody_mu: gravitational parameter of the central body
        """
        self.r_0 = r_0
        self.r_1 = r_1
        self.centerbody_mu = centerbody_mu
        delta = difTAnom([r_0, r_1])

        mr_0 = np.linalg.norm(r_0) # Magnitude of position vectors
        mr_1 = np.linalg.norm(r_1)  
        c = np.linalg.norm(r_1-r_0)  #chord
        
        am = (c + mr_0 + mr_1)/4 # semi-major axis for minimum energy
        s = 2*am
        betam = 2*np.arcsin( np.sqrt((s-c) / (2*am)) )
        tm = np.sqrt(s**3/8) * (np.pi - betam + np.sin(betam)) / \
            np.sqrt(centerbody_mu)
        
        #Obtain a
        self.a_L = spy.optimize.fsolve(self.__LambertKepler, am, \
            [r_0,r_1,c,am,delta, transfertime, tm])
        self.alpha, self.beta = self.__LambertAlphaBeta(2*am, c, \
            self.a_L, delta, transfertime, tm)

    def TerminalVelVect(self):
        """
        TerminalVelVect: calculate the terminal velocity vectors
        """
        # Terminal velocity vectors
        A = np.sqrt( self.centerbody_mu /(4*self.a_L) ) / np.tan(self.alpha/2)
        if self.beta == 0:
            B = 0
        else:
            B = np.sqrt(self.centerbody_mu /(4*self.a_L)) / np.tan(self.beta/2)
        
        u1 = self.r_0 / np.linalg.norm(self.r_0) #unit vector in the direction of r1
        u2 = self.r_1 / np.linalg.norm(self.r_1) #unit vector in the direction of r2
        # unit vector associated with the chord:
        uc = (self.r_1-self.r_0)/(np.linalg.norm(self.r_1-self.r_0)) 
        
        v_0 = (B+A)*uc+(B-A)*u1
        v_1 = (B+A)*uc+(A-B)*u2
        return v_0, v_1

    def __LambertKepler(self, a, x):
        """
        LambertKepler: solve Kepler's equation for the case of the Lambert's 
        problem'
        Inputs:
            a: initial guess of semi-major axis
            x: additional arguments [r1,r2,c,am,nu,t,tm]
                r1: position of first point
                r2: position of second point
                c: chord that joins the two positions
                am: semi-major axis of the minimum-energy transfer
                nu: difference in true anomaly between the planets
                t: time of flight (s)
                tm: time of minimium-energy transfer
            Outputs:
                Func: equation to solve
        """
        [r1, r2, c, am, nu, t, tm] = x
        s = 2*am # Semiperimeter of the space triangle
        alpha,beta = self.__LambertAlphaBeta(s, c, a, nu, t, tm)
        
        Func = - ( alpha - beta - np.sin(alpha) + np.sin(beta)) +\
            np.sqrt(self.centerbody_mu / a**3) * t # Lambert's equation:
        return Func

    def __LambertAlphaBeta(self, s, c, a, nu, t, tm):
        """
        LambertAlphaBeta: obtain the alpha and beta angles of the Lambert method.
        Inputs:
            s: semiperimeter of the space triangle
            c: chord that joins the points
            a: semi-major axis of the transfer orbit
            nu: difference in true anomaly of the planets
            t: time of tranfer
            tm: time of minimium-energy transfer
        Outputs:
            alpha: angle for the Lambert algorithm
            beta: angle for the Lambert algorithm
        """
        beta = 2 * np.arcsin( np.sqrt(( s - c) / ( 2 * a))) 
    #     if nu > np.pi and nu < (2*np.pi):  #Eliminate correction to go using the shorter side
    #         beta = -beta

        alpha = 2 * np.arcsin( np.sqrt( s / ( 2 * a )))
        # Do not correct because we want to go always in the same direction (counterclockwise)
        if t > tm:    # In case is to the direction where is longer
            alpha = 2 * np.pi - alpha
        
        return alpha, beta

# # Repeating orbits

# class repitingOrbit: # Review
#     """
#     repitingOrbit: contains the parameters necessary to obtain a repiting orbit.
#     """
#     def __init__(self, j, k, Body, Cts,*args, **kwargs):
#         """
#         Constructor
#         INPUTS:
#             j: Orbital revolutions necessary for repetition
#             k: Earth days for 1 repiting orbit of the satellite
#             Body: main body about which the orbit is performed
#             Cts: list of constants
#         ADDITIONAL ARGUMENTS:
#                 a: semi-major axis of the repiting orbit
#                 e: eccentricity of the repiting orbit
#                 i: inclination angle of the repiting orbit
#         """
#         self.j = j
#         self.k = k
        
#         self.a = kwargs.get('a', 'No_a') 
#         self.e = kwargs.get('e', 'No_e') 
#         self.i = kwargs.get('i', 'No_i') 
        

#     def fsolve_Simple(self,x,param):
#         """
#         fsolve_Simple: solve the system of equations to obtain a, e or i. Equations taken from J. R. Wertz, Mission geometry: orbit and constellation design and management: 
#         spacecraft  orbit  and  attitude  systems.
#         INPUTS:
#             x: a, e or i for the initial guess for the solver
#             param:
#                 Body: main body about which the orbit is performed
#                 j: Orbital revolutions necessary for repetition
#                 k: Earth days for 1 repiting orbit of the satellite
#                 name: 'a','e' or 'i' specifying what is the x given
#         OUTPUTS:
#             F: function to solve
#         """
#         i = self.i
#         e = self.e
#         a = self.a
#         [Body,Cts,j,k,name] = param
        
#         #Select the variable used as input
#         if name == 'a':
#             a = x
#         elif name == 'i':
#             i = x
#         elif name == 'e':
#             e = x
#         else:
#             print('Warning: not enough values')
        
#         T = 2*np.pi*np.sqrt(a**3/Body.mu) #Period of the satellite orbit
#         deltaL1 = -2 *np.pi*T/Body.T #rad/rev. Difference in longitude 1
#         deltaL2 = -3*np.pi*Cts.J2*Body.R**2* np.cos(i) / (a**2 * (1-e**2)**2) #rad/rev. Difference in longitude 2
        
#         F = k/j*2*np.pi - abs(deltaL1+deltaL2) #Function to solve
#         return F
    
#     def get_a_Simple(self, Body,Cts, x0):
#         """
#         get_a_Simple: method to solve 'a' given the other two taking into consideration
#         only the effect of J2 on the right ascension of the ascending node.
#         INPUTS: 
#             Body: main body about which the orbit is performed
#             x0: initial guess to input for the semi-major axis
#         OUTPUTS:
#             solution for a
#         """
#         return spy.optimize.fsolve(self.fsolve_Simple,x0,[Body,Cts,self.j,self.k,'a'])
    
#     def get_e_Simple(self, Body,Cts, x0):
#         """
#         get_e_Simple: method to solve 'e' given the other two taking into consideration
#         only the effect of J2 on the right ascension of the ascending node.
#         INPUTS: 
#             Body: main body about which the orbit is performed
#             x0: initial guess to input for the eccentricity
#         OUTPUTS:
#             solution for e
#         """
#         return spy.optimize.fsolve(self.fsolve_Simple,x0,[Body,Cts,self.j,self.k,'e'])
    
#     def get_i_Simple(self, Body,Cts, x0):
#         """
#         get_i_Simple: method to solve 'i' given the other two taking into consideration
#         only the effect of J2 on the right ascension of the ascending node.
#         INPUTS: 
#             Body: main body about which the orbit is performed
#             x0: initial guess to input for the inclination angle
#         OUTPUTS:
#             solution for i
#         """
#         return spy.optimize.fsolve(self.fsolve_Simple,x0,[Body,Cts,self.j,self.k,'i'])
         
    
#     def get_a(self, Body,Cts, N_iter):
#         """
#         get_a: method to solve 'a' given the other two taking into consideration
#         the effect of J2 on the right ascension of the ascending node (RAAN), the argument 
#         of the periapsis (omega) and the Mean anomaly (M).  Equations taken from J. R. Wertz,
#         Mission geometry: orbit and constellation design and management: spacecraft  orbit  
#         and  attitude  systems.
#         INPUTS: 
#             Body: main body about which the orbit is performed
#             N_iter: number of iterations to perform
#         OUTPUTS:
#             a: solution for the semi-major axis
#             n: solution for the mean motion
#         """
#         e = self.e
#         i = self.i
        
#         a = Body.mu**(1/3) * (Cts.DSid* self.k / (self.j*2*np.pi))**(2/3) #km. Initial guess for a
#         Ldot = 360 #deg/day. Rate of change of the longitude

#         for x in range(N_iter):
#             RAANdot = -3/2 * Cts.J2 *np.sqrt(Body.mu) *Body.R**2 * a**(-3.5)* np.cos(i)*(1-e**2)**(-2)  * 180/np.pi* Cts.DSid #deg/day. Rate of change of RAAN
#             omegadot = 3/4 * Cts.J2 *np.sqrt(Body.mu) *Body.R**2 * a**(-3.5)* (5*np.cos(i)**2-1)*(1-e**2)**(-2)*Cts.DSid*180/np.pi #deg/day. Rate of change of omega
#             Mdot = 3/4 * Cts.J2 *np.sqrt(Body.mu) *Body.R**2 * a**(-3.5)* (3*np.cos(i)**2-1)*(1-e**2)**(-1.5)*Cts.DSid*180/np.pi #deg/day. Rate of change of M
#             n = self.j/ self.k *(Ldot-RAANdot)-(omegadot+Mdot) #deg/day. mean motion.
#             a = (Body.mu/ (n*np.pi/(180*Cts.DSid))**2 )**(1/3) #km. New value for the semi-major axis
#         self.a = a
#         self.n = n*np.pi/(180*Cts.DSid)
#         return [self.a, self.n]
    
# #     def fsolve_i(self,i,param):
# #         a = self.a
# #         e = self.e
# #         [Body,n] = param

# #         Ldot = 360 #deg/day
# #         RAANdot = -3/2 * Cts.J2 *np.sqrt(Body.mu) *Body.R**2 * a**(-3.5)* np.cos(i)*(1-e**2)**(-2)  *180/np.pi* Cts.DSid
# #         omegadot = 3/4 * Cts.J2 *np.sqrt(Body.mu) *Body.R**2 * a**(-3.5)* (5*np.cos(i)**2-1)*(1-e**2)**(-2)*Cts.DSid*180/np.pi
# #         Mdot = 3/4 * Cts.J2 *np.sqrt(Body.mu) *Body.R**2 * a**(-3.5)* (3*np.cos(i)**2-1)*(1-e**2)**(-1.5)*Cts.DSid*180/np.pi
        
# #         F = self.j/self.k*(Ldot-RAANdot)-(omegadot+Mdot)-n
# #         return F
    
# #     def get_i(self, Body):
# #         self.n = Cts.DSid*180/np.pi*np.sqrt(Body.mu/self.a**3)

# #         i = spy.optimize.fsolve(self.fsolve_i,30*np.pi/180,[Body,self.n])
# #         self.i = i[0]*180/np.pi
# #         return [self.i]



class repeatingOrbit:
    """
    repeatingOrbit: contains the calculations for a repeating orbit
    """
    def __init__(self, Body, Cts, *args, **kwargs):
        """
        Constructor
        INPUTS:
            Body: main body about which the orbit is performed
            Cts: list of constants
        ADDITIONAL ARGUMENTS:
            j: Orbital revolutions necessary for repetition
            k: Earth days for 1 repiting orbit of the satellite
            a: semi-major axis of the repiting orbit
            e: eccentricity of the repiting orbit
            i: inclination angle of the repiting orbit
        """
        self.j = kwargs.get('j', 'No_j') 
        self.k = kwargs.get('k', 'No_k') 
        
        if self.j != 'No_j' and self.k != 'No_k':
            self.k_j = self.k / self.j
        else:
            self.k_j = kwargs.get('k_j', 'No_k,No_j') 
        
        self.a = kwargs.get('a', 'No_a') 
        self.e = kwargs.get('e', 'No_e') 
        self.i = kwargs.get('i', 'No_i') 
        
        self.Body = Body
        self.Cts = Cts
        
    def fsolve_Simple(self, x, param):
        """
        fsolve_Simple: solve the system of equations to obtain a, e or i. 
        Equations taken from J. R. Wertz, Mission geometry: orbit and 
        constellation design and management: spacecraft  orbit  and  attitude 
        systems.
        INPUTS:
            x: a, e or i for the initial guess for the solver
            param:
                Body: main body about which the orbit is performed
                j: Orbital revolutions necessary for repetition
                k: Earth days for 1 repiting orbit of the satellite
                name: 'a','e' or 'i' specifying what is the x given
        OUTPUTS:
            F: function to solve
        """
        i = self.i
        e = self.e
        a = self.a
        [Body,Cts,name] = param
        
        #Select the variable used as input
        if self.k_j != 'No_k,No_j':
            if name == 'a':
                a = x
            elif name == 'i':
                i = x
            elif name == 'e':
                e = x
            elif name == 'k_j':
                k_j = x
            else:
                print('Warning: not enough values')
        else:
            print('Warning: not enough values')
        
        T = 2*np.pi*np.sqrt(a**3/Body.mu) #Period of the satellite orbit
        deltaL1 = -2 *np.pi * T / Body.T #rad/rev. Difference in longitude 1
        deltaL2 = -3 * np.pi *Cts.J2 * Body.R**2 * np.cos(i) / \
            (a**2 * (1-e**2)**2) #rad/rev. Difference in longitude 2
        
        F = self.k_j*2*np.pi - abs(deltaL1 + deltaL2) #Function to solve
        return F
    
    def get_k_j_Simple(self, x0):
        """
        get_k_j_Simple: method to solve 'a' given the other two taking into 
        consideration
        only the effect of J2 on the right ascension of the ascending node.
        INPUTS: 
            x0: initial guess to input for the semi-major axis
        OUTPUTS:
            solution for a
        """
        return spy.optimize.fsolve(self.fsolve_Simple,x0,[self.Body,self.Cts,'k_j'])
    
    def get_a_Simple(self, x0):
        """
        get_a_Simple: method to solve 'a' given the other two taking into 
        consideration only the effect of J2 on the right ascension of the 
        ascending node.
        INPUTS: 
            x0: initial guess to input for the semi-major axis
        OUTPUTS:
            solution for a
        """
        self.a = spy.optimize.fsolve(self.fsolve_Simple,x0,[self.Body,self.Cts,'a'])[0]
        self.h = self.a - self.Body.R
        self.T = 2*np.pi*np.sqrt(self.a**3/self.Body.mu)
        return self.a
    
    def get_e_Simple(self, x0):
        """
        get_e_Simple: method to solve 'e' given the other two taking into 
        consideration only the effect of J2 on the right ascension of the 
        ascending node.
        INPUTS: 
            x0: initial guess to input for the eccentricity
        OUTPUTS:
            solution for e
        """
        self.e = spy.optimize.fsolve(self.fsolve_Simple, x0, \
            [self.Body,self.Cts,'e'])
        return self.e
    
    def get_i_Simple(self, x0):
        """
        get_i_Simple: method to solve 'i' given the other two taking into 
        consideration only the effect of J2 on the right ascension of the 
        ascending node.
        INPUTS: 
            x0: initial guess to input for the inclination angle
        OUTPUTS:
            solution for i
        """
        self.i = spy.optimize.fsolve(self.fsolve_Simple, x0, \
            [self.Body,self.Cts,'i'])
        return self.i
         
    def Table_Simple(self):
        """
        Table_Simple: create latex table with the necessary outputs. 
        For the simplified case. (Approach 2).
        INPUTS: 
            j: Orbital revolutions necessary for repetition
            k: Earth days for 1 repiting orbit of the satellite
            e: eccentricity of the repiting orbit
            i: inclination angle of the repiting orbit
        OUTPUTS:
            h: height of the orbit
            i: inclination angle
            T: period of the orbit
        """            
        print(r'h & i & T & k/j \\')
        print(r'%f & %f & %f & %f \\'%(self.h,self.i*180/np.pi,self.T,(self.k_j)))

        return self.a, self.h, self.T


# # Dual axis 



class dualAxis:
    def __init__(self, *args, **kwargs):
        """
        dual axis: class containing the necessary steps to solve a dual Axis spherical Triangle method.
        INPUTS:
            omega: angular velocities of points S and P.
            rho: angular distances of point S (1) and P (2). 
            psi: azimuth of point S (1) and P (2). 
            pis_0: initial aximuth of point S (1) and P (2). 
        """
        self.omega_1 = kwargs.get('omega_1', 0)
        self.omega_2 = kwargs.get('omega_2', 0)
        
        self.rho_1 = kwargs.get('rho_1', 0)
        self.rho_2 = kwargs.get('rho_2', 0)
        
        
        #Initial case and time given
        self.psi_1_0 = kwargs.get('psi_1_0', 0)
        self.psi_2_0 = kwargs.get('psi_2_0', 0)
        self.t = kwargs.get('t', 0)
        
        #Directly given
        self.psi_1 = kwargs.get('psi_1', self.psi_1_0+self.omega_1*self.t)
        self.psi_2 = kwargs.get('psi_2', self.psi_2_0+self.omega_2*self.t)
        
        while self.psi_1 >= 2*np.pi:
            self.psi_1 -= 2*np.pi
        while self.psi_2 >= 2*np.pi:
            self.psi_2 -= 2*np.pi
            
        if self.psi_1 < 0:
            self.psi_1 += 2*np.pi
        if self.psi_2 < 0:
            self.psi_2 += 2*np.pi
        
    def alpha(self): #slide 33   
        """
        alpha: function to obtain the angle alpha and delta alpha from the initial data
        INPUTS: 
        OUTPUTS:
        """
        rho1 = self.rho_1
        rho2 = self.rho_2
        psi2 = self.psi_2
        psi1 = self.psi_1    
                    
        delta_co = np.arccos( np.cos(rho1)*np.cos(rho2) + np.sin(rho1)*np.sin(rho2)*np.cos(psi2))
        self.delta = np.pi/2 - delta_co        
        
        
        acos = (np.cos(rho2) - np.cos(rho1)*np.sin(self.delta)) / (np.sin(rho1)*np.cos(self.delta))
        
        H = H_fun(psi2)
        self.delta_alpha = acos2(acos, -H)
                    
        self.alpha = psi1 + self.delta_alpha
        while self.alpha >= 2*np.pi:
            self.alpha -= 2*np.pi
        if self.alpha < 0:
            self.alpha += 2*np.pi
            
    def EulerAxis(self): #slide 31
        """
        EulerAxis: function to obtain the angle delta_E. the angular velocity and the angular distance of the Euler axis.
        INPUTS: 
        OUTPUTS:
        """
        om1 = self.omega_1
        om2 = self.omega_2
        rho1 = self.rho_1
        rho2 = self.rho_2

        self.delta_E_co = np.arctan( om2*np.sin(rho1) / (om1 + om2*np.cos(rho1)) )
        self.omega_E = om2*np.sin(rho1)/np.sin(self.delta_E_co)
#         print("check: %0.4f = %0.4f = %0.4f"% (om1 + om2, om2*np.sin(rho1)/np.sin(self.delta_E_co) , np.sqrt(om1**2+om2**2+2*om1*om2*np.cos(rho1))) )    
        self.rho_E = np.arccos( np.cos(self.delta_E_co) * np.sin(self.delta) +
                              np.sin(self.delta_E_co)*np.cos(self.delta)*np.cos(self.delta_alpha))
        
    def vel(self):
        """
        vel: function to obtain the velocity of point P and the angles PSI and delta PSI.
        INPUTS: 
        OUTPUTS:
        """
        self.v = self.omega_E*np.sin(self.rho_E)
        
        acos = (np.cos(self.delta_E_co) - np.cos(self.rho_E)*np.sin(self.delta)) / (np.sin(self.rho_E)*np.cos(self.delta))
        H = H_fun(self.delta_alpha)
        self.delta_PSI = acos2(acos,H)
        if self.delta_PSI <0:
            self.delta_PSI += 2*np.pi 
        self.PSI = self.delta_PSI -np.pi/2
        if self.PSI <0:
            self.PSI += 2*np.pi 
        
#     def latLon(self):    
        
        
    def print_fun(self):
        """
        print_fun: function to print the results.
        INPUTS: 
        OUTPUTS:
        """
        print('\hline')
        print( r'$\rho_1$ = %0.2f $^{\circ}$, $\rho_2$ = %0.2f $^{\circ}$ \\'%(AL_BF.rad2deg(self.rho_1),AL_BF.rad2deg(self.rho_2)))
        print( r'$\varphi_1$ = %0.2f $^{\circ}$, $\varphi_2$ = %0.2f $^{\circ}$ \\'%(AL_BF.rad2deg(self.psi_1),AL_BF.rad2deg(self.psi_2)))
        print( r'$\omega_1$ = %0.2f rad/s, $\omega_2$ = %0.2f rad/s \\'%(self.omega_1,self.omega_2))
        
        print('\hline')
        print( r'$\delta$ = %0.2f $^{\circ}$ \\'%AL_BF.rad2deg(self.delta))
        print(r'$\Delta \alpha$ = %0.2f $^{\circ}$ \\'%AL_BF.rad2deg(self.delta_alpha))
        print(r'$\alpha$ = %0.2f $^{\circ}$ \\'%AL_BF.rad2deg(self.alpha) )
        print( r"$\delta_E'$ = %0.2f $^{\circ}$ \\"%AL_BF.rad2deg(self.delta_E_co))
        print(r'$\rho_E$ = %0.2f $^{\circ}$ \\'%AL_BF.rad2deg(self.rho_E))
        print(r'$\omega_E$ = %0.2f rad/s \\'%self.omega_E )
        print(r'$v$ = %0.2f rad/s \\'%self.v )
        print(r'$\Delta \Psi$ = %0.2f $^{\circ}$ \\'%AL_BF.rad2deg(self.delta_PSI))
        print(r'$\Psi$ = %0.2f $^{\circ}$ \\'%AL_BF.rad2deg(self.PSI))
        
        print('\hline')

def H_fun(x):
    """
    H_fun: function to decide which angle to be chosen by the acos2.
    INPUTS: 
        x: angle 
    OUTPUTS:
        H: sign
    """    
    while x >= 2*np.pi:
        x -= 2*np.pi           
    if x>= 0 and x <= np.pi:
        H = 1
    else:
        H = -1
    return H
    
def acos2(x,H):
    """
    acos2: function to retrieve the correct angle from the acos.
    INPUTS: 
        x: angle 
        H: sign
    OUTPUTS:
        H: sign
    """  
    y = H*np.arccos(x)
    while y >= 2*np.pi:
        y -= 2*np.pi
    if y < 0:
        y += 2*np.pi
    return y

class satellite:
    """
    satellite: satellite object
    """
    def __init__(self, Body, *args, **kwargs):
        """
        Constructor
        INPUTS:
            Body: central body
        ADDITIONAL INPUTS:
            h: height of the satellite
            lat: latitude
            long: longitude
        """
        self.Body = Body
        self.h = kwargs.get('h', 0)
        self.lat = kwargs.get('lat', 0)
        self.long = kwargs.get('long', 0)
        
        self.TRIA = kwargs.get('TRIA', 0) #Dual axis triangle
        if self.TRIA != 0:
            self.TRIA.alpha()          
            self.long = AL_BF.ConvertRange(self.TRIA.alpha, -np.pi, np.pi)
            self.lat = AL_BF.ConvertRange(self.TRIA.delta, -np.pi/2, np.pi/2)

        self.a = self.h + self.Body.R
        self.T = 2*np.pi*np.sqrt(self.a**3/self.Body.mu)
        self.n = 2*np.pi / self.T
        
    def coverageMax(self, Eps_min): #Chapter 9 book
        """
        coverageMax: calculate maximum coverage
        INPUTS:
            Eps_min:
        """
        self.rho = np.arcsin(self.Body.R/(self.Body.R + self.h))
        self.eta_max = np.arcsin(np.sin(self.rho)*np.cos(Eps_min))
        self.lambda_max = np.pi/2 - Eps_min - self.eta_max
        self.D_max = self.Body.R * np.sin(self.lambda_max) / np.sin(self.eta_max)
        
    def coverage(self, target):
        """
        coverage: calculate coverage with target
        INPUTS:
            target: vector with lat and long of target    
        """
        delta_long = -(self.long - target[1])
        self.lambda_tgt = np.arccos( np.cos(np.pi/2 - target[0])*np.cos(np.pi/2-self.lat) +                                np.sin(np.pi/2-target[0])*np.sin(np.pi/2-self.lat)*np.cos(delta_long) )#eq 9.9
        self.phi_tgt = acos2(( np.cos(np.pi/2-target[0]) -np.cos(np.pi/2-self.lat)*np.cos(self.lambda_tgt) )/ ( np.sin(np.pi/2-self.lat)*np.sin(self.lambda_tgt) ), H_fun(delta_long) )

        self.eta = np.arctan(np.sin(self.rho)*np.sin(self.lambda_tgt)/(1-np.sin(self.rho)*np.cos(self.lambda_tgt)))
        self.epsilon = np.arccos(np.sin(self.eta)/np.sin(self.rho))
        self.D = self.Body.R * np.sin(self.lambda_tgt)/np.sin(self.eta)

        if self.lambda_tgt <= self.lambda_max:
            self.CheckCov = 1
        else:
            self.CheckCov = 0
        
    def plotFootprint(self, angle):
        """
        plotFootprint: plot the footprint of a satellite
        INPUTS:
            angle: vector with the angles around the satellite at
            which coverage wants to be plotted
        """
        lambda_0 = self.lambda_max
        
        point_cov = np.zeros([len(angle),2])
        for i in range(len(angle)):
            point_cov[i,0] = self.long + lambda_0*np.cos(angle[i])
            point_cov[i,1] = self.lat + lambda_0*np.sin(angle[i])

            if point_cov[i,1] <= -np.pi/2:
                point_cov[i,1] = -np.pi/2 - (point_cov[i,1] + np.pi/2)
                if point_cov[i,0] > 0:
                    point_cov[i,0] -= np.pi
                else:
                    point_cov[i,0] += np.pi

            elif point_cov[i,1] >= np.pi/2:
                point_cov[i,1] = np.pi/2 - (point_cov[i,1] - np.pi/2)
                if point_cov[i,0] > 0:
                    point_cov[i,0] -= np.pi
                else:
                    point_cov[i,0] += np.pi

            if point_cov[i,0] <= -np.pi:
                point_cov[i,0] = np.pi - (-np.pi - point_cov[i,0])
            elif point_cov[i,0] >= np.pi:
                point_cov[i,0] = -np.pi + (point_cov[i,0] - np.pi)

        return point_cov
    
# img = plt.imread("ConstellationCoverage.png")
# fig, ax = plt.subplots(figsize=(10,10))
# ax.imshow(img,)


def ConstellationDesign(sat):
    """
    ConstellationDesign: for satellites in the same orbit but with a
        phase angle (in time)
    """
    N_const = round(np.pi / sat.lambda_max)
    if (round(np.pi / sat.lambda_max) - np.pi / sat.lambda_max) < 0:
        N_const += 1
    S_const = 2*np.pi / N_const
    t_phase = S_const / sat.n
    return t_phase , int(N_const)


# # Lambert targetter
class LambertTrajectory: # Review
    """
    LambertTrajectory: 
    """
    def __init__(self, r0, rf, Body, *args, **kwargs):
        """
        INPUTS:
            r0: position vector of departure planet
            rf: position vector of arrival planet
            Body: central body
        ADDITIONAL ARGUMENTS:
            tflight: time of flight
            smAxis: semi-major axis
        """
        self.Body = Body
        self.r0 = r0
        self.rf = rf
        self.tf = kwargs.get('tflight', None)
        self.a = kwargs.get('smAxis', None)
        
        if self.tf != None:
            self.delta_theta = self.LambertDiffAngle()  
            self.Lambert()
#             alpha,beta = self.Lambert()
#             self.v0, self.vf = self.LambertTermVelVec()
        
    def LambertDiffAngle(self):
        """
        LambertDiffAngle: calculate the difference in angle of the points that are going to be used to create the orbit
                Assumes it is a prograde trajectori.
        Inputs: 
            r0: position of the first point
            rf: position of the second point
        Outputs: 
            diffAngle: angle between the points
        """
        r0 = self.r0
        rf = self.rf

        cosAngle0f = np.dot(r0,rf) / (np.linalg.norm(r0)*np.linalg.norm(rf)) #cosine of the angle between planets
        sinAngle0f = np.linalg.norm(np.cross(r0,rf)) / (np.linalg.norm(r0)*np.linalg.norm(rf)) #sine of the angle between planets

#         ref = np.array([1,0,0])
#         angle0 = np.dot(r0,ref) / (np.linalg.norm(r0)) #cosine of the angle between planets
#         anglef = np.dot(rf,ref) / (np.linalg.norm(rf)) #cosine of the angle between planets
        
        if sinAngle0f >= 0:
#         if (anglef-angle0) >= 0:
        # To decide the cuadrant of the difference in true anomaly
            diffAngle = np.arccos(cosAngle0f)
        else:
            diffAngle = 2*np.pi - np.arccos(cosAngle0f)
        
        return diffAngle
    
    def F_lamb(self, x, param):
        
        [T, q] = param
                
        E_lam = x**2 - 1 
        y = np.sqrt(abs(E_lam))
        z = np.sqrt(1 - q**2 + q**2 * x**2)
        
        f = y * (z - q*x)
        g = x*z - q*E_lam

        
        if E_lam < 0: #Elliptic orbit
            d = np.arctan2(f,g)
            
        elif E_lam > 0: #Hyperbolic orbit
            d = np.log(f + g)
            
        else:
            return None

    #         print(E_lam,T,q,z,d,y)
        F = T - 2 * (x - q*z - d/y) / (E_lam)
    
        return F 
    
#     def LambertAlphaBeta(self,s,c,tm,a):
#         """
#         LambertAlphaBeta: obtain the alpha and beta angles of the Lambert method.
#         Inputs:
#             s: semiperimeter of the space triangle
#             c: chord that joins the points
#             a: semi-major axis of the transfer orbit
#             nu: difference in true anomaly of the planets
#             t: time of tranfer
#             tm: time of minimium-energy transfer
#         Outputs:
#             alpha: angle for the Lambert algorithm
#             beta: angle for the Lambert algorithm
#         """
#         beta = 2*np.arcsin(np.sqrt((s-c)/(2*abs(a)))) 
#         if self.delta_theta > np.pi and self.delta_theta < (2*np.pi):  #Eliminate correction to go using the shorter side
#             beta = -beta

#         alpha = 2*np.arcsin(np.sqrt(s/(2*abs(a))))
# #         Do not correct because we want to go always in the same direction (counterclockwise)
#         if self.tf > tm:    # In case is to the direction where is longer
#             alpha = 2*np.pi-alpha

#         return alpha,beta

    def Lambert(self):
        
        r0_mag = np.linalg.norm(self.r0)
        rf_mag = np.linalg.norm(self.rf)
        
        self.c = np.linalg.norm(self.rf - self.r0)  #chord
        s = (r0_mag + rf_mag + self.c) / 2
        T = np.sqrt(8*self.Body.mu / s**3) * self.tf
        q = np.sqrt( r0_mag * rf_mag) / s * np.cos(self.delta_theta / 2)
        
        x0 = 0.0
        x = spy.optimize.fsolve(self.F_lamb, x0, [T, q])
        self.a = s / (2* (1 - x**2))
        
        am = s/2
        betam = 2*np.arcsin(np.sqrt((s-self.c) / (2*am)))
        tm = np.sqrt(s**3/8)*(np.pi - betam + np.sin(betam)) / np.sqrt(self.Body.mu)

        self.energy = -self.Body.mu/(2*abs(self.a))
#         alpha, beta = self.LambertAlphaBeta(2*am, c, tm, self.a)
        
#         return (alpha,beta)
    

#     def LambertTermVelVec(self,alpha,beta):
#         """
#         LambertTermVelVec: obtain the terminal velocity vectors of teh trajectory between the two points
#         Inputs: 
#             r1: position of first point
#             r2: position of second point
#             a: semi-major axis of the transfer orbit
#             alpha: Lambert's problem angle
#             beta: Lambert's problem angle
#         Outputs:
#             v1: velocity vector at position 1
#             v2: velocity vector at position 2
#         """
#         print('alphabeta', alpha, beta)
        
#         A = np.sqrt(self.Body.mu/(4*abs(self.a))) / np.tan(alpha/2)
#         print(A,alpha)
#         if beta == 0:
#             B = 0
#         else:
#             B = np.sqrt(self.Body.mu/(4*abs(self.a))) / np.tan(beta/2)

#         u1 = self.r0 /np.linalg.norm(self.r0) #unit vector in the direction of r1
#         u2 = self.rf /np.linalg.norm(self.rf) #unit vector in the direction of r2
#         uc = (self.rf - self.r0) / (np.linalg.norm(self.rf - self.r0)) #unit vector associated with the chord

#         print((B+A)*uc, (B-A)*u1)
#         v1 = (B+A)*uc + (B-A)*u1
#         v2 = (B+A)*uc + (A-B)*u2

#         return v1,v2   
    

    



# #Trying terminal velocity vectors
# class LambertTrajectory:
#     """
#     INPUTS:
#         r0: position vector of departure planet
#         rf: position vector of arrival planet
#     ARGS:
#         tf: time of flight
#     """
#     def __init__(self, r0, rf, Body, *args, **kwargs):
        
#         self.Body = Body
#         self.r0 = r0
#         self.rf = rf
#         self.tf = kwargs.get('tflight', None)
#         self.a = kwargs.get('smAxis', None)
        
#         if self.tf != None:
#             self.delta_theta = self.LambertDiffAngle()  
#             self.Lambert()
#             self.v0, self.vf = self.LambertTermVelVec()
        
#     def LambertDiffAngle(self):
#         """
#         LambertDiffAngle: calculate the difference in angle of the points that are going to be used to create the orbit
#                 Assumes it is a prograde trajectori.
#         Inputs: 
#             r0: position of the first point
#             rf: position of the second point
#         Outputs: 
#             diffAngle: angle between the points
#         """
#         r0 = self.r0
#         rf = self.rf

#         cosAngle0f = np.dot(r0,rf) / (np.linalg.norm(r0)*np.linalg.norm(rf)) #cosine of the angle between planets
#         sinAngle0f = np.linalg.norm(np.cross(r0,rf)) / (np.linalg.norm(r0)*np.linalg.norm(rf)) #sine of the angle between planets

#         ref = np.array([1,0,0])
#         angle0 = np.dot(r0,ref) / (np.linalg.norm(r0)) #cosine of the angle between planets
#         anglef = np.dot(rf,ref) / (np.linalg.norm(rf)) #cosine of the angle between planets
        
#         if (anglef-angle0) >= 0:
#         # To decide the cuadrant of the difference in true anomaly
#             diffAngle = np.arccos(cosAngle0f)
#         else:
#             diffAngle = 2*np.pi - np.arccos(cosAngle0f)
        
#         return diffAngle
    
#     def F_lamb(self, x, param):
        
#         [T, q] = param
                
#         E_lam = x**2 - 1 
#         y = np.sqrt(abs(E_lam))
#         z = np.sqrt(1 - q**2 + q**2 * x**2)
        
#         f = y * (z - q*x)
#         g = x*z - q*E_lam

        
#         if E_lam < 0: #Elliptic orbit
#             d = np.arctan2(f,g)
            
#         elif E_lam > 0: #Hyperbolic orbit
#             d = np.log(f + g)
            
#         else:
#             return None

#     #         print(E_lam,T,q,z,d,y)
#         F = T - 2 * (x - q*z - d/y) / (E_lam)
    
#         return F 
    
# #     def LambertAlphaBeta(self,s,c,tm,a):
# #         """
# #         LambertAlphaBeta: obtain the alpha and beta angles of the Lambert method.
# #         Inputs:
# #             s: semiperimeter of the space triangle
# #             c: chord that joins the points
# #             a: semi-major axis of the transfer orbit
# #             nu: difference in true anomaly of the planets
# #             t: time of tranfer
# #             tm: time of minimium-energy transfer
# #         Outputs:
# #             alpha: angle for the Lambert algorithm
# #             beta: angle for the Lambert algorithm
# #         """
# #         beta = 2*np.arcsin(np.sqrt((s-c)/(2*abs(a)))) 
# #         if self.delta_theta > np.pi and self.delta_theta < (2*np.pi):  #Eliminate correction to go using the shorter side
# #             beta = -beta

# #         alpha = 2*np.arcsin(np.sqrt(s/(2*abs(a))))
# # #         Do not correct because we want to go always in the same direction (counterclockwise)
# #         if self.tf > tm:    # In case is to the direction where is longer
# #             alpha = 2*np.pi-alpha

# #         return alpha,beta

#     def Lambert(self):
        
#         r0_mag = np.linalg.norm(self.r0)
#         rf_mag = np.linalg.norm(self.rf)
        
#         self.c = np.linalg.norm(self.rf - self.r0)  #chord
#         s = (r0_mag + rf_mag + self.c) / 2
#         T = np.sqrt(8*self.Body.mu / s**3) * self.tf
#         q = np.sqrt( r0_mag * rf_mag) / s * np.cos(self.delta_theta / 2)
        
#         x0 = 0.0
#         x = spy.optimize.fsolve(self.F_lamb, x0, [T, q])
#         self.a = s / (2* (1 - x**2))
        
#         am = s/2
#         betam = 2*np.arcsin(np.sqrt((s-self.c) / (2*am)))
#         tm = np.sqrt(s**3/8)*(np.pi - betam + np.sin(betam)) / np.sqrt(self.Body.mu)

#         self.energy = -self.Body.mu/(2*abs(self.a))
# #         alpha, beta = self.LambertAlphaBeta(2*am, c, tm, self.a)
        
# #         return (alpha,beta)
    

# #     def LambertTermVelVec(self,alpha,beta):
# #         """
# #         LambertTermVelVec: obtain the terminal velocity vectors of teh trajectory between the two points
# #         Inputs: 
# #             r1: position of first point
# #             r2: position of second point
# #             a: semi-major axis of the transfer orbit
# #             alpha: Lambert's problem angle
# #             beta: Lambert's problem angle
# #         Outputs:
# #             v1: velocity vector at position 1
# #             v2: velocity vector at position 2
# #         """
# #         print('alphabeta', alpha, beta)
        
# #         A = np.sqrt(self.Body.mu/(4*abs(self.a))) / np.tan(alpha/2)
# #         print(A,alpha)
# #         if beta == 0:
# #             B = 0
# #         else:
# #             B = np.sqrt(self.Body.mu/(4*abs(self.a))) / np.tan(beta/2)

# #         u1 = self.r0 /np.linalg.norm(self.r0) #unit vector in the direction of r1
# #         u2 = self.rf /np.linalg.norm(self.rf) #unit vector in the direction of r2
# #         uc = (self.rf - self.r0) / (np.linalg.norm(self.rf - self.r0)) #unit vector associated with the chord

# #         print((B+A)*uc, (B-A)*u1)
# #         v1 = (B+A)*uc + (B-A)*u1
# #         v2 = (B+A)*uc + (A-B)*u2

# #         return v1,v2   
    
#     def eSolver(self,e):
        
#         r0 = self.r0
#         rf = self.rf
        
#         ref = np.array([1,0,0])
#         angle0 = np.arccos( np.dot(r0,ref) / (np.linalg.norm(r0))) #cosine of the angle between planets
#         anglef = np.arccos( np.dot(rf,ref) / (np.linalg.norm(rf))) #cosine of the angle between planets
        
#         F = self.tf - np.sqrt(self.a**3 / self.Body.mu)* (2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(anglef/2)) - e*np.sqrt(1-e**2)* np.sin(anglef)/(1+e*np.cos(anglef)) - (2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(angle0/2)) - e*np.sqrt(1-e**2)* np.sin(angle0)/(1+e*np.cos(angle0))) )
#         print(e,F)
#         return F
    
#     def LambertTermVelVec(self):
        
#         self.e = spy.optimize.fsolve(self.eSolver, 0.9)
#         print('e',self.e)
        
#         p = self.a * (1 - self.e**2)
#         r0_mag = np.linalg.norm(self.r0)
#         rf_mag = np.linalg.norm(self.rf)
        
#         vc = self.c *np.sqrt(self.Body.mu * p) / (r0_mag * rf_mag* np.sin(self.delta_theta))
#         vp = np.sqrt(self.Body.mu / p ) *(1- np.cos(self.delta_theta)) / np.sin(self.delta_theta)

#         u1 = self.r0 /r0_mag #unit vector in the direction of r1
#         u2 = self.rf /rf_mag #unit vector in the direction of r2
#         uc = (self.rf - self.r0) / self.c #unit vector associated with the chord


#         v0 = vc*uc + vp*u1
#         vf = vc*uc - vp*u2
        
#         return v0,vf


# In[7]:


# class LambertTrajectory:
#     """
#     INPUTS:
#         r0: position vector of departure planet
#         rf: position vector of arrival planet
#     ARGS:
#         tf: time of flight
#     """
#     def __init__(self, r0, rf, Body, *args, **kwargs):
        
#         self.Body = Body
#         self.r0 = r0
#         self.rf = rf
#         self.tf = kwargs.get('tflight', '0')
#         self.a = kwargs.get('smAxis', '0')
        
#         #######################################
#         ## 1. Given the time of flight ########
#         #######################################
#         if self.tf != 0:
#             self.delta_theta = self.LambertDiffAngle()  
#             print('diff angle', self.delta_theta*180/np.pi)
#             alphaLamb,betaLamb = self.Lambert()
#             self.v0, self.vf = self.LambertTermVelVec(alphaLamb,betaLamb)

#     def LambertDiffAngle(self):
#         """
#         LambertDiffAngle: calculate the difference in angle of the points that are going to be used to create the orbit
#                 Assumes it is a prograde trajectori.
#         Inputs: 
#             r0: position of the first point
#             rf: position of the second point
#         Outputs: 
#             diffAngle: angle between the points
#         """
#         r0 = self.r0
#         rf = self.rf

#         cosAngle0f = np.dot(r0,rf) / (np.linalg.norm(r0)*np.linalg.norm(rf)) #cosine of the angle between planets
#         sinAngle0f = np.linalg.norm(np.cross(r0,rf)) / (np.linalg.norm(r0)*np.linalg.norm(rf)) #sine of the angle between planets

#         ref = np.array([1,0,0])
#         angle0 = np.dot(r0,ref) / (np.linalg.norm(r0)) #cosine of the angle between planets
#         anglef = np.dot(rf,ref) / (np.linalg.norm(rf)) #cosine of the angle between planets
        
#         if (anglef-angle0) >= 0:
#         # To decide the cuadrant of the difference in true anomaly
#             diffAngle = np.arccos(cosAngle0f)
#         else:
#             diffAngle = 2*np.pi - np.arccos(cosAngle0f)
        
#         return diffAngle
    

#     def LambertAlphaBeta(self,s,c,tm,a):
#         """
#         LambertAlphaBeta: obtain the alpha and beta angles of the Lambert method.
#         Inputs:
#             s: semiperimeter of the space triangle
#             c: chord that joins the points
#             a: semi-major axis of the transfer orbit
#             nu: difference in true anomaly of the planets
#             t: time of tranfer
#             tm: time of minimium-energy transfer
#         Outputs:
#             alpha: angle for the Lambert algorithm
#             beta: angle for the Lambert algorithm
#         """
#         beta = 2*np.arcsin(np.sqrt((s-c)/(2*a))) 
#         if self.delta_theta > np.pi and self.delta_theta < (2*np.pi):  #Eliminate correction to go using the shorter side
#             beta = -beta

#         alpha = 2*np.arcsin(np.sqrt(s/(2*a)))
#         # Do not correct because we want to go always in the same direction (counterclockwise)
#         if self.tf > tm:    # In case is to the direction where is longer
#             alpha = 2*np.pi-alpha

#         return alpha,beta

#     def LambertKepler(self,a,x):
#         """
#         LambertKepler: solve Kepler's equation for the case of the Lambert's problem'
#         Inputs:
#             a: initial guess of semi-major axis
#             x: additional arguments [r1,r2,c,am,nu,t,tm]
#                 r1: position of first point
#                 r2: position of second point
#                 c: chord that joins the two positions
#                 am: semi-major axis of the minimum-energy transfer
#                 nu: difference in true anomaly between the planets
#                 t: time of flight (s)
#                 tm: time of minimium-energy transfer
#             Outputs:
#                 Func: equation to solve
#         """
#         [c,am,s,tm] = x
        
#         alpha, beta = self.LambertAlphaBeta(s,c,tm, a)

#         Func = -(alpha-beta-np.sin(alpha)+np.sin(beta)) + np.sqrt( self.Body.mu / a**3)* self.tf # Lambert's equation:
        
#         return Func

#     def Lambert(self): # Solve Lambert's problem
#         """
#         Lambert: solve Lambert's problem
#         Inputs:
#             nu: difference in true anomaly between the planets
#         Outputs:
#             a: semi-major axis of the transfer orbit
#             alpha: Lambert's problem angle
#             beta: Lambert's problem angle
#         """
#         r0 = self.r0
#         rf = self.rf
        
#         mr1 = np.linalg.norm(r0) # Magnitude of position vectors
#         mr2 = np.linalg.norm(rf)  

#         c = np.linalg.norm(rf-r0)  #chord

#         am = (c + mr1 + mr2) /4 # semi-major axis for minimum energy
#         s = 2*am
#         betam = 2*np.arcsin(np.sqrt( (s-c) / (2*am)) )
#         tm = np.sqrt(s**3/8)*(np.pi - betam + np.sin(betam)) / np.sqrt(self.Body.mu)

#         print('tm', tm/(3600*24),am/1.496e11)
        
#         #Obtain a
#         a0 = am
#         self.a = spy.optimize.fsolve(self.LambertKepler,a0,[c, am, s,tm])
#         alpha, beta = self.LambertAlphaBeta(2*am, c, tm, self.a)
        
#         return (alpha,beta)
    
#     def LambertTermVelVec(self,alpha,beta):
#         """
#         LambertTermVelVec: obtain the terminal velocity vectors of the trajectory between the two points
#         Inputs: 
#             r1: position of first point
#             r2: position of second point
#             a: semi-major axis of the transfer orbit
#             alpha: Lambert's problem angle
#             beta: Lambert's problem angle
#         Outputs:
#             v1: velocity vector at position 1
#             v2: velocity vector at position 2
#         """
#         A = np.sqrt(self.Body.mu/(4*self.a)) / np.tan(alpha/2)
#         if beta == 0:
#             B = 0
#         else:
#             B = np.sqrt(self.Body.mu/(4*self.a)) / np.tan(beta/2)

#         u1 = self.r0 /np.linalg.norm(self.r0) #unit vector in the direction of r1
#         u2 = self.rf /np.linalg.norm(self.rf) #unit vector in the direction of r2
#         uc = (self.rf - self.r0) / (np.linalg.norm(self.rf - self.r0)) #unit vector associated with the chord

#         v1 = (B+A)*uc + (B-A)*u1
#         v2 = (B+A)*uc + (A-B)*u2

#         return v1,v2


################################################################################
# Pork-chop plots
################################################################################
def porkchop3(r, Body, tf, t0, nameFile, orderTransfer):
    """
    porkchop3
    INPUTS:
        r:
        Body: central body
        tf: time of flight
        t0: launch date
        nameFile: name of the file to save the picture
        orderTransfer
    """
    AL_BF.writeData(orderTransfer,'w',nameFile)
    Energy_total = np.zeros([len(tf[:,0]), len(t0)])
    t_total = np.zeros(len(tf[:,0]))

    for x in range(len(t0)): #For each t0
        
        for i in range(len(tf[:,0])):
            Energy = 0

            for j in range(len(r) - 1): #For each leg
                t_f = tf[i,j]

                r1 = r[j]
                r2 = r[j+1]
                ri = r1[t0[x], :]
                rf = r2[t0[x], :]

                Lambert = LambertTrajectory(ri, rf, Body, tflight = t_f * 24 * 3600 )
                Energy  +=  Lambert.energy
            
            Energy_total[i,x] = Energy
        
            t_total[i] = np.sum(tf[i,:])
            
            data = np.concatenate(([Energy_total[i,x]],tf[i,:]))
            data = np.append(data, [t0[x], t_total[i]])
            AL_BF.writeData(data,'a+', nameFile)
    
    X,Y = np.meshgrid(t0,t_total)
    return X,Y,Energy_total

def randomTransferTimes(numberPoints, r, tfi_interval, tfo_interval):
    """
    randomTransferTimes: choose random transfer times
    INPUTS:
        numberPoints:
        r: 
        tfi_interval:
        tfo_interval:
    """
    tf = np.zeros([numberPoints,len(r)])
    for j in range(len(r)-2):
        tf[:,j] = np.random.randint(tfi_interval[0],tfi_interval[-1], size = numberPoints)
    tf[:,len(r)-2] = np.random.randint(tfo_interval[0],tfo_interval[-1], size = numberPoints)
  
    #Add the total time to the last column
    for x in range(numberPoints):
        tf[x,-1] = sum(tf[x,:])
    
    #Sort by total time
    tf = tf[tf[:,-1].argsort()]
    return tf

def randomFlybies(planets):
    """
    randomFlybies: create a random set of flybys
    INPUTS:
        planets: sequence of planets to be mixed
    OUTPUTS:
        r: vector of planets
        x: number of flybys
    """
    numberFlybies = np.random.randint(0,4)

    r = [r_E]
    x = np.zeros(numberFlybies)
    for i in range(numberFlybies):
        x[i] = np.random.randint(0,3)
        r += [planets[int(x[i])]]
    r += [r_J]

    return r, x

def allPossibleFybies(planets, maxFlybies):
    """
    allPossibleFlybies: mix all possible flybys
    INPUTS:
        planets: [r_E,r_V,r_M]
        maxFlybies: maximum number of flybys
    OUTPUTS:
        r:
        x: order of flybies
    """
    numberCombinations = sum(len(planets)**i for i in range(4)) #Number possibilities
    
    for j in range(numberCombinations):
        
        r = [r_E] #Starting at Earth
        x = np.zeros(numberFlybies)
        for i in range(numberFlybies):
            x[i] = np.random.randint(0,3)
            r += [planets[int(x[i])]]
        r += [r_J]
        
    return r, x

