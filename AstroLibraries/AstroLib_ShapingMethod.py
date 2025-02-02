import matplotlib.pyplot as plt
import numpy as np
import scipy as spy

import time

from . import AstroLib_Basic as AL_BF
from . import AstroLib_2BP as AL_2BP
from . import AstroLib_Ephem as AL_Eph

Cts = AL_BF.ConstantsBook() #Table of constants

class shapingMethod:
    def __init__(self, mu, *args, **kwargs):
        # https://www.esa.int/gsp/ACT/doc/ARI/ARI%20Study%20Report/ACT-RPT-MAD-ARI-05-4106-Spiral%20Trajectories%20in%20Global%20Optimisation.pdf
        # https://arc.aiaa.org/doi/pdf/10.2514/1.21796?casa_token=NH4awyVRgR4AAAAA:sYa1RlUomlRBOCrW9HBg6xRIqoZlEGsBKay-KASEmV5jrwWBEMTwSrzSTBjL5oc_odkqlVde
        # Initial and final radious are fixed
        self.mu = mu 

    def getGamma(self, k2, r1, r2, theta, num):
        """
        getGamma: obtain minimum and maximum value of gamma and create a vector
        """
        delta = 2*(1-np.cos(k2*theta))/k2**4- np.log(r1/r2)**2

        gamma1_min = np.arctan( k2/2 * ( -np.log(r1/r2) / np.tan(k2*theta/2) - np.sqrt(delta)))
        gamma1_max = np.arctan( k2/2 * ( -np.log(r1/r2) / np.tan(k2*theta/2) + np.sqrt(delta)))

        if (gamma1_max-gamma1_min) > 0: # valid interval
            h = num*(gamma1_max-gamma1_min) #Interval
            gamma1 = np.arange(gamma1_min, gamma1_max +h, h)
        else:
            gamma1 = np.array([gamma1_max])

        return gamma1

    def simpson(self, f,a,b,n, args):
        """
        simpson: solve an integral approximately using Simpson's 1/3 composite method
        """
        dx = (b-a) / n
        x = np.linspace(a,b,n+1)
        y ,args = f(x, args)
        S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])

        return S, args

    def f(self, theta, args):
        """
        f: function with the integral to be solved by Simpson's method
        """
        k0,k1, k2, phi = args
        
        q = 0
        mu = Cts.mu_S / Cts.AU**3
        
        s = np.sin(k2*theta + phi)
        c = np.cos(k2*theta + phi)

        r = k0 * np.exp(q*theta + k1*s)
        gamma = np.arctan(q + k1*k2*c)
   
        fsol = np.sqrt(r**3 * (np.tan(gamma)**2 + k1*k2**2*s +1 ) / mu)

        args = [theta, k1*s, s, c, gamma, r*Cts.AU, fsol**2]
    
        return fsol, args

    def S(self, k2, r1, r2, psi, N, gamma1, steps):
        """
        S: exposins function to calculate the main parameters
        """
        theta_b = psi + 2*np.pi*N
 
        k1 = np.sqrt( ( (np.log(r1/r2) + np.tan(gamma1)/k2*np.sin(k2*theta_b) ) / \
               (1-np.cos(k2*theta_b)) )**2 + np.tan(gamma1)**2/k2**2 )

        sign_k1 = np.sign( np.log(r1/r2) + np.tan(gamma1)/k2*np.sin(k2*theta_b) )
        k1 *= sign_k1

        phi = np.arccos(np.tan(gamma1) / (k1*k2))
        k0 = r1 / np.exp(k1*np.sin(phi))

        args0 = [k0,k1, k2, phi]
        TOF, args = self.simpson(self.f, 0, theta_b, steps, args0) #call simpson's method

        theta, k1_s, s, c, gamma, r, fsol = args
        param = [k1, phi, k0, theta, k1_s, s, c, gamma, r, fsol]

        return TOF, param

    def calculateExposin(self, N, gamma1,  r1, r2, psi):
        steps = 300
        TOF, param = self.S(self.k2,r1, r2, psi, N, gamma1, steps)
        [k1, phi, k0, theta, k1_s, s, c, gamma, r, fsol] = param

        self.N = N
        self.k0 = k0
        self.k1 = k1
        self.phi = phi
        self.TOF = TOF

        self.gamma1 = gamma1

    def gammaOpt(self, gamma,TOF, lim):
        """
        gammaOpt: calculate the intersection with the reference point
        """
        index = np.where(abs(TOF-lim) == min(abs(TOF-lim)) )
        if abs(TOF[index]  - lim)> 10 or len(index[0]) == 0:
            return 0,0
        else:
            return gamma[index], TOF[index]

    # def calculatePower(self):

    def terminalVel(self, r1, r2, psi):

        theta_b = psi + 2 * np.pi * self.N

        # gamma 1 is known. gamma2:
        # gamma2 = np.arctan( -np.tan(gamma1) - k2*np.log(r1/r2) / np.tan(k2*theta_b/2)  )
        self.gamma2 = np.arctan(self.k1*self.k2*np.cos(self.k2*theta_b+self.phi))

        # magnitude of the velocities
        theta_dot_1 = np.sqrt( self.mu/r1**3 / \
                    (np.tan(self.gamma1)**2 + self.k1*self.k2**2*np.sin(self.phi) + 1) )
        r1_dot = r1*theta_dot_1* ( self.k1*self.k2*np.cos(self.phi))
        v1 = theta_dot_1/ np.cos(self.gamma1)

        theta_dot_1 *= r1
        # r1_dot= v1*np.sin(gamma1)

        # print("Must be equal",v1, r1_dot/np.sin(gamma1))

        theta_dot_2 = np.sqrt( self.mu/r2**3 / \
                    (np.tan(self.gamma2)**2 + self.k1*self.k2**2*np.sin(self.k2*theta_b+self.phi) + 1) )
        r2_dot = r2*theta_dot_2* ( self.k1*self.k2*np.cos(self.k2*theta_b+self.phi))
        v2 = theta_dot_2/ np.cos(self.gamma2)

        theta_dot_2 *= r2 # polar coordinates tangential velocity
        # r2_dot = v2*np.sin(gamma2)

    # return [v1, gamma1], [v2, gamma2], a_T
        return [r1_dot, theta_dot_1], [r2_dot, theta_dot_2]

    def calculateThrustProfile(self, N, r1, r2):
        
        steps = 300

        # THRUST PROFILE: thrust changes with gamma and theta
        theta = np.linspace(0, 2*np.pi*(self.N+1), 2000)
        r = np.zeros(len(theta))
        gamma = np.zeros(len(theta))
        a = np.zeros(len(theta))
        t = np.zeros(len(theta))
        
        for i in range(len(theta)):
            r[i] = self.k0 * np.exp( self.k1*np.sin(self.k2*(theta[i]) +self.phi ) )
            gamma[i] = np.arctan(self.k1*self.k2*np.cos(self.k2*theta[i]+self.phi))
            
            s = np.sin(self.k2*(theta[i]+self.phi))
            a[i] = self.mu/r[i]**2 * np.tan(gamma[i])/(2*np.cos(gamma[i])) *\
            ( 1/(np.tan( gamma[i])**2 + self.k1*self.k2**2*s + 1) -\
            (self.k2**2*(1-2*self.k1*s))/(np.tan(gamma[i])**2+self.k1*self.k2**2*s+1)**2 )

            args0 = [self.k0, self.k1, self.k2, self.phi]
            t[i], args = self.simpson(self.f, 0, theta[i], steps, args0) #call simpson's method

        # plt.plot(AL_BF.rad2deg(theta), a,'o-')
        # plt.grid(alpha = 0.5)
        # plt.show()



        self.t_f = np.polyfit(t, a, 11) # acceleration as a function of time
        plt.plot(t, a,'o-')

        a_i = np.zeros(len(t))
        for i in range(len(t)):
            a_i[i] = self.accelerationAtTime(t[i])

        plt.plot(t, a_i, color = 'red')

        plt.grid(alpha = 0.5)
        plt.show()


        return t, a

    def accelerationAtTime(self, t):
        a_i = 0
        for exp in range(len(self.t_f)):
            a_i += self.t_f[exp] * t**(len(self.t_f)-1-exp)

        return a_i

        

    def plot_sphere(self, r1, r2, psi):
        steps = 300

        theta = np.linspace(0, 2*np.pi*(self.N+1), 2000)
        r = np.zeros((len(theta)))
        x = np.zeros((len(theta)))
        y = np.zeros((len(theta)))

        for i in range(len(theta)):
            r[i] = self.k0 * np.exp( self.k1*np.sin(self.k2*(theta[i]) + self.phi ) )
            
            x[i] = r[i]*np.cos(theta[i])
            y[i] = r[i]*np.sin(theta[i])

        # x = np.arange(0,len(theta),1)
        # plt.plot(x, r)
        plt.scatter(r1, 0, marker = 'o')
        plt.scatter(r2*np.cos(psi), r2*np.sin(psi),marker = 'o')
        plt.plot(x,y)
        
        plt.xlim((-2*r2,2*r2))
        plt.ylim((-2*r2, 2*r2))
        plt.grid(alpha = 0.5)
        plt.axis('equal')
        
        plt.show()

    def calculategamma1(self, r1, r2, psi, t_t, k2, plot = False):
        N = np.arange(0,4,1)
        lim = t_t # in days

        self.k2 = k2

        steps = 300
        num = 1/1000  

        if plot == True:
            labelSize = 15
            fig, ax = plt.subplots(1,1, figsize = (9,6))
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
            hspace=0.40,wspace=0.25)

        # Save gamma and TOF 
        self.gammaOptim_v = np.zeros(len(N))
        self.TOFopt_v = np.zeros(len(N))
        for i in range(len(N)):
            theta = psi + 2 * np.pi * N[i]
            gamma1 = self.getGamma(self.k2, r1, r2, theta, num)

            TOF = np.zeros(len(gamma1))
            for j in range(len(gamma1)):
                TOF[j], param = self.S(self.k2,r1, r2, psi, N[i], gamma1[j], steps)
                TOF[j] = AL_BF.sec2days(TOF[j])
            
            self.gammaOptim_v[i], self.TOFopt_v[i] = self.gammaOpt(gamma1, TOF, lim)

            if plot == True:
                plt.plot(gamma1, TOF,\
                    label = r'N = %i, $\gamma_m$=%0.4f rad, $\gamma_M$=%0.4f rad '%(N[i],gamma1[0],gamma1[-1]))
                if self.gammaOptim_v[i] == 0:
                    plt.scatter(self.gammaOptim_v[i], self.TOFopt_v[i] ,label = 'No solution')
                else:
                    plt.scatter(self.gammaOptim_v[i], self.TOFopt_v[i], s= 100,\
                        label = '$\gamma_1$ = %0.4f rad, TOF = %0.4f days'%\
                        (self.gammaOptim_v[i],self.TOFopt_v[i]))

        if plot == True:
            xAxis = np.linspace(gamma1[0], gamma1[-1],num= 100)
            yAxis = np.ones(len(xAxis))*lim
            plt.plot(xAxis, yAxis,'r-')

            plt.xlabel('$\gamma_1$',size = labelSize)
            plt.ylabel('TOF (days)',size = labelSize)
            plt.ylim([0,2*lim])
            plt.tick_params(axis ='both', which='major', labelsize = labelSize)
            plt.tick_params(axis ='both', which='minor', labelsize = labelSize)
            plt.legend()
            plt.grid(alpha=0.5)
            plt.show()

        return self.gammaOptim_v

if __name__ == "__main__":

    # Verify with example
    print("VERIFICATION")
    r1 = 1
    r2 =  5
    psi = np.pi/2
    k2 = 1/4

    eSin = AL_Sh.shapingMethod(sun.mu / Cts.AU_m**3)
    gammaOptim_v = eSin.start(r1, r2, psi, 365*1.5, k2) # verified
    Ni = 1
    print(gammaOptim_v)
    eSin.plot_sphere(r1, r2, psi, gammaOptim_v[Ni], Ni) # verified

    # Real life case
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

    # Calculate trajectory of the bodies based on ephem
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')
    
    date0 = np.array([27,1,2018,0])
    t0 = AL_Eph.DateConv(date0,'calendar') #To JD
    t_t = 350

    r_E, v_E = earthephem.eph( t0.JD_0 )
    r_M, v_M = marsephem.eph(t0.JD_0+ t_t )

    r_1 = r_E 
    r_2 = r_M 
    r_1_norm = np.linalg.norm( r_1 )
    r_2_norm = np.linalg.norm( r_2 )
    
    dot = np.dot(r_1[0:2], r_2[0:2])      # dot product between [x1, y1] and [x2, y2]
    det = r_1[0]*r_2[1] - r_2[0]*r_1[1]     # determinant
    psi = np.arctan2(det, dot) 
    psi = AL_BF.convertRange(psi, 'rad', 0 ,2*np.pi)
    
    k2 = 1/12

    eSin = AL_Sh.shapingMethod(sun.mu / AL_BF.AU**3)
    gammaOptim_v = eSin.calculategamma1(r_1_norm / AL_BF.AU, r_2_norm / AL_BF.AU, psi, \
        t_t, k2, plot = False)
    
    Ni = 1
    eSin.calculateExposin(Ni, gammaOptim_v[Ni],r_1_norm / AL_BF.AU, r_2_norm / AL_BF.AU, psi )
    eSin.plot_sphere(r_1_norm / AL_BF.AU, r_2_norm / AL_BF.AU, psi)
    v1, v2 = eSin.terminalVel(r_1_norm / AL_BF.AU, r_2_norm / AL_BF.AU, psi)
    t, a_T = eSin.calculateThrustProfile(r_1_norm / AL_BF.AU, r_2_norm / AL_BF.AU, psi)

    # To heliocentric coordinates (2d approximation)
    def to_helioc(r, vx, vy):
        v_body = np.array([vx, vy, 0])
        # Convert to heliocentric
        angle = np.arctan2(r[1], r[0])
        v_h = AL_BF.rot_matrix(v_body, angle, 'z')

        return v_h

    v_1 = to_helioc(r_1, v1[0]*AL_BF.AU, v1[1]*AL_BF.AU)
    v_2 = to_helioc(r_2, v2[0]*AL_BF.AU, v2[1]*AL_BF.AU) 

    print("Helioc vel", v_1, v_2)
    print("with respect to body ", v_1-v_E, v_2-v_M)

    v_1_E = np.linalg.norm(v_1-v_E)
    v_2_M = np.linalg.norm(v_2-v_M)

    print("Final vel", v_1_E, v_2_M)
    
    ################################################3
    # Propagate with terminal velocity vectors
    SF = CONFIG.SimsFlan_config()    
    Fit = Fitness(Nimp = SF.Nimp)

    decv = np.zeros(len(SF.bnds))
    decv[6] = t0.JD_0
    decv[7] = AL_BF.days2sec(t_t)
    decv[0:3] = AL_BF.convert3dvector(v_1-v_E,'cartesian')
    decv[3:6] = AL_BF.convert3dvector(v_2- v_M,'cartesian')

    print('decv', decv[0:9])
    
    for i in range(SF.Nimp):
        # Acceleration on segment is average of extreme accelerations
        t_i = AL_BF.days2sec(t_t) / (SF.Nimp+1) *i
        t_i1 = AL_BF.days2sec(t_t) / (SF.Nimp+1) *(i+1)

        #find acceleration at a certain time
        a_i = eSin.accelerationAtTime(t_i)
        a_i1 = eSin.accelerationAtTime(t_i1)
        
        a = (a_i+a_i1)/2 # find the acceleration at a certain time
        print("a", a_i, a_i1, a)
        deltav_i = AL_BF.days2sec(t_t) / (SF.Nimp+1) * a*AL_BF.AU
        print(deltav_i)
        decv[8+3*i] = deltav_i /100
        decv[8+3*i+1] = 0
        decv[8+3*i+2] = 0

    Fit.calculateFeasibility(decv, plot = True, thrust = 'tangential')
    Fit.printResult()
 