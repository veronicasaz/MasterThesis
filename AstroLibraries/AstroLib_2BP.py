import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

from . import AstroLib_Basic as AL_BF

# import ipynb.fs.full.AstroLibrary_BasicFunctions as AL_BF #Use other ipynb documents

################################################################################
# CREATE BODY WITH ITS CHARACTERISTICS
################################################################################
class Spacecraft:
    def __init__(self, *args, **kwargs):
        # Propulsive parameters
        self.Isp = kwargs.get('Isp', 3100)
        self.m_dry = kwargs.get('m_dry', 747)
        self.T = kwargs.get('T', 90e-3)
        self.g0 = kwargs.get('g0', AL_BF.g0)

    def MassChange(self, m, deltav): 
        """
        MassChange: change (decrease) mass of the spacecraft when an impulse is applied
        Inputs:
            m: current mass of the spacecraft (fuel+drymass)
            deltav: magnitude of the impulse applied
        Outputs:
            mf: final mass
        """
        mf = m * np.exp( -deltav/(self.g0 * self.Isp) )
        return mf

    def MassChangeInverse(self, m, deltav):  
        """
        MassChangeInverse: change (increase) the mass to obtaine the mass of the spacecraft before
                an impulse was applied
        Inputs: 
            m: current mass of the spacecraft (fuel+drymass)
            deltav: magnitude of the impulse applied
        Outputs:
            mf: final mass
        """
        mi = m/np.exp( -deltav /(self.g0 * self.Isp) )
        return mi

class Body:
    """
    Body: contains the main characteristics of a celestial body.
        name: name of the body.
        mass: mass of the body
        radius: radius of the body
        color: color to plot the body
        
        Additional parameters:
            mu: gravitational parameter of the body. If it is not given, it is calculated as G*Mass
    """
    def __init__(self, name, color, Mass = 0, radius = 0, *args, **kwargs):
        self.name = name
        self.R = radius
        self.M = Mass
        self.color = color
        self.mu = kwargs.get('mu', self.M * AL_BF.ConstantsBook().G) #if it is not given, it is calculated with the mass
        self.orb = kwargs.get('orbit', 'No information of the orbit') # Orbit of the body

    def addOrbit(self,orbit):
        """

        """
        self.orb = orbit


################################################################################
# CREATE ORBIT AND ITS CHARACTERISTICS
# Valid for an elliptical orbit. Application of the 2 Body Problem.
################################################################################
class CircularOrbit:
    def __init__(self, a, CentralBody, *args, **kwargs):
        self.a = a
        self.CentralBody = CentralBody
        self.Body = kwargs.get('Body', 'Unknown')
        
        # Mean motion, period of the orbit
        self.n = np.sqrt(self.CentralBody.mu / self.a**3) 
        self.T = 2*np.pi / self.n


#Review Kepler 2000: not working
class BodyOrbit:
    """
    orbitParam: contains the main characteristics of an ELLIPTICAL orbit. Application of the 2 Body problem. 
        Body: main body about which the orbit is performed
        x: Two main cases are possible:
            Keplerian elements as inputs:
                a: semi-major axis (m)
                e: eccentricity of the orbit
                i: inclination of the orbit
                RAAN: right ascension of the ascending node
                omega: argument of the periapsis
                M: mean anomaly
                units: 'deg' indicates that the angles have to be transformed to radians
            Cartesian elements:
                x,y,z,xdot,ydot,zdot: position and velocity at a given time
        typeParam:
            'Keplerian_J2000': x is a matrix with:
                row 0: Keplerian elements at J2000
                row 1: centennial rates of change of those elements
                row 2: date in Julian centuries
            'Keplerian': to use Keplerian parameters as inputs
            'Cartesian': to use Cartesian parameters as inputs
    """
    def __init__(self, x, typeParam, Body, *args, **kwargs):
        
        self.CentralBody = Body
        
        ######################################
        #### Case 1: Keplerian Elements J2000 + centennial rates
        ######################################
        # Obtain Keplerian elements at a given date (in Julian Centuries)
        
        units_input = kwargs.get('unitsI', 'rad')
        
        if typeParam == 'Keplerian_J2000':
            a,e,i,L,varpi,RAAN = x[0,:] + x[1,:]*x[2,0]
            omega = varpi - RAAN
            M = L - varpi 
            

        ######################################
        #### Case 1: Keplerian Elements given
        ######################################
        elif typeParam == 'Keplerian':
            self.a,self.e,self.i,self.RAAN,self.omega,self.M = x

        if typeParam == 'Keplerian' or typeParam == 'Keplerian_J2000':
            
            if units_input == 'deg': #Operations are done in radians
                [self.i,self.RAAN,self.omega,self.M] = [AL_BF.deg2rad(angle) for angle in [self.i,self.RAAN,self.omega,self.M]]

            #Obtain the eccentric and true anomaly
            self.theta, self.E, self.M = Kepler(self.M, self.e,'Mean')
            
            #Obtain the cartesian coordinates
            x = self.Kepl2Cart()
            self.r = x[0:3]
            self.v = x[3:]
            self.h = np.cross(self.r, self.v)
                
        ######################################
        #### Case 3: Cartesian Elements given"
        ######################################
        elif typeParam == 'Cartesian':
            state = x #in m and m/s
            
            self.r = state[0:3] # position vector
            self.v = state[3:] # velocity vector
            self.r_mag = np.linalg.norm(self.r) # magnitude of the position vector 
            self.v_mag = np.linalg.norm(self.v) # magnitude of the velocity vector 
            
            self.h = np.cross(self.r, self.v)
            self.h_mag = np.linalg.norm(self.h) 
            self.N = np.cross(np.array([0,0,1]),self.h) 
            self.N_mag = np.linalg.norm(self.N)
            
            #Keplerian elements
            self.a = 1 / (2/self.r_mag-self.v_mag**2/Body.mu) #semi-major axis. Negative if e>1
            self.e_vec = np.cross(self.v,self.h)/Body.mu - self.r/self.r_mag #eccentricity vector
            self.e = np.linalg.norm(self.e_vec) #eccentricity of the orbit

            self.i = np.arccos(self.h[2]/self.h_mag) #inclination of the orbit

            N_xy = np.sqrt(self.N[0]**2 + self.N[1]**2)
            if N_xy == 0.:
                self.RAAN = 0
            else:
                self.RAAN = np.arctan2(self.N[1]/N_xy , self.N[0]/N_xy) #right ascension of the ascending node

            if self.e == 0.0 or self.N_mag == 0.0: 
                self.omega = 0.0
            else:
                self.omega = np.arccos(np.dot(self.e_vec/self.e , self.N/self.N_mag)) #argument of the perihelium
                if (np.dot(np.cross(self.N/self.N_mag , self.e_vec),self.h)) <= 0: 
                    self.omega = -self.omega
            
            if self.r_mag == 0.0 or self.e == 0.0:
                self.theta = 0.0
            else:
                self.theta = np.arccos(np.dot(self.r/self.r_mag , self.e_vec/self.e)) #true anomaly
                if (np.dot(np.cross(self.e_vec,self.r),self.h)) <= 0: 
                    self.theta = -self.theta

            if self.e < 1:
                self.theta, self.E, self.M = Kepler(self.theta,self.e,'True') #Mean anomaly
            else:
                self.theta, self.E, self.M = Keplerh(self.theta,self.e,'True') #Mean anomaly
            
        else:
            print('Warning: Specify the type of elements')

        ######################################
        ## More parameters of the orbit
        ######################################
        #Create a vector with all the keplerian elements
                
        self.KeplerElem = np.array([self.a,self.e,self.i,self.RAAN,self.omega,self.M])         
        self.KeplerElem_deg = np.zeros(6)
        self.KeplerElem_deg[0:2] = [self.a,self.e]
        for item, angle in enumerate([self.i,self.RAAN,self.omega,self.M]):
            self.KeplerElem_deg[2+item] = AL_BF.rad2deg(angle)

        #Mean motion, period of the orbit
        self.n = np.sqrt(Body.mu/self.a**3) 
        self.T = 2*np.pi * np.sqrt(self.a**3/Body.mu)
        
        #Geometry of the elliptic trajectory
        self.p = self.a * (1-self.e**2)
        self.rp = self.a * (1-self.e)
        self.ra = self.a * (1+self.e)
        self.b = self.a * np.sqrt(1-self.e**2)


    def Kepl2Cart(self, *args, **kwargs):
        """
        Kepl2Cart: transform from Keplerian elements to cartesian coordinates.
        Inputs: 
        Outputs:
            [x,y,z,xdot,ydot,zdot]: state of the body given by its position and velocity vectors.
        """
        givenElements = kwargs.get('givenElements', 'False') 
        if type(givenElements) != str:
            [a,e,i,RAAN,omega,M,theta] = givenElements
        else:
            [a,e,i,RAAN,omega,M,theta] = np.array([self.a,self.e,self.i,self.RAAN,self.omega,self.M,self.theta]) 

        l1 = np.cos(RAAN)*np.cos(omega) - np.sin(RAAN)*np.sin(omega)*np.cos(i)
        l2 = -np.cos(RAAN)*np.sin(omega) - np.sin(RAAN)*np.cos(omega)*np.cos(i)
        m1 = np.sin(RAAN)*np.cos(omega) + np.cos(RAAN)*np.sin(omega)*np.cos(i)
        m2 = -np.sin(RAAN)*np.sin(omega) + np.cos(RAAN)*np.cos(omega)*np.cos(i)
        n1 = np.sin(omega)*np.sin(i)
        n2 = np.cos(omega)*np.sin(i)

        H = np.sqrt(self.CentralBody.mu*a*(1-e**2)) #magnitude of the angular momentum
        r = a*(1-e*np.cos(self.E)) #magnitude of the position vector at a given moment

        #Position vector
        x = l1*r*np.cos(theta) + l2*r*np.sin(theta) 
        y = m1*r*np.cos(theta) + m2*r*np.sin(theta)
        z = n1*r*np.cos(theta) + n2*r*np.sin(theta)

        #Velocity vector
        xdot = self.CentralBody.mu/H * (-l1*np.sin(theta) + l2*(e+np.cos(theta)))
        ydot = self.CentralBody.mu/H * (-m1*np.sin(theta) + m2*(e+np.cos(theta)))
        zdot = self.CentralBody.mu/H * (-n1*np.sin(theta) + n2*(e+np.cos(theta)))

        return [x,y,z,xdot,ydot,zdot]

    
    def Propagation(self,t, typeParam, *args, **kwargs):
        """
        Propagation: propagate the mean anomaly M in time.
        Inputs: 
            t: time in seconds at which the position wants to be known with
                respect to the given state in the init
        Outputs: 
            M: new mean anomaly
        """
        self.M += self.n*t # Propagate in time
        self.M = AL_BF.correctAngle(self.M, 'rad') # Correct for more than 306deg
            
        if self.e < 1:
            self.theta, self.E, self.M = Kepler(self.M, self.e,'Mean')
        else:
            self.theta, self.E, self.M = Keplerh(self.M, self.e,'Mean')
        # self.theta = Kepler(self.M, self.e,'Mean')[0]
        
        x = np.array([self.a,self.e,self.i,self.RAAN,self.omega,self.M,self.theta])
        if typeParam == 'Keplerian':
            return x
        else:
            #Obtain the cartesian coordinates
            x = self.Kepl2Cart(givenElements = x)
            return x
    
    def atPeriapsis(self):
        self.rp, self.vp = VisVivaEq(self.CentralBody, self.a, r = self.rp)
        self.tp = self.M / self.n # time since passage through periapsis

    # def solveKeplerUniversal(self, Xi, x_add):
    #     [t, alpha] = x_add
    #     self.atPeriapsis() 
    #     r0 = self.rp
    #     vr0 = self.vp
    #     t0 = self.tp
    #     f = - np.sqrt(self.CentralBody.mu) * (t + t0) + \
    #         r0*vr0 / np.sqrt(self.CentralBody.mu) * Xi**2 * Stumpff_C(alpha* Xi**2) +\
    #         (1 - alpha* r0) * Xi**3 * Stumpff_S( alpha* Xi**2) +\
    #         r0 * Xi
    #     return f

    # def PropagationUniversal(self, t, typeParam, *args, **kwargs):
    #     """
    #     Propagation: propagate the mean anomaly M in time. Valid for ellipses, hyperbola, parabola
    #     Inputs: 
    #         t: time in seconds at which the position wants to be known with
    #             respect to the given state in the init
    #     Outputs: 
    #         M: new mean anomaly
    #     """
    #     # Find value of XI: universal anomaly
    #     alpha = 1/self.a 
    #     Xi_0 = np.sqrt(self.CentralBody.mu) * abs(alpha) * t
    #     Xi = spy.fsolve(self.solveKeplerUniversal, Xi_0, args = [t, alpha, ])
    

    def PlotOrbit(self):
        """
        PlotOrbit: plot the relative position of the planets at a given date for each one
        Inputs: 
            Body: central Body
        Outputs:
            xEl_E: x coordinate of the ellipse that represents the orbit
            yEl_E: y coordinate of the ellipse that represents the orbit
        """
        # Plot ellipse
        xEllipse = np.linspace(-self.a,self.a, num = 2500)
        yPos = np.sqrt(self.b**2*(1-xEllipse**2/self.a**2)) 
        yNeg = -np.sqrt(self.b**2*(1-xEllipse**2/self.a**2)) 
        xEl_E = np.append(xEllipse,xEllipse[::-1])
        xEl_E -= self.a*self.e*np.ones(len(xEl_E))
        yEl_E = np.append(yPos,yNeg)

    #     # Rotate ellipse
    #     ELL0=np.zeros((len(xEl_E0),3)) #Ellipse coordinates
    #     ELL0[:,0] = xEl_E0
    #     ELL0[:,1] = yEl_E0
    #     Q, nan,nan = transforMatrix(x,(x[4]-x[3])*np.pi/180,[0,0,0],[0,0,0])

    #     ELL = np.dot(ELL0,np.transpose(Q))
    #     xEl_E = ELL[:,0]
    #     yEl_E = ELL[:,1]

        fig, ax = plt.subplots(1,1, figsize = (9,8))
        # Plot orbit
        plt.plot(xEl_E,yEl_E,color = 'black',linestyle = '--') # Plot ellipse
        plt.plot([self.r[0],0],[self.r[1],0],color = 'black') #Plot line from planet to the Sun
        plt.plot(self.r[0],self.r[1], marker = "o",markersize = 20, color = 'black', alpha = 0.5,label = 'Position') # Plot planet

        plt.plot(0,0,Markersize = 695508/15000,color= self.CentralBody.color,marker="o")
        
        plt.title('2D Trajectory',size = 22)

        plt.axis("equal")
#         plt.xticks(np.arange(-3e11, 3e11, 1.0e11))
#         plt.yticks(np.arange(-3e11, 3e11, 1.0e11))
        plt.xlabel('(m)',size = 25)
        plt.ylabel('(m)',size = 25)

        plt.tick_params(axis ='both', which='major', labelsize = 15)
        plt.tick_params(axis ='both', which='minor', labelsize = 15)
        text = ax.xaxis.get_offset_text()
        text.set_size(15) #
        text = ax.yaxis.get_offset_text()
        text.set_size(15) #

#         lgnd = plt.legend(bbox_to_anchor=(0.5, 0), loc='lower left',ncol=2, mode="expand", borderaxespad=0.,fontsize=16)
#         lgnd.legendHandles[0]._legmarker.set_markersize(20)
        
    #     plt.savefig("_%i.png"%i,dpi=200,bbox_inches='tight')
        plt.show()


################################################################################
# STUMPFF FUNCTIONS
################################################################################
def Stumpff_C(z):
    if z > 0:
        f = ( 1 - np.cos(np.sqrt(z)) ) / z
    elif z == 0:
        f = 1/2
    else:
        f = ( np.cosh(np.sqrt(-z)) - 1 ) / (-z)
    return f

def Stumpff_S(z): 
    if z > 0:
        f = ( np.sqrt(z) - np.sin(np.sqrt(z)) ) / z**(3/2)
    elif z == 0:
        f = 1/6
    else:
        f = ( np.sinh(np.sqrt(-z)) - np.sqrt(-z)  ) / (-z)**(3/2)
    return f

################################################################################
# PROPAGATE ALONG ORBIT FG PARAMS
################################################################################
def fgSolveE(E,x):
    """
    fgSolveE: solve Kepler's equation to obtain the Eccentric anomaly
    Inputs: 
        E: initial guess of the eccentric anomaly
        x: arguments to put in the equation
            x[0]: E0. Initial eccentric anomaly of the spacecraft
            x[1]: M0. Initial mean nomaly of the spacecraft
            x[2]: M. Current mean anomaly of the spacecraft
            x[3]: e. Eccentricity of the orbit
            x[4]: a. Semi-major axis of the propagation orbit
            x[5]: SV. State vector of the planet. [x,y,z,vx,vy,vz,m0,ind]
                x: position in the x axis
                y: position in the y axis
                z: position in the z ais
                vx : velocity in the x axis
                vy : velocity in the y axis
                vz : velocity in the z axis
                m0: mass at that point
                ind: indicator to know if there is an impulse
    Outputs:
        f: function to solve
    """
    [E0,M0,M,e,a,SV,mu] = x
    r = np.linalg.norm(SV[0:3])
    
    sigma0 = np.dot(SV[0:3],SV[3:6])/np.sqrt(mu) # Value to put in next equation
    f = E-E0-(1-r/a)*np.sin(E-E0)+sigma0/np.sqrt(a)*(1-np.cos(E-E0))-(M-M0) # Modified Kepler's equation
    # for the case in which we are not in perigee
    return f
            

def fgPropagate(orbit,SV,t): 
    """
    fgPropagate: propagate the mean anomaly M in time.
    Inputs: 
        param: list of parameters. [a,e,E0,nu0,M0,n]
            E0: previous eccentric anomaly
            nu0: previous true anomaly
            M0: previous mean anomaly
            n: proper motion
        SV: state vector at the previous point [x,y,z,vx,vy,vz,m0,ind]
                x: position in the x axis
                y: position in the y axis
                z: position in the z ais
                vx : velocity in the x axis
                vy : velocity in the y axis
                vz : velocity in the z axis
        t: difference in time from the previous point
    Outputs: 
            E: new eccentric anomaly
            M: new mean anomaly
    """
    
    M = orbit.M + orbit.n * t
    E = spy.fsolve(fgSolveE,orbit.E,[orbit.E,orbit.M,M,orbit.e,orbit.a,SV,orbit.Body.mu]) # Obtain new E
    E = float(E)
    
    return [E,M]

def fgSV(orbit,param,t):
    """
    fgSV: obtain new state vector from the previous one and the parameters of the orbit.
    Inputs:
        SV0: state vector at the previous point [x,y,z,vx,vy,vz,m,ind]
            x: position in the x axis
            y: position in the y axis
            z: position in the z ais
            vx: velocity in the x axis
            vy: velocity in the y axis
            vz: velocity in the z axis
            m: mass at that point
            ind: indicator to know if there is an impulse
        param: parameters of the point of the propagation
        paramF: parametrs of the next point of the propagation
            a: semi-major axis of the propagation orbit
            e: eccentricity of the propagation orbit
            E: new eccentric anomaly
            nu: new true anomaly
            M: new mean anomaly
            n: proper motion
        t: difference in time from the previous time
    Outputs:
        SV: new state vector
    """  
    E0 = orbit.E
    E = param[0]

    r0 = np.linalg.norm(orbit.r)

    # f and g coefficients
    sigma0 = np.dot(orbit.r,orbit.v)/np.sqrt(orbit.Body.mu)
    f = 1 - orbit.a / r0*(1-np.cos(E-E0))
    g = (orbit.a * sigma0*(1-np.cos(E-E0)) + r0*np.sqrt(orbit.a)*np.sin(E-E0))/np.sqrt(orbit.Body.mu)
    
    #New position
    SV = np.zeros(6)
    SV[0:3] = f*orbit.r + g*orbit.v
    r = np.linalg.norm(SV[0:3])
    
    #f',g'
    ff = -np.sqrt(orbit.Body.mu * orbit.a) / (r*r0)*(np.sin(E-E0))
    gg = 1 - orbit.a / r *(1-np.cos(E-E0))
    
    #New velocity
    SV[3:6] = ff*orbit.r + gg*orbit.v
    
    return SV


################################################################################
# MORE
################################################################################
def VisVivaEq(Body,a,*args,**kwargs):
    """
    VisVivaEq: obtain position from velocity or vice versa with the vis viva or energy equation.
    inputs:
        Body: body around which the orbit is performed
        a: semi-major axis of the orbit
        **kwargs:
            r: magnitude of the position vector
            v: magnitude of the velocity vector
    Outputs: 
        r: magnitude of the position vector
        v: magnitude of the velocity vector
    """
    r = kwargs.get('r', None) 
    v = kwargs.get('v', None)
    
    if r == None and v != None: #v is given
        r = Body.mu / (Body.mu/(2*a)+v**2/2) 
    elif  r != None and v == None: #r is given
        v = 2*np.sqrt(-Body.mu/(2*a) + Body.mu/r)
    else: 
        print('Warning: Not enough arguments or too many arguments given')
    return r,v


def KeplerSolve(E0, x):
    """
    Kepler: function to solve Kepler's equation knowing the mean anomaly. To obtain the eccentric anomaly.
    Inputs:
        E0: initial guess of the solution
        x: vector containing other parameters of the equation:
            x[0]: Mean anomaly (rad)
            x[1]: Eccentricity
    Outputs:
        F: function to solve (rad)
    """
    #Parametrs of the equation
    M = x[0]
    e = x[1]

    #Equation to solve
    F = E0-e*np.sin(E0)-M
    F = AL_BF.correctAngle(F, 'rad')

    return F


def Kepler(angle,e,name):
    """
    Kepler: function to calculate true, eccentric and mean anomaly given one of them and the 
    eccentricity of the orbit.
    Inputs:
        angle: initial angle known. It can be any of the three mentioned
        e: eccentricity of the orbit (magnitude)
        name: specify which the initial angle given is:
            'Mean': mean anomaly
            'True': true anomaly
            'Eccentric': eccentric anomaly
    Outputs: 
        theta: true anomaly
        E: eccentric anomaly
        M: mean anomaly
    """
    if name == 'Mean':
        M = angle
        E = spy.fsolve(KeplerSolve,M,[M,e]) #Solve Kepler's equation
        E = E[0] 
        theta = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2)) #Calculate true anomaly
#         if abs(theta) >= 2*np.pi: #Correction in case is higher than 360
#             a = theta//(2*np.pi)
#             theta = theta-2*np.pi*a  
        return(theta,E,M)

    elif name == 'Eccentric':
        E = angle 
        theta = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2)) #Calculate true anomaly
        M = E-e*np.sin(E) #Kepler's equation
        return(theta,E,M)
    
    elif name == 'True':
        theta = angle
        E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(theta/2)) #Calculate eccentric anomaly
        M = E-e*np.sin(E) #Kepler's equation
        return(theta,E,M)
    
    else:
        print('Warning: Include a valid name for the angle given')



def KeplerhSolve(E0, x):
    """
    Kepler: function to solve Kepler's equation knowing the mean anomaly. To obtain the eccentric anomaly.
    Inputs:
        E0: initial guess of the solution
        x: vector containing other parameters of the equation:
            x[0]: Mean anomaly (rad)
            x[1]: Eccentricity
    Outputs:
        F: function to solve (rad)
    """
    #Parametrs of the equation
    M = x[0]
    e = x[1]

    #Equation to solve
    F = e*np.sinh(E0) - E0- M
    F = AL_BF.correctAngle(F, 'rad')
    return F

def Keplerh(angle, e, name):
    """
    Keplerh: function to calculate true, eccentric and mean anomaly given one of them and the 
    eccentricity of the orbit. For an orbit of e >= 1
    Inputs:
        angle: initial angle known. It can be any of the three mentioned
        e: eccentricity of the orbit (magnitude)
        name: specify which the initial angle given is:
            'Mean': mean anomaly
            'True': true anomaly
            'Eccentric': eccentric anomaly
    Outputs: 
        theta: true anomaly
        E: eccentric anomaly
        M: mean anomaly
    """
    print("Hyperbola")

    if name == 'Mean':
        M = angle
        E = spy.fsolve(KeplerhSolve, M, [M,e]) #Solve Kepler's equation
        E = E[0] 
        theta = 2*np.arctan( np.sqrt( (e+1) / (e-1) ) * np.tan(E/2) )
        return(theta, E, M)

    elif name == 'Eccentric':
        E = angle 
        theta = 2 * np.arctan( np.sqrt( (e+1) / (e-1) ) * np.tan(E/2) ) #Calculate true anomaly
        M = e * np.sinh(E) - E #Kepler's equation
        return(theta, E, M)
    
    elif name == 'True':
        theta = angle
        E = 2 * np.arctanh( np.sqrt( (e-1) / (e+1) ) * np.tan(theta/2) ) #Calculate eccentric anomaly
        M = e * np.sinh(E) - E #Kepler's equation
        return(theta, E, M)
    
    else:
        print('Warning: Include a valid name for the angle given')
        