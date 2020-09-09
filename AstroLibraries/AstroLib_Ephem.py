import matplotlib.pyplot as plt
import numpy as np



# ## Date conversion: Julian Days

class DateConv: 
    """
    DateConv: conversion of date
    """
    def __init__(self, date, typeDate):
        """
        Constructor: 
        INPUTS:
            date: date to be input
            typeDate: type of date used as input
                calendar: the date is a vector [Day, Month, Year, UT]
                MJD: date is the value of the modified julian days
                JD: date is the value of the julian days
        """
        if typeDate == "calendar":
            self.date = date
            self.J0 = self.JulianDays(date)
            self.JD = self.J0 + date[-1]/24   #UT represents the time
            self.T0 = (self.JD - 2451545) / 36525 # Julian century:
            self.MJD = self.JD - 2400000.5 #Modified Julian day
            self.JD_0 = self.JD - 2451545.0 #JD since J2000
            
        elif typeDate == "MJD":
            self.JD = date + 2400000.5
            self.MJD = date
            self.date = self.Calendar(self.JD)
            self.JD_0 = self.JD - 2451545.0 #JD since J2000
            
        elif typeDate == "JD":
            self.date = self.Calendar(date)
            self.JD = date
            self.MJD = self.JD - 2400000.5 
            self.JD_0 = self.JD - 2451545.0 #JD since J2000
        elif typeDate == "JD_0":
            self.JD = date + 2451545.0
            self.MJD = self.JD - 2400000.5 
            self.JD_0 = date
            self.date = self.Calendar(self.JD)
        else:
            print("Warning: introduce a valid type of date")
            
    def JulianDays(self, date): 
        """
        JulianDays: Transform the date into Julian Days
        INPUTS: 
            date: vector containining the date in the form [day, month, year,UT]
        OUTPUTS:
            J0: number of Julian Days corresponding to the date
        """        
        # Boulet formula to calculate the Julian days. // to obtain integers 
        J0 = 367*date[2] - ( 7*( date[2]+ ( (date[1]+9)/12 ) //1 )/4) //  \
            1+ ( 275 * date[1]/9 ) // 1 + date[0] + 1721013.5
        return J0
    
    def Calendar(self, date):
        """
        Calendar: transform JD into date
        INPUTS:
            date: number of julian days
        OUTPUTS: 
            vector [Day, Month, Year,UT] being UT = 0
        """
        JD0 = date + 0.5
        L = ( JD0 + 68569 ) // 1
        N = ( 4 * L / 146097 ) // 1
        L = L - ((146097 * N + 3 ) / 4) // 1
        I = ((4000 * (L + 1) ) / 1461001) // 1
        L = L -(1461*I/4) // 1 + 31
        J = (80 * L / 2447) // 1
        D = L - (2447 * J / 80) // 1
        L = (J / 11) // 1
        M = J + 2- 12 * L
        Y = 100* ( N - 49 ) + I + L
        return [D, M, Y, 0]


class EphemerisJ2000: # Fix
    """
    EphemerisJ2000: [a (m) ,e, i ,Omega (longitude of ascending node), varpi 
    (longitude of perihelium), L (mean longitude)]    m, deg, deg/Cy
    """
    def __init__(self):

#         Mercury0   = np.array([0.38709927*1.49597871e11, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 252.25032350])
#         MMercury0 = np.array([0.00000037*1.49597871e11,  0.00001906, -0.00594749, -0.12534081, 0.16047689, 149472.67411175])
                     
#         self.Earth0 = np.array([1.00000261*1.49597871e11, 0.01671022 , -0.00054346   ,-5.11260389, 102.94719, 100.46435])
#         self.EEarth0 = np.array([-0.00000005*1.49597871e11 ,-0.00003661, -0.01337178,-0.24123856,  0.31795260, 35999.37306329     ])

#         self.Mars0 = np.array([1.52371034*1.49597871e11, 0.09339410, 1.85181869  ,49.71320984, 336.04084, 355.45332]) 
#         self.MMars0 = np.array([0.00000097*1.49597871e11, 0.00009149, -0.00724757    , -0.26852431, 0.45223625 ,19140.29934243])

        # a, e, i , L, longitude of perihelion, longitude of ascending node:
        # a, e, i , M+ omega+RAAN, omega+RAAN, RAAN
        self.AU = 1.49597871e11
        self.Mercury0 = np.array([0.38709927*self.AU , 0.20563593, 7.00497902, \
            252.25032350, 77.45779628, 48.33076593])
        self.MMercury0 = np.array([0.00000037*self.AU , 0.00001906, -0.00594749,\
             149472.67411175, 0.16047689, -0.12534081])
        self.Venus0 = np.array([0.72333566*self.AU , 0.00677672, 3.39467605, \
            181.97909950, 131.60246718, 76.67984255])
        self.VVenus0 = np.array([0.00000390*self.AU , -0.00004107, -0.00078890,\
             58517.81538729, 0.00268329, -0.27769418])
        self.Earth0 = np.array([1.00000261*self.AU , 0.01671123, -0.00001531, \
            100.46457166, 102.93768193, 0.0])
        self.EEarth0 = np.array([0.00000562*self.AU , -0.00004392, -0.01294668, \
            35999.37244981, 0.32327364, 0.0])
        self.Mars0 = np.array([1.52371034*self.AU , 0.09339410, 1.84969142,\
             -4.55343205, -23.94362959, 49.55953891])
        self.MMars0 = np.array([0.00001847*self.AU , 0.00007882, -0.00813131,\
             19140.30268499, 0.44441088, -0.29257343])
        self.Jupiter0 = np.array([5.20288700*self.AU , 0.04838624, 1.30439695, \
            34.39644051, 14.72847983, 100.47390909])
        self.JJupiter0 = np.array([-0.00011607*self.AU , -0.00013253, \
            -0.00183714, 3034.74612775, 0.21252668, 0.20469106])
        self.Saturn0 = np.array([9.53667594*self.AU , 0.05386179, 2.48599187, \
            49.95424423, 92.59887831, 113.66242448])
        self.SSaturn0 = np.array([-0.00125060*self.AU , -0.00050991, 0.00193609,\
             1222.49362201, -0.41897216, -0.28867794])
        self.Uranus0 = np.array([19.18916464*self.AU , 0.04725744, 0.77263783,\
             313.23810451, 170.95427630, 74.01692503])
        self.UUranus0 = np.array([-0.00196176*self.AU , -0.00004397,\
             -0.00242939, 428.48202785, 0.40805281, 0.04240589])
        self.Neptune0 = np.array([30.06992276*self.AU , 0.00859048, 1.77004347,\
             -55.12002969, 44.96476227, 131.78422574])
        self.NNeptune0 = np.array([0.00026291*self.AU , 0.00005105, 0.00035372,\
             218.45945325, -0.32241464, -0.00508664])
        self.Pluto0 = np.array([39.48211675*self.AU, 0.24882730, 17.14001206,\
             238.92903833, 224.06891629, 110.30393684])
        self.PPluto0 = np.array([-0.00031596*self.AU , 0.00005170, 0.00004818,\
             145.20780515, -0.04062942, -0.01183482])


# ## Ephemeris from file

class EphemerisFile:
    """
    EphemerisFile: take ephemeris from file. In km and km/s
    """
    def __init__(self):

        self.ephem_E = np.genfromtxt('AstroLibraries_0/PlanetaryEphem/earthState.txt', \
            delimiter = None)
        self.ephem_M = np.genfromtxt('AstroLibraries_0/PlanetaryEphem/marsState.txt', \
            delimiter = None)
        self.ephem_V = np.genfromtxt('AstroLibraries_0/PlanetaryEphem/venusState.txt', \
            delimiter = None)
        self.ephem_J = np.genfromtxt('AstroLibraries_0/PlanetaryEphem/jupiterState.txt',\
             delimiter = None)
        
    
    def rv(self, ephem, units, returnValue):
        """
        rv: from ephemeris obtain position and velocity.
        INPUTS:
            ephem: ephemeris
            units: 'km' or 'm'
            returnValue: 'r', 'v', 'rv'
        OUTPUTS:
            r, v or state: depending on returnValue
        """
        r = np.zeros([len(ephem[:,0])//3, 3])
        v = np.zeros(np.shape(r))
        state = np.zeros([len(ephem[:,0])//3, 6])

        if units == 'm':
            factor = 1000
        else:
            factor = 1
            
        for i in range(len(ephem[:,0])//3):
            r[i,:] = ephem[3*i+1,:] * factor
            v[i,:] = ephem[3*i+2,:] * factor   
            state[i,:] = np.concatenate((r[i,:],v[i,:]))
        
        if returnValue == 'r':
            return r
        elif returnValue == 'v':
            return v
        else:
            return state
    
    
    def rv_atDate(self, state, date):
        """
        rv_atDate: position and velocity at a time
        INPUTS:
            state: position and velocity
            date: time in julian days from 2000
        OUTPUTS:
            state[date]: state at date
        """
        return state[date]

## Change in systems of reference
def rot(angle, axis):
    """
    rot: create a rotation matrix around an axis for an angle
    INPUTS:
        angle:angle to be rotated
        axis: axis around which it is rotated: x = 1, y = 2, z = 3
    OUTPUTS:
        R: rotation matrix

    """
    if axis == 1: 
        R = np.array([[1, 0, 0],
                [0, np.cos(angle), np.sin(angle)],
                [0, -np.sin(angle), np.cos(angle)]])
    elif axis == 2:
        R = np.array([[np.cos(-angle), 0, np.sin(-angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(-angle)]])
    elif axis == 3:
        R = np.array([[np.cos(angle), np.sin(angle), 0],
                    [-np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    return R

class FramesOfReference:
    """
    FramesOfReference: transform between perifocal and heliocentric frame
    INPUTS:
        x: 3d vector 
        kep: vector [omega, i, RAAN]
        typeOrigin: type of the input vector
            helioc: heliocentric fram
            perif: perifocal frame
        typeGoal: frame of the output vector. Same types as Origin
    OUTPUTS:
        x2: transformed vector in the other frame
    """
    def __init__(self, kep, typeOrigin):
        """
        Constructor:
        INPUTS: 
            kep: [argument of the periapsis, inclination and right ascension of 
            the ascending node]
            typeOrigin: frame of reference, either "perif" or "helioc"
        """
        omega, i, RAAN = kep
        self.typeOrigin = typeOrigin

        R3_omega = rot(omega, 3)
        R1_i = rot(i, 1)
        R3_RAAN = rot(RAAN, 3)

        self.R_h2p = np.dot(np.dot(R3_omega,R1_i),R3_RAAN)

    def transform(self, x, typeGoal):
        """
        transform: transform between one frame to the other
        INPUTS:
            x: vector to be transformed
            typeGoal: type to be transformed to
        OUTPUTS:
            x2: transformed vector
        """
        
        if typeGoal == 'perif':
            x2 = np.dot(self.R_h2p, x)
        elif typeGoal == 'helioc':
            x2 = np.dot( np.transpose(self.R_h2p), x)
        else:
            x2 = None
            print('Error: Wrong input type of reference frame')

        return x2


