import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy
import pykep as pk


# ## Simple functions
def deg2rad(x):
    """
    deg2rad: convert from degrees to radians
    INPUTS:
        x: angle in degrees
    OUTPUTS:
        angle in radians
    """
    return x*np.pi/180

def rad2deg(x):
    """
    rad2deg: convert from radians to degrees
    INPUTS:
        x: angle in radians
    OUTPUTS:
        angle in degrees
    """
    return x/np.pi*180

def days2sec(x):
    """
    days2sec: convert from days to seconds
    INPUTS:
        x: time in days
    OUTPUTS:
        time in seconds
    """
    return x*3600*24

def sec2days(x):
    """
    sec2days: convert from seconds to days
    INPUTS:
        x: time in seconds
    OUTPUTS:
        time in days
    """
    return x/ (3600*24)

def correctAngle(x, units):  
    """
    correctAngle: correct angle for more than 2 pi radians or 360deg. 
    INPUTS:
        x: angle to correct 
        units:
            'deg': degrees
            'rad': radians
    OUTPUTS:
        x: corrected angle
    """
    if units == 'deg':
        angle_ref = 360
    elif units == 'rad':
        angle_ref = 2*np.pi
    else:
        print("WARNING: Not a valid angle")

    if x >= angle_ref: # Correction in case is higher than 360
        a = x //angle_ref # How many entire revolutions
        return x - angle_ref*a
    else:
        return x

    
def convertRange(x, units, minRange, maxRange):
    """
    convertRange: convert an angle so that it is within the range
    INPUTS:
        x: angle to be converted
        units: units of the input angle
        minRange: minimum value of the range
        maxRange: maximum value of the range

    OUTPUTS:
        angleCorrected: corrected angle
    """

    if units == 'deg':
        angle_ref = 360
    elif units == 'rad':
        angle_ref = 2*np.pi
    else:
        print("WARNING: Not a valid angle")

    if x > maxRange:
        angleCorrected = x-maxRange + minRange
    elif x < minRange:
        angleCorrected = x -minRange +maxRange
            
    return angleCorrected


def writeData(data, mode, title):
    """
    writeData: write data into file
    INPUTS:
        data: data to be written (use np.concatenate and append)
        mode: type of file mode: w (write), a+(append), r (read)
        title: filename
    """
    str_data = '  '.join(str(i) for i in data)
    with open(title, mode) as fo:
        fo.write(str_data)
        fo.write("\n")

################################################################################
## Tables of constants
################################################################################

# pykep Constants
AU = pk.AU
mu_S = pk.MU_SUN
mu_E = pk.MU_EARTH
v_E = pk.EARTH_VELOCITY
g0 = pk.G0


#Constants from Books (different than pykep )
class ConstantsBook:
    """
    Table of constants taken from  J. R. Wertz, Mission geometry: orbit and 
    constellation design and management: spacecraft  orbit  and  attitude  systems.    
    """
    def __init__(self):
        """
        Constructor:
        """
        self.G = 6.67408e-11 #m^3kg^-1s^-2 Universal Gravitational parameter

        self.AU = 149597870.66 #km Astronomical unit
        self.DSid = 86164.1004 #s Lenght of the sidereal day
        self.i_Delfi = 98.0 #deg Inclination orbit Delfi-c3
        self.i_ENVISAT = 98.5 #deg Inclination orbit ENVISAT
        self.i_GPS = 55.0 #deg Inclination orbit GPS
        self.i_ISS = 51.9 #deg Inclination orbit ISS
        self.J2 = 1082.63e-6 #J2 coefficient gravity field Earth
        self.J22 = 1.82e-6 #J2,2 coefficient gravity field Earth

        self.R_E = 6378.136e3 #m Radius of Earth 
        self.R_J = 71492e3 #m Radius Jupiter
        self.R_M = 3397e3 #m Radius Mars
        self.R_S = 6.96033e8 #m Radius Sun
        self.d_SJ = 5.20336301 #AU distance Sun-Jupiter
        self.d_SM = 1.52366231 #AU distance Sun-Mars
        self.SC = 1367 #W/m^2 Solar constant

        self.lat_Baikonur = 45.6 #deg latitude Baikonur
        self.lat_Delft = 52.0 #deg latitude Delft
        self.lat_KSC = 28.5 #deg latitude Kennedy Space Center
        self.lat_Plesetsk = 62.8 #deg latitude Plesetsk
        self.long_Delft = 4.3 #deg longitude Delft

        self.gravCoef_E = -14.9 #deg Coefficient gravity field Earth
        self.mu_E = 398600.441 #km^3/s^2 Gravitational parameter Earth 
        self.mu_E_m = 3.98600441e14 #m^3/s^2 Gravitational parameter Earth 
        self.mu_J = 1.2669e8 #km^3/s^2 Gravitational parameter Jupiter
        self.mu_M = 4.2832e4 #km^3/s^2 Gravitational parameter Mars
        self.mu_M_m = 4.2832e13 #m^3/s^2 Gravitational parameter Mars
        self.mu_S = 1.327178e11 #km^3/s^2 Gravitational parameter Sun
        self.mu_S_m = 1.327178e20 #m^3/s^2 Gravitational parameter Sun        
