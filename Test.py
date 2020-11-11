import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import pykep as pk
from pykep.orbit_plots import plot_planet, plot_lambert
from pykep import AU, DAY2SEC


from FitnessFunction_normalized import Fitness, Propagate
import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_2BP as AL_2BP
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_Plots as AL_Plot
from AstroLibraries import AstroLib_Trajectories as AL_TR


import LoadConfigFiles as CONFIG

# # In[]
# ##############################################
# ############# VALIDATION #####################
# ##############################################
Cts = AL_BF.ConstantsBook()

def pykepephem():
    # earth = pk.planet.jpl_lp('earth')
    # marsephem = pk.planet.jpl_lp('mars')

    earth = pk.planet.jpl_lp('earth')

    K_0 = earth.eph(0)
    # print(K_0)
    earth.mu_central_body
    print(earth.mu_self)
    elem = earth.osculating_elements(pk.epoch(2345.3, 'mjd2000'))
    print(elem)
    print(np.array(elem))


##############################################
# Change frame of reference
##############################################
def changeReferenceFrame():
    # From Howard Curtis: pg218
    # For a given earth orbit, the elements are h 1⁄4 80,000 km 2 /s, e 1⁄4 1.4, i 1⁄4 30  , U 1⁄4 40  , u 1⁄4 60  , and q 1⁄4 30  .
    # Using Algorithm 4.5, find the state vectors r and v in the geocentric equatorial frame.
    h = 80000 * 1e6 #m^2/s
    e = 1.4
    i = AL_BF.deg2rad(30)
    RAAN = AL_BF.deg2rad(40)
    omega = AL_BF.deg2rad(60)
    theta = AL_BF.deg2rad(30)

    # rv in perifocal
    r_norm = h**2/Cts.mu_E_m / (1+e*np.cos(theta))
    r = r_norm* np.array([np.cos(theta), np.sin(theta),0])
    v = Cts.mu_E_m / h * np.array([-np.sin(theta), e+ np.cos(theta),0])
    print(r, v) # Validated 

    # r, v in geocentric
    Frame1 = AL_Eph.FramesOfReference([omega, i, RAAN], 'perif')
    r2 = Frame1.transform(r, 'helioc')
    v2 = Frame1.transform(v, 'helioc')
    print(r2, v2) # Validated

    # Back to perifocal
    r3 = Frame1.transform(r2, 'perif')
    v3 = Frame1.transform(v2, 'perif')
    print(r3, v3) # Validated: same as initial


def test_convertAngleForm():
    vector0 = np.array([90,0,0])
    vector1 = AL_BF.convert3dvector(vector0, "cartesian")
    print(vector1) # validated
    vector0 = np.array([90,90,90])
    vector1 = AL_BF.convert3dvector(vector0, "cartesian")
    print(vector1) # validated
    vector0 = np.array([155, 0.78, 0.61])
    vector1 = AL_BF.convert3dvector(vector0, "polar")
    print(vector1) # validated

    print(np.arcsin(-0.7071067811865475))
    vector0 = np.array([-90,-90,-90])
    vector1 = AL_BF.convert3dvector(vector0, "cartesian")
    print(vector1*AL_BF.rad2deg(1)) 

def test_convertRange():
    angle = -3.02
    print(AL_BF.convertRange(angle, 'deg', 0, 360))

##############################################
# Conversion Cartesian-Keplerian and vice-versa
##############################################
def CartKeplr():
    # From Mission Geometry slides
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)

    x = np.array([8751268.4691, -7041314.6869, 4846546.9938, \
                    332.2601039, -2977.0815768,-4869.8462227])
    x2 = np.array([-2700816.139,-3314092.801, 5266346.4207,\
                    5168.606551,-5597.546615,-868.8784452])
    x3 = np.array([10157768.1264, -6475997.0091, 2421205.9518,\
                1099.2953996, 3455.1059240, 4355.0978095])

    K = np.array([12158817.9615, 0.014074320051, 52.666016957,\
                    323.089150643, 148.382589129, 112.192638384])

    K2 = np.array([12269687.5912,  0.004932091570, 109.823277603,\
                134.625563565, 106.380426142, 301.149932402])

    print("CARTESIAN-KEPLERIAN")

    trajectory1 = AL_2BP.BodyOrbit(x, 'Cartesian', earth )
    print("Result vector 1")
    print(trajectory1.KeplerElem_deg) # Validated

    trajectory12 = AL_2BP.BodyOrbit(x2, 'Cartesian', earth)
    print("Result vector 2")
    print(trajectory12.KeplerElem_deg) # Validated

    trajectory13 = AL_2BP.BodyOrbit(x3, 'Cartesian', earth)
    print("Result vector 3")
    print(trajectory13.KeplerElem_deg) # Validated

    print("KEPLERIAN-CARTESIAN")

    trajectory2 = AL_2BP.BodyOrbit(K, 'Keplerian', earth, unitsI = 'deg')
    print("Result vector 1")
    print(trajectory2.r, trajectory2.v ) # Validated

    trajectory22 = AL_2BP.BodyOrbit(trajectory12.KeplerElem_deg, 'Keplerian', earth, unitsI = 'deg')
    print("Result vector 2")
    print(trajectory22.r, trajectory22.v ) # Validated

    trajectory23 = AL_2BP.BodyOrbit(K2, 'Keplerian', earth, unitsI = 'deg')
    print("Result vector 3")
    print(trajectory23.r, trajectory23.v ) # Validated`

##############################################
# Propagation
##############################################
def propagateHohmann():
    # with a hohmann orbit the necessary parameters are found, propagating 
    # we should get to the planet
    # ephemeris retrieval
    date0 = np.array([27,1,2016,0])
    t_0 = AL_Eph.DateConv(date0,'calendar') #To JD
    transfertime = 258.8493

    # Create bodies
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

    # Calculate trajectory of the bodies based on ephem
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')
    r_E, v_E = earthephem.eph(t_0.JD_0)
    r_M, v_M = marsephem.eph(t_0.JD_0 + transfertime)

    orbit_E = AL_2BP.BodyOrbit(np.append(r_E, v_E), 'Cartesian', sun)
    earth.addOrbit(orbit_E)
    orbit_M = AL_2BP.BodyOrbit(np.append(r_M, v_M), 'Cartesian', sun)
    mars.addOrbit(orbit_M)

    # Create transfer in the first moment
    transfer1 = AL_TR.SystemBodies([earth, mars])

    t, r_E, v_E, r_M, v_M = transfer1.findDateWith180deg(date0, transfertime, [earthephem, marsephem],usingCircular=False)
    date_1 = AL_Eph.DateConv(t,'JD_0') #To JD
    transfer1.HohmannTransfer(r_0 = r_E, r_1 = r_M)

    # Propagate
    r0 = np.array(r_E)
    v0 = transfer1.vp_H * np.array(v_E) / np.linalg.norm(v_E)

    orbit_sp = AL_2BP.BodyOrbit(np.append(r0,v0), 'Cartesian', sun)
    x = orbit_sp.Propagation(transfer1.timeflight_H, 'Cartesian') # Coordinates after propagation

    # Compare with supposed solutions
    print("HOHMANN")
    vf = transfer1.va_H
    rf = np.array(r_M)
    print(rf)
    print("PROPAGATION")
    x = np.array(x)
    print(x)

    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    ax.scatter(0,0,0, color = 'yellow', s = 100)
    ax.scatter(r0[0],r0[1],r0[2], color = 'blue')
    ax.scatter(rf[0],rf[1],rf[2], color = 'red')
    ax.scatter(x[0], x[1], x[2], color = 'green')

    AL_Plot.set_axes_equal(ax)
    plt.show() # Validated: more or less. Points are close to each other

    # orbit_sp.Plot(transfer1.KeplerElem, r0, 1, 'green','spacecraft', 10)

##############################################
# Lambert    
##############################################
def Lambert():
    #### Howard Curtis page 254
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    r_1 = np.array([5000, 10000, 2100]) * 1000
    r_2 = np.array([-14600, 2500, 7000]) * 1000

    lambert = AL_TR.Lambert(r_1, r_2, 3600, earth.mu)
    v1, v2 = lambert.TerminalVelVect()
    print(v1, v2) #Validated


def propagateLambert():
    ### Using ephemeris
    # Lambert trajectory obtain terminal velocity vectors
    date0 = np.array([27,1,2016,0])
    t_0 = AL_Eph.DateConv(date0,'calendar') #To JD
    transfertime = 250

    # Create bodies
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

    # Calculate trajectory of the bodies based on ephem
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')
    r_E, v_E = earthephem.eph(t_0.JD_0)
    r_M, v_M = marsephem.eph(t_0.JD_0 + transfertime)

    orbit_E = AL_2BP.BodyOrbit(np.append(r_E, v_E), 'Cartesian', sun)
    earth.addOrbit(orbit_E)
    orbit_M = AL_2BP.BodyOrbit(np.append(r_M, v_M), 'Cartesian', sun)
    mars.addOrbit(orbit_M)

    # Create transfer in the first moment
    lambert = AL_TR.Lambert(np.array(r_E), np.array(r_M), AL_BF.days2sec(transfertime), sun.mu)
    v_0, v_f = lambert.TerminalVelVect()
    print(v_0)

    # Propagate
    orbit_sp = AL_2BP.BodyOrbit(np.append(r_E, v_0), 'Cartesian', sun)
    x = orbit_sp.Propagation(AL_BF.days2sec(transfertime), 'Cartesian') # Coordinates after propagation

    r0 = np.array(r_E)
    rf = np.array(r_M)

    print('Mars', r_M, 'Propagation', x, 'Error', abs(rf - x[0:3])) # Almost zero

    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    ax.scatter(0,0,0, color = 'yellow', s = 100)
    ax.scatter(r0[0],r0[1],r0[2], color = 'blue')
    ax.scatter(rf[0],rf[1],rf[2], color = 'red')
    ax.scatter(x[0], x[1], x[2], color = 'green')

    AL_Plot.set_axes_equal(ax)
    plt.show() # Validated

# ##############################################
# # Propagation Universal
# ##############################################
def propagateUniversalLambert():
    ### Using ephemeris
    # Lambert trajectory obtain terminal velocity vectors
    date0 = np.array([27,1,2016,0])
    t_0 = AL_Eph.DateConv(date0,'calendar') #To JD
    transfertime = 250

    # Create bodies
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

    # Calculate trajectory of the bodies based on ephem
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')
    r_E, v_E = earthephem.eph(t_0.JD_0)
    r_M, v_M = marsephem.eph(t_0.JD_0 + transfertime)

    orbit_E = AL_2BP.BodyOrbit(np.append(r_E, v_E), 'Cartesian', sun)
    earth.addOrbit(orbit_E)
    orbit_M = AL_2BP.BodyOrbit(np.append(r_M, v_M), 'Cartesian', sun)
    mars.addOrbit(orbit_M)

    # Create transfer in the first moment
    lambert = AL_TR.Lambert(np.array(r_E), np.array(r_M), AL_BF.days2sec(transfertime), sun.mu)
    v_0, v_f = lambert.TerminalVelVect()
    print(v_0)

    # Propagate
    orbit_sp = AL_2BP.BodyOrbit(np.append(r_E, v_0), 'Cartesian', sun)
    x = orbit_sp.PropagationUniversal(AL_BF.days2sec(transfertime), 'Cartesian') # Coordinates after propagation

    r0 = np.array(r_E)
    rf = np.array(r_M)

    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    ax.scatter(0,0,0, color = 'yellow', s = 100)
    ax.scatter(r0[0],r0[1],r0[2], color = 'blue')
    ax.scatter(rf[0],rf[1],rf[2], color = 'red')
    ax.scatter(x[0], x[1], x[2], color = 'green')

    AL_Plot.set_axes_equal(ax)
    plt.show() 

def findValidLambert():
    SF = CONFIG.SimsFlan_config()

    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')

    counter = 0

    valid = False
    while valid == False:
        decv = np.zeros(len(SF.bnds))
        for i in range(6,8): 
            decv[i] = np.random.uniform(low = SF.bnds[i][0], \
                high = SF.bnds[i][1], size = 1)

        r_E, v_E = earthephem.eph(decv[6])
        r_M, v_M = marsephem.eph(decv[6] + AL_BF.sec2days(decv[7]) )

        # Create transfer in the first moment
        nrevs = 2
        l = pk.lambert_problem(r1 = r_E, r2 = r_M, tof = decv[7], \
            cw = False, mu =  Cts.mu_S_m, max_revs=nrevs)
        v1 = np.array(l.get_v1())
        v2 = np.array(l.get_v2())

        v_i_prev = 1e12 # Excessive random value
        for rev in range(len(v1)):
            v_i = np.linalg.norm(v1[rev] - np.array(v_E)) # Relative velocities for the bounds 
            v_i2 = np.linalg.norm(v2[rev] - np.array(v_M))
            # Change to polar for the bounds
            if v_i >= SF.bnds[0][0] and  v_i <= SF.bnds[0][1] and \
            v_i2 >= SF.bnds[3][0] and  v_i2 <= SF.bnds[3][1]:
                print('decv')
                print(v1[rev]-v_E,v2[rev]-v_M)
                decv[0:3] = AL_BF.convert3dvector(v1[rev]-v_E, "cartesian")
                decv[3:6] = AL_BF.convert3dvector(v2[rev]-v_M, "cartesian")
                print(decv[0:6])
                valid = True

                print('rev', rev, v1[rev], v2[rev])

        counter += 1
        print(counter)
    return decv, l

def propagateSimsFlanagan():
    "Test the propagation of SimsFlanagan back and forth using the velocities from Lambert"
    ### Using ephemeris
    # Lambert trajectory obtain terminal velocity vectors
    SF = CONFIG.SimsFlan_config()

    # Create bodies
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

    # Calculate trajectory of the bodies based on ephem
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')
    
    decv, l = findValidLambert()

    print(decv)

    Fit = Fitness(Nimp = SF.Nimp)
    Fit.calculateFeasibility(decv)
    Fit.printResult()

    # We plot
    mpl.rcParams['legend.fontsize'] = 10

    # Create the figure and axis
    fig = plt.figure(figsize = (16,5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter([0], [0], [0], color=['y'])

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter([0], [0], [0], color=['y'])
    ax2.view_init(90, 0)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter([0], [0], [0], color=['y'])
    ax3.view_init(0,0)

    t1 = SF.t0.JD
    t2 = t1 + AL_BF.sec2days(decv[7])

    for ax in [ax1, ax2, ax3]:
        # Plot the planet orbits
        # plot_planet(earth, t0=t1, color=(0.8, 0.8, 1), legend=True, units=AU, axes=ax)
        # plot_planet(mars, t0=t2, color=(0.8, 0.8, 1), legend=True, units=AU, axes=ax)

        # Plot the Lambert solutions
        axis = plot_lambert(l, color='b', legend=True, units=AU, axes=ax)
        # axis = plot_lambert(l, sol=1, color='g', legend=True, units=AU, axes=ax)
        # axis = plot_lambert(l, sol=2, color='g', legend=True, units=AU, axes=ax)

    plt.show()

def propagateSimsFlanaganForward():
    "Test the propagation of SimsFlanagan forward using the velocities from Lambert"
    ### Using ephemeris
    # Lambert trajectory obtain terminal velocity vectors
    SF = CONFIG.SimsFlan_config()

    # Create bodies
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

    # Calculate trajectory of the bodies based on ephem
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')
    
    decv, l = findValidLambert()
    print(decv)

    r_M, v_M = marsephem.eph(decv[6] + AL_BF.sec2days(decv[7]) )

    Fit = Propagate(Nimp = SF.Nimp)
    Fit.prop(decv, plot = True)
    print("Final", Fit.rv_final)
    print("Theoretical final", r_M,  AL_BF.convert3dvector(decv[3:6], "polar")+v_M)

def test_Exposin():

    # Verify with example
    print("VERIFICATION")
    r1 = 1
    r2 =  5
    psi = np.pi/2
    k2 = 1/4

    # eSin = AL_TR.shapingMethod(sun.mu / Cts.AU_m**3)
    # gammaOptim_v = eSin.start(r1, r2, psi, 365*1.5, k2) # verified
    # Ni = 1
    # print(gammaOptim_v)
    # eSin.plot_sphere(r1, r2, psi, gammaOptim_v[Ni], Ni) # verified

    # Real life case
    sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
    earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
    mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

    # Calculate trajectory of the bodies based on ephem
    earthephem = pk.planet.jpl_lp('earth')
    marsephem = pk.planet.jpl_lp('mars')
    
    date0 = np.array([27,1,2018,0])
    t0 = AL_Eph.DateConv(date0,'calendar') #To JD
    t_t = 250

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
    
    k2 = 1/4

    eSin = AL_TR.shapingMethod(sun.mu / Cts.AU_m**3)
    gammaOptim_v = eSin.start(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi, t_t, k2)
    
    Ni = 1
    eSin.plot_sphere(r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi, gammaOptim_v[Ni], Ni)
    v1, v2, a_T = eSin.calculateVel(Ni, gammaOptim_v[Ni], r_1_norm / Cts.AU_m, r_2_norm / Cts.AU_m, psi)
    print('a', a_T*Cts.AU_m)

    print("body coord", v1, v2)

    # print(v1[0]*Cts.AU_m, v1[1], v2[0]*Cts.AU_m, v2[1])

    # To heliocentric coordinates (2d approximation)
    def to_helioc(r, vx, vy):
        v_body = np.array([vx, vy, 0])
        # Convert to heliocentric
        angle = np.arctan2(r[1], r[0])
        v_h = AL_BF.rot_matrix(v_body, angle, 'z')

        return v_h

    v_1 = to_helioc(r_1, v1[0]*Cts.AU_m, v1[1]*Cts.AU_m)
    v_2 = to_helioc(r_2, v2[0]*Cts.AU_m, v2[1]*Cts.AU_m) # r1 was assumed to be in y = 0

    print(v_1, v_2)
    # def to_helioc(r, v, gamma):
    #     v_body = np.array([v*np.sin(gamma), \
    #                        v*np.cos(gamma),\
    #                        0])

    #     # Convert to heliocentric
    #     angle = np.arctan2(r[1], r[0])
    #     v_h = AL_BF.rot_matrix(v_body, angle, 'z')

    #     return v_h
    # v_1 = to_helioc(r_1, v1[0]*Cts.AU_m, v1[1])
    # v_2 = to_helioc(r_2, v2[0]*Cts.AU_m, v2[1])
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
    
    deltav_i = t_t / (SF.Nimp+1) * a_T*Cts.AU_m
    for i in range(SF.Nimp):
        decv[8+3*i] = deltav_i
        decv[8+3*i+1] = 0
        decv[8+3*i+2] = 0

    Fit.calculateFeasibility(decv, plot = True, thrust = 'tangential')
    Fit.printResult()

if __name__ == "__main__":
    # pykepephem()
    # changeReferenceFrame()
    # test_convertAngleForm()
    #  test_convertRange()
    # CartKeplr()
    # propagateHohmann()
    # Lambert()
    # propagateLambert()
    # print("Universal propagation")
    # propagateUniversalLambert()
    # propagateSimsFlanagan()
    # propagateSimsFlanaganForward()
    test_Exposin()