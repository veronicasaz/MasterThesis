import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

import pykep as pk

import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_2BP as AL_2BP
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_Plots as AL_Plot
from AstroLibraries import AstroLib_Trajectories as AL_TR


# # In[]
# ##############################################
# ############# VALIDATION #####################
# ##############################################
Cts = AL_BF.ConstantsBook()

# # In[]
# ##############################################
# # Change frame of reference
# ##############################################
# # From Howard Curtis: pg218
# # For a given earth orbit, the elements are h 1⁄4 80,000 km 2 /s, e 1⁄4 1.4, i 1⁄4 30  , U 1⁄4 40  , u 1⁄4 60  , and q 1⁄4 30  .
# # Using Algorithm 4.5, find the state vectors r and v in the geocentric equatorial frame.
# h = 80000 * 1e6 #m^2/s
# e = 1.4
# i = AL_BF.deg2rad(30)
# RAAN = AL_BF.deg2rad(40)
# omega = AL_BF.deg2rad(60)
# theta = AL_BF.deg2rad(30)

# # rv in perifocal
# r_norm = h**2/Cts.mu_E_m / (1+e*np.cos(theta))
# r = r_norm* np.array([np.cos(theta), np.sin(theta),0])
# v = Cts.mu_E_m / h * np.array([-np.sin(theta), e+ np.cos(theta),0])
# print(r, v) # Validated 

# # r, v in geocentric
# Frame1 = AL_Eph.FramesOfReference([omega, i, RAAN], 'perif')
# r2 = Frame1.transform(r, 'helioc')
# v2 = Frame1.transform(v, 'helioc')
# print(r2, v2) # Validated

# # Back to perifocal
# r3 = Frame1.transform(r2, 'perif')
# v3 = Frame1.transform(v2, 'perif')
# print(r3, v3) # Validated: same as initial

# # In[]
# ##############################################
# # Conversion Cartesian-Keplerian and vice-versa
# ##############################################
# # From Mission Geometry slides
# sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
# earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)

# x = np.array([8751268.4691, -7041314.6869, 4846546.9938, \
#                 332.2601039, -2977.0815768,-4869.8462227])
# x2 = np.array([-2700816.139,-3314092.801, 5266346.4207,\
#                 5168.606551,-5597.546615,-868.8784452])
# x3 = np.array([10157768.1264, -6475997.0091, 2421205.9518,\
#             1099.2953996, 3455.1059240, 4355.0978095])

# K = np.array([12158817.9615, 0.014074320051, 52.666016957,\
#                 323.089150643, 148.382589129, 112.192638384])

# K2 = np.array([12269687.5912,  0.004932091570, 109.823277603,\
#             134.625563565, 106.380426142, 301.149932402])

# print("CARTESIAN-KEPLERIAN")

# trajectory1 = AL_2BP.BodyOrbit(x, 'Cartesian', earth )
# print("Result vector 1")
# print(trajectory1.KeplerElem_deg) # Validated

# trajectory12 = AL_2BP.BodyOrbit(x2, 'Cartesian', earth)
# print("Result vector 2")
# print(trajectory12.KeplerElem_deg) # Validated

# trajectory13 = AL_2BP.BodyOrbit(x3, 'Cartesian', earth)
# print("Result vector 3")
# print(trajectory13.KeplerElem_deg) # Validated

# print("KEPLERIAN-CARTESIAN")

# trajectory2 = AL_2BP.BodyOrbit(K, 'Keplerian', earth, unitsI = 'deg')
# print("Result vector 1")
# print(trajectory2.r, trajectory2.v ) # Validated

# trajectory22 = AL_2BP.BodyOrbit(trajectory12.KeplerElem_deg, 'Keplerian', earth, unitsI = 'deg')
# print("Result vector 2")
# print(trajectory22.r, trajectory22.v ) # Validated

# trajectory23 = AL_2BP.BodyOrbit(K2, 'Keplerian', earth, unitsI = 'deg')
# print("Result vector 3")
# print(trajectory23.r, trajectory23.v ) # Validated

# ##############################################
# # Propagation
# ##############################################
# # with a hohmann orbit the necessary parameters are found, propagating 
# # we should get to the planet
# # ephemeris retrieval
# date0 = np.array([27,1,2016,0])
# t_0 = AL_Eph.DateConv(date0,'calendar') #To JD
# transfertime = 258.8493

# # Create bodies
# sun = AL_2BP.Body('sun', 'yellow', mu = Cts.mu_S_m)
# earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
# mars = AL_2BP.Body('mars', 'red', mu = Cts.mu_M_m)

# # Calculate trajectory of the bodies based on ephem
# earthephem = pk.planet.jpl_lp('earth')
# marsephem = pk.planet.jpl_lp('mars')
# r_E, v_E = earthephem.eph(t_0.JD_0)
# r_M, v_M = marsephem.eph(t_0.JD_0 + transfertime)

# orbit_E = AL_2BP.BodyOrbit(np.append(r_E, v_E), 'Cartesian', sun)
# earth.addOrbit(orbit_E)
# orbit_M = AL_2BP.BodyOrbit(np.append(r_M, v_M), 'Cartesian', sun)
# mars.addOrbit(orbit_M)

# # Create transfer in the first moment
# transfer1 = AL_TR.SystemBodies([earth, mars])

# t, r_E, v_E, r_M, v_M = transfer1.findDateWith180deg(date0, transfertime, [earthephem, marsephem],usingCircular=False)
# date_1 = AL_Eph.DateConv(t,'JD_0') #To JD
# transfer1.HohmannTransfer(r_0 = r_E, r_1 = r_M)

# # Propagate
# r0 = np.array(r_E)
# v0 = transfer1.vp_H * np.array(v_E) / np.linalg.norm(v_E)

# orbit_sp = AL_2BP.BodyOrbit(np.append(r0,v0), 'Cartesian', sun)
# x = orbit_sp.Propagation(transfer1.timeflight_H, 'Cartesian') # Coordinates after propagation

# # Compare with supposed solutions
# print("HOHMANN")
# vf = transfer1.va_H
# rf = np.array(r_M)
# print(rf)
# print("PROPAGATION")
# x = np.array(x)
# print(x)

# fig = plt.figure()
# ax = fig.gca(projection = '3d')

# ax.scatter(0,0,0, color = 'yellow', s = 100)
# ax.scatter(r0[0],r0[1],r0[2], color = 'blue')
# ax.scatter(rf[0],rf[1],rf[2], color = 'red')
# ax.scatter(x[0], x[1], x[2], color = 'green')

# AL_Plot.set_axes_equal(ax)
# plt.show() # Validated: more or less. Points are close to each other

# # orbit_sp.Plot(transfer1.KeplerElem, r0, 1, 'green','spacecraft', 10)

# ##############################################
# # Lambert    
# ##############################################
# #### Howard Curtis page 254
# earth = AL_2BP.Body('earth', 'blue', mu = Cts.mu_E_m)
# r_1 = np.array([5000, 10000, 2100]) * 1000
# r_2 = np.array([-14600, 2500, 7000]) * 1000

# lambert = AL_TR.Lambert(r_1, r_2, 3600, earth.mu)
# v1, v2 = lambert.TerminalVelVect()
# print(v1, v2) #Validated


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

fig = plt.figure()
ax = fig.gca(projection = '3d')

ax.scatter(0,0,0, color = 'yellow', s = 100)
ax.scatter(r0[0],r0[1],r0[2], color = 'blue')
ax.scatter(rf[0],rf[1],rf[2], color = 'red')
ax.scatter(x[0], x[1], x[2], color = 'green')

AL_Plot.set_axes_equal(ax)
plt.show() # Validated
