import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np


def rot(angle, axis):
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

o =np.array([0,0,0])
x_g = np.array([10,0,0])
y_g = np.array([0,10,0])
z_g = np.array([0,0,10])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

omega = 0
i = 98.13*3.14/180
RAAN = 122.61*3.14/180
h = 6

# ax.scatter(0,1,0)
# ax.scatter(0,0,1)


R3_omega = rot(omega, 3)
R1_i = rot(i, 1)
R3_RAAN = rot(RAAN, 3)
R3_2 = rot(-np.pi/2, 3)
R2_2 = rot(-np.pi/2, 2)

rotMatrix = np.transpose( np.dot(np.dot(np.dot(np.dot(R3_2, R2_2),R3_omega),R1_i),R3_RAAN) )

xrot = np.dot(rotMatrix, x_g) 
yrot = np.dot(rotMatrix, y_g)
zrot = np.dot(rotMatrix, z_g)

ax.plot3D([0,x_g[0]],[0,x_g[1]],[0,x_g[2]],'r--')
ax.plot3D([0,y_g[0]],[0,y_g[1]],[0,y_g[2]],'g--')
ax.plot3D([0,z_g[0]],[0,z_g[1]],[0,z_g[2]],'k--')

ax.plot3D([0,xrot[0]],[0,xrot[1]],[0,xrot[2]],'r')
ax.plot3D([0,yrot[0]],[0,yrot[1]],[0,yrot[2]],'g')
ax.plot3D([0,zrot[0]],[0,zrot[1]],[0,zrot[2]],'k')


plt.show()

