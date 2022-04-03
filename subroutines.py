import numpy as np
import matplotlib.pyplot as plt

#  Equations of motion
def dynamics(x, t, para):
    # unpack states and parameters
    phi, theta, dphi, dtheta = x
    a, b, m, d = [para[i] for i in ['a', 'b', 'm', 'd']]

    # metric tensor
    g = np.matrix([[(a*np.sin(theta))**2, 0], [0, 1.0/2.0*(a**2+b**2+(a**2-b**2)*np.cos(2*theta))]])

    # compute damping forces
    v = np.matrix([[dphi], [dtheta]])
    f_d = -d*np.dot(g,v)

    # compute time derivatives
    f = f_d
    ddphi = f[0]/((a*np.sin(theta))**2) - 2*np.cos(theta)/np.sin(theta)*dtheta*dphi
    ddtheta = (2*f[1] + (a**2-b**2)*np.sin(2*theta)*dtheta**2 + a**2*np.sin(2*theta)*dphi**2) /\
              (a**2 + b**2 + (a**2-b**2)*np.cos(2*theta))
    dxdt = [dphi, dtheta, ddphi, ddtheta]
    return dxdt


# Convert ellipsoidal coordinate solution to Cartesian coordinates
def ellipsoid2cartesian(sol, para):
    phi, theta, _, _ = [sol[:,i] for i in range(sol.shape[1])]
    a, b = [para[i] for i in ['a', 'b']]
    N = sol.shape[0]
    r = np.zeros((N,3))

    s = a * np.sin(theta) / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    c = b * np.cos(theta) / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    L = a * b / np.sqrt((a * c) ** 2 + (b * s) ** 2)
    r[:, 0] = L * s * np.cos(phi)
    r[:, 1] = L * s * np.sin(phi)
    r[:, 2] = L * c
    return r

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])