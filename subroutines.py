import numpy as np
import matplotlib.pyplot as plt

#  Equations of motion
def dynamics(x, t, para):
    # unpack states and parameters
    n, a, b, m, d, p = [para[i] for i in ['n', 'a', 'b', 'm', 'd', 'p']]
    phi, theta, dphi, dtheta = [x[i*n:(i+1)*n] for i in range(4)]


    # Jacobian matrix
    J = np.array([[-a * np.sin(theta) * np.sin(phi), a * np.cos(theta) * np.cos(phi)],
                   [a * np.cos(phi) * np.sin(theta), a * np.cos(theta) * np.sin(phi)],
                   [np.zeros(n), -b * np.sin(theta)]]).transpose((2,0,1))

    # metric tensor
    g = np.array([[(a*np.sin(theta))**2, np.zeros(n)], [np.zeros(n), 1.0/2.0*(a**2+b**2+(a**2-b**2)*np.cos(2*theta))]]).transpose((2,0,1))
    v = np.array([[dphi], [dtheta]]).transpose((2,0,1))  # velocity vector

    # compute damping forces
    f_d = -d*(g@v)

    # compute propulsive forces
    dr = J@v
    dr_norm = np.sqrt(np.sum(dr*dr,axis=1)).repeat(2,axis=1).reshape((n,2,1))
    f_p = p * (J.transpose((0,2,1))@dr)/(dr_norm+1e-16)

    # compute time derivatives
    f = f_p + f_d

    ddphi = f[:,0,:].T/((a*np.sin(theta))**2) - 2*np.cos(theta)/np.sin(theta)*dtheta*dphi
    ddtheta = (2*f[:,1,:].T + (a**2-b**2)*np.sin(2*theta)*dtheta**2 + a**2*np.sin(2*theta)*dphi**2) /\
              (a**2 + b**2 + (a**2-b**2)*np.cos(2*theta))

    dxdt = np.concatenate((dphi, dtheta, ddphi.flatten(), ddtheta.flatten()), axis=0)
    return dxdt


# Convert ellipsoidal coordinate solution to Cartesian coordinates
def ellipsoid2cartesian(sol, para):
    n, a, b = [para[i] for i in ['n', 'a', 'b']]
    phi, theta, _, _ = [sol[:, i * n:(i + 1) * n] for i in range(4)]
    N = sol.shape[0]
    r = np.zeros((N, n, 3))

    s = a * np.sin(theta) / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    c = b * np.cos(theta) / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    L = a * b / np.sqrt((a * c) ** 2 + (b * s) ** 2)
    r[:,:,0] = L * s * np.cos(phi)
    r[:,:,1] = L * s * np.sin(phi)
    r[:,:,2] = L * c
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