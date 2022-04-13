import numpy as np
import matplotlib.pyplot as plt

#  Equations of motion
def dynamics(x, t, para, pbar, state):
    # unpack states and parameters
    n, a, b, d, p, tau_c = [para[i] for i in ['n', 'a', 'b', 'd', 'p', 'tau_c']]
    phi, theta, psi = [x[i * n:(i + 1) * n] for i in range(3)]


    # Jacobian matrix
    J = np.array([[-a * np.sin(theta) * np.sin(phi), a * np.cos(theta) * np.cos(phi)],
                   [a * np.cos(phi) * np.sin(theta), a * np.cos(theta) * np.sin(phi)],
                   [np.zeros(n), -b * np.sin(theta)]]).transpose((2,0,1))

    Jn0 = np.sqrt(a**2*np.sin(theta)**2)  # norm of first column vector in J
    Jn1 = np.sqrt(1/2*(a**2 + b**2 + (a**2-b**2)*np.cos(2*theta)))  # norm of second column vector in J

    # compute propulsive forces
    wphi = np.sin(psi) / Jn0
    wtheta = np.cos(psi) / Jn1
    v = J@np.array([[wphi], [wtheta]]).transpose((2,0,1))  # propulsive vector
    v_norm = np.sqrt(np.sum(v*v,axis=1)).repeat(2,axis=1).reshape((n,2,1))
    f_p = p * (J.transpose((0,2,1))@v)/v_norm

    # compute interaction forces
    f_i = repulsion(x, para, J)

    # compute rotation of psi due to contact forces
    f_i_on = 1*(np.sum(np.abs(f_i),axis=1)>0)  # check for collision
    alpha = np.arctan2(f_i[:,0,:].flatten()*Jn0,f_i[:,1,:].flatten()*Jn1)  # collision force angle
    mod_alpha = np.mod(alpha,2*np.pi)
    mod_psi = np.mod(psi, 2 * np.pi)
    da = mod_psi-mod_alpha
    mod_psi = mod_psi + 2*np.pi*(da<-np.pi)
    mod_alpha = mod_alpha + 2*np.pi*(da>np.pi)
    dpsi_i = f_i_on.flatten() * (mod_alpha-mod_psi) / (tau_c * np.pi)  # angular speed due to collision

    # compute time derivatives
    f = f_p + f_i
    dphi = f[:,0,:].T/(d*(a*np.sin(theta))**2)
    dtheta = (2*f[:,1,:].T) / (d*(a**2 + b**2 + (a**2-b**2)*np.cos(2*theta)))
    dpsi = -np.cos(theta)/np.sin(theta)*dphi*Jn0/Jn1 + dpsi_i
    dxdt = np.concatenate((dphi.flatten(), dtheta.flatten(), dpsi.flatten()), axis=0)

    # update progress bar
    last_t, dt = state
    ni = int((t - last_t) / dt)
    pbar.update(ni)
    state[0] = last_t + dt * ni

    return dxdt


# compute repulsive forces between particles as given by the function f_r
def repulsion(x, para, J):
    n, a, b, r_b, f_r= [para[i] for i in ['n', 'a', 'b', 'r_b', 'f_r']]
    phi, theta = [x[i * n:(i + 1) * n] for i in range(2)]

    # compute Euclidean distance
    r = ellipsoid2cartesian(np.array([x]), para)[0,:,:]
    drx = r[:,0:1]-r[:,0:1].T; dry = r[:,1:2]-r[:,1:2].T; drz = r[:,2:3]-r[:,2:3].T
    dist = np.sqrt(drx**2 + dry**2 + drz**2) + a*100*np.eye(n)

    # compute inter-particle forces
    f_norm = -f_r(dist)

    # compute force vectors
    fx = np.sum(f_norm*drx/dist,axis=0); fy = np.sum(f_norm*dry/dist,axis=0); fz = np.sum(f_norm*drz/dist,axis=0)
    f_c = np.array([[fx, fy, fz]]).T

    # Project to ellipsoid and convert to ellipsoidal coordinate system
    f_e = J.transpose((0,2,1))@f_c

    return f_e

# Convert ellipsoidal coordinate solution to Cartesian coordinates
def ellipsoid2cartesian(sol, para):
    n, a, b = [para[i] for i in ['n', 'a', 'b']]
    phi, theta = [sol[:, i * n:(i + 1) * n] for i in range(2)]
    N = sol.shape[0]
    r = np.zeros((N, n, 3))

    s = a * np.sin(theta) / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    c = b * np.cos(theta) / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    L = a * b / np.sqrt((a * c) ** 2 + (b * s) ** 2)
    r[:,:,0] = L * s * np.cos(phi)
    r[:,:,1] = L * s * np.sin(phi)
    r[:,:,2] = L * c
    return r


# Maximally separate particles on given ellipsoid
def separate_all(x, para):
    n, a, b = [para[i] for i in ['n', 'a', 'b']]
    phi, theta, psi = [x[i*n:(i+1)*n] for i in range(3)]

    para_tmp = para.copy()
    para_tmp['f_r'] = lambda d: 1e3*a**2/n/d**2

    N = 1000
    for i in range(N):
        # Jacobian matrix
        J = np.array([[-a * np.sin(theta) * np.sin(phi), a * np.cos(theta) * np.cos(phi)],
                      [a * np.cos(phi) * np.sin(theta), a * np.cos(theta) * np.sin(phi)],
                      [np.zeros(n), -b * np.sin(theta)]]).transpose((2, 0, 1))

        # take a separation step
        s = repulsion(x, para_tmp, J)
        phi = (phi + s[:,0,:].T).flatten()
        theta = (theta + s[:,1,:].T).flatten()
        x = np.concatenate((phi, theta, psi), axis=0)

    return x

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

# compute particle shape on sphere for animation
# ne is the number of vertices that form the particle outline
def create_ellipse(r, q ,ne, para):
    a, b, r_b, n = [para.item()[i] for i in ['a', 'b', 'r_b', 'n']]
    phi, theta, psi = [q[i] for i in range(3)]
    J1 = np.array([[-a * np.sin(theta) * np.sin(phi)], [a * np.cos(phi) * np.sin(theta)], [0*theta]])/np.sqrt(a**2*np.sin(theta)**2)  # norm of first column vector in J
    J2 = np.array([[ a * np.cos(theta) * np.cos(phi)], [a * np.cos(theta) * np.sin(phi)], [-b * np.sin(theta)]])/np.sqrt(1/2*(a**2 + b**2 + (a**2-b**2)*np.cos(2*theta)))  # norm of second column vector in J

    s = np.linspace(2*np.pi, 0, ne)
    poly = r + (r_b*(J2*np.cos(s) + J1*np.sin(s))).T

    return poly