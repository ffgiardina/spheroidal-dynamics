import numpy as np
import matplotlib.pyplot as plt

#  Equations of motion
def dynamics(x, t, para, pbar, state):
    # unpack states and parameters
    n, a, b, d, d_rot, p = [para[i] for i in ['n', 'a', 'b', 'd', 'd_rot', 'p']]
    phi, theta, psi = [x[i * n:(i + 1) * n] for i in range(3)]

    # compute cached variables
    cache = compute_cached_vars(x, para)
    jacobian, jac_norm_0, jac_norm_1 = [cache[i] for i in ['J', 'Jn0', 'Jn1']]

    # compute propulsive forces
    w_phi = np.sin(psi) / jac_norm_0  # polar particle orientation. Component of local coordinate vector y on manifold
    w_theta = np.cos(psi) / jac_norm_1  # polar particle orientation. Component of local coordinate vector x on manifold
    v = jacobian@np.array([[w_phi], [w_theta]]).transpose((2,0,1))  # propulsive vector
    v_norm = np.sqrt(np.sum(v*v,axis=1)).repeat(2,axis=1).reshape((n,2,1))
    f_p = p * (jacobian.transpose((0,2,1))@v)/v_norm

    # compute interaction forces f_i and torques t_i
    f_i, t_i = repulsion(x, para, cache)

    # compute right-hand side of dynamical system
    f = f_p + f_i
    d_phi = f[:,0,:].T/(d*(a*np.sin(theta))**2)
    d_theta = (2*f[:,1,:].T) / (d*(a**2 + b**2 + (a**2-b**2)*np.cos(2*theta)))
    d_psi = -np.cos(theta)/np.sin(theta)*d_phi*jac_norm_0/jac_norm_1 + t_i/d_rot
    dx_dt = np.concatenate((d_phi.flatten(), d_theta.flatten(), d_psi.flatten()), axis=0)

    # update progress bar
    last_t, dt = state
    ni = int((t - last_t) / dt)
    pbar.update(ni)
    state[0] = last_t + dt * ni

    return dx_dt


# compute variables to be reused and store them in dictionary
def compute_cached_vars(x, para):
    n, a, b = [para[i] for i in ['n', 'a', 'b']]
    phi, theta, psi = [x[i * n:(i + 1) * n] for i in range(3)]

    # Jacobian matrix
    jacobian = np.array([[-a * np.sin(theta) * np.sin(phi), a * np.cos(theta) * np.cos(phi)],
                  [a * np.cos(phi) * np.sin(theta), a * np.cos(theta) * np.sin(phi)],
                  [np.zeros(n), -b * np.sin(theta)]]).transpose((2, 0, 1))

    # norms of first and second column vector in J
    jac_norm_0 = np.sqrt(a ** 2 * np.sin(theta) ** 2)
    jac_norm_1 = np.sqrt(1 / 2 * (a ** 2 + b ** 2 + (a ** 2 - b ** 2) * np.cos(2 * theta)))

    return {'J': jacobian, 'Jn0': jac_norm_0, 'Jn1': jac_norm_1}


# compute repulsive forces and torques between particles as given by the function f_r
def repulsion(x, para, cache):
    # unpack all parameters and variables
    n, a, b, l_a, l_b, f_r= [para[i] for i in ['n', 'a', 'b', 'l_a', 'l_b', 'f_r']]
    phi, theta, psi = [x[i * n:(i + 1) * n] for i in range(3)]
    jacobian, jac_norm_0, jac_norm_1 = [cache[i] for i in ['J', 'Jn0', 'Jn1']]

    # compute Euclidean distance
    r = ellipsoid2cartesian(np.array([x]), para)[0,:,:]

    # normalize Jacobian vectors
    jac_0 = jacobian[:, :, 0] / np.array([jac_norm_0]).T
    jac_1 = jacobian[:, :, 1] / np.array([jac_norm_1]).T

    # front and rear centers of particle spheres
    r_front = r+(l_a-l_b)/2*(jac_1*np.array([np.cos(psi)]).T + jac_0*np.array([np.sin(psi)]).T)
    r_rear = r-(l_a-l_b)/2*(jac_1*np.array([np.cos(psi)]).T + jac_0*np.array([np.sin(psi)]).T)

    r_all = np.concatenate((r_front, r_rear), axis=0)

    # all sphere center distances in a 2n x 2n matrix
    drx = r_all[:, 0:1] - r_all[:, 0:1].T; dry = r_all[:, 1:2] - r_all[:, 1:2].T; drz = r_all[:, 2:3] - r_all[:, 2:3].T

    # compute distances and make sure to mask diagonal entries (distance between sphere and itself is 0)
    mask = np.block([[np.eye(n), np.eye(n)],[np.eye(n), np.eye(n)]])
    dist = np.sqrt(drx ** 2 + dry ** 2 + drz ** 2) + a * 100 * mask

    # compute inter-particle forces
    f_norm = f_r(dist)

    # compute force vectors
    fx_m = f_norm * drx / dist;fy_m = f_norm * dry / dist;fz_m = f_norm * drz / dist
    fx = np.sum(fx_m, axis=1);fy = np.sum(fy_m, axis=1);fz = np.sum(fz_m, axis=1)
    f_c = np.array([[fx[0:n]+fx[n:], fy[0:n]+fy[n:], fz[0:n]+fz[n:]]]).T

    # compute torques
    normal = np.array([jac_1[:, 1] * jac_0[:, 2] - jac_1[:, 2] * jac_0[:, 1],
                       jac_1[:, 2] * jac_0[:, 0] - jac_1[:, 0] * jac_0[:, 2],
                       jac_1[:, 0] * jac_0[:, 1] - jac_1[:, 1] * jac_0[:, 0]])  # surface normals at particle positions
    r_cx = r_all[:,0:1].T - drx / 2; r_cy = r_all[:,1:2].T - dry / 2; r_cz = r_all[:,2:3].T - drz / 2  # force contact points
    r_cmx = np.concatenate((r[:, 0:1],r[:, 0:1]), axis=0) - r_cx # force contact point to particle vector
    r_cmy = np.concatenate((r[:, 1:2],r[:, 1:2]),axis=0) - r_cy
    r_cmz = np.concatenate((r[:, 2:3],r[:, 2:3]),axis=0) - r_cz
    t1 = np.array([np.sum(fy_m * r_cmz - fz_m * r_cmy, axis=1),
                   np.sum(fz_m * r_cmx - fx_m * r_cmz, axis=1),
                   np.sum(fx_m * r_cmy - fy_m * r_cmx, axis=1)])  # cross product of force vectors and contact vectors
    t_c = np.sum((t1[:,0:n]+t1[:,n:]) * normal, axis=0)

    # Project to ellipsoid and convert to ellipsoidal coordinate system
    f_e = jacobian.transpose((0,2,1))@f_c

    return f_e, t_c


# Convert ellipsoidal coordinates to Cartesian coordinates
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


# Separate particles (in N steps) on given ellipsoid using gradient descent
def separate_all(x, N, para):
    n, a, b, l_a, l_b = [para[i] for i in ['n', 'a', 'b', 'l_a', 'l_b']]
    phi, theta, psi = [x[i*n:(i+1)*n] for i in range(3)]

    # squared-distance force law
    f_r = lambda d: 1e3*a**2/n/d**2

    for i in range(int(N)):
        # Jacobian matrix
        cache = compute_cached_vars(x, para)
        jacobian = cache['J']

        # compute inter-particle distances
        r = ellipsoid2cartesian(np.array([x]), para)[0, :, :]
        drx = r[:, 0:1] - r[:, 0:1].T;dry = r[:, 1:2] - r[:, 1:2].T;drz = r[:, 2:3] - r[:, 2:3].T
        dist = np.sqrt(drx ** 2 + dry ** 2 + drz ** 2) + a * 100 * np.eye(n)

        # compute inter-particle gradients
        f_norm = -f_r(dist)

        # compute gradient vectors
        fx = np.sum(f_norm * drx / dist, axis=0);fy = np.sum(f_norm * dry / dist, axis=0);fz = np.sum(f_norm * drz / dist, axis=0)
        f_c = np.array([[fx, fy, fz]]).T

        # Project to ellipsoid and convert to ellipsoidal coordinate system
        f_e = jacobian.transpose((0, 2, 1)) @ f_c

        # take a separation step
        phi = (phi + f_e[:,0,:].T).flatten()
        theta = (theta + f_e[:,1,:].T).flatten()
        x = np.concatenate((phi, theta, psi), axis=0)

    return x


# compute the heading vectors of all particles
def get_particle_heading(x, para):
    a, b, l_a, l_b, n = [para[i] for i in ['a', 'b', 'l_a', 'l_b', 'n']]
    psi = np.array([x[2 * n:(2 + 1) * n]]).T

    cache = compute_cached_vars(x, para)
    jacobian, jac_norm_0, jac_norm_1 = [cache[i] for i in ['J', 'Jn0', 'Jn1']]

    jac_0 = jacobian[:, :, 0] / np.array([jac_norm_0]).T
    jac_1 = jacobian[:, :, 1] / np.array([jac_norm_1]).T

    return jac_1*np.cos(psi) + jac_0*np.sin(psi)

# compute the position of boundary points (for animation purposes)
def particle_boundary(r, q ,ne, para):
    a, b, l_a, l_b, n = [para[i] for i in ['a', 'b', 'l_a', 'l_b', 'n']]
    phi, theta, psi = [q[i] for i in range(3)]

    # norms of first and second column vector in J
    jac_0 = np.array([[-a * np.sin(theta) * np.sin(phi)], [a * np.cos(phi) * np.sin(theta)], [0 * theta]]) / np.sqrt(
        a ** 2 * np.sin(theta) ** 2)
    jac_1 = np.array(
        [[a * np.cos(theta) * np.cos(phi)], [a * np.cos(theta) * np.sin(phi)], [-b * np.sin(theta)]]) / np.sqrt(
        1 / 2 * (a ** 2 + b ** 2 + (a ** 2 - b ** 2) * np.cos(2 * theta)))

    # center points of spheres
    c1 = np.array([r]).T+(l_a-l_b)/2*(jac_1*np.cos(psi) + jac_0*np.sin(psi))
    c2 = np.array([r]).T-(l_a-l_b)/2*(jac_1*np.cos(psi) + jac_0*np.sin(psi))

    # compute particle boundary points
    s = np.linspace(0, 2 * np.pi, ne)
    poly1 = c1.T + l_b*(jac_1*(np.cos(s)*np.cos(psi) - np.sin(psi)*np.sin(s))).T + l_b*(jac_0*(np.cos(s)*np.sin(psi) + np.cos(psi)*np.sin(s))).T
    poly2 = c2.T + l_b*(jac_1*(np.cos(s)*np.cos(psi) - np.sin(psi)*np.sin(s))).T + l_b*(jac_0*(np.cos(s)*np.sin(psi) + np.cos(psi)*np.sin(s))).T

    return np.concatenate((poly1,poly2))


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
