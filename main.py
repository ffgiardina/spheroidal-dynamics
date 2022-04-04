import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.integrate import odeint
from subroutines import *

# Parameters
para = dict()
n = 100  # number of bacteria
a = 1  # major axis a of ellipsoid
b = 0.5  # minor axis a of ellipsoid
m = 1.0  # mass of bacteria
d = 0.5  # damping coefficient
p = 0.1  # propulsive force
r_b = 0.1  # particle radius
f_r = lambda d: 0.01/d**2  # repulsive force function

para['n'] = n; para['m'] = 1.0; para['a'] = a; para['b'] = b; para['d'] = d; para['p'] = p
para['r_b'] = r_b; para['f_r'] = f_r

# Initial conditions
np.random.seed(1)
phi0 = np.random.rand(n)*np.pi-np.pi/2
theta0 = np.random.rand(n)*np.pi
dphi0 = np.random.rand(n)-1/2
dtheta0 = np.random.rand(n)-1/2
x0 = np.concatenate((phi0, theta0, dphi0, dtheta0), axis=0)

# Solve problem
tend = 100  # final time
N = 1001  # number of solution time points
t = np.linspace(0, tend, N)
sol = odeint(dynamics, x0, t, args=(para,))

# Convert solution to cartesian coordinates
r = ellipsoid2cartesian(sol, para)

# Compute particle velocities
# dr = (r[2:,:,:] - r[1:-1,:,:])/(tend/N)
# v = np.sqrt(np.sum(dr*dr,axis=2))

# Animate results
fig = plt.figure(1)
ax = plt.axes(projection='3d')
plt.axis('off')

u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
x = a*np.cos(u)*np.sin(v)
y = a*np.sin(u)*np.sin(v)
z = b*np.cos(v)

ax.plot_surface(x, y, z, cmap=cm.Spectral, alpha=0.5, cstride=1, rstride=1, linewidth=0, antialiased=True)
ax.set_box_aspect([1, 1, 1])
set_axes_equal(ax)

points, = ax.plot(r[0, :, 0], r[0, :,  1], r[0, :,  2], marker="o", c='black', linestyle = 'None',)
line, = ax.plot(r[0, 1, 0], r[0, 1, 1], r[0, 1, 2], c='black', linewidth=0.2)

# Updating function, to be repeatedly called by the animation
def update(it):
    # obtain point coordinates
    r0 = r[it,:,:]
    # set point's coordinates
    points.set_data(r0[:, 0],r0[:, 1])
    points.set_3d_properties(r0[:, 2],'z')

    # Rotate plot
    ax.view_init(elev=90, azim=it/N*0*360)

    line.set_data(r[0:it,0,0],r[0:it,0,1])
    line.set_3d_properties(r[0:it,0,2], 'z')

    #return points, line
    return points, line, ax

repspeed = 10
anim = FuncAnimation(fig, update, interval=1000.0*tend/N/repspeed, blit=False, repeat=False,
                    frames=N)

plt.show()

#anim.save('dynamics.mp4', writer = 'ffmpeg', fps = 30)
