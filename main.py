import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.integrate import odeint
from subroutines import *

# Parameters
para = dict()
a = 1  # major axis a of ellipsoid
b = 0.5  # minor axis a of ellipsoid
m = 1.0  # mass of bacteria

para['m'] = 1.0; para['a'] = a; para['b'] = b

# Initial conditions
x0 = [0, np.pi/2, 1, 2]  # phi, theta, dphi, dtheta

# Solve problem
tend = 100  # final time
N = 1001  # number of solution time points
t = np.linspace(0, tend, N)
sol = odeint(dynamics, x0, t, args=(para,))

# Convert solution to cartesian coordinates
r = ellipsoid2cartesian(sol, para)

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

points, = ax.plot(r[0, 0], r[0, 1], r[0, 2], marker="o", c='black')
line, = ax.plot(r[0, 0], r[0, 1], r[0, 2], c='black')

# Updating function, to be repeatedly called by the animation
def update(it):
    # obtain point coordinates
    r0 = r[it,:]
    # set point's coordinates
    points.set_data([r0[0]],[r0[1]])
    points.set_3d_properties([r0[2]], 'z')

    # Rotate plot
    ax.view_init(azim=it/N*5*360)

    line.set_data(r[0:it,0],r[0:it,1])
    line.set_3d_properties(r[0:it,2], 'z')

    #return points, line
    return points, line, ax

repspeed = 20
anim = FuncAnimation(fig, update, interval=1000.0*tend/N/repspeed, blit=False, repeat=False,
                    frames=N)

#ax.plot(r[0:, 0],r[0:, 1],r[0:, 2], c='black')
plt.show()

#anim.save('dynamics.mp4', writer = 'ffmpeg', fps = 30)
