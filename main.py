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
f_r = lambda d: 0.02/d**2  # repulsive force function

para['n'] = n; para['m'] = 1.0; para['a'] = a; para['b'] = b; para['d'] = d; para['p'] = p
para['r_b'] = r_b; para['f_r'] = f_r

# Initial conditions
np.random.seed(1)
phi0 = np.random.rand(n)*np.pi-np.pi/2
theta0 = np.random.rand(n)*np.pi
dphi0 = np.random.rand(n)-1/2
dtheta0 = np.random.rand(n)-1/2
x0 = np.concatenate((phi0, theta0, dphi0, dtheta0), axis=0)

x0 = separate_all(x0, para)

# Solve problem
tend = 20  # final time
N = 1001  # number of solution time points
t = np.linspace(0, tend, N)
sol = odeint(dynamics, x0, t, args=(para,))

# Convert solution to cartesian coordinates
r = ellipsoid2cartesian(sol, para)

# Save results
p = para.copy(); del p['f_r']; p['N'] = N; p['tend'] = tend
np.savez('results.npz', trajectories=r, parameters=p)

# Compute particle velocities
# dr = (r[2:,:,:] - r[1:-1,:,:])/(tend/N)
# v = np.sqrt(np.sum(dr*dr,axis=2))

# Animation
from animation import animate
animate(video=False)

