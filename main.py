from scipy.integrate import odeint
from subroutines import *
from tqdm import tqdm

# Parameters (all SI units)
n = 20  # number of particles
a = 1e-5  # major axes a of spheroid
b = a*0.95  # minor axis a of spheroid
m = 1e-15  # mass of particle
relax = 1e-10  # relax stiffness to improve numerical integration speed
E = relax * 3e5  # elastic modulus (for particle collisions)
l_a = 3e-6  # particle major axis length
l_b = 1e-6  # particle minor axis length
f_r = lambda d: np.sqrt(8/9*np.abs(d-2*l_b)**3*E**2*l_b)*(d-2*l_b<0)  # sphere-sphere Hertzian repulsion
mu = 1e-3  # dynamic viscosity of water at room temperature
v_b = 1e-5  # particle speed
d = 6*np.pi*mu*l_a*v_b  # damping coefficient (Stoke's law)
d_rot = 8*np.pi*mu*l_a**3 * relax * 1e4  # rotational damping coefficient for a sphere (Stoke's law. see Happel and Brenner p.173)
p = v_b*d  # propulsive force

# pack up all parameters
para = {'n': n, 'm': m, 'a': a, 'b': b, 'd': d, 'd_rot': d_rot, 'p': p, 'l_a': l_a, 'l_b': l_b,
        'f_r': f_r}

# Initial conditions
np.random.seed(1)
phi0 = np.random.rand(n)*2*np.pi-np.pi
theta0 = np.random.rand(n)*np.pi
psi0 = np.random.rand(n)*2*np.pi  # polar orientation in local coordinates of tangent plane
x0 = np.concatenate((phi0, theta0, psi0), axis=0)
x0 = separate_all(x0, 1e3, para)  # separate particles to avoid initial overlaps

# Solve problem
tend = 10  # final time
N = 1001  # number of solution time points
t = np.linspace(0, tend, N)

s_e = 4*np.pi*(((a*a)**1.6 + 2*(a*b)**1.6)/3)**(1/1.6)  # spheroid surface area
print(f'Running simulation at packing fraction {np.round(2*n*l_b**2*np.pi/s_e,2)}')  # display packing fraction

with tqdm(total=100, unit="‰") as pbar:
    sol = odeint(dynamics, x0, t, args=(para, pbar, [0, tend/100]))

# Convert solution to Cartesian coordinates
r = ellipsoid2cartesian(sol, para)

# Save results
para['N'] = N; para['tend'] = tend; del para['f_r']
np.savez('results.npz', solution=sol, trajectories=r, parameters=para)

# Animation
from animation import animate
animate(False)

