from scipy.integrate import odeint
from subroutines import *
from tqdm import tqdm

# Parameters
para = dict()
n = 20  # number of particles
a = 100e-6  # major axes a of spheroid
b = a*0.7  # minor axis a of spheroid
m = 1e-15  # mass of particle
E = 1e-11 * 3e5  # elastic modulus (for particle collisions)
r_b = 3e-6  # particle radius
f_r = lambda d: np.sqrt(8/9*np.abs(d-2*r_b)**3*E**2*r_b)*(d-2*r_b<0)  # sphere-sphere Hertzian repulsion
mu = 1e-3  # dynamic viscosity of water at room temperature
v_b = 10e-6  # particle speed
d = 6*np.pi*mu*r_b*v_b  # damping coefficient (Stoke's law)
p = v_b*d  # propulsive force
l_c = (9/8*p**2/(E**2*r_b))**(1/3)  # contact length scale
tau_c = 0.1 * l_c/v_b  # contact time scale

# pack up all parameters
para['n'] = n; para['m'] = m; para['a'] = a; para['b'] = b; para['d'] = d; para['p'] = p
para['r_b'] = r_b; para['f_r'] = f_r; para['tau_c'] = tau_c

# Initial conditions
np.random.seed(3)
phi0 = np.random.rand(n)*2*np.pi-np.pi
theta0 = np.random.rand(n)*np.pi
psi0 = np.random.rand(n)*2*np.pi  # polar orientation in local coordinates of tangent plane
x0 = np.concatenate((phi0, theta0, psi0), axis=0)
x0 = separate_all(x0, para)  # separate particles to avoid overlaps

# Solve problem
tend = 20  # final time
N = 1001  # number of solution time points
t = np.linspace(0, tend, N)

s_e = 4*np.pi*(((a*a)**1.6 + 2*(a*b)**1.6)/3)**(1/1.6)  # spheroid surface area
print(f'Running simulation at packing fraction {np.round(n*r_b**2*np.pi/s_e,2)}')  # display packing fraction
with tqdm(total=100, unit="‰") as pbar:
    sol = odeint(dynamics, x0, t, args=(para, pbar, [0, tend/100]))

# Convert solution to Cartesian coordinates
r = ellipsoid2cartesian(sol, para)

# Save results
para_tmp = para.copy(); del para_tmp['f_r']; para_tmp['N'] = N; para_tmp['tend'] = tend
np.savez('results.npz', solution=sol, trajectories=r, parameters=para_tmp)

# Compute total spin of system along z-axis
drdt = (r[1:,:,:] - r[:-1,:,:])/(tend/N)  # velocity vectors
rot = np.zeros(N-1)
ez = np.array([[0],[0],[1]])  # z-axis
for i in range(N-1):
    rot_all = np.array([r[i,:,1]*drdt[i,:,2] - r[i,:,2]*drdt[i,:,1],
                        r[i,:,2]*drdt[i,:,0] - r[i,:,0]*drdt[i,:,2],
                        r[i,:,0]*drdt[i,:,1] - r[i,:,1]*drdt[i,:,0]])
    rot[i] = np.dot(ez.T,np.sum(rot_all,axis=1))

plt.figure(0)
plt.plot(rot)
plt.xlabel('t')
plt.ylabel('Total spin in z')

# Animation
from animation import animate
animate(False)

