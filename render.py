import numpy as np
import pyvista as pv
from subroutines import *

# Load data
data = np.load('results.npz', allow_pickle=True)
r = data['trajectories']
para = data['parameters']
sol = data['solution']

a = para.item()['a']; b = para.item()['b']; N = para.item()['N']; tend = para.item()['tend']
r_b = para.item()['r_b']; n = para.item()['n']

# set animation properties
pv.set_plot_theme("document")
plotter = pv.Plotter(window_size=([1024, 768]))

# axes arrows
arrowx = pv.Arrow((-a, 0, -b*1.1), (2*a, 0, 0), shaft_radius=0.01, tip_length=0.1, tip_radius=0.03, scale='auto')
arrowy = pv.Arrow((0, -a, -b*1.1), (0, 2*a, 0), shaft_radius=0.01, tip_length=0.1, tip_radius=0.03, scale='auto')
plotter.add_mesh(arrowx, color='k')
plotter.add_mesh(arrowy, color='k')

# ground plane
plane = pv.Polygon((0,0,-b*1.1), radius=a, normal=(0,0,1.0), n_sides=100)
plotter.add_mesh(plane, color='k', opacity=0.1)

# spheroid
spheroid = pv.ParametricEllipsoid(a,a,b)
plotter.add_mesh(spheroid, color='seashell', smooth_shading=True, opacity=0.85)

# particles
pdata = pv.PolyData(r[0,:,:])
pdata['orig_sphere'] = np.arange(n)
particle = pv.Sphere(radius=r_b, phi_resolution=20, theta_resolution=20)
pc = pdata.glyph(scale=False, geom=particle)
plotter.add_mesh(pc, cmap='Greens', smooth_shading=True, opacity=1, show_scalar_bar=False)

# set camera properties
plotter.camera.zoom(1.3)

# create animation and save as mp4
plotter.open_movie("dynamics.mp4")
length = 5  # in seconds
fps = 24  # frames per second
nrot = 0.5  # number of rotations of axes during animation
n_frames = int(np.floor(N/fps*length))
increment = int(np.floor(fps/length))
for i in range(n_frames):
    pdata = pv.PolyData(r[i*increment, :, :])
    pdata['orig_sphere'] = np.arange(n)
    pc = pdata.glyph(scale=False, geom=particle)
    plotter.update_coordinates(pc.points)

    plotter.camera.azimuth = 360* nrot*i/n_frames

    plotter.write_frame()

plotter.close()