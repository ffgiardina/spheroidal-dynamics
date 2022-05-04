# renders a video of ellipsoidal particles on a spheroid using pyvista
import numpy as np
import pyvista as pv
from subroutines import *

# parameters of video
length = 5  # in seconds
fps = 24  # frames per second
nrot = 0.25  # number of rotations around z-axis in video

# Load data
data = np.load('results.npz', allow_pickle=True)
r = data['trajectories']
para_loaded = data['parameters']
sol = data['solution']

para = para_loaded.item()
a, b, N, tend, l_a, l_b, n = [para[k] for k in ['a', 'b', 'N', 'tend', 'l_a', 'l_b', 'n']]
phi, theta, psi = [sol[:, i * n:(i + 1) * n] for i in range(3)]

# set animation properties
pv.set_plot_theme("document")
plotter = pv.Plotter(window_size=([1024, 768]), lighting='three lights')

# axes arrows
arrow_x = pv.Arrow((-a, 0, -b*1.1), (2*a, 0, 0), shaft_radius=0.01, tip_length=0.1, tip_radius=0.03, scale='auto')
arrow_y = pv.Arrow((0, -a, -b*1.1), (0, 2*a, 0), shaft_radius=0.01, tip_length=0.1, tip_radius=0.03, scale='auto')
plotter.add_mesh(arrow_x, color='k')
plotter.add_mesh(arrow_y, color='k')

# ground plane
plane = pv.Polygon((0,0,-b*1.1), radius=a, normal=(0,0,1.0), n_sides=100)
plotter.add_mesh(plane, color='k', opacity=0.1)

# spheroid
spheroid = pv.ParametricEllipsoid(a,a,b)
plotter.add_mesh(spheroid, color='seashell', smooth_shading=True, opacity=0.85)

# particles
pdata = pv.PolyData(r[0,:,:])
pdata['orig_sphere'] = np.arange(n)
particle = pv.ParametricEllipsoid(xradius=l_a*0.6, yradius=l_b, zradius=l_b)
pc = pdata.glyph(scale=False, geom=particle)
plotter.add_mesh(pc, cmap='Purples', smooth_shading=False, opacity=1, show_scalar_bar=False)

# set camera properties
plotter.camera.zoom(1.3)

# create animation and save as mp4
plotter.open_movie("dynamics.mp4")
n_frames = int(np.floor(N/fps*length))
increment = int(np.floor(fps/length))
for i in range(n_frames):
    pdata = pv.PolyData(r[i*increment, :, :])
    pdata['orig_sphere'] = np.arange(n)
    heading = get_particle_heading(sol[i*increment, :], para)
    pdata.point_data.set_array(heading, 'headings')
    pc = pdata.glyph(orient = 'headings',scale=False, geom=particle)

    plotter.update_coordinates(pc.points)
    plotter.camera.azimuth = 360* nrot*i/n_frames
    plotter.write_frame()

plotter.close()