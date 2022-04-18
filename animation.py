from matplotlib.animation import FuncAnimation
from matplotlib import cm
from subroutines import *
import math

def animate(save_video=False):
    # Load data
    data = np.load('results.npz', allow_pickle=True)
    r = data['trajectories']
    para = data['parameters']
    sol = data['solution']

    # Animation parameters
    replay_speed = 10
    a = para.item()['a']; b = para.item()['b']; N = para.item()['N']; tend = para.item()['tend']
    r_b = para.item()['r_b']; n = para.item()['n']

    phi, theta, psi = [sol[:, i * n:(i + 1) * n] for i in range(3)]

    # Animate results
    fig = plt.figure(1)
    ax = plt.axes(projection='3d', computed_zorder=True)
    plt.axis('off')

    # Ellipsoid mesh
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    x = a*np.cos(u)*np.sin(v)
    y = a*np.sin(u)*np.sin(v)
    z = b*np.cos(v)

    ax.plot_surface(x, y, z, cmap=cm.gist_earth, alpha=0.5, cstride=1, rstride=1, linewidth=0, antialiased=True,
                    zorder=1)

    # ground plane
    us = a*np.cos(u)*np.sin(v); vs = a*np.sin(u)*np.sin(v)
    ax.plot_surface(us, vs, -a*np.ones(us.shape), cmap=cm.bone, alpha=0.1, cstride=1, rstride=1, linewidth=0, antialiased=True,zorder=0)

    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

    # lines to be updated
    line, = ax.plot(r[0, 1, 0], r[0, 1, 1], r[0, 1, 2], c='black', linewidth=2)
    axis1, = ax.plot([-a,a], [0,0], [-a,-a], alpha=0.4,  c='black', linewidth=1)
    axis2, = ax.plot([0, 0], [-a, a], [-a, -a], alpha=0.4, c='black', linewidth=1)

    # particle shapes
    ne = 20  # number of vertices on polygon
    particles = []
    for i in range(n):
        q = [phi[0, i], theta[0, i], psi[0, i]]
        pts = create_ellipse(r[0,i,:], q, ne, para)
        particles.append(ax.plot(pts[:,0], pts[:,1], pts[:,2], c='black', linewidth=1))

    # time
    axtext = fig.add_axes([0.0, 0.95, 0.1, 0.05])
    axtext.axis("off")
    time = axtext.text(0.5, 0.5, str(0), ha="left", va="top")

    # Updating function, to be repeatedly called by the animation
    def update(it):
        # obtain point coordinates
        r0 = r[it,:,:]

        # update particle coordinates
        for i in range(n):
            q = [phi[it,i], theta[it,i], psi[it,i]]
            pts = create_ellipse(r0[i,:], q, ne, para)
            particles[i][0].set_data(pts[:,0], pts[:,1])
            particles[i][0].set_3d_properties(pts[:,2], 'z')

        # Rotate plot
        ax.view_init(elev=20, azim=1 * it/N*360)

        line.set_data(r[0:it,0,0],r[0:it,0,1])
        line.set_3d_properties(r[0:it,0,2], 'z')

        time.set_text('t=' + str(np.round(tend*it/N,1)) + 's')

        return particles, line, ax, time


    anim = FuncAnimation(fig, update, interval=math.ceil(1000.0*tend/N/replay_speed), blit=False, repeat=False,
                        frames=N)

    plt.show()

    if save_video:
        anim.save('animation.mp4', writer = 'ffmpeg', fps = 30)


if __name__ == "__main__":
    animate()