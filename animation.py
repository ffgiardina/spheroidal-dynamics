from matplotlib.animation import FuncAnimation
from matplotlib import cm
from subroutines import *
import math

def animate(save_video=False):
    # Load data
    data = np.load('results.npz', allow_pickle=True)
    r = data['trajectories']
    para = data['parameters']

    # Animation parameters
    replay_speed = 10
    a = para.item()['a']; b = para.item()['b']; N = para.item()['N']; tend = para.item()['tend']
    r_b = para.item()['r_b']

    # Animate results
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    plt.axis('off')

    # Ellipsoid mesh
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    x = a*np.cos(u)*np.sin(v)
    y = a*np.sin(u)*np.sin(v)
    z = b*np.cos(v)

    ax.plot_surface(x, y, z, cmap=cm.Spectral, alpha=0.5, cstride=1, rstride=1, linewidth=0, antialiased=True)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

    # points and lines to be updated
    points, = ax.plot(r[0, :, 0], r[0, :,  1], r[0, :,  2], marker="o", c='black', linestyle = 'None', markersize=10)
    line, = ax.plot(r[0, 1, 0], r[0, 1, 1], r[0, 1, 2], c='black', linewidth=1)

    # time
    axtext = fig.add_axes([0.0, 0.95, 0.1, 0.05])
    axtext.axis("off")
    time = axtext.text(0.5, 0.5, str(0), ha="left", va="top")

    # Updating function, to be repeatedly called by the animation
    def update(it):
        # obtain point coordinates
        r0 = r[it,:,:]
        # set point's coordinates
        points.set_data(r0[:, 0],r0[:, 1])
        points.set_3d_properties(r0[:, 2],'z')

        # Rotate plot
        ax.view_init(elev=20, azim=0 * it/N*360)

        line.set_data(r[0:it,0,0],r[0:it,0,1])
        line.set_3d_properties(r[0:it,0,2], 'z')

        time.set_text('t=' + str(np.round(tend*it/N,1)) + 's')

        #return points, line
        return points, line, ax, time,


    anim = FuncAnimation(fig, update, interval=math.ceil(1000.0*tend/N/replay_speed), blit=False, repeat=False,
                        frames=N)

    plt.show()

    if save_video:
        anim.save('animation.mp4', writer = 'ffmpeg', fps = 30)


if __name__ == "__main__":
    animate()