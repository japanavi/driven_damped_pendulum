import matplotlib.animation as animation
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def ddp(y, t, gamma, omega, omega_0, beta):
    phi, alpha = y
    dydt = [alpha, -2 * beta * alpha - omega_0**2 *
            np.sin(phi) + gamma * omega_0**2 * np.cos(omega * t)]
    return dydt


gamma = 0.9
omega = 2 * np.pi
omega_0 = 1.5 * omega
beta = omega_0 / 4

y0 = [0.0, 0.0]
dt = 0.05
t = np.arange(0, 8, dt)

sol = odeint(ddp, y0, t, args=(gamma, omega, omega_0, beta))

phi = sol[:, 0]
phi_dot = sol[:, 1]

fig = plt.figure(figsize=(10, 10), constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
# Font Size
FS = 20

ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1])

ax1.set_title("Driven Damped Pendulum", fontsize=FS, pad=10)
ax1.grid(linestyle="--")
ax1.axis([-1.5, 1.5, -1.5, 1.5])

line1, = ax1.plot([], [], 'o-', lw=2)
line2, = ax2.plot([], [], 'o-', lw=2, c='r')

time_template = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, fontsize=FS)

ax2.set_title("Angle ($\phi$) with Respect to Vertical", fontsize=FS, pad=10)
ax2.plot(phi, t, zorder=0)
ax2.set_xlabel('$\phi$(t)', fontsize=FS)
ax2.set_ylabel('t', rotation=0, fontsize=FS, labelpad=10)

x_s = np.sin(phi)
y_s = -np.cos(phi)


def animate(i):
    thisx = [0, x_s[i]]
    thisy = [0, y_s[i]]

    line1.set_data(thisx, thisy)
    line2.set_data([phi[i]], [t[i]])
    time_text.set_text(time_template % (i*dt))
    return line1, line2, time_text


ani = animation.FuncAnimation(fig, animate, len(sol), interval=dt*1000, blit=True)
plt.show()
# Uncomment line below to save gif of simulation
# ani.save('ddp.gif', writer='imagemagick')
