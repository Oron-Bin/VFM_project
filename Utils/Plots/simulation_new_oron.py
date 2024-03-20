import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def vibration_phi(z, t, omega, tau_f, theta, r, m, miu):
    phi, omega_val = z
    f_c = 5 * np.sin(omega * t) - miu * omega_val
    dzdt = [omega_val, (f_c * (r + np.cos(theta) * np.sin(theta) - r * np.sin(theta) * np.cos(theta)) + tau_f) / (m * (r ** 2))]
    return dzdt

def vibration_x(x, t, omega, theta, M, miu, beta):
    x_val, x_dot = x
    F_K = (7.91/1000) * np.sin(240 * t) + (32.33/1000)
    f_c = (34.56/1000)* np.cos(240 * t) + (1.2/1000)
    dzdt = [x_dot, ((f_c- F_K - M*g*np.sin(beta))* np.cos(theta)) / M]
    return dzdt

def vibration_y(y, t, omega, theta, m, miu,F_N, F_K):
    y_val, y_dot = y
    # f_c = (34.56/1000)* np.cos(240 * t) + (1.2/1000)
    F_K = (7.91 / 1000) * np.sin(240 * t) + (32.33 / 1000)
    F_N = (34.56/ 1000) * np.sin(240 * t) + (141.2 / 1000)
    dydt = [y_dot, ((F_N - F_K - M*g*np.sin(beta)) * np.sin(theta)) / M]
    return dydt

tau_f = 10
r = 0.005
m = 14/1000#   kg or 22 gram
theta = 0
omega = 2*np.pi / 10  # Frequency of 0.1 Hz
miu = 0.229 # Friction coefficient
F_K = 0.0
F_N = 0.0
M = 0.0
dx =0.0
dy = 0.0
y0 = [1.0, 1.0]
x0 = [0.0, 0.0]
z0 = [np.pi - 0.1, 0.0]
t = np.linspace(0, 10, 101)

sol_phi = odeint(vibration_phi, z0, t, args=(omega, tau_f, theta, r, m, miu))
sol_x = odeint(vibration_x, x0, t, args=(omega, theta, m, miu))
sol_y = odeint(vibration_y, y0, t, args=(omega, theta, m, miu))

fig, axs = plt.subplots(3)

def update_phi(frame):
    axs[0].cla()
    axs[0].plot(t[:frame], sol_phi[:frame, 0], 'b', label='phi(t)')
    axs[0].plot(t[:frame], sol_phi[:frame, 1], 'g', label='omega(t)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('Phi/Omega')
    axs[0].legend(loc='best')
    axs[0].grid()

def update_x(frame):
    axs[1].cla()
    axs[1].plot(t[:frame], sol_x[:frame, 0], 'b', label='x(t)')
    axs[1].plot(t[:frame], sol_x[:frame, 1], 'g', label='v(t)')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('X/Velocity')
    axs[1].legend(loc='best')
    axs[1].grid()

def update_y(frame):
    axs[2].cla()
    axs[2].plot(t[:frame], sol_y[:frame, 0], 'b', label='y(t)')
    axs[2].plot(t[:frame], sol_y[:frame, 1], 'g', label='v(t)')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('Y/Velocity')
    axs[2].legend(loc='best')
    axs[2].grid()

ani_phi = FuncAnimation(fig, update_phi, frames=len(t), interval=50)
ani_x = FuncAnimation(fig, update_x, frames=len(t), interval=50)
ani_y = FuncAnimation(fig, update_y, frames=len(t), interval=50)

plt.show()