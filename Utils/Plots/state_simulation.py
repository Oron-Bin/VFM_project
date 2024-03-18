import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Constants
M = 0.14
beta_deg = 15 * np.pi / 180
teta_deg = 0 * np.pi / 180
f = 100
g = 9.81
tau_f = 1
# Convert angles to radians
beta_rad = beta_deg
teta_rad = teta_deg

elapsed_time = 0.0

# Define the differential equations
def diff_eq(state, t):
    x, x_dot, y, y_dot, phi, phi_dot = state

    # Compute M*sin(phi)*sin(beta) and M*cos(phi)*sin(beta)
    M_sin_phi_sin_beta = M * np.sin(phi) * np.sin(beta_rad)
    M_cos_phi_sin_beta = M * np.cos(phi) * np.sin(beta_rad)

    f_g_x = -M_cos_phi_sin_beta *g
    f_g_y = -M_sin_phi_sin_beta*g
    # Equations of motion
    x_dot_dot = (f * np.cos(teta_rad) + f_g_x) / M
    y_dot_dot = (f * np.sin(teta_rad) + f_g_y) / M
    # Angular acceleration equation
    phi_dot_dot = (f*(x*np.sin(teta_rad)-y*np.cos(teta_rad)-(x*f_g_y - y*f_g_x )) +tau_f) / (M*(x**2 + y**2)+ 0.01 )

    return [x_dot, x_dot_dot, y_dot, y_dot_dot,phi_dot, phi_dot_dot]  # The last value is phi_dot (constant)


# Initial conditions
initial_state = [0, 0, 0, 0, 0 ,0]  # x, x_dot, y, y_dot, phi
time = np.linspace(0, 100, 1000)  # Adjust the time range and number of time points as needed

# Solve the differential equations
solution = odeint(diff_eq, initial_state, time)

# Extract the x and y coordinates
x = solution[:, 0]
# print(x)
y = solution[:, 2]
# print(y)
phi = solution[:,4]
phi_dot = solution[:,5]
# print(phi)
# print(len(phi))
# Create the animation
fig, ax = plt.subplots()


def update(frame):
    global elapsed_time
    elapsed_time = time[frame]
    ax.clear()
    ax.plot(time[:frame],phi[:frame])
    # ax.set_xlabel('time')
    print(time[frame], phi[frame], phi_dot[frame])
    ax.set_ylabel('phi')
    # ax.plot(x[:frame], y[:frame])
    ax.set_xlabel('t')
    # ax.set_ylabel('Y')
    ax.set_title('Projectile Motion')
    ax.axis('equal')


ani = FuncAnimation(fig, update, frames=len(time), interval=10)
plt.show()