import sympy as smp
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define symbols
x, y, teta, beta, f_c, m, g, tau_f = smp.symbols('x y teta beta f_c m g tau_f')

# Define equations
F = -(m * g) * smp.sin(beta)
F_x = F * smp.cos(teta)
F_y = F * smp.sin(teta)

eq1 = (F_x + f_c * smp.cos(teta))/m
eq2 = (F_y + f_c * smp.sin(teta))/m
eq3 = (f_c * (x * smp.sin(teta) - y * smp.cos(teta)) - (x * F_y - y * F_x) + tau_f) / (m * (x ** 2 + y ** 2))

# Substitute teta=0
eq1_sub = eq1.subs(teta ,0)
eq2_sub = eq2.subs(teta, 0)
eq3_sub = eq3.subs(teta, 0)

print(eq1_sub)
print(eq2_sub)
print(eq3_sub)

# Convert equations to numerical functions
eq1_func = smp.lambdify((x, y, beta,  f_c, m, g, tau_f), eq1_sub)
eq2_func = smp.lambdify((x, y, beta,  f_c, m, g, tau_f), eq2_sub)
eq3_func = smp.lambdify((x, y, beta,  f_c, m, g, tau_f), eq3_sub)


# Define parameters
beta_val = 20.0  # Assuming beta is constant
# phi_val = 0.0  # Assuming phi is constant
f_c_val = 10.0  # Assuming f_c is constant
m_val = 14.0  # Assuming mass m is constant
g_val = 9.81  # Assuming gravitational acceleration g is constant
tau_f_val = 1.0  # Assuming tau_f is constant

# Define initial conditions
x0 = 1.0
y0 = 1.0
v0 = 1.0  # Initial velocity
phi0 = 10.0 # Initial
# Time vector
t = np.linspace(0, 10, 100)  # Simulation time from 0 to 10 seconds

# Define function to integrate
def func(z, t):
    x, y , phi = z
    ddx_dt = eq1_func(x, y, beta_val, f_c_val, m_val, g_val, tau_f_val)
    ddy_dt = eq2_func(x, y, beta_val, f_c_val, m_val, g_val, tau_f_val)
    ddphi_dt = eq3_func(x, y, beta_val,f_c_val,m_val, g_val, tau_f_val)
    return [ddx_dt, ddy_dt, ddphi_dt]

# Integrate the equations of motion
sol = odeint(func, [x0, y0, phi0], t)


def update_plot(frame_num):
    scat.set_offsets([t[frame_num], sol[frame_num, 2]])  # Update time and phi

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, max(t))  # Set x-axis limits based on the time vector
ax.set_ylim(0, max(sol[:, 2]))  # Set y-axis limits based on phi
# ax.set_ylim(min(sol[:, 2]), max(sol[:, 2]))  # Set y-axis limits based on phi
ax.set_xlabel('Time')
ax.set_ylabel('omega')
ax.set_title('omega vs Time')

# Plot the marker representing the current time and phi
scat = ax.scatter([], [], color='red', marker='o')

# Set up the animation
ani = FuncAnimation(fig, update_plot, frames=len(t), interval=50)

# Display the animation
plt.show()
# Plot x and y positions versus time
# import matplotlib.pyplot as plt
#
# plt.plot(t, sol[:, 0], label='x')
# plt.plot(t, sol[:, 1], label='y')
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.title('Simulation of x and y positions vs time')
# plt.legend()
# plt.grid(True)
# plt.show()