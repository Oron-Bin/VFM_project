import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
w = 240   # Frequency
R = 50.0 # mm
M = 14.0 # kg
m = 1.2  # kg
l = 4.0  # mm
beta = 0.0
g = 9.81 # m/s^2
miu_k = 0.3 # Friction coefficient
miu_s = 0.8 # Static Friction coefficient
fb = 1.2
lamda = 0.8
lamda_2 = 0.8
teta = 30

def calculate_forces(t_values):
    f_c_values = (m*l*(w**2))/1000.0*np.cos(np.deg2rad(w*t_values))*lamda
    F_N_values = M*g*np.cos(np.deg2rad(beta)) -((m*l*(w**2))*np.sin(np.deg2rad(w*t_values))*lamda_2/1000.0)
    # F_k_values = miu_k*F_N_values + fb*t_values
    F_k_values = miu_k*F_N_values
    F_s_values = np.full_like(t_values, miu_s*(M * g + (m*l*(w**2))*lamda_2/1000.0))
    return f_c_values, F_k_values, F_s_values


def system_equations(t, x, f_c_values, F_k_values):
    phi, omega, r, vr = x
    f_c = np.interp(t, t_values, f_c_values)
    F_k = np.interp(t, t_values, F_k_values )
    F_k = F_k + fb* np.abs(r)

    amp_f_c = (m*l*(w**2))/1000.0*lamda

    if amp_f_c <= F_k :
        dphi_dt = omega
        domega_dt = 0
        dr_dt = 0
        dvr_dt = 0
    else:
        F = f_c - F_k
        dphi_dt = omega
        domega_dt = 2 * F * r * np.sin(np.deg2rad(teta)) / (M * (R ** 2))
        dr_dt = vr
        dvr_dt = (F * np.cos(np.deg2rad(teta)) + M * r * omega ** 2) / M

    return [dphi_dt, domega_dt, dr_dt, dvr_dt]

# Initial conditions
r0 = 30.0
phi0 = 0.0
vr0 = 0.0
omega0 = 0.0
initial_conditions = [phi0, omega0, r0, vr0]

# Time span
t_span = (0, 12)
t_values = np.linspace(*t_span, 120)

# Calculate forces
f_c_values, F_k_values, F_s_values = calculate_forces(t_values)

# Solve equations
sol = solve_ivp(lambda t, x: system_equations(t, x, f_c_values, F_k_values), t_span, initial_conditions, dense_output=True)

# Evaluate solution
sol_values = sol.sol(t_values)

# Find the index where F_k_values + fb*np.abs(sol_values[2]) exceeds F_s_values
index = np.argmax(F_k_values + fb*np.abs(sol_values[2]) > F_s_values)

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot forces up to the point where F_k_values + fb*np.abs(sol_values[2]) exceeds F_s_values
axs[0].plot(t_values[:index], f_c_values[:index], label='F_c')
axs[0].plot(t_values[:index], F_s_values[:index], label='F_s')
axs[0].plot(t_values[:index], F_k_values[:index] + fb*np.abs(sol_values[2][:index]), label='F_k', color='orange')
axs[0].set_ylabel('Force (mN)')
axs[0].set_title('Forces vs Time')
axs[0].legend()
axs[0].grid(True)

# Plot F_s values after the condition is met
axs[0].plot(t_values[index:], F_s_values[index:], linestyle='--', color='grey')
axs[0].plot(t_values, f_c_values, label='F_c', color='magenta')
# Plot other variables
axs[1].plot(t_values, sol_values[0]*(180/np.pi), label='phi', color='m')
axs[1].plot(t_values, sol_values[1]*(180/np.pi), label='omega', color='orange')
axs[1].set_ylabel('Values (degree, degree/s)')
axs[1].set_title('Phi and Omega')
axs[1].legend()
axs[1].grid(True)

# Modify r_dot values after the condition is met
sol_values[3][index:] = 0

axs[2].plot(t_values, sol_values[2]/10, label='r', color='m')
axs[2].plot(t_values, sol_values[3]/10, label='r_dot', color='orange')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Values (cm, cm/s)')
axs[2].set_title('r and r_dot')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
