import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

# Define global variables
w = 240   # Frequency
R = 50.0 # mm
M = 14.0 # kg
m = 1.2  # kg
l = 4.0  # mm
beta = 0.0
g = 9.81 # m/s^2
miu_k = 0.2 # Friction coefficient
miu_s = 0.8 # Static Friction coefficient
fb = 3
lamda = 0.9
lamda_2 = 0.5
static_friction = miu_s *M*g
epsilon = 20
print("static friction", static_friction)
teta = 40

initial_conditions = [0.0 ,0.0, 10.0, 0.0]
# Variable to store the last valid value of r
last_valid_values = None

def forces(t):
    f_c = (m*l*(w**2))/1000.0*np.cos(np.deg2rad(w*t))*lamda
    F_N = M*g*np.cos(np.deg2rad(beta)) -(m*l*(w**2))/1000.0*np.sin(np.deg2rad(w*t))*lamda_2
    F_k = miu_k*F_N
    F_g = M * g * np.sin(np.deg2rad(beta))
    return [f_c, F_N, F_k, F_g]

# Function to define the system of ODEs
def system_of_odes(t, variables):
    global last_valid_values
    phi, omega, r, vr = variables

    Fk = forces(t)[2] + fb * t
    Fc = forces(t)[0]
    F = Fc - Fk
    print(F)
    if F < -293:
        if last_valid_values is None:
            last_valid_values = [phi, omega, r, vr]
        elif last_valid_values[2] < r:  # Compare current r with stored r
            last_valid_values = [phi, omega, r, vr]

    dr_dt = vr
    dphi_dt = omega
    dvr_dt = (F * np.cos(np.deg2rad(teta)) + M * r * omega ** 2) / M
    domega_dt = 2 * F * r * np.sin(np.deg2rad(teta)) / (M * (R ** 2))

    return [dphi_dt, domega_dt, dr_dt, dvr_dt]

t_span = (0, 50)
# Function to perform simulation
solution = solve_ivp(system_of_odes, t_span, initial_conditions, dense_output=True, t_eval=np.linspace(0, 50, 1000))

last_phi, last_omega, last_r, last_vr = last_valid_values
print("Last valid values at F < 0:")
print("Phi:", last_phi)
print("Omega:", last_omega)
print("r:", last_r)
print("Velocity_r:", last_vr)


# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Phi vs Time
axs[0, 0].plot(solution.t, solution.y[0])
axs[0, 0].set_title('Phi vs Time')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Phi')

# Plot 2: Omega vs Time
axs[0, 1].plot(solution.t, solution.y[1])
axs[0, 1].set_title('Omega vs Time')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Omega')

# Plot 3: r vs Time
axs[1, 0].plot(solution.t, solution.y[2])
axs[1, 0].set_title('r vs Time')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('r')

# Plot 4: vr vs Time
axs[1, 1].plot(solution.t, solution.y[3])
axs[1, 1].set_title('Velocity_r vs Time')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Velocity_r')

plt.tight_layout()
plt.show()