#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
# from matplotlib.animation import FuncAnimation
#
# # Define constants
# miu = 0.15  # Friction coefficient
# fb = 2.18*10   # Friction constant
# M = 14.0    # Mass
# theta = 10   # Angle
# g = 9.81    # Gravity
#
# # Define parameters
# w = 240     # Frequency
# l = 4.0     # mm
# m = 1.2     # mkg
# beta = 5.0
# # Calculate additional constants
# F_g = M * g * np.sin(np.deg2rad(beta))
#
# # Define the function for the second-order ODE
# def dvr_dt(y, t):
#     r, vr = y
#
#     F_N = ((m * l * (w ** 2)) / 1000.0) * np.sin(np.deg2rad(w * t)) + (M * np.cos(np.deg2rad(theta)) + m) * g
#     F_K = miu * F_N
#     f_c = ((m * l * (w ** 2)) / 1000.0) * np.cos(np.deg2rad(w * t)) + m * g
#
#     if f_c == F_K + miu * fb * r * np.cos(np.deg2rad(theta)) + F_g:
#         vr = 0.0
#         f_c = F_K + miu * fb * r * np.cos(np.deg2rad(theta)) + F_g
#
#     return [vr, (f_c - (F_K + miu * fb * r * np.cos(np.deg2rad(theta))) - F_g) / M]
#
# # Solve the differential equation using odeint
# t = np.linspace(0, 100, 1000)  # Time span
# initial_conditions = [0.0, 0.0]  # Initial conditions: r(0) = 0, vr(0) = 0
# solution = odeint(dvr_dt, initial_conditions, t)
#
# # Animation
# fig, ax = plt.subplots()
# line, = ax.plot([], [], lw=2)
#
# # Set fixed limits for the y-axis
# ax.set_ylim(-20, 10)
#
# def init():
#     ax.set_xlim(0, 100)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Displacement (r)')
#     ax.set_title('Solution to $\ddot{r} = \\frac{{f_c - (F_K + \mu \cdot f_b \cdot r \cdot \cos(\\theta)) - F_g}}{{M}}$')
#     return line,
#
# def update(frame):
#     line.set_data(t[:frame], solution[:frame, 0])
#     return line,
#
# # Adjust the frame rate (interval) of the animation
# frame_rate = 50  # in milliseconds
# ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=frame_rate)
# plt.show()
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define global variables
w = 240   # Frequency
tau_f = 100
theta = 20
R = 50.0 #mm
M = 14.0#   m kg
I_com = (1/2)*(M*R*R)
m = 1.2  #mkg
l = 4.0 # mm
beta = 5.0
g = 9.81 #m/s^2
miu = 0.15 # Friction coefficient
dx = 10.0 # mm
dy = 10.0 # mm
fb = 10
initial_conditions = [0.0 ,0.0, 10.0, 0.0, 20.0, 0.0] # [phi_0, omega_0, x0, vx_0, y_0, vy_0]

# Define the system of ODEs
def system_of_odes(t, variables):
    phi, omega, x, vx, y, vy = variables

    F_N = ((m * l * (w ** 2)) / 1000.0) * np.sin(np.deg2rad(w * t)) + (M * np.cos(np.deg2rad(beta)) + m) * g
    F_K = miu * F_N
    F_g = M * g * np.sin(np.deg2rad(beta))
    f_c = ((m * l * (w ** 2)) / 1000.0) * np.cos(np.deg2rad(w * t)) + m * g
    distance = np.sqrt(x ** 2 + y ** 2)

    dx_dt = vx
    dy_dt = vy

    amplitude_fc = (m * l * (w ** 2)) / (1000.0) +(M * np.cos(np.deg2rad(beta)) + m) * g
    min_amplitude_res_y = miu * ((m * l * (w ** 2)) / (1000.0) - (M * np.cos(np.deg2rad(beta))) - (m * g) - (fb*10*y) - (M * g * np.sin(np.deg2rad(beta))))
    min_amplitude_res_x = miu * ((m * l * (w ** 2)) / (1000.0) - (M * np.cos(np.deg2rad(beta))) - (m * g) - (fb * 10 * x) - ( M * g * np.sin(np.deg2rad(beta))))

    if amplitude_fc < min_amplitude_res_x:
        dx_dt = 0.0
        dvx_dt = 0.0
    else:
        dvx_dt = ((f_c - (F_K + miu * fb * x) - F_g) * (np.cos(np.deg2rad(theta)))) / M

    if amplitude_fc < min_amplitude_res_y:
        dy_dt = 0.0
        dvy_dt = 0.0
    else:
        dvy_dt = ((f_c - (F_K + miu * fb * y) - F_g) * (np.sin(np.deg2rad(theta)))) / M
    dphi_dt = omega
    domega_dt = ((f_c - (F_K + miu * fb * distance)) * (dx * np.sin(np.deg2rad(theta)) - dy * np.cos(np.deg2rad(theta))) - tau_f) / (I_com + (M * (x ** 2 + y ** 2)))

    return [dphi_dt, domega_dt, dx_dt, dvx_dt, dy_dt, dvy_dt]

# Time span for the integration
t_span = (0, 50)

# Solve the system of ODEs
solution = solve_ivp(system_of_odes, t_span, initial_conditions, dense_output=True, t_eval=np.linspace(0, 50, 1000))

# Calculate r(t)
r = np.sqrt(solution.y[2]**2 + solution.y[4]**2) / 10  # divide by 10 to convert from mm to cm

# Plot r(t)
plt.figure()
plt.plot(solution.t, r)
plt.xlabel('Time (t)')
plt.ylabel('r(t) (cm)')
plt.title('Plot of r(t)')
plt.grid(True)
plt.show()
