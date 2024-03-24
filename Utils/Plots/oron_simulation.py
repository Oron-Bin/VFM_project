import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def system_of_odes(t, variables, w, tau_f, theta, M, m, miu, g, l ,I_com, beta, dx, dy):
    F_N = ((m*l*(w**2))/1000.0) * np.sin(np.deg2rad(w * t)) + (M*np.cos(np.deg2rad(beta)) + m)*g #mN
    F_K = miu* F_N
    f_c = ((m*l*(w**2))/1000.0)* np.cos(np.deg2rad(w * t)) + m*g

    phi, omega, x, vx, y, vy = variables
    dx_dt = vx
    dvx_dt = ((f_c - F_K - M*g*np.sin(np.deg2rad(beta)))* (np.cos(np.deg2rad(theta)))) / M
    dy_dt = vy
    dvy_dt = ((f_c - F_K - M*g*np.sin(np.deg2rad(beta))) * (np.sin(np.deg2rad(theta)))) / M
    dphi_dt = omega
    domega_dt = ((f_c -F_K)*(dx*np.sin(np.deg2rad(theta)) - dy*np.cos(np.deg2rad(theta))) - tau_f) / ((I_com + (M*(x**2 + y**2))))
    return [dphi_dt,domega_dt, dx_dt, dvx_dt, dy_dt, dvy_dt]

# Initial conditions
initial_conditions = [np.pi ,0.0, 30.0, 0.0, 0.0, 0.0]  # [phi_0, omega_0, x0, vx_0, y_0, vy_0]

# Parameters
tau_f = 100
R = 50.0 #mm
M = 14.0#   m kg
I_com = (1/2)*(M*R*R)
m = 1.2  #mkg
l = 5.0 # mm
beta = 0.0
g = 9.81 #m/s^2
theta = 90
w = 240   # Frequency
miu = 0.229 # Friction coefficient
dx = 5.0
dy = 5.0
parameters = (w, tau_f, theta, M, m, miu, g, l ,I_com, beta, dx, dy)  #

# Time span for the integration
t_span = (0, 100)

# Solve the system of ODEs
solution = solve_ivp(system_of_odes, t_span, initial_conditions, args=parameters, dense_output=True, t_eval=np.linspace(0, 100, 1000))

# Animation function
def animate(i):
    ax_phi.clear()
    ax_phi.plot(solution.t[:i], solution.y[0][:i] *(180/np.pi), label='phi(t)', color='blue')
    ax_phi.plot(solution.t[:i], solution.y[1][:i], label='phi_dot(t)', color='green')
    ax_phi.set_xlabel('Time (t)')
    ax_phi.set_ylabel('Phi(deg)')
    ax_phi.set_title('Evolution of Phi over Time')
    ax_phi.grid(True)

    ax_x.clear()
    ax_x.plot(solution.t[:i], solution.y[2][:i], label='x(t)', color='red')
    ax_x.plot(solution.t[:i], solution.y[3][:i], label='x_dot', color='green')
    ax_x.set_xlabel('Time (t)')
    ax_x.set_ylabel('x(mm)')
    ax_x.set_title('Evolution of x over Time')
    ax_x.grid(True)

    ax_y.clear()
    ax_y.plot(solution.t[:i], solution.y[4][:i], label='y(t)(mm)', color='green')
    ax_y.set_xlabel('Time (t)')
    ax_y.set_ylabel('y(mm)')
    ax_y.set_title('Evolution of y over Time')
    ax_y.grid(True)

# Create subplots
fig, (ax_phi, ax_x, ax_y) = plt.subplots(3, 1, figsize=(10, 18))

# Create animation
ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=10)

# Show animation
plt.show()
