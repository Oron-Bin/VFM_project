import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define global variables
w = 240   # Frequency
tau_f = 100
theta = 0
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
fb = 2.18
initial_conditions = [0.0 ,0.0, 30.0, 0.0, 10.0, 0.0] # [phi_0, omega_0, x0, vx_0, y_0, vy_0]

# Define the system of ODEs
def system_of_odes(t, variables):
    phi, omega, x, vx, y, vy = variables

    F_N = ((m * l * (w ** 2)) / 1000.0) * np.sin(np.deg2rad(w * t)) + (M * np.cos(np.deg2rad(beta)) + m) * g
    F_K = miu * F_N
    F_g = M * g * np.sin(np.deg2rad(beta))
    f_c = ((m * l * (w ** 2)) / 1000.0) * np.cos(np.deg2rad(w * t)) + m * g
    distance = np.sqrt(x ** 2 + y ** 2)

    f_res_x = -miu * (((m * l * (w ** 2)) / 1000.0) * np.sin(np.deg2rad(w * t)) + (M * np.cos(np.deg2rad(beta)) + m) * g + (fb * dx)) + M * g * np.sin(np.deg2rad(beta))
    f_res_y = -miu * (((m * l * (w ** 2)) / 1000.0) * np.sin(np.deg2rad(w * t)) + (M * np.cos(np.deg2rad(beta)) + m) * g + (fb * dy)) + M * g * np.sin(np.deg2rad(beta))

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

# Animation function
def animate(i):
    ax_phi.clear()
    ax_phi.plot(solution.t[:i], solution.y[0][:i]*(180/np.pi), label='phi(t)', color='blue')
    ax_phi.plot(solution.t[:i], solution.y[1][:i], label='omega(t)', color='green')
    ax_phi.set_xlabel('Time (t)')
    ax_phi.set_ylabel('Phi(deg)')
    ax_phi.set_title('Evolution of Phi over Time')
    ax_phi.grid(True)
    ax_phi.legend()

    ax_x.clear()
    ax_x.plot(solution.t[:i], solution.y[2][:i]/10, label='x(t)', color='blue')
    ax_x.plot(solution.t[:i], solution.y[3][:i]/10, label='velocity_x(t)', color='green')
    ax_x.set_xlabel('Time (t)')
    ax_x.set_ylabel('x(cm)')
    ax_x.set_title('Evolution of x over Time')
    ax_x.grid(True)
    ax_x.legend()

    ax_y.clear()
    ax_y.plot(solution.t[:i], solution.y[4][:i]/10, label='y(t)', color='blue')
    ax_y.plot(solution.t[:i], solution.y[5][:i]/10, label='velocity_y(t)', color='green')
    ax_y.set_xlabel('Time (t)')
    ax_y.set_ylabel('y(cm)')
    ax_y.set_title('Evolution of y over Time')
    ax_y.grid(True)
    ax_y.legend()

    # Calculate f_res
    f_c_animation = ((m * l * (w ** 2)) / 1000.0) * np.cos(np.deg2rad(w * solution.t[:i])) + m * g
    f_res_x_animtaion = -miu * (((m * l * (w ** 2)) / 1000.0) * np.sin(np.deg2rad(w * solution.t[:i])) + (M * np.cos(np.deg2rad(beta)) + m) * g + (fb*10*(solution.y[2][:i]))) + M * g * np.sin(np.deg2rad(beta))
    f_res_y_animtaion = -miu * (((m * l * (w ** 2)) / 1000.0) * np.sin(np.deg2rad(w * solution.t[:i])) + (M * np.cos(np.deg2rad(beta)) + m) * g + (fb*10*(solution.y[4][:i]))) + M * g * np.sin(np.deg2rad(beta))

    # # Plot f_c and f_k
    ax_fc.clear()
    ax_fc.plot(solution.t[:i], f_c_animation, label='f_c', color='orange')
    ax_fc.plot(solution.t[:i], f_res_x_animtaion , label='f_res', color='purple')
    ax_fc.set_xlabel('Time (t)')
    ax_fc.set_ylabel('Forces (mN)')
    ax_fc.set_title('Evolution of f_c and f_k_total over Time')
    ax_fc.legend()
    ax_fc.grid(True)

# Create subplots
fig, (ax_phi, ax_x, ax_y, ax_fc) = plt.subplots(4, 1, figsize=(10, 18))

# Create animation
ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=10)

# Show animation
plt.show()
