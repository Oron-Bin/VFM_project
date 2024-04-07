import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def system_of_odes(t, variables, parameters):
    phi, omega, x, vx, y, vy = variables
    w, tau_f, theta, M, m, miu, g, l, I_com, beta, dx, dy, fb = parameters

    F_N = (m * l * (w ** 2)) / 1000.0 * np.sin(np.deg2rad(w * t)) + (M * np.cos(np.deg2rad(beta)) + m) * g
    F_K = miu * F_N
    F_g = M * g * np.sin(np.deg2rad(beta))
    f_c = ((m * l * (w ** 2)) / 1000.0) * np.cos(np.deg2rad(w * t)) + m * g

    dx_dt = vx
    dvx_dt = ((f_c - F_K - F_g) * np.cos(np.deg2rad(theta))) / M
    dy_dt = vy

    if np.isclose(f_c, (F_K + miu * fb * y) + F_g):
        dvy_dt = 0.0
    else:
        dvy_dt = ((f_c - (F_K + miu * fb * y) - F_g) * np.sin(np.deg2rad(theta))) / M

    dphi_dt = omega
    domega_dt = ((f_c - (F_K + miu * fb * y)) * (dx * np.sin(np.deg2rad(theta)) - dy * np.cos(np.deg2rad(theta))) - tau_f) / (I_com + (M * (x ** 2 + y ** 2)))

    return [dphi_dt, domega_dt, dx_dt, dvx_dt, dy_dt, dvy_dt]

class Vibration_animator:
    def __init__(self, solution, ax_phi, ax_x, ax_y, ax_fc, miu, m, w, M, l, beta, g, fb):
        self.solution = solution
        self.ax_phi = ax_phi
        self.ax_x = ax_x
        self.ax_y = ax_y
        self.ax_fc = ax_fc
        self.miu = miu
        self.m = m
        self.w = w
        self.M = M
        self.l = l
        self.beta = beta
        self.g = g
        self.fb = fb

    def animate(self, i):
        # Clear previous plots
        self.ax_phi.clear()
        self.ax_x.clear()
        self.ax_y.clear()
        self.ax_fc.clear()

        # Plot new data for each subplot
        self.ax_phi.plot(self.solution.t[:i], self.solution.y[0][:i]*(180/np.pi), label='phi(t)', color='blue')
        self.ax_phi.plot(self.solution.t[:i], self.solution.y[1][:i], label='phi_dot(t)', color='green')
        self.ax_phi.set_xlabel('Time (t)')
        self.ax_phi.set_ylabel('Phi(deg)')
        self.ax_phi.set_title('Evolution of Phi over Time')
        self.ax_phi.grid(True)

        self.ax_x.plot(self.solution.t[:i], self.solution.y[2][:i]/10, label='x(t)', color='blue')
        self.ax_x.plot(self.solution.t[:i], self.solution.y[3][:i]/10, label='x_dot', color='green')
        self.ax_x.set_xlabel('Time (t)')
        self.ax_x.set_ylabel('x(cm)')
        self.ax_x.set_title('Evolution of x over Time')
        self.ax_x.grid(True)

        self.ax_y.plot(self.solution.t[:i], self.solution.y[4][:i]/10, label='y(t)(cm)', color='blue')
        self.ax_y.plot(self.solution.t[:i], self.solution.y[5][:i]/10, label='y_dot', color='green')
        self.ax_y.set_xlabel('Time (t)')
        self.ax_y.set_ylabel('y(cm)')
        self.ax_y.set_title('Evolution of y over Time')
        self.ax_y.grid(True)

        # Calculate f_c for each time step
        fc_values = [(self.m * self.l * (self.w ** 2)) / 1000.0 * np.cos(np.deg2rad(self.w * t)) + self.m * self.g for t in self.solution.t[:i]]
        self.ax_fc.plot(self.solution.t[:i], fc_values, label='f_c', color='orange')

        self.ax_fc.set_xlabel('Time (t)')
        self.ax_fc.set_ylabel('Forces')
        self.ax_fc.set_title('Evolution of f_c over Time')
        self.ax_fc.legend()
        self.ax_fc.grid(True)

def main():
    # Initial conditions
    initial_conditions = [0.0 ,0.0, 30.0, 0.0, 10.0, 0.0] # [phi_0, omega_0, x0, vx_0, y_0, vy_0]

    # Parameters
    tau_f = 100
    R = 50.0 #mm
    M = 14.0#   m kg
    I_com = (1/2)*(M*R*R)
    m = 1.2  #mkg
    l = 4.0 # mm
    beta = 5.0
    g = 9.81 #m/s^2
    theta = 90
    w = 240   # Frequency
    miu = 0.15 # Friction coefficient
    dx = 10.0 # mm
    dy = 10.0 # mm
    fb = 2.18
    parameters = (w, tau_f, theta, M, m, miu, g, l ,I_com, beta, dx, dy, fb)  #

    # Time span for the integration
    t_span = (0, 50)

    # Solve the system of ODEs
    # Solve the system of ODEs
    solution = solve_ivp(system_of_odes, t_span, initial_conditions, args=(parameters,), dense_output=True,
                         t_eval=np.linspace(0, 50, 1000))

    # Initialize animator
    fig, (ax_phi, ax_x, ax_y, ax_fc) = plt.subplots(4, 1, figsize=(10, 18))
    animator = Vibration_animator(solution, ax_phi, ax_x, ax_y, ax_fc, *parameters[5:])

    # Create animation
    ani = FuncAnimation(fig, animator.animate, frames=len(solution.t), interval=10)

    # Show animation
    plt.show()

if __name__ == "__main__":
    main()
