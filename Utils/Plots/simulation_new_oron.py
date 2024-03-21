import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def vibration_phi(z, t, omega, tau_f, theta, M, m, miu, g, l ,I_com, beta, dx, dy, x, y):
    phi, phi_dot = z
    F_N = (m*l*(omega**2)) * np.sin(np.deg2rad(omega * t)) + (M*np.cos(np.deg2rad(beta)) + m)*g
    F_K = miu* F_N
    f_c = (m*l*(omega**2))* np.cos(np.deg2rad(omega * t)) + m*g
    dzdt = [phi_dot, ((f_c -F_K)*(dx*np.sin(np.deg2rad(theta)) - dy*np.cos(np.deg2rad(theta))) - tau_f) / (I_com + (M*(x**2 + y**2)))]
    return dzdt

def vibration_x(x, t, omega, theta, M, m, miu, g,l, beta):
    x_val, x_dot = x
    F_N = (m*l*(omega^2)) * np.sin(np.deg2rad(omega * t)) + (M*np.cos(np.deg2rad(beta)) + m)*g
    F_K = miu* F_N
    f_c = (m*l*(omega^2))* np.cos(np.deg2rad(omega * t)) + m*g
    dzdt = [x_dot, ((f_c - F_K - M*g*np.sin(np.deg2rad(beta)))* (np.cos(np.deg2rad(theta)))) / M]
    return dzdt

def vibration_y(y, t, omega, theta, M, m, miu, g,l, beta):
    y_val, y_dot = y
    F_N = (m*l*(omega^2)) * np.sin(np.deg2rad(omega * t)) + (M*np.cos(np.deg2rad(beta)) + m)*g
    F_K = miu* F_N
    f_c = (m*l*(omega^2))* np.cos(np.deg2rad(omega * t)) + m*g
    dydt = [y_dot, ((f_c - F_K - M*g*np.sin(np.deg2rad(beta))) * (np.sin(np.deg2rad(theta)))) / M]
    return dydt

tau_f = -100
R = 50.0 #mm
M = 14.0#   m kg
I_com = (1/2)*(M*R*R)
m = 1.2  #mkg
l = 5.0 # mm
beta = 0.0
g = 9.81*1000 #mm/s^2
theta = 90
omega = 240   # Frequency of 0.1 Hz
miu = 0.229 # Friction coefficient
dx = 3.0
dy = 3.0
y0 = [-30.0, 0.0]
x0 = [30.0, 0.0]
z0 = [0.0, 0.0]
t = np.linspace(0, 100, 100)

# print(t)


sol_x = odeint(vibration_x, x0, t, args=(omega, theta, M, m, miu, g,l, beta))
sol_y = odeint(vibration_y, y0, t, args=(omega, theta, M, m, miu, g,l, beta))
sol_phi = odeint(vibration_phi, z0, t, args=(omega, tau_f, theta, M, m, miu, g, l ,I_com, beta, dx, dy,1))
for i in range(0,100):
    print(sol_y[i][0])
fig, axs = plt.subplots(3)

def update_phi(frame):
    axs[2].cla()
    axs[2].plot(t[:frame], sol_phi[:frame, 0], 'b', label='phi(t)')
    axs[2].plot(t[:frame], sol_phi[:frame, 1], 'g', label='omega(t)')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('Phi/Omega')
    axs[2].legend(loc='best')
    axs[2].grid()

def update_x(frame):
    axs[0].cla()
    axs[0].plot(t[:frame], sol_x[:frame, 0], 'b', label='x(t)(mm)')
    axs[0].plot(t[:frame], sol_x[:frame, 1], 'g', label='v(t)(mm/s)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('X/Velocity')
    axs[0].legend(loc='best')
    axs[0].grid()

def update_y(frame):
    axs[1].cla()
    axs[1].plot(t[:frame], sol_y[:frame, 0]*1000, 'b', label='y(t)(mm)')
    axs[1].plot(t[:frame], sol_y[:frame, 1], 'g', label='v(t)(mm/s)')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Y/Velocity')
    axs[1].legend(loc='best')
    axs[1].grid()

ani_phi = FuncAnimation(fig, update_phi, frames=len(t), interval=50)
ani_x = FuncAnimation(fig, update_x, frames=len(t), interval=50)
ani_y = FuncAnimation(fig, update_y, frames=len(t), interval=50)

plt.show()