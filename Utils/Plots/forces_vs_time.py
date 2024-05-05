import numpy as np
import matplotlib.pyplot as plt

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
fb = 2.5
lamda = 0.8
lamda_2 = 0.8
teta = 40

def forces(t):
    f_c = (m*l*(w**2))/1000.0*np.cos(np.deg2rad(w*t))*lamda
    F_N = M*g*np.cos(np.deg2rad(beta)) -((m*l*(w**2))*np.sin(np.deg2rad(w*t))*lamda_2/1000.0)
    F_k = miu_k*F_N + fb*t
    F_g = M * g * np.sin(np.deg2rad(beta))
    F_s = miu_s * (M * g + (m*l*(w**2))*lamda_2/1000.0)
    return f_c, F_k, F_s

t_values = np.linspace(0, 50, 500)
f_c_values, F_k_values, F_s_values = [], [], []

for t in t_values:
    f_c, F_k, F_s = forces(t)
    f_c_values.append(f_c)
    F_k_values.append(F_k)
    F_s_values.append(F_s)

plt.plot(t_values, f_c_values, label='$f_c$')
plt.plot(t_values, F_k_values, label='$F_N$')
plt.plot(t_values, F_s_values, label='$F_s$')
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('Forces vs Time')
plt.legend()
plt.grid(True)
plt.show()
