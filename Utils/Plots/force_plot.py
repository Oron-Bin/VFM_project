
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from matplotlib.path import Path
import matplotlib.patches as patches

#parameters of the model

l = 0.004
m = 0.02
w = 200
fb = 5.
mu_s = 0.8
mu_k = 0.7

t = np.linspace(0, .1 ,1000) #this function sample 1000 points in interval of 0.1

# fN = m*l*w**2 * np.sin(w*t) + fb
# f =  m*l*w**2 * np.cos(w*t) #- mu_s * fN 

def model(z, t, M): ## model equation

    fN = m*l*w**2 * np.sin(w*t) + fb #normal force
    f = m*l*w**2 * np.cos(w*t) #- sgn*ff

    if f > mu_s * fN: #slip mode
        ff = mu_k * fN if np.linalg.norm(z[1]) > 1e-3 else 0.
        Fx = f - np.sign(z[1])*ff
    else: #stick mode
        Fx = f

    x = z[0]
    x_tag= z[1]

    x_tag2= Fx/m
    return np.array([x_tag, x_tag2])

def model_F(z, t, M): ## model equation

    fN = m*l*w**2 * np.sin(w*t) + fb

    f = m*l*w**2 * np.cos(w*t) #- sgn*ff

    if f > mu_s * fN:
        ff = mu_k * fN if np.linalg.norm(z[1]) > 1e-3 else 0.
        Fx = f - np.sign(z[1])*ff
    else:
        Fx = f

    x = z[0]
    x_tag= z[1]

    x_tag2= Fx/m
    return Fx, fN, f

# initial condition
x0 = np.array([0., 0.0])
# Motor Amplitude
# A = 1.2*(10**(-3))*0.75 * 0.1
# object dimensions in meter
M = np.pi/4
# Initial motor angle
theta = 0.#np.pi/4# -(0)/4

X = odeint(model, x0, t, args=(M,)) #function that solve numerical differnt equation

Fx, fN, f = [], [], []
for x, tt in zip(X, t):
    Fxi, fNi, fi = model_F(x, tt, M)
    Fx.append(Fxi)
    fN.append(fNi)
    f.append(fi)

Fx = np.array(Fx)
fN = np.array(fN)
f = np.array(f)

# plt.figure(figsize = (7,3.5))
fig, axs = plt.subplots(2, figsize=(7,3.))

axs[0].plot(t, f, '-k', label = '$f$')
axs[0].plot(t, fN, '--k', label = '$f_N$')
axs[0].set_xlim([0,np.max(t)])
axs[0].set_xticks([])
# axs[0].set_yticks([-6e-5,0,6e-5])
# axs[0].set_yticklabels(['','$f_b$',''])
axs[0].set_ylabel('force', fontsize=14)
axs[0].legend(fontsize=14)


axs[1].plot(t, X[:,0], 'k')
axs[1].set_xticks([])
axs[1].set_xlabel('time', fontsize=14)
axs[1].set_xlim([0,np.max(t)])
# axs[1].set_ylim([0,0.04])
# axs[1].set_yticks([0, 0.02, 0.04])
# axs[1].set_yticklabels([0,'',''])
axs[1].set_ylabel('$x$', fontsize=14)
axs[1].grid()


# axs[2].plot(t, (f - mu_s*fN)/M)
# axs[2].set_xlim([0,1])

# plt.savefig('stsl_plt.png')
plt.show()