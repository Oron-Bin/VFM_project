import sympy as smp
import numpy as np
import matplotlib.pyplot as plt


x, y, teta, beta, phi, f_c, m, g, tau_f= smp.symbols('x y teta beta phi f_c m g tau_f')

#define the parameters of the system:
# f_c = 5
# m = 0.14
# g = 9.81

F = -(m*g)*smp.sin(beta)
F_x = F*smp.cos(phi)
F_y = F*smp.sin(phi)

eq1 = F_x + f_c*smp.cos(teta)
eq2 = F_y + f_c*smp.sin(teta)
eq3 = (f_c*(x*smp.sin(teta)-y*(smp.cos(teta))-(x*F_y - y*F_x )) +tau_f) / (m*(x**2 + y**2))

x_sols = smp.solve(eq1, teta)
y_sols = smp.solve(eq2, teta)
# print(x_sols[-1])
# print(y_sols[-1])
# sol = smp.solve([eq1, eq2], [teta])
# print(sol)
# sols = smp.solve([eq1, eq2], [teta, beta])
# print(sols)