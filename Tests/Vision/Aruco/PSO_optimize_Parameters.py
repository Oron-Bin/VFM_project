import numpy as np
from scipy.optimize import minimize
# import findangle
import matplotlib.pyplot as plt
import pandas as pd
import csv

# filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-07-01-22-51-22.csv'
filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-07-01-22-50-03.csv'
df = pd.read_csv(filename)


with open(filename, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read and store the header row
    converted_data = [header]  # Include the header row in the converted data

    for row in reader:
        converted_row = []
        for index, value in enumerate(row):
            if index == 0 and value != '':  # Convert only if it's the first column and not empty
                converted_row.append(float(value))
            else:
                converted_row.append(value)
        converted_data.append(converted_row)

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(converted_data)

df = pd.read_csv(filename).dropna()
# print(df)
####loading data from angle test results



# Given constants
# D1 = 0.016  # known value of D1
# D2 = 0.013  # known value of D2
M = 14/1000 #kg
delta_t= 0.15 #sec


# Given measurements
pos_x = np.array(df['Pos_x'])
pos_y = np.array(df['Pos_y'])
x_dot = np.array(df['x_dot'])
y_dot = np.array(df['y_dot'])
phi_dot = np.array(df['phi_dot'])
teta = np.array(np.deg2rad(df['Motor_angle']))[:-1]
# teta = teta[:-1]

# print(np.array(pos_x))

# F = np.array([0, 1.1, 3.58, 5.2, 6.8, 8.5, 9.9, 10.6])
# F=F*10

# Objective function with L=loss and conditions for the notmal force
def objective(x):
    # C, K, L = x
    f_C, tau_f = x

    # eq1= K * C * T1 - F * np.cos(T3 / 2) * (D1+2*D2*np.sin(T2 / 2)+D2*np.sin(T3 / 2)) - F * np.sin(T3 / 2) * (D2*np.cos(T3 / 2)+2*D2*np.cos(T2 / 2)+1.5*D2)
    # eq2 = K * T2 - F * np.cos(T3 / 2) * (D1+D2*np.sin(T3 / 2)) - F * np.sin(T3 / 2) * (D2*np.cos(T3 / 2)+2*D2)
    # eq3= K * T3 - F * np.cos(T3 / 2) * D1 - F * np.sin(T3 / 2) * D2

    # eq1= (x_dot[1:] - x_dot[:-1])/delta_t - (x[0]/M)*np.cos(teta)
    eq2 = (y_dot[1:] - y_dot[:-1])/delta_t - (x[0] / M) * np.sin(teta)
    # eq3 = (phi_dot[1:] - phi_dot[:-1])/delta_t - (x[0] * (pos_x * np.sin(teta) - pos_y * np.cos(teta)) + x[1]) / (M * (pos_x**2 + pos_y**2))
    # return np.sum(np.square(eq1))
    # return np.sum(np.square(eq1)) + np.sum(np.square(eq2)) + np.sum(np.square(eq3))
    return np.sum(np.square(eq2))
# Initial guess for C, K, and L
x0 = np.array([1, 3])
# x0 = np.array([1.0, 1.0, 0.5]).flatten()
# Bounds on C, K, and L
bounds = [(0.00001 , 1000), (0.1, 10)]
# Minimize the objective function
result = minimize(objective, x0, bounds=bounds)
# result = minimize(objective, x0, bounds=bounds, method='SLSQP')

# Extract the optimized values
f_c_opt , tau_f_opt = result.x

# Print the optimized values
print("Optimized values:")
print("f_C:", f_c_opt)
print("tau_f:", tau_f_opt)


print("##################")

#OPOSSITE DIRECTION

#
# import numpy as np
# from scipy.optimize import root
#
# def equation1(T1, T2, T3, F, D1, K, L):
#     return T3 - ( F * np.cos(T3 / 2) * D1 + F * np.sin(T3 / 2) * D2) / K
#     # return T3 - ( F * np.cos(T3 / 2) * 0.012 + F * np.sin(T3 / 2) * 0.01) / K
#
#
# def equation2(T1, T2, T3, F, D1,D2, K, L):
#     if T2*180/np.pi > 30:
#         #AFTER D2 CORECTION
#         return T2 - (F * np.cos(T3 / 2) * (D1+D2*np.sin(T3 / 2)) + F * np.sin(T3 / 2) * (D2*np.cos(T3 / 2)+2*D2) + F*np.sin(T2 / 2)*D2) / K
#
#     else:
#         return T2 - (F * np.cos(T3 / 2) * (D1+D2*np.sin(T3 / 2)) + F * np.sin(T3 / 2) * (D2*np.cos(T3 / 2)+2*D2)) / K
#
# def equation3(T1, T2, T3, F, D1,D2, K, L, C):
#     if T2*180/np.pi > 30:
#         return T1 - (F * np.cos(T3 / 2) * (D1+2*D2*np.sin(T2 / 2)+D2*np.sin(T3 / 2)) + F * np.sin(T3 / 2) * (D2*np.cos(T3 / 2)+2*D2*np.cos(T2 / 2)+1.5*D2) + F*np.sin(T2 / 2)*(D2*np.cos(T2 / 2)+1.5*D2)) / (K * C)
#     else:
#         return T1 - (F * np.cos(T3 / 2) * (D1+2*D2*np.sin(T2 / 2)+D2*np.sin(T3 / 2)) + F * np.sin(T3 / 2) * (D2*np.cos(T3 / 2)+2*D2*np.cos(T2 / 2)+1.5*D2)) / (K * C)
#
# # Define a function representing the system of equations in the form F(x) = 0
# def equations_to_solve(variables, F, D1, K, L, C):
#     n = len(F)  # Number of F values
#     f = np.zeros((3 * n,))  # Initialize the array to store the equation values
#     for i in range(n):
#         T1 = variables[i]
#         T2 = variables[n + i]
#         T3 = variables[2 * n + i]
#         f[i] = equation1(T1, T2, T3, F[i], D1, K, L)
#         f[n + i] = equation2(T1, T2, T3, F[i], D1,D2, K, L)
#         f[2 * n + i] = equation3(T1, T2, T3, F[i], D1,D2, K, L, C)
#     return f
#
# # Use scipy.optimize.root to find the solution for each value in F
# F_values = F  # List of F values
# D1 = 0.016
#
# K = 4.47145
# C = 6.14839
# # C = 7.46
# L = 0.415
# # K = 4.22
#
# guess = np.zeros((3 * len(F_values),))  # Initial guess for T1, T2, and T3
#
# result = root(equations_to_solve, guess, args=(F_values, D1, K, L, C))
#
#
#
# T1_values = result.x[:len(F_values)]
# T2_values = result.x[len(F_values):2 * len(F_values)]
# T3_values = result.x[2 * len(F_values):]
# print(T2)
# print("T1 values:", T1_values*180/np.pi)
# print("T2 values:", T2_values)
# print("T3 values:", T3_values)
#
#
#
# #
# # # Plotting T3 and T3b
# # plt.figure(figsize=(8, 6))
# # plt.plot(F_values/10, T3_values, marker='o', label='model')
# # plt.plot(F_values/10, T3, marker='o', label='recordings')
# # plt.xlabel('Force [N]')
# # plt.ylabel('Teta3')
# # plt.title('Finger angle3 - Model VS Recordings')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# #
# # # Plotting T2 and T2b
# # plt.figure(figsize=(8, 6))
# # plt.plot(F_values/10, T2_values, marker='o', label='model')
# # plt.plot(F_values/10, T2, marker='o', label='recordings')
# # plt.xlabel('Force [N]')
# # plt.ylabel('Teta2')
# # plt.title('Finger angle2 - Model VS Recordings')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# #
# # # Plotting T1 and T1b
# # plt.figure(figsize=(8, 6))
# # plt.plot(F_values/10, T1_values, marker='o', label='model')
# # plt.plot(F_values/10, T1, marker='o', label='recordings')
# # plt.xlabel('Force [N]')
# # plt.ylabel('Teta1')
# # plt.title('Finger angle1 - Model VS Recordings')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
#
# import matplotlib.pyplot as plt
#
# #####converting to radians:
# teta0_6=teta0_6*np.pi/180
# teta1_6=teta1_6*np.pi/180
# teta2_6=teta2_6*np.pi/180
# teta3_6=teta3_6*np.pi/180
# teta0_7=teta0_7*np.pi/180
# teta1_7=teta1_7*np.pi/180
# teta2_7=teta2_7*np.pi/180
# teta3_7=teta3_7*np.pi/180
# teta0_8=teta0_8*np.pi/180
# teta1_8=teta1_8*np.pi/180
# teta2_8=teta2_8*np.pi/180
# teta3_8=teta3_8*np.pi/180
# teta0_9=teta0_9*np.pi/180
# teta1_9=teta1_9*np.pi/180
# teta2_9=teta2_9*np.pi/180
# teta3_9=teta3_9*np.pi/180
# teta0_10=teta0_10*np.pi/180
# teta1_10=teta1_10*np.pi/180
# teta2_10=teta2_10*np.pi/180
# teta3_10=teta3_10*np.pi/180
#
# # variables = [teta0_6, teta1_6, teta2_6, teta3_6, teta0_7, teta1_7, teta2_7, teta3_7, teta0_8, teta1_8, teta2_8, teta3_8, teta0_9, teta1_9, teta2_9, teta3_9, teta0_10, teta1_10, teta2_10, teta3_10]
# #
# # for i in range(len(variables)):
# #     variables[i] = variables[i] * np.pi / 180
#
#
#
# # Create a single figure with subplots
# fig, axes = plt.subplots(3, 1, figsize=(8, 18))
#
# # Plotting T3 and T3b
# axes[0].plot(F_values/10, T3_values, marker='o', label='model')
# axes[0].plot(F_values/10, T3, marker='o', label='recordings')
# ### ploting all the lines:
# axes[0].plot(F_values/10,teta3_6[0:8], marker='o', label='6')
# axes[0].plot(F_values/10,teta3_7[0:8], marker='o', label='7')
# axes[0].plot(F_values/10,teta3_8[0:8], marker='o', label='8')
# axes[0].plot(F_values/10,teta3_9[0:8], marker='o', label='9')
# axes[0].plot(F_values/10,teta3_10[0:8], marker='o', label='10')
# ####
# axes[0].set_xlabel('Force [N]')
# axes[0].set_ylabel('Teta3')
# axes[0].set_title('Finger angle3 - Model VS Recordings')
# axes[0].legend()
# axes[0].grid(True)
#
# # Plotting T2 and T2b
# axes[1].plot(F_values/10, T2_values, marker='o', label='model')
# axes[1].plot(F_values/10, T2, marker='o', label='recordings')
# ### ploting all the lines:
# axes[1].plot(F_values/10,teta2_6[0:8], marker='o', label='6')
# axes[1].plot(F_values/10,teta2_7[0:8], marker='o', label='7')
# axes[1].plot(F_values/10,teta2_8[0:8], marker='o', label='8')
# axes[1].plot(F_values/10,teta2_9[0:8], marker='o', label='9')
# axes[1].plot(F_values/10,teta2_10[0:8], marker='o', label='10')
# ####
# axes[1].set_xlabel('Force [N]')
# axes[1].set_ylabel('Teta2')
# axes[1].set_title('Finger angle2 - Model VS Recordings')
# axes[1].legend()
# axes[1].grid(True)
#
# # Plotting T1 and T1b
# axes[2].plot(F_values/10, T1_values, marker='o', label='model')
# axes[2].plot(F_values/10, T1, marker='o', label='recordings')
# ### ploting all the lines:
# axes[2].plot(F_values/10,teta1_6[0:8], marker='o', label='6')
# axes[2].plot(F_values/10,teta1_7[0:8], marker='o', label='7')
# axes[2].plot(F_values/10,teta1_8[0:8], marker='o', label='8')
# axes[2].plot(F_values/10,teta1_9[0:8], marker='o', label='9')
# axes[2].plot(F_values/10,teta1_10[0:8], marker='o', label='10')
# ####
# axes[2].set_xlabel('Force [N]')
# axes[2].set_ylabel('Teta1')
# axes[2].set_title('Finger angle1 - Model VS Recordings')
# axes[2].legend()
# axes[2].grid(True)
#
# # Adjust the spacing between subplots
# plt.tight_layout()
#
# # Display the plots
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
# ##### ploter of finger scale
#
# def plot_lines(teta1, teta2, teta3, length, color):
#     for angle1, angle2, angle3 in zip(teta1, teta2, teta3):
#         # Compute the angles relative to angle1 and angle2
#         angle2_relative = angle1 + angle2
#         angle3_relative = angle2_relative + angle3
#
#         # Convert angles from degrees to radians
#         angle1_rad = np.deg2rad(angle1)
#         angle2_rad = np.deg2rad(angle2_relative)
#         angle3_rad = np.deg2rad(angle3_relative)
#
#         # Compute the x and y coordinates of the lines' endpoints
#         x1 = [0, length * np.cos(angle1_rad)]
#         y1 = [0, length * np.sin(angle1_rad)]
#
#         x2 = [x1[1], x1[1] + length * np.cos(angle2_rad)]
#         y2 = [y1[1], y1[1] + length * np.sin(angle2_rad)]
#
#         x3 = [x2[1], x2[1] + length * np.cos(angle3_rad)]
#         y3 = [y2[1], y2[1] + length * np.sin(angle3_rad)]
#
#         # Plot the lines
#         plt.plot(x1, y1, color=color, linestyle='-', marker='o', markersize=5, markerfacecolor='red',
#                  label='Line 1 ({} degrees)'.format(angle1))
#         plt.plot(x2, y2, color=color, linestyle='-', marker='o', markersize=5, markerfacecolor='red',
#                  label='Line 2 ({} degrees)'.format(angle2_relative))
#         plt.plot(x3, y3, color=color, linestyle='-', marker='o', markersize=5, markerfacecolor='red',
#                  label='Line 3 ({} degrees)'.format(angle3_relative))
#
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Lines at Different Angles')
#     plt.grid(True)
#     # plt.legend()
#
#
# # Usage example
# teta1 = T1_values*180/np.pi+90
# teta2 = T2_values*180/np.pi
# teta3 = T3_values*180/np.pi
# length = 1  # Length of the lines
#
# teta1_real = T1*180/np.pi+90
# teta2_real = T2*180/np.pi
# teta3_real = T3*180/np.pi
#
# # Create a new figure
# # plt.figure()
# #
# # # Plot the first set of lines in black color
# # plot_lines(teta1, teta2, teta3, length, 'k')
# #
# # # Plot the second set of lines in green color
# # plot_lines(teta1_real, teta2_real, teta3_real, length, 'g')
#
# indexes = [0, 1, 2, 3,4, 5,7]
#
# # Create a new figure
# plt.figure()
#
# # Plot the first set of lines for the chosen indexes
# plot_lines([teta1[i] for i in indexes], [teta2[i] for i in indexes], [teta3[i] for i in indexes], length, 'k')
#
# # Plot the second set of lines for the chosen indexes
# plot_lines([teta1_real[i] for i in indexes], [teta2_real[i] for i in indexes], [teta3_real[i] for i in indexes], length, 'g')
#
#
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Lines at Different Angles')
# plt.grid(True)
# plt.axis('equal')  # Set equal aspect ratio for x and y axes
# plt.show()