import numpy as np
from scipy.optimize import minimize
# import findangle
import matplotlib.pyplot as plt
import pandas as pd
import csv


# filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-07-01-22-51-22.csv'
filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-07-01-22-50-03.csv'
# filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-06-29-09-57-57.csv'
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

####loading data from angle test results



# Given constants

M = 14/1000 #kg


# Given measurements
pos_x = np.array(df['Pos_x'])
pos_y = np.array(df['Pos_y'])
x_dot = np.array(df['x_dot'])
y_dot = np.array(df['y_dot'])
phi_dot = -np.array(df['phi_dot'])
teta = np.array(np.deg2rad(df['Motor_angle']))[:-1]
time = np.array(df['Time'])
phi = np.array(df['Orientation'])
delta_t = time[1:]- time[:-1]

# print('delta t is',delta_t)

# Objective function with L=loss and conditions for the notmal force
def objective(x):
    f_C, tau_f = x
    #
    eq1= (x_dot[1:] - x_dot[:-1])/delta_t - (x[0]/M)*np.cos(teta)
    eq2 = (y_dot[1:] - y_dot[:-1])/delta_t - (x[0] / M) * np.sin(teta)
    eq3 = (phi_dot[1:]-phi_dot[:-1])/delta_t - (x[0] * ((pos_x[:-1])* np.sin(teta) - (pos_y[:-1]) * np.cos(teta)) + x[1]) / \
          (M * ((pos_x[:-1])**2 + (pos_y[:-1])**2)+ 0.00001)


    return  np.sum(np.square(eq3)) + np.sum(np.square(eq1)) + np.sum(np.square(eq2)) # return np.sum(np.square(eq2))
    # return  np.sum(np.square(eq3))
# Initial guess for C, K, and L
x0 = np.array([0.001, 1])

bounds = [(0.0001 , 10), (0.000001, 5)]
# Minimize the objective function
result = minimize(objective, x0, bounds=bounds)
# result = minimize(objective, x0, bounds=bounds, method='SLSQP')

# Extract the optimized values
f_c_opt , tau_f_opt = result.x

# Print the optimized values
print("Optimized values:")
print("f_C:", f_c_opt , 'N')
print("tau_f:", tau_f_opt, 'Nm')

# print(phi_dot[193])
# check if there is any connection:
print((phi_dot[193]) + delta_t[193]*(f_c_opt * ((pos_x[193])* np.sin(teta[193]) - (pos_y[193]) * np.cos(teta[193])) + tau_f_opt) / \
          (M * ((pos_x[193])**2 + (pos_y[193])**2)+ 0.00001))
print("##################")

#calculate the error between the path:


#calulate the error between the csv files


#calculate the error between the equations that optimized


def equations_of_motion(x, y, x_dot, y_dot, teta, delta_t, f_c, tau_f ,phi , phi_dot):
    x_acc = (f_c/M) * np.cos(teta)
    y_acc = (f_c/M) * np.sin(teta)
    phi_acc = (f_c * ((x) * np.sin(teta) - (y) * np.cos(teta))+ tau_f) / (M * ((x)**2 + (y)**2) + 0.00001)

    x_dot_new = x_dot + delta_t * x_acc
    y_dot_new = y_dot + delta_t * y_acc
    phi_dot_new = -phi_dot + delta_t * phi_acc

    x_new = x + delta_t * x_dot_new
    y_new = y + delta_t * y_dot_new
    phi_new = phi + delta_t * phi_dot_new

    return x_new , y_new , phi_new ,x_dot_new , y_dot_new , phi_dot_new
#
optimized_path_x = [pos_x[0]]
optimized_path_y = [pos_y[0]]
x_dot_new = x_dot[0]
y_dot_new = y_dot[0]
#
for i in range(len(delta_t)):
    x_new, y_new, phi_new ,x_dot_new , y_dot_new , phi_dot_new = equations_of_motion(optimized_path_x[-1], optimized_path_y[-1],
                                                              x_dot_new, y_dot_new, teta[i], delta_t[i],
                                                              f_c_opt, tau_f_opt, phi , phi_dot)

    optimized_path_x.append(x_new)
    optimized_path_y.append(y_new)
#
# print()
# print(optimized_path_x)
# print(pos_x.reshape(-1,1))
# print(optimized_path_y)

# Calculate the errors
phi_error = np.abs(phi - phi_new)
pos_x_error = np.abs(pos_x - np.array(optimized_path_x))
pos_y_error = np.abs(pos_y - np.array(optimized_path_y))

# Calculate the mean and maximum errors
mean_phi_error = np.mean(phi_error)
max_phi_error = np.max(phi_error)
mean_pos_x_error = np.mean(pos_x_error)
max_pos_x_error = np.max(pos_x_error)
mean_pos_y_error = np.mean(pos_y_error)
max_pos_y_error = np.max(pos_y_error)

print("Mean error between phi and phi_new:", mean_phi_error )
print("Maximum error between phi and phi_new:", max_phi_error)
print("Mean error between pos_x and x_new:", mean_pos_x_error)
print("Maximum error between pos_x and x_new:", max_pos_x_error)
print("Mean error between pos_y and y_new:", mean_pos_y_error)
print("Maximum error between pos_y and y_new:", max_pos_y_error)


#plotting:
phi_rad = np.radians(phi)
dx = np.cos(phi_rad)
dy = np.sin(phi_rad)

phi_rad_new = np.radians(phi_new)
dx_new = np.cos(phi_new)
dy_new = np.sin(phi_new)
#
plt.plot(pos_x, pos_y, marker='o', linestyle='-', color='b', label='Real Path')
plt.quiver(pos_x, pos_y, dx, dy, scale=15, color='r')
#
# # Plot the optimized path in green
plt.plot(optimized_path_x, optimized_path_y, marker='o', linestyle='-', color='g', label='Optimized Path')
plt.quiver(optimized_path_x, optimized_path_y, dx_new, dy_new, scale=15, color= 'r')  # Same orientation arrows for optimized path
#
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Real Path vs. Optimized Path')
plt.legend()
#
# # Set the limits for the plot
x_margin = 0.005
y_margin = 0.005
# x_min, x_max = min(min(pos_x), min(optimized_path_x)) - x_margin, max(max(pos_x), max(optimized_path_x)) + x_margin
# y_min, y_max = min(min(pos_y), min(optimized_path_y)) - y_margin, max(max(pos_y), max(optimized_path_y)) + y_margin
plt.xlim(-0.01, 0.1)
plt.ylim(-0.1, 0.15)
#
plt.grid()
plt.show()
