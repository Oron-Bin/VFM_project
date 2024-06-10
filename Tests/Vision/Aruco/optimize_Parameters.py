import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd
import csv

filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-08-28-14-17-35.csv'
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

# Given measurements
pos_x = np.array(df['Pos_x'])
pos_y = np.array(df['Pos_y'])
radius = np.array(np.sqrt(df['Pos_x'] ** 2 + df['Pos_y'] ** 2))
r_dot = np.array(np.sqrt(df['x_dot'] ** 2 + df['y_dot'] ** 2))
x_dot = np.array(df['x_dot'])
y_dot = np.array(df['y_dot'])
teta = np.array(np.deg2rad(df['delta_teta']))[:-1]
time = np.array(df['Time'])
num = time[1:] - time[:-1]
delta_t = 0.12
M = 18  # g

# Define system constants
fb = 0.1  # Example value for fb
m = 1.0  # Example value for m
l = 1.0  # Example value for l
w = 1.0  # Example value for w
lamda = 1.0  # Example value for lamda
R = 1.0  # Example value for R

# Define initial conditions
initial_phi = 0.0  # Example initial value for phi
initial_omega = 0.0  # Example initial value for omega
initial_r = radius[0]  # Starting radius
initial_vr = r_dot[0]  # Starting radial velocity

# Use the time array from the CSV file
t_values = time


def system_equations(t, x, f_c_values, F_k_values):
    phi, omega, r, vr = x
    f_c = np.interp(t, t_values, f_c_values)
    F_k = np.interp(t, t_values, F_k_values)
    F_k = F_k + fb * np.abs(r)

    amp_f_c = (m * l * (w ** 2)) / 1000.0 * lamda

    if amp_f_c <= F_k:
        # print('time static', t)
        dphi_dt = omega
        domega_dt = 0
        dr_dt = 0
        dvr_dt = 0
    else:
        F = f_c - F_k
        dphi_dt = omega
        domega_dt = 2 * F * r * np.sin(np.deg2rad(teta)) / (M * (R ** 2))
        dr_dt = vr
        dvr_dt = (F * np.cos(np.deg2rad(teta)) + M * r * omega ** 2) / M

    return [dphi_dt, domega_dt, dr_dt, dvr_dt]


# Define objective function
def objective(params):
    f_c_values, F_k_values = params[:len(t_values)], params[len(t_values):]

    x0 = [initial_phi, initial_omega, initial_r, initial_vr]
    sol = solve_ivp(lambda t, y: system_equations(t, y, f_c_values, F_k_values), [time[0], time[-1]], x0, t_eval=time)

    simulated_r = sol.y[2]
    error = np.sum((simulated_r - radius) ** 2)

    return error


# Initial guesses for f_c_values and F_k_values
initial_guesses = np.concatenate([np.ones(len(t_values)), np.ones(len(t_values))])

# Bounds and constraints (if any)
bounds = [(0, None)] * len(initial_guesses)

# Optimization
result = minimize(objective, initial_guesses, bounds=bounds, method='L-BFGS-B')

# Extract optimized values
optimized_f_c_values = result.x[:len(t_values)]
optimized_F_k_values = result.x[len(t_values):]

# Print results
print("Optimized f_c values:", optimized_f_c_values)
print("Optimized F_k values:", optimized_F_k_values)
