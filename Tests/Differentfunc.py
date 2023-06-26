import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import scipy as scipy

# Define the physical equation
def equation(parameters, x, y, teta):
    a, c, d ,e  = parameters
    M = 14 *(10**-3) #kg
    F_c,tau_f, f_g_X, f_g_y = a ,c, d, e
    return (1/M*((x**2 + y**2))) * ((a * x * np.sin(teta)) - (y * np.cos(teta)) - (x * e - y * d) + c)

# Define the objective/loss function
def objective(parameters):
    predicted_output = equation(parameters, x_data, y_data, teta_data)
    error = predicted_output - output_data
    return error

# Read data from CSV file
data = pd.read_csv('/home/roblab20/Desktop/videos/data_oron/data_oron_2023-06-26-15-57-28.csv')
x_data = data['Pos_x'].values
y_data = data['Pos_y'].values
teta_data = data['Motor angle'].values
output_data = data['Orientation'].values

# Initial guess for the parameters
initial_parameters = np.random.rand(4)
# print(initial_parameters)
# Run the optimization
result = least_squares(objective, initial_parameters)

# Retrieve the optimized parameter values
optimized_parameters = result.x

# Analyze the results
# print("Optimized parameters:", optimized_parameters)

print('VIBRATION_FORCE:',optimized_parameters[0],'N')
print('gravitaion_force:',np.array(optimized_parameters[2],optimized_parameters[3]),'N')
print('torsional friction:',optimized_parameters[1],'Nm')
