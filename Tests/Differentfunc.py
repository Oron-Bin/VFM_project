import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import scipy as scipy

# F_c,tau_f, f_g = 0 , 0 , [0,0]
# Define the physical equation
def equation(parameters, x, y, teta):
    a, c, d ,e  = parameters
    M = 14 *(10**-3) #kg

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
F_c,tau_f, f_g = optimized_parameters[0], optimized_parameters[1], [optimized_parameters[2],optimized_parameters[3]]# Analyze the results
# print("Optimized parameters:", optimized_parameters)

print('VIBRATION_FORCE:',F_c,'N')
print('gravitaion_force:',f_g,'N')
print('torsional friction:',tau_f,'Nm')
