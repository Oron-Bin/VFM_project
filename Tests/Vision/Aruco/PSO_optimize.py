# import numpy as np
# import scipy.optimize as optimize
# import pandas as pd
#
# filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-06-29-10-21-11.csv'
# df = pd.read_csv(filename)
#
# def f(params, X, Y, teta, M):
#     fc, tau_f = params
#     denominator = M * (X**2 + Y**2)
#     numerator = fc * (X * np.sin(teta) - Y * np.cos(teta)) + tau_f
#
#     # Handle potential division by zero
#     with np.errstate(divide='ignore', invalid='ignore'):
#         result = np.divide(numerator, denominator)
#         result[~np.isfinite(result)] = 0  # Set non-finite values to 0
#
#     return np.sum(result)
#
# def optimize_params(X, Y, teta):
#     M = 14/1000
#
#     def objective(params):
#         return f(params, X, Y, teta, M)
#
#     initial_guess = [1, 1]
#     result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
#
#     if result.success:
#         fitted_params = result.x
#         print("Optimized Parameters: ", fitted_params)
#     else:
#         raise ValueError(result.message)
#
# # Example usage with your data
# X = df['Pos_x']
# Y = df['Pos_y']
# teta = df['Motor_angle']
#
# optimize_params(X, Y, teta)

import csv
import pandas as pd

filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-06-29-10-21-11.csv'

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

df = pd.read_csv(filename)
# print(df)


