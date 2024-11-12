import os
import pandas as pd
import numpy as np

# from Tests.Vision.circledetection import distance

# Define the path to the folder containing the CSV files
folder_path = '/home/roblab20/Desktop/article_videos/data_full_algo'
# folder_path = '/home/roblab20/Desktop/article_videos/full_rec_data'

# Initialize empty lists to store the errors
orientation_errors = []
distance_errors = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Get the last value from the 'Orientation Error' and 'Distance Error' columns
        last_orientation_error = df['Orientation Error'].iloc[-1]
        # last_distance_error = np.abs(df['R_Desire'].iloc[-1] - df['Actual_R'].iloc[-1])
        last_distance_error = df['Distance Error'].iloc[-1]

        # Append the errors to the respective lists
        orientation_errors.append(last_orientation_error)
        # distance_errors.append(last_distance_error)
        distance_errors.append(last_distance_error/2)
# The lists 'orientation_errors' and 'distance_errors' now contain the last values from each file
print("Orientation Errors:", orientation_errors)
print("Distance Errors:", distance_errors)

# orientation_errors = [3, 4, 4, 4, 3, 5, 4, 5, 6, 3, 4, 2, 4, 4, 4, 5, 4, 4, 3, 5, 2, 5, 3, 5, 3, 0, 4, 6, 3, 2]
orientation_mean = np.mean(orientation_errors)
distance_mean = np.mean(distance_errors)
orientation_std = np.std(orientation_errors)
distance_std = np.std(distance_errors)

print("Orientation Errors:", orientation_mean)
print("Distance Errors:", distance_mean)


print("Orientation std", orientation_std)
print("Distance std:", distance_std)