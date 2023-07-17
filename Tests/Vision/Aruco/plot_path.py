import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

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

pos_x = np.array(df['Pos_x'])
pos_y = np.array(df['Pos_y'])
phi = np.array(df['Orientation'])

phi_rad = np.radians(phi)

# delta_t = time[1:]- time[:-1]

dx = np.cos(phi_rad)
dy = np.sin(phi_rad)


plt.figure(figsize=(8, 6))
plt.plot(pos_x, pos_y, marker='o', linestyle='-', color='b')
plt.quiver(pos_x, pos_y, dx, dy, scale=15, color='r', label='Orientation')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Path Plot')
plt.legend()

x_margin = 0.005  # Extend the x-axis limits by 1 unit on each side
y_margin = 0.005  # Extend the y-axis limits by 1 unit on each side
plt.xlim(min(pos_x) - x_margin, max(pos_x) + x_margin)
plt.ylim(min(pos_y) - y_margin, max(pos_y) + y_margin)

plt.grid()
plt.show()

