import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
CSV_FILE_PATH = "/home/roblab20/Desktop/data_from_gui/data_2.csv"

# Load the data from the CSV file
data = pd.read_csv(CSV_FILE_PATH)

# Calculate the time values
# Assuming the length of the data corresponds to the number of frames
num_frames = len(data)
time_values = [i / 15 for i in range(num_frames)]

# Plot the parameters
plt.figure(figsize=(12, 8))

# Plot each parameter
plt.subplot(4, 1, 1)
plt.plot(time_values, data['Center'] , label='Center', color='b')
plt.title('Center Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Center')
plt.grid()
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_values, data['Orientation Angle'], label='Orientation Angle', color='g')
plt.title('Orientation Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Orientation Angle (degrees)')
plt.grid()
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time_values, data['Percent'], label='Percent', color='r')
plt.title('Percent Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Percent (%)')
plt.grid()
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time_values, data['Rad Angle'], label='Rad Angle', color='m')
plt.title('Rad Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Rad Angle (radians)')
plt.grid()
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()




# CSV_FILE_PATH = "/home/roblab20/Desktop/oron_data_}.csv"
#
# # Create CSV file and write headers
# with open(CSV_FILE_PATH, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Center', 'Orientation Angle', 'Percent', 'Rad Angle'])