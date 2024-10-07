import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
CSV_FILE_PATH = "/home/roblab20/Desktop/article_videos/data_full_algo/data_2024-10-07-18-59-32.csv"

# Load the data from the CSV file
data = pd.read_csv(CSV_FILE_PATH)

# Calculate the time values
# Assuming the length of the data corresponds to the number of frames
num_frames = len(data)
time_values = [i / 20 for i in range(num_frames)]

# Plot the parameters
plt.figure(figsize=(12, 8))

# Plot each parameter
plt.subplot(3, 1, 1)
plt.plot(time_values, data['Control angle'] , label='Control Angle', color='b')
plt.title('Control Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg')
# plt.xlim(0, 360)  # Set x-axis limits from 0 to 6
# plt.ylim(0, -200)
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_values, data['Radius'], label='Radius', color='m')
plt.title('Pos of COM')
plt.xlabel('Time (s)')
plt.ylabel('Pos of COM')
plt.grid()
plt.legend()


plt.subplot(3, 1, 3)
plt.plot(time_values, data['Orientation Angle'], label='Orientation Angle', color='g')
plt.title('Orientation Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Orientation Angle (degrees)')
# plt.ylim(200, 50)
plt.grid()
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

