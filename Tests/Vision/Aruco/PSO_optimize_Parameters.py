import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

filename = '/home/roblab20/Desktop/videos/data_oron/data_oron_2023-06-29-10-21-11.csv'
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

df = pd.read_csv(filename)
# print(df)

stop_iter = 50

# print(X**2)

########### Objective function ###########

def Weight(x):
    f_c, tau_f = x[0], x[1]
    M = 14 / 1000  # gram
    Pos_x = df['Pos_x']
    Pos_y = df['Pos_y']
    teta = np.deg2rad(df['Motor_angle'])
    denominator = M * (Pos_x**2 + Pos_y**2)
    numerator = x[0] * (Pos_x * np.sin(teta) - Pos_y * np.cos(teta)) + x[1]
    result = np.divide(numerator, denominator)
    return result


def generate_valid_particles(n_particles, x_min, x_max):
    X = []
    count = 0
    while count < n_particles:
        x = np.random.rand(ndim, 1) * (x_max - x_min) + x_min
        X.append(x)
        count += 1

    X = np.array(X).reshape(n_particles,-1).T
    return X

def f(X):
    O = []
    for x in X.T:
        O.append(Weight(x))

    return np.array(O)

########### Define parameters ###########
# x = [L, R, t]
x_min = np.array([0.00001, 0.00001]).reshape(-1,1) #
x_max = np.array([10000, 10000]).reshape(-1,1) #
ndim = len(x_min)

#### Hyper-parameter of the algorithm ###
c1 = 0.1
c2 = 0.1
w = 0.8

# Create particles
n_particles = 100
np.random.seed(100)
# X = np.random.rand(ndim, n_particles) * (x_max - x_min) + x_min # Random particles within the bounds
X = generate_valid_particles(n_particles, x_min, x_max)
V = np.random.randn(ndim, n_particles) * 0.2 # Random initial velocity

# Initialize data
pbest = X # Initialize personal best as first generation
pbest_obj = f(X)
gbest = pbest[:, pbest_obj.argmin()] # Global best particles
gbest_obj = pbest_obj.min()

# plt.figure()
G = []

for j in range(100):
    plt.clf()
    # Update params
    r1, r2 = np.random.rand(2)
    V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
    X_temp = X + V
    for i in range(n_particles): # Check for constraints
        if np.all(X_temp[:,i] > x_min.reshape(-1,)) and np.all(X_temp[:,i] < x_max.reshape(-1,)) :
            X[:,i] = X_temp[:,i].copy()
    obj = f(X)
    pbest[:, np.where(pbest_obj >= obj)] = X[:, np.where(pbest_obj >= obj)]
    # pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()] # Update global minimum
    gbest_obj = pbest_obj.min()
    print('Iteration', j, 'f_c=' + str(gbest[0]), 'tau_f=' + str(gbest[1]))
    G.append(gbest_obj)

    if np.all(np.array(G[-stop_iter:])==G[-1]):
        break

print()
print('f_c = ' + str(round(gbest[0],3)) + ' N')
print('tau_f = ' + str(round(gbest[1],3)) + 'Nm')

