import numpy as np
import matplotlib.pyplot as plt

x = np.round(np.linspace(-30,30,45)) #sample interval of points and round them
y = np.linspace(-30, 30, 45) #sample interval of points and round them

def path_generator(x,y): #generate a path
    path = []
    for i in x:
        point = [i,y[0]]
        path.append(point)
    for i in y:
        point = [x[-1],i]
        path.append(point)
    for i in reversed(x):
        point = [i,y[-1]]
        path.append(point)
    for i in reversed(y):
        point = [x[0],i]
        path.append(point)
    return path


t = np.arange(0,2*np.pi, 0.1)
y = -30*np.sin(t)**3
x = 30*np.cos(t)-10*np.cos(2*t)-4*np.cos(3*t)-2*np.cos(4*t)
x1 = x[:round(len(x)/2)].tolist()
x2 = x[round(len(x)/2):].tolist()
y1 = y[:round(len(y)/2)].tolist()
y2 = y[round(len(y)/2):].tolist()
y2.extend(y1)
x2.extend(x1)

# t = np.arange(0,2*np.pi, 0.1)
# x = 30*np.sin(t)
# y = 30*np.cos(t)
# mylist = [1,2,3,4]
# x = list(reversed(mylist))
# print(x)
# print(y)
plt.plot(y2,x2)
plt.show()

# print(path_generator(x,y))
