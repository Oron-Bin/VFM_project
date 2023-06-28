import pandas as pd


def shortest_way(num_1, num_2):
    """Function for finding the shortest motor path to destanation"""

    if abs(num_1 - num_2) < 180:
        return num_2 - num_1
    else:
        if num_1 > num_2:
            return abs(num_1 - num_2 - 360)
        else:
            return abs(num_1 - num_2) - 360


data = pd.read_csv('/home/roblab20/Desktop/videos/data_oron/data_oron_2023-06-26-15-57-28.csv')
pixel_factor = 1/2000

data.at[0, 'x_dot'] = 0
data.at[0, 'y_dot'] = 0
data.at[0, 'phi_dot'] = 0
data.at[0, 'Pos_x'] = 0
data.at[0, 'Pos_y'] = 0

data.at[len(data['Orientation'] -1 ), 'phi_dot'] = 0
data.at[len(data['Orientation'] -1 ), 'x_dot'] = 0
data.at[len(data['Orientation'] -1 ), 'y_dot'] = 0

# data.at[0, 'Motor'] = 0
data['Pos_x'] = data['Pos_x'] * pixel_factor
data['Pos_y'] = data['Pos_y'] * pixel_factor


# data.at[-1, 'phi_dot'] = 0
# data.at[-1, 'x_dot'] = 0
# data.at[-1, 'y_dot'] = 0

for i in range(1,len(data['Orientation']) -1):
    sub_phi = shortest_way(data['Orientation'][i],data['Orientation'][i-1])
    sub_x = (data['Pos_x'][i] - data['Pos_x'][i - 1])
    sub_y = (data['Pos_y'][i] - data['Pos_y'][i - 1])
    sub_time = data['Time'][i] - data['Time'][i-1]
    char_phi = sub_phi/sub_time
    char_x = sub_x / sub_time
    char_y = sub_y / sub_time
    data.at[i, 'phi_dot'] = char_phi
    data.at[i, 'x_dot'] = char_x
    data.at[i, 'y_dot'] = char_y
    data.at[i,'delta_teta'] = data.at[i+1, 'delta_teta']



data.to_csv('/home/roblab20/Desktop/oron.csv', index=False)





