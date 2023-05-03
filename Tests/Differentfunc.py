#this is an auxiliary function that find the shortest way to go from angle 1 to angle 2

def diff_fun(x,y):
  if abs(x-y) <= 180:
      return y-x
  else:
    if x>y:
      return 360-abs(y-x)
    else:
      return abs(y-x)-360


print(diff_fun(-180,360))


# def shortest_motor_path(self, output):
#   """Function for finding the shortest motor path to destanation"""
#
#   if abs(self.last_angle - output) < 180:
#     return output - self.last_angle  ## - output
#   else:
#     if self.last_angle > output:
#       return abs(self.last_angle - output - 360)
#     else:
#       return abs(self.last_angle - output) - 360


#i checked - the same code

