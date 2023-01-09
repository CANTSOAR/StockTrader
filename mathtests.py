import matplotlib.pyplot as plt
import numpy as np
import random
import math


def create_data(num_points, dimensions, rad = 5, center = [10, 10, 10]):
  r = [rad]
  theta = []
  phi = []
  
  theta_scale = (random.random() - .5) * 10
  phi_scale = (random.random() - .5) * 10

  for i in range(num_points):
    r.append(r[-1] + random.random() * (random.random() - .5))
    theta.append(0.1 * (theta_scale * i + (random.random() - .5)))
    phi.append(0.1 * (phi_scale * i + (random.random() - .5)))

  r = r[1:]
  x = []
  y = []
  z = []

  if dimensions == 3:
    for i in range(num_points):
      x.append(r[i] * math.cos(theta[i]) * math.sin(phi[i]) + center[0])
      y.append(r[i] * math.sin(theta[i]) * math.sin(phi[i]) + center[1])
      z.append(r[i] * math.cos(phi[i]) + center[2])

    return x, y, z

  if dimensions == 2:
    for i in range(num_points):
      x.append(r[i] * math.cos(theta[i]) + center[0])
      y.append(r[i] * math.sin(theta[i]) + center[1])

    return x, y


def create_vector_field(*system, show = True):
  vectortails = []
  vectorheads = []
  allpoints = []

  for axis in system:
    axis = np.array(axis)
    vectortails.append(axis[:-1])
    vectorheads.append(axis[1:] - axis[:-1])
    allpoints.append(axis)

  vectortails = np.array(vectortails)
  vectorheads = np.array(vectorheads)

  fig = plt.figure()
  if len(allpoints) - 2:
    plot = fig.add_subplot(111, projection="3d")
    plot.quiver(*vectortails,
                *vectorheads,
                color=color_gradient(len(vectortails[0])),
                length=1,
                arrow_length_ratio=0)
  else:
    plot = fig.add_subplot(111)
    plot.quiver(*vectortails,
                *vectorheads,
                scale=1,
                scale_units="xy",
                angles="xy",
                color=color_gradient(len(vectortails[0])))
  
  if show:
    plt.show()


def color_gradient(points):

  colors = []
  for point in range(points):
    colors.append((
      point / points,
      1 - point / points,
      1 - point / points,
    ))

  return colors


def simplify_vector_field(*system, scale=10, show = True):
  axis_bounds = []
  axis_positions = []
  axis_changes = []
  
  for axis in system:
    axis = np.array(axis)
    axis_bounds.append([min(axis), max(axis)])
    axis_positions.append(list(axis[:-1]))
    axis_changes.append(list(axis[1:] - axis[:-1]))

  threeD = True

  if len(axis_bounds) == 2:
    threeD = False
    axis_bounds.append([0, 0])
    axis_positions.append([0 for x in system[0]])
    axis_changes.append([0 for x in system[0][1:]])

  D = axis_positions.copy()
  R = axis_changes.copy()

  avgD = []  #domain of the simplified field
  avgR = []  #range of simple vector lenghts

  x_axis_range = axis_bounds[0][1] - axis_bounds[0][0]
  y_axis_range = axis_bounds[1][1] - axis_bounds[1][0]
  z_axis_range = axis_bounds[2][1] - axis_bounds[2][0]
  
  for i in range(scale):
    for j in range(scale):
      for k in range(scale):
        tempx = []
        tempy = []
        tempz = []

        toremove = []

        for val in range(len(D[0])):
          if axis_bounds[0][0] + x_axis_range * (i - .5) / (scale - 1) <= D[0][val] and D[0][val] <= axis_bounds[0][0] + x_axis_range * (i + .5) / (scale - 1) and axis_bounds[1][0] + y_axis_range * (j - .5) / (scale - 1) <= D[1][val] and D[1][val] <= axis_bounds[1][0] + y_axis_range * (j + .5) / (scale - 1) and axis_bounds[2][0] + z_axis_range * (k - .5) / (scale - 1) <= D[2][val] and D[2][val] <= axis_bounds[2][0] + z_axis_range * (k + .5) / (scale - 1):
            tempx.append(R[0][val])
            tempy.append(R[1][val])
            tempz.append(R[2][val])
            toremove.append(val)

        for val in toremove[::-1]:
          D[0].pop(val)
          D[1].pop(val)
          D[2].pop(val)

          R[0].pop(val)
          R[1].pop(val)
          R[2].pop(val)

        avgD.append([axis_bounds[0][0] + x_axis_range * i / (scale - 1), axis_bounds[1][0] + y_axis_range * j / (scale - 1), axis_bounds[2][0] + z_axis_range * k / (scale - 1)])
        if tempx:
          avgR.append([mean(tempx), mean(tempy), mean(tempz)])
        else:
          avgR.append([0, 0, 0])

  avgD = np.array(avgD).T.tolist()
  avgR = np.array(avgR).T.tolist()
  
  fig = plt.figure()

  if threeD:

    plot = fig.add_subplot(111, projection="3d")
    plot.quiver(*avgD,
                *avgR,
                length=1,
                arrow_length_ratio=0,
                color = color_gradient(len(avgD[0])))
  else:

    plot = fig.add_subplot(111)
    plot.quiver(*avgD[:-1],
                *avgR[:-1],
                scale=1,
                scale_units="xy",
                angles="xy",
                color = color_gradient(len(avgD[0])))

  if show:
    plt.show()


def predict(*system, input, timesteps = 1, scale = 10, give_updates = False, give_warnings = True):
  predicted_path = [np.array(input)]

  for step in range(timesteps):
    axis_bounds = []
    D = []
    R = []

    for axis in system:
      axis = np.array(axis)
      axis_bounds.append([int(min(axis)), int(max(axis))])
      D.append(list(axis[:-1]))
      R.append(list(axis[1:] - axis[:-1]))

    threeD = True

    if len(axis_bounds) == 2:
      threeD = False
      axis_bounds.append([0, 0])
      D.append([0 for x in system[0]])
      R.append([0 for x in system[0][1:]])
      input.append(0)

    i = input[0]
    j = input[1]
    k = input[2]

    local_x_change = []
    local_y_change = []
    local_z_change = []

    og_scale = scale
    give_warnings_temp = give_warnings

    while not local_x_change:
      if og_scale  / scale> 8 and give_warnings_temp:
        print("CAUTION, DATA SURROUNGING", i, j, k, "MIGHT NOT BE SUFFICIANT FOR ACCURATE PREDICTIONS WITH THIS SCALE (consider increasing data or lowering scale)")
        give_warnings_temp = False

      for val in range(len(D[0])):
        if i - .5 / scale <= D[0][val] and D[0][val] <= i + .5 / scale and j - .5 / scale <= D[1][val] and D[1][val] <= j + .5 / scale and k - .5 / scale <= D[2][val] and D[2][val] <= k + .5 / scale:
          local_x_change.append(R[0][val])
          local_y_change.append(R[1][val])
          local_z_change.append(R[2][val])

      if local_x_change:
        local_change = [mean(local_x_change), mean(local_y_change), mean(local_z_change)]
      else:
        scale /= 2

    scale = og_scale
    prediction = np.array(input) + np.array(local_change)

    if not threeD:
      prediction = prediction[:-1]

    if give_updates:
      print("given input", input, "the system is predicted to progress to", prediction, "in the next timestep")
      print(timesteps - step - 1, "timesteps remaining for final prediction")

    predicted_path.append(prediction)
    input = prediction.tolist()

    if not len(predicted_path) % 1000:
      print(len(predicted_path), "points predicted")

  return prediction, np.array(predicted_path).T.tolist()

def mean(list):
  return sum(list) / len(list)

#threesystem = create_data(3000, 3)
#twosystem = threesystem[:-1]
#
#create_vector_field(*threesystem)
#simplify_vector_field(*threesystem, scale = 21)
#prediction, path = predict(*twosystem, input = [15, 15], timesteps = 1000, scale = 20, give_warnings = False)
#create_vector_field(*path)