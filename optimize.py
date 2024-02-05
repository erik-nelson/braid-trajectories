import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple

# Given an input set of num_agents x num_timestamps x 2 trajectories in 2D space, optimize them
# s.t. the start/end positions remain fixed, the trajectories don't collide with one another, and
# the trajectories each achieve their shortest path length. 
def Optimize(trajectories: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
  # Store start and end positions for hard constraints.
  start_positions = []
  end_positions = []
  num_trajectories = len(trajectories)
  num_timestamps = len(trajectories[0])
  for i in range (num_trajectories):
    start_positions.append(trajectories[i][0])
    end_positions.append(trajectories[i][-1])

  # Define objective function: minimize total distance traveled.
  def objective_function(positions_flat):
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    total_distance = 0
    for i in range(num_trajectories):
        for t in range(num_timestamps - 1):
            total_distance += np.linalg.norm(positions[i, t] - positions[i, t+1])**2
    return total_distance

  # Define constraints: No collisions.
  def collision_constraints(positions_flat):
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    constraint_values = []
    for t in range(num_timestamps):
        for i in range(num_trajectories):
            for j in range(i+1, num_trajectories):
                constraint_values.append(np.linalg.norm(positions[i, t] - positions[j, t])**2 - 0.25)  # Collision constraint
    return np.array(constraint_values)

  # Define constraints: Start and end positions.
  def start_end_constraints(positions_flat):
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    constraint_values = []
    for i in range(num_trajectories):
        constraint_values.append(positions[i, 0, 0] - start_positions[i][0])  # Start positions (x-coordinate)
        constraint_values.append(positions[i, 0, 1] - start_positions[i][1])  # Start positions (y-coordinate)
        constraint_values.append(positions[i, -1, 0] - end_positions[i][0])   # End positions (x-coordinate)
        constraint_values.append(positions[i, -1, 1] - end_positions[i][1])   # End positions (y-coordinate)
    return np.array(constraint_values)

  # Minimize the objective function subject to constraints.
  result = minimize(objective_function, np.array(trajectories).flatten(), constraints=[
    {'type': 'ineq', 'fun': collision_constraints},
    {'type': 'eq', 'fun': start_end_constraints}
  ])

  # Reshape the optimized positions.
  trajectories = result.x.reshape((num_trajectories, num_timestamps, 2))
  return trajectories