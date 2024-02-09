import numpy as np
from scipy.optimize import minimize
import time
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

  # Define callback that prints optimization status.
  def print_callback(x):
    print_callback.iteration += 1
    print_callback.last_time = print_callback.curr_time
    print_callback.curr_time = time.time()
    if print_callback.iteration % 5 == 0:
      np.set_printoptions(precision=2, threshold=10)
      iter_dt = print_callback.curr_time - print_callback.last_time
      total_dt = print_callback.curr_time - print_callback.start_time
      print(f"Iteration {print_callback.iteration} (iter_dt={iter_dt:.2f} (s), total_dt={total_dt:.2f} (s)):")
      print(f"  (soft) Total path length objective cost = {objective_function(x):.2f}")
      print(f"  (hard) Collision inequality constraint costs = {collision_constraints(x)}")
      print(f"  (hard) Start/end equality constraint costs = {start_end_constraints(x)}")
      print("")
  print_callback.iteration = 0
  print_callback.start_time = time.time()
  print_callback.curr_time = time.time()

  # Define objective function: minimize total distance traveled.
  def objective_function(positions_flat):
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    diffs = positions[:, :-1] - positions[:, 1:]
    total_distance = np.sum(np.linalg.norm(diffs, axis=2) ** 2)
    return total_distance

  # Define gradient of objective function. This just makes optimization a little faster,
  # since we won't be using finite differences.
  def grad_objective_function(positions_flat):
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    diff = positions[:, :-1] - positions[:, 1:]
    gradient = np.zeros_like(positions)
    gradient[:, :-1] += 2 * diff
    gradient[:, 1:] -= 2 * diff
    return gradient.flatten()

  # Define constraints: No collisions.
  def collision_constraints(positions_flat):
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    # Reshape positions array to make it compatible for vectorized operations.
    positions_reshaped = positions.transpose(1, 0, 2)  # Shape: (num_timestamps, num_trajectories, 2)
    # Compute pairwise distances between all agents at all timestamps.
    differences = positions_reshaped[:, :, None, :] - positions_reshaped[:, None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    # Create a mask to exclude self-distances and double-counting.
    mask = np.triu(np.ones((num_trajectories, num_trajectories), dtype=bool), k=1)
    # Calculate collision constraints for all timestamps.
    constraint_values = (distances**2 - 0.1)[..., mask]
    return np.concatenate(constraint_values)
    
  ''' TODO(erik): Implement (vectorized) collision constraint gradient to speed up computation...
  # Define gradient of collision constraints.
  def grad_collision_constraints(positions_flat):
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    gradient = np.zeros_like(positions_flat)
    for t in range(num_timestamps):
        for i in range(num_trajectories):
            for j in range(i+1, num_trajectories):
                diff = positions[i, t] - positions[j, t]
                distance = np.linalg.norm(diff)
                if distance <= 0.1:  # If trajectories are too close, penalize them
                    gradient[i*num_timestamps*2 + t*2 : i*num_timestamps*2 + t*2 + 2] += 2 * diff
                    gradient[j*num_timestamps*2 + t*2 : j*num_timestamps*2 + t*2 + 2] -= 2 * diff
    return gradient
  '''

  # Define constraints: Start and end positions.
  def start_end_constraints(positions_flat):  
    positions = positions_flat.reshape((num_trajectories, num_timestamps, 2))  # Reshape to 3D array
    # Compute start and end constraints for all trajectories.
    start_constraints = (positions[:, 0, :] - np.array(start_positions)).flatten()
    end_constraints = (positions[:, -1, :] - np.array(end_positions)).flatten()
    constraint_values = np.concatenate([start_constraints, end_constraints])
    return constraint_values

  # Minimize the objective function subject to constraints.
  result = minimize(fun=objective_function, 
                    x0=np.array(trajectories).flatten(), 
                    jac=grad_objective_function,
                    callback=print_callback,
                    constraints=[
                      {'type': 'ineq', 'fun': collision_constraints},#, 'jac': grad_collision_constraints},
                      {'type': 'eq', 'fun': start_end_constraints}
                    ],
                    options={'disp': True, 'maxiter': 100},
                    method='SLSQP')

  # Reshape the optimized positions.
  trajectories = result.x.reshape((num_trajectories, num_timestamps, 2))
  return trajectories