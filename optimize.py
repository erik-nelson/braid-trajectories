import braid_group
import braid
import numpy as np
from scipy.optimize import minimize


N = 4 # Number of agents.
time_stamps = 21
g0 = braid_group.Generator(0) 
i1 = braid_group.InverseGenerator(1)
g2 = braid_group.Generator(2)
word = braid_group.Word(g0.Compose(i1).Compose(g0).Compose(g2))
braid = braid.Braid.Create(word=word, num_strands=N)

# Extract initial trajectories from braid.
initial_trajectories = []
for i in range(N):
  trajectory = []
  for t in range(time_stamps):
    pose = braid.Strand(i).AtTime(t / (time_stamps - 1))
    trajectory.append((pose[0], pose[1]))
  initial_trajectories.append(trajectory)

# Store start and end positions for hard constraints.
start_positions = []
end_positions = []
for i in range (N):
  start_positions.append(initial_trajectories[i][0])
  end_positions.append(initial_trajectories[i][-1])

# Define objective function: minimize total distance traveled
def objective_function(positions_flat):
    positions = positions_flat.reshape((N, time_stamps, 2))  # Reshape to 3D array
    total_distance = 0
    for i in range(N):
        for t in range(time_stamps - 1):
            total_distance += np.linalg.norm(positions[i, t] - positions[i, t+1])**2
    return total_distance

# Define constraints: No collisions
def collision_constraints(positions_flat):
    positions = positions_flat.reshape((N, time_stamps, 2))  # Reshape to 3D array
    constraint_values = []
    for t in range(time_stamps):
        for i in range(N):
            for j in range(i+1, N):
                constraint_values.append(np.linalg.norm(positions[i, t] - positions[j, t])**2 - 0.25)  # Collision constraint
    return np.array(constraint_values)

# Define constraints: Start and end positions
def start_end_constraints(positions_flat):
    positions = positions_flat.reshape((N, time_stamps, 2))  # Reshape to 3D array
    constraint_values = []
    for i in range(N):
        constraint_values.append(positions[i, 0, 0] - start_positions[i][0])  # Start positions (x-coordinate)
        constraint_values.append(positions[i, 0, 1] - start_positions[i][1])  # Start positions (y-coordinate)
        constraint_values.append(positions[i, -1, 0] - end_positions[i][0])   # End positions (x-coordinate)
        constraint_values.append(positions[i, -1, 1] - end_positions[i][1])   # End positions (y-coordinate)
    return np.array(constraint_values)

# Minimize the objective function subject to constraints
result = minimize(objective_function, np.array(initial_trajectories).flatten(), constraints=[
    {'type': 'ineq', 'fun': collision_constraints},
    {'type': 'eq', 'fun': start_end_constraints}
])

# Reshape the optimized positions
optimized_positions = result.x.reshape((N, time_stamps, 2))

# Print optimized positions
for i in range(N):
    print(f"Agent {i+1} positions: {optimized_positions[i]}")

# Print total distance traveled
print("Total distance traveled:", result.fun)


import utils
utils.PlotTrajectories3D(utils.BraidToTrajectory(braid, time_stamps), time_stamps, 'before.png')
utils.PlotTrajectories3D(optimized_positions, time_stamps, 'after.png')
