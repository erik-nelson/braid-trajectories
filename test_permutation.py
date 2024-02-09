import permutation
import numpy as np

# Permutation on 1-agent system -----------------------------------------------
start_positions = [np.array([0, 0])]
end_positions = [np.array([0, 0])]
p = permutation.permutation_for_system(start_positions, end_positions) 
assert p == (0,)

start_positions = [np.array([-12.8, 16.5])]
end_positions = [np.array([20.5, -14.8])]
p = permutation.permutation_for_system(start_positions, end_positions) 
assert p == (0,)

# Permutation on a 2-agent system ---------------------------------------------
start_positions = [np.array([-1, -1]), np.array([1, 1])]
end_positions = [np.array([-1, 1]), np.array([1, -1])]
p = permutation.permutation_for_system(start_positions, end_positions)
assert p == (0, 1)

# When viewed along the +y direction, the permutation changes.
p = permutation.permutation_for_system(start_positions, end_positions, direction=np.array([0, 1]))
assert p == (1, 0)

# Permutation on a 3-agent system ---------------------------------------------
start_positions = [np.array([-1, 0]), np.array([0, -1]), np.array([1, -1])]
end_positions = [np.array([2, 0]), np.array([0, 1]), np.array([1, 1])]
p = permutation.permutation_for_system(start_positions, end_positions)
assert p == (1, 2, 0)

# When viewed along a direction close to +y (slightly tilted), the permutation is
# the same for this system. Note that viewing along exactly the +y axis leads multiple
# start and end points to be projected to the same point along the direction vector,
# which is an error for us.
p = permutation.permutation_for_system(start_positions, end_positions, direction=np.array([0.05, 1]))
assert p == (1, 2, 0)