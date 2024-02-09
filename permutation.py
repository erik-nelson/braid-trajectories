import numpy as np
from typing import List, Tuple


def start_end_permutations(start_positions: List[np.ndarray],
                           end_positions: List[np.ndarray],
                           direction: np.ndarray = np.array([1, 0])) -> Tuple[Tuple[int], Tuple[int]]:
  # Get distance for each start point along projection direction.
  start_distances = [np.dot(s, direction) for s in start_positions]
  assert len(set(start_distances)) == len(start_distances) # Ensure projected values are unique.
  start_order = np.argsort(start_distances)

  # Get distances for each end point along projection direction.
  end_distances = [np.dot(e, direction) for e in end_positions]
  assert len(set(end_distances)) == len(end_distances) # Ensure projected values are unique.
  end_order = np.argsort(end_distances)

  return (start_order, end_order)


# Given a set of 2D start positions and 2D end positions, return the braid permutation
# for this system. The permutation is found by projecting start positions onto an input
# direction vector and ordering them along the vector, then repeating this procedure for
# end positions and finding the resulting permutation from the start ordering to end ordering.
#
# Example:
#
#  Start configuration:         | End configuration:
#                               |
#     a                         |             c
#                    b          |        b
#                               |               
#            c                  |        
#                               |                 a
#   ------> direction           |  --------> direction
#
# In the start configuration, we have:
#     proj(a, direction) < proj(c, direction) < proj(b, direction)
#
# thus our starting agent configuration is [a, c, b]. In the end configuration, we have:
#     proj(b, direction) < proj(c, direction) < proj(a, direction)
#
# thus our end agent configuration is [b, c, a]. The permutation that takes [a, c, b] --> [b, c, a] is given by
# the output tuple (2, 1, 0), since:
# - The agent at starting index 0 (a) is sent to ending index 2
# - The agent at starting index 1 (c) is sent to ending index 1
# - The agent at starting index 2 (b) is sent to ending index 0
#      
def permutation_for_system(start_positions: List[np.ndarray], 
                           end_positions: List[np.ndarray], 
                           direction: np.ndarray = np.array([1, 0])) -> Tuple[int]:
  start_order, end_order = start_end_permutations(start_positions, end_positions, direction)
  permutation = [end_order[start_order[i]] for i in range(len(start_positions))]
  return tuple(permutation)