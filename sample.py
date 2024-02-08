import braid_group
from collections import deque
from typing import List, Tuple
import copy
import numpy as np

# When taking a step during a search in the space of braids, that step may make our braid
# more sorted, less sorted, or the same amount of sorted compared to a goal permutation.
# Do not take a step if it would make our braid more unsorted than this value, compared
# to the previous braid. Note that this value needs to be incremented by 2 to have any effect
# because a swap on two strands can only make unsortedness decrease by 2, stay the same, or 
# increase by 2.
UNSORTEDNESS_THRESHOLD = 0

# Helper function that computes a metric for how sorted a list is, compared to a 
# goal configuration. Given an input list and goal list, each of the same size and
# containing the same values but in different orders, compute:
#
#   sum_i | i - goal.index(values[i]) |
#
# In other words, for each value, how far is it away from its final destination?
#
def unsortedness(values: Tuple[int], goal: Tuple[int]) -> int:
  assert len(values) == len(goal)
  total = 0
  for i in range(len(values)):
    total += np.abs(i - goal.index(values[i]))
  return total


# Given a word on the braid group of `num_strands` strands, determine the permutation
# that this word induces.
def permutation_for_word(word: braid_group.Word, num_strands: int) -> Tuple[int]:
  permutation = list(range(num_strands))
  
  # Convert from single character to word if needed.
  if not isinstance(word, braid_group.Word):
    word = braid_group.Word(word)

  for c in word.characters:
    if isinstance(c, braid_group.Identity): continue
    # The i'th generator flips the i'th and i+1'th strands. We don't actually care if this
    # is a generator or an inverse generator.
    permutation[c.i], permutation[c.i+1] = permutation[c.i+1], permutation[c.i]
 
  return tuple(permutation)


# A class representing a path while breadth-first-searching through braid words. The
# path represents a sequence of generators and inverse generators composed against 
# one another (i.e. a word in the braid group), as well as a history of all permutations
# that have been achieved by sub-words in the past along this path.
class Path:
  def __init__(self, num_strands: int):
    # We start with the identity word (e.g. at the origin in the lattice of generators),
    # with the permutation for the identity word, i.e. the identity permutation.
    self.num_strands = num_strands
    self.word = braid_group.Identity()
    self.permutations = [permutation_for_word(self.word, self.num_strands)]

  def try_visit(self, character: braid_group.Character, goal: Tuple[int]) -> bool:
    next_word = self.word.Compose(character)
    p = permutation_for_word(next_word, self.num_strands)

    # Did this step achieve our goal permutation? If so don't check anything else.
    if p == goal:
      self.permutations.append(p)
      self.word = next_word
      return True

    # Don't visit this word if we have seen a word with the same permutation before.
    if p in self.permutations:
      return False

    # Don't visit this word if it is "more unsorted" than our previous state.
    if unsortedness(p, goal) - unsortedness(self.permutations[-1], goal) > UNSORTEDNESS_THRESHOLD:
      return False
    
    # Visit this word. We have not seen its permutation before.
    self.permutations.append(p)
    self.word = next_word
    return True



# Takes as input a "permutation" on N index values, where each value in the input tuple
# is in [0, N), and each value represents the permuted location of a braid strand that
# starts at that index. For example:
#
# (1, 2, 0) is a permutation on 3 elements that represents the function P s.t.:
# P(0) = 1, P(1) = 2, P(2) = 0
#
# In other words this example input list expresses the permutation:
#  [0 0 1]
#  [1 0 0] 
#  [0 1 0]
#
# This function then samples candidate braids that achieve the input permutation.
# Using the example permutation above, here are several such braids:
#
#  P(2)P(0)P(1)      P(2)P(0)P(1)
#   x   x   x         x   x   x
#    \ /    |         |    \ /
#     \     |         |     /
#    / \    |         |    / \
#   x   x   x         x   x   x
#   |    \ /           \ /    |
#   |     /             \     |
#   |    / \           / \    |
#   x   x   x         x   x   x
#   0   1   2         0   1   2
#
#  inv(g1) * g2      g2 * inv(g1)
#
# To do this we perform a breadth first tree search in the space of braid generators
# and inverse generators, stopping when we encounter a braid that matches the required
# permutation, or when subject to other early stopping conditions.
#
# TODO(erik): Configuration options we can add that would allow this algorithm to be
# more flexible / tunable:
# - Max unsortedness threshold.
# - Max number of times of visiting same permutation (currently tolerance=0).
# - Max search depth.
def sample_braids(permutation: Tuple[int]) -> List[braid_group.Word]:

  num_strands = len(permutation)

  # Candidate "directions" that we can take at each time include the generators and
  # their inverses.
  dirs = ([braid_group.Generator(i) for i in range(num_strands - 1)] + 
          [braid_group.InverseGenerator(i) for i in range(num_strands - 1)])

  # Breadth first search over generators and inverse generators for a word (a path) that
  # achieves the right permutation.
  paths = deque([Path(num_strands)])
  matches = []
  if paths[0].permutations[-1] == permutation:
    # Insert the identity braid if it matches the goal permutation.
    matches.append(paths[0].word)

  while paths:
    path = paths.popleft()

    for d in dirs:
      # Try to move along this direction.
      new_path = copy.deepcopy(path)
      if not new_path.try_visit(character=d, goal=permutation):
        continue

      # Check if moving in this direction got us to our goal permutation.
      if new_path.permutations[-1] == permutation:
        matches.append(new_path.word)
        continue
      
      paths.append(new_path)

  # Convert to words, in case any matches were a single character.
  return [braid_group.Word(match) for match in matches]