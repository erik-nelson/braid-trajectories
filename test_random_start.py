import braid_group
import braid
import itertools
import numpy as np
import optimize
import os
import permutation
import random
import sample
import shutil
import tqdm
import utils

# Clean output directory.
output_dir = 'random_start'
if os.path.exists(output_dir):
  shutil.rmtree(output_dir)
for d in itertools.product(['before', 'after'], ['images', 'gifs']):
  os.makedirs(f"{output_dir}/{d[0]}/{d[1]}")

# Sample a start and end position for each agent from a multivariate normal distribution.
num_agents = 5
print(f"Samping start/end positions for {num_agents} agents.")

# Cap start and end positions to lie on unit circle.
mean = [0, 0]
covariance = [[1, 0], [0, 1]]
start_positions = [row / np.linalg.norm(row) for row in np.random.multivariate_normal(mean, covariance, size=num_agents)]
end_positions = [row / np.linalg.norm(row) for row in np.random.multivariate_normal(mean, covariance, size=num_agents)]

print(f"Start positions: {np.array(start_positions)}")
print(f"End positions: {np.array(end_positions)}")

# Determine the resulting system permutation.
start_order, end_order = permutation.start_end_permutations(start_positions, end_positions)
P = permutation.permutation_for_system(start_positions, end_positions)
print(f"Got system permutation: {P}")

# Find braid words that match this permutation.
print("Searching for braid words that fit this permutation (this may take a minute)...")
words = sample.sample_braids(goal_permutation=P, stop_after_num_matches=1000)
print(f"Stopped after finding {len(words)} suitable braid words.")
words = random.sample(words, min(len(words), 20))
print(f"Sampled down to {len(words)} suitable braid words.")

# For each word, construct an initial, braid, optimize it, and display it.
num_timestamps = 75
for word in tqdm.tqdm(words):
  # Create an initial trajectory from the braid.
  print("Initializing trajectory...")
  next_braid = braid.Braid.Create(word=word, num_strands=num_agents)
  initial_trajectories = utils.BraidToTrajectory(braid=next_braid, 
                                                 num_timestamps=num_timestamps, 
                                                 num_segments=len(word.characters) + 1)

  # Attach the start and end positions for each agent to the braid.
  # TODO(erik): Util function for this.
  for i in range(num_agents):
    initial_trajectories[i].insert(0, start_positions[start_order[i]])
    initial_trajectories[i].append(end_positions[end_order[i]])

  # Optimize the trajectory.
  print("Optimizing trajectory...")
  optimized_trajectories = optimize.Optimize(initial_trajectories)

  # Save initial and optimized trajectory.
  print("Storing resulting figures...")
  word_str = word.__str__().replace(' ', '')
  word_str = word_str[:min(len(word_str), 100)]
  utils.PlotTrajectories3D(initial_trajectories, output_dir + '/before/images/' + word_str + '.png')
  utils.AnimateTrajectories(initial_trajectories, output_dir + '/before/gifs/' + word_str + '.gif')
  utils.PlotTrajectories3D(optimized_trajectories, output_dir + '/after/images/' + word_str + '.png')
  utils.AnimateTrajectories(optimized_trajectories, output_dir + '/after/gifs/' + word_str + '.gif')