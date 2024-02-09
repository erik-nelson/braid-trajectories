import braid_group
import braid
import optimize
import sample
import tqdm
import utils

# We are hard coding start and end points. They induce the permutation (1, 2, 0)
# (i.e. by looking at the ordering of X values in the initial and final positions).
# Sample all valid braids with this permutation.
num_agents = 3
num_timestamps = 50
words = sample.sample_braids((1, 2, 0))
for word in tqdm.tqdm(words):
  next_braid = braid.Braid.Create(word=word, num_strands=num_agents)
  initial_trajectories = utils.BraidToTrajectory(next_braid, num_timestamps)

  # Connect start locations to work space. Simulate 3 agent crossing scenario.
  initial_trajectories[0].insert(0, (-1.0, 0.0))
  initial_trajectories[1].insert(0, (0.0, -1.0))
  initial_trajectories[2].insert(0, (1.0, -1.0))
  num_timestamps += 1

  # Connect end locations to work space. X positions ordered as 1, 2, 0 (permutation induced by crossing).
  initial_trajectories[0].append((2.0, 0.0))
  initial_trajectories[1].append((0.0, 1.0))
  initial_trajectories[2].append((1.0, 1.0))

  num_timestamps += 1

  optimized_trajectories = optimize.Optimize(initial_trajectories)

  # Print optimized trajectories.
  for i in range(num_agents):
    print(f"Agent {i+1} trajectory:\n{optimized_trajectories[i].T}")

  # Save initial and optimized trajectory.
  word_str = word.__str__().replace(' ', '')
  utils.PlotTrajectories3D(initial_trajectories, 'three_agents/before_' + word_str + '.png')
  utils.AnimateTrajectories(initial_trajectories, 'three_agents/before_' + word_str + '.gif')
  utils.PlotTrajectories3D(optimized_trajectories, 'three_agents/after_' + word_str + '.png')
  utils.AnimateTrajectories(optimized_trajectories, 'three_agents/after_' + word_str + '.gif')