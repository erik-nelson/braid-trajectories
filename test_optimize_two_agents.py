import braid_group
import braid
import optimize
import utils

num_agents = 2
num_timestamps = 21
g0 = braid_group.Generator(0) 
word = braid_group.Word(g0.Compose(g0).Compose(g0))
braid = braid.Braid.Create(word=word, num_strands=num_agents)

initial_trajectories = utils.BraidToTrajectory(braid, num_timestamps)

# Connect start locations to work space. Simulate 2 agent crossing scenario.
initial_trajectories[0].insert(0, (-1.0, 0.0))
initial_trajectories[1].insert(0, (0.0, -1.0))
num_timestamps += 1

# Connect end locations to work space. X positions ordered as 2, 1 (permutation induced by crossing).
initial_trajectories[0].append((1.0, 0.0))
initial_trajectories[1].append((0.0, 1.0))
num_timestamps += 1

optimized_trajectories = optimize.Optimize(initial_trajectories)

# Print optimized trajectories.
for i in range(num_agents):
    print(f"Agent {i+1} trajectory: {optimized_trajectories[i]}")

# Save initial and optimized trajectory.
word_str = word.__str__().replace(' ', '')
utils.PlotTrajectories3D(initial_trajectories, num_timestamps, 'two_agents/before_' + word_str + '.png')
utils.AnimateTrajectories(initial_trajectories, 'two_agents/before_' + word_str + '.gif')
utils.PlotTrajectories3D(optimized_trajectories, num_timestamps, 'two_agents/after_' + word_str + '.png')
utils.AnimateTrajectories(optimized_trajectories, 'two_agents/after_' + word_str + '.gif')