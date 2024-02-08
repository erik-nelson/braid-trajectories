import braid_group
import braid
import optimize
import utils

num_agents = 4
num_timestamps = 21
g0 = braid_group.Generator(0) 
i1 = braid_group.InverseGenerator(1)
g2 = braid_group.Generator(2)
word = braid_group.Word(g0.Compose(i1).Compose(g0).Compose(g2).Compose(g0.Inverse()).Compose(i1.Inverse()))
braid = braid.Braid.Create(word=word, num_strands=num_agents)

initial_trajectories = utils.BraidToTrajectory(braid, num_timestamps)

# Connect start locations to work space. X positions ordered as 0, 1, 2, 3.
# Y positions can be arbitrary.
initial_trajectories[0].insert(0, (-1.4, -0.5))
initial_trajectories[1].insert(0, (-0.6, 0.8))
initial_trajectories[2].insert(0, (1.2, 0.3))
initial_trajectories[3].insert(0, (1.4, -0.5))
num_timestamps += 1

# Connect end locations to work space. X positions ordered as 3, 1, 2, 0 (the permutation induced by the braid we are using).
# Y positions can be arbitrary.
initial_trajectories[0].append((0.7, 0.6))
initial_trajectories[1].append((0.0, -1.0))
initial_trajectories[2].append((0.5, 0.0))
initial_trajectories[3].append((-0.4, -0.5))
num_timestamps += 1

optimized_trajectories = optimize.Optimize(initial_trajectories)

# Print optimized trajectories.
for i in range(num_agents):
    print(f"Agent {i+1} trajectory: {optimized_trajectories[i]}")

# Save initial and optimized trajectory.
utils.PlotTrajectories3D(initial_trajectories, num_timestamps, 'before.png')
utils.AnimateTrajectories(initial_trajectories, 'before.gif')
utils.PlotTrajectories3D(optimized_trajectories, num_timestamps, 'after.png')
utils.AnimateTrajectories(optimized_trajectories, 'after.gif')