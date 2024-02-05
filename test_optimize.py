import braid_group
import braid
import optimize
import utils

num_agents = 4
num_timestamps = 21
g0 = braid_group.Generator(0) 
i1 = braid_group.InverseGenerator(1)
g2 = braid_group.Generator(2)
word = braid_group.Word(g0.Compose(i1).Compose(g0).Compose(g2))
braid = braid.Braid.Create(word=word, num_strands=num_agents)

initial_trajectories = utils.BraidToTrajectory(braid, num_timestamps)
optimized_trajectories = optimize.Optimize(initial_trajectories)

# Print optimized trajectories.
for i in range(num_agents):
    print(f"Agent {i+1} trajectory: {optimized_trajectories[i]}")

# Save initial and optimized trajectory.
utils.PlotTrajectories3D(initial_trajectories, num_timestamps, 'before.png')
utils.PlotTrajectories3D(optimized_trajectories, num_timestamps, 'after.png')