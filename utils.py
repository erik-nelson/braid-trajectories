import braid
import braid_group
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Generates a set of agent trajectories for a given input braid. The output is an array of size:
#    num_agents x num_timestamps x 2
# where output[a][t][i] accesses the i'th coordinate at the t'th timestamp of the a'th agent.
def BraidToTrajectory(braid: braid.Braid, num_timestamps = 10) -> List[List[Tuple[float, float]]]:
  ts = np.linspace(0, 1, num_timestamps)
  trajectories = []
  num_agents = len(braid.strands)
  for i in range(num_agents):
    trajectory = []
    for t in ts:
      position = braid.Strand(i).AtTime(t)
      trajectory.append((position[0], position[1]))
    trajectories.append(trajectory)
  return trajectories

# Plot a set of input trajectories to an output file. This produces a 2x2 grid of subplots showing
# various cross sections of the (x, y, t) input trajectories.
def PlotTrajectories3D(trajectories: List[List[Tuple[float, float]]], 
                       num_timestamps = 10, 
                       save_file: str = 'braid_3d.png'):
  ts = np.linspace(0, 1, num_timestamps)
  num_agents = len(trajectories)
  xs = {agent_idx: [] for agent_idx in range(num_agents)}
  ys = {agent_idx: [] for agent_idx in range(num_agents)}

  for agent in range(num_agents):
    for position in trajectories[agent]:
      xs[agent].append(position[0])
      ys[agent].append(position[1])

  fig = plt.figure()
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
  colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

  # Set up 2D x-t cross section plot (top left corner).
  ax_xt = axes[0, 0]
  for i in range(num_agents):
    ax_xt.plot(xs[i], ts, color=colors[i])
  ax_xt.set_xlim([0, num_agents - 1])
  ax_xt.set_ylim([0, 1])
  ax_xt.set_xlabel('X')
  ax_xt.set_ylabel('Time')
  ax_xt.grid(True)
  plt.axis('equal')

  # Set up 2D y-t cross section plot (top right corner).
  ax_yt = axes[0, 1]
  for i in range(num_agents):
    ax_yt.plot(ys[i], ts, color=colors[i])
  ax_yt.set_ylim([0, 1])
  ax_yt.set_xlabel('Y')
  ax_yt.set_ylabel('Time')
  ax_yt.grid(True)
  plt.axis('equal')

  # Set up 2D x-y cross section plot (bottom right corner).
  ax_xy = axes[1, 0]
  for i in range(num_agents):
    ax_xy.plot(xs[i], ys[i], color=colors[i])
  ax_xy.set_xlim([0, num_agents - 1])
  ax_xy.set_xlabel('X')
  ax_xy.set_ylabel('Y')
  ax_xy.grid(True)
  plt.axis('equal')

  # Set up 3D plot (bottom right corner).
  axes[1, 1].set_visible(False)
  ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
  for i in range(num_agents):
    ax_3d.plot(xs[i], ys[i], ts, color=colors[i])
  ax_3d.set_xlabel('X')
  ax_3d.set_ylabel('Y')
  ax_3d.set_zlabel('Time')
  ax_3d.set_xlim([0, num_agents - 1])
  ax_3d.set_zlim([0, 1])
  ax_3d.set_box_aspect([1, 1, 3])

  plt.tight_layout()
  plt.savefig(save_file)